# import os
# from pathlib import Path
import random
import torch
import multiprocessing as mp
import pandas as pd
import albumentations as A
from PIL import Image
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate
import numpy as np
import logging
import zipfile
import io
from collections import OrderedDict

# from src.utils.start_tensorboard import start_tensorboard


logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{TIMESTAMP}")
MODEL_CHECKPOINT = "microsoft/trocr-small-handwritten"
CUSTOM_LOADER_DATASET = "ImageNet"
VALIDATION_SPLIT_SIZE = 0.05
RANDOM_SEED = 42
final_csv_path = Path(rf"d:\datasets\rus\dataset_full_index.csv") # Путь к файлу датасета

LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")
MAX_CACHE_ZIP_FILES = 8


# --- Определяем пайплайн аугментаций --- Это сильный набор аугментаций для борьбы с переобучением
train_transforms = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Affine(translate_percent=0.0625, scale=0.1, rotate=5, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
])


# --- Класс датасета для больших данных ---
class BigCyrillicHandwrittenDataset(Dataset):
    def __init__(self, df, processor, transforms=None, max_target_length=128, max_cache_size=8):
        """
        :param df: DataFrame с колонками 'zip_path' и 'image_path' (внутренний)
        :param processor: TrOCRProcessor
        :param max_cache_size: Максимальное количество ZIP-архивов для кэширования в RAM
        """
        self.df = df
        self.processor = processor
        self.transforms = transforms
        self.max_target_length = max_target_length
        self.max_cache_size = max_cache_size
        self.cache = OrderedDict()  # Кэш для хранения открытых объектов ZipFile или их содержимого (байтов) . Используем OrderedDict для реализации LRU-поведения
        logging.info(f"Инициализирован BigDataset с кэшем на {max_cache_size} архивов.")

    def __len__(self):
        return len(self.df)

    def _get_archive_data(self, zip_path):
        """
        Загружает или извлекает данные архива из кэша (LRU-стратегия).
        Возвращает словарь: {внутреннее_имя_файла: байты_изображения}
        """
        zip_path_str = str(zip_path)

        if zip_path_str in self.cache:
            # 1. Кэш-Хит: Перемещаем в конец (самый свежий)
            self.cache.move_to_end(zip_path_str)
            return self.cache[zip_path_str]

        # 2. Кэш-Мис: Загружаем новый архив
        logging.info(f"Загрузка ZIP-архива в RAM: {zip_path_str}")

        # Проверяем лимит кэша
        if len(self.cache) >= self.max_cache_size:
            # LRU: Удаляем самый старый элемент (первый)
            lru_key, _ = self.cache.popitem(last=False)
            logging.warning(f"Кэш заполнен ({self.max_cache_size}). Удален самый старый архив: {lru_key}")

        # Загрузка всего содержимого ZIP в словарь байтов
        new_cache_entry = {}
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    # Фильтруем папки и не-изображения, если необходимо
                    if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        new_cache_entry[name] = zf.read(name)
        except Exception as e:
            logging.error(f"Ошибка при чтении или кэшировании ZIP {zip_path}: {e}")
            raise

        # Сохраняем новый архив в кэш
        self.cache[zip_path_str] = new_cache_entry
        return new_cache_entry

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        zip_path = row['zip_path']
        internal_file_name = row['image_path']
        text = row['text']

        # 1. Получаем кэшированные байты архива
        archive_data = self._get_archive_data(zip_path)

        # 2. Получаем байты нужного изображения
        if internal_file_name not in archive_data:
            logging.error(f"Файл {internal_file_name} не найден в кэшированном архиве {zip_path}")
            # Возвращаем None или поднимаем ошибку, в зависимости от желаемого поведения
            return self.__getitem__(
                random.randint(0, len(self.df) - 1))  # Простая стратегия: взять случайный другой

        image_bytes = archive_data[internal_file_name]

        # 3. Декодируем байты в объект PIL.Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 4. Применяем аугментации (как в вашем исходном коде)
        if self.transforms:
            image_np = self.transforms(image=np.array(image))['image']
            image = Image.fromarray(image_np)

        # 5. Обработка процессором (как в вашем исходном коде)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


print(f"\nЗагрузка модели '{MODEL_CHECKPOINT}'...")
processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)  # --- Загрузка модели и процессора ---
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # --- Синхронизация словаря и конфигурации --- Это КЛЮЧЕВОЙ шаг при дообучении на новом языке!
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.config.decoder.dropout = 0.3  # Настраиваем dropout для регуляризации
model.config.decoder.attention_dropout = 0.3
# model.config.encoder.dropout = 0.1

model = model.to(device)
print("\n✅ Модель и процессор загружены и сконфигурированы.")


df = pd.read_csv(final_csv_path)
train_df, eval_df = train_test_split(df, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED)
train_dataset = BigCyrillicHandwrittenDataset(df=train_df, processor=processor, transforms=train_transforms, max_cache_size=MAX_CACHE_ZIP_FILES)  # --- Создание экземпляров датасета ---
eval_dataset = BigCyrillicHandwrittenDataset(df=eval_df, processor=processor, max_cache_size=2)  # Валидация без аугментаций
print(f"\nДанные разделены с применением BigCyrillicHandwrittenDataset. Обучение: {len(train_dataset)}, Валидация: {len(eval_dataset)}")

print("✅ Dataset и аугментации определены.")

cer_metric = evaluate.load("cer")  # --- Подготовка метрики CER ---

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id  # Заменяем -100 на pad_token_id перед декодированием
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)  # Декодируем
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


print("\n✅ Функция для расчета метрик готова.")


def main(*args, **kwargs):
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        predict_with_generate=True,
        per_device_train_batch_size=1,  # 64 для RTX4000ada, 48 для T4 и L4, 96 для А100 (VRAM 26 из 40)
        per_device_eval_batch_size=1,  # 96 для RTX4000ada
        # fp16=True,  # Используем смешанную точность для ускорения

        # --- настройки логирования ---
        logging_dir=str(LOG_DIR),
        logging_strategy="steps",
        logging_steps=100,  # Частое логирование для плавных графиков
        eval_strategy="steps",
        eval_steps=500,    # Частая валидация для мониторинга
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        # report_to=["tensorboard"],

        # --- Гиперпараметры ---
        num_train_epochs=10,
        learning_rate=1e-6,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",

        # --- Управление лучшей моделью ---
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # --- параметры для улучшенного логирования ---
        logging_first_step=True,  # Логируем первый шаг
        logging_nan_inf_filter=False,  # Логируем все значения
        eval_accumulation_steps=5,  # Для стабильности оценки
        # dataloader_pin_memory=torch.cuda.is_available(),  # Ускорение загрузки данных
        dataloader_num_workers=2,    # Параллельная загрузка
    )

    class EnhancedValidationCallback(TrainerCallback):
        def __init__(self,
                     checkpoint_dir,
                     processor,
                     log_every=100,
                     num_samples=5,
                     early_stopping_patience=3):
            """
            callback для валидации и логирования

            :param checkpoint_dir: Директория для сохранения лучших моделей
            :param processor: Процессор для обработки текста
            :param log_every: Частота логирования (в шагах)
            :param num_samples: Количество примеров для визуализации
            :param early_stopping_patience: Терпение для ранней остановки
            """
            self.checkpoint_dir = Path(checkpoint_dir)
            self.processor = processor
            self.log_every = log_every
            self.num_samples = num_samples
            self.early_stopping_patience = early_stopping_patience
            self.best_cer = float('inf')
            self.epochs_no_improve = 0
            self.writer = None  # Будет инициализирован при первом вызове

        def init_writer(self, logs):
            """Ленивая инициализация SummaryWriter"""
            if self.writer is None and 'tensorboard' in logs:
                self.writer = logs['tensorboard']
                print(f"📊 Инициализирован логгер TensorBoard")

        def on_evaluate(self, args, state, control, **kwargs):
            """Вызывается после каждого этапа валидации"""
            metrics = kwargs.get('metrics', {})
            global_step = state.global_step

            # Пропускаем если не наш шаг логирования
            if global_step % self.log_every != 0:
                return

            # Инициализация логгера
            self.init_writer(kwargs.get('logs', {}))

            # Получаем метрики
            cer = metrics.get('eval_cer', float('inf'))
            predictions = metrics.get('eval_predictions', [])
            labels = metrics.get('eval_labels', [])

            # Логируем CER
            if self.writer:
                self.writer.add_scalar("Val/cer", cer, global_step)

            # Логируем примеры предсказаний
            if len(predictions) > 0 and self.writer:
                n_samples = min(self.num_samples, len(predictions))
                indices = random.sample(range(len(predictions)), n_samples)

                for i, idx in enumerate(indices):
                    pred_text = self.processor.decode(predictions[idx], skip_special_tokens=True)
                    true_text = self.processor.decode(labels[idx], skip_special_tokens=True)

                    if self.writer:
                        self.writer.add_text(
                            f"Val/sample_{i}",
                            f"True: {true_text}\nPred: {pred_text}",
                            global_step
                        )

            # Сохранение лучшей модели
            if cer < self.best_cer:
                self.best_cer = cer
                self.epochs_no_improve = 0

                # Создаем директорию если не существует
                best_model_dir = self.checkpoint_dir / "best_cer_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)

                # Сохраняем модель и процессор
                kwargs['model'].save_pretrained(best_model_dir)
                self.processor.save_pretrained(best_model_dir)

                if args.local_rank in [-1, 0]:  # Только для главного процесса
                    print(f"🎯 Новый лучший CER: {cer:.4f}. Модель сохранена в {best_model_dir}")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"⚠️ CER не улучшается {self.epochs_no_improve} эпох подряд")

            # Вывод в консоль
            if args.local_rank in [-1, 0]:
                print(f"Validation {datetime.now().strftime('%Y-%m-%d_%H-%M')} @ step {global_step} - CER: {cer:.4f} | Best CER: {self.best_cer:.4f}")

        def on_train_end(self, args, state, control, **kwargs):
            """Закрываем ресурсы при завершении"""
            if self.writer:
                self.writer.close()
                print("✅ Логгер TensorBoard закрыт")

    callback = EnhancedValidationCallback(
        checkpoint_dir=OUTPUT_DIR,
        processor=processor,
        log_every=200,  # Логировать каждые 200 шагов
        num_samples=5,  # Показывать 5 примеров
        early_stopping_patience=5  # Останавливать после 5 эпох без улучшений
    )

    # --- Инициализация Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[callback],
        data_collator=default_data_collator,
    )

    resume_training = bool(get_last_checkpoint(OUTPUT_DIR))
    trainer.train(resume_from_checkpoint=resume_training)

    # --- Финальное сохранение ---
    print(f"\n✅ Обучение завершено! {datetime.now().strftime('%Y-%m-%d_%H-%M')} Сохраняем лучшую модель...")
    trainer.save_model(str(OUTPUT_DIR / "best_model"))
    processor.save_pretrained(str(OUTPUT_DIR / "best_model"))
    print(f"🎉 Модель сохранена в: {OUTPUT_DIR / 'best_model'}")


if __name__ == "__main__":
    # --- Запуск обучения ---
    print(f"\n🚀 НАЧИНАЕМ ОБУЧЕНИЕ! {datetime.now().strftime('%Y-%m-%d_%H-%M')}")
    print(f"Логи TensorBoard будут доступны в: {LOG_DIR}")
    print("Для просмотра запустите: tensorboard --logdir=указанный_выше_путь")
    # Установите spawn метод для multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
