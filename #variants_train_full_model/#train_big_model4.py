import os
import random
import torch
import pandas as pd
import albumentations as A
from PIL import Image
from pathlib import Path
from datetime import datetime
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

from src.utils.start_tensorboard import start_tensorboard
from src.utils.settings import settings_train


logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# пути
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{TIMESTAMP}")
MODEL_CHECKPOINT = "microsoft/trocr-small-handwritten"
VALIDATION_SPLIT_SIZE = 0.05
RANDOM_SEED = 42

train_csv_path = Path(rf"d:\datasets\rus\datasets\train.csv") # Путь к файлу train датасета
val_csv_path = Path(rf"d:\datasets\rus\datasets\val.csv") # Путь к файлу val датасета
images_dir_path = Path(rf"{settings_train.dataset_path}\images")  # Исходные файлы
LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")



# --- Запуск TensorBoard в Internet ---
start_tensorboard()  # как вариант:  start_cloudflare_tunnel()

# --- Определяем пайплайн аугментаций --- Это сильный набор аугментаций для борьбы с переобучением
train_transforms = A.Compose([
    A.Rotate(limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Affine(translate_percent=0.0625, scale=0.1, rotate=5, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
])


# --- Класс датасета ---
class CyrillicHandwrittenDataset(Dataset):
    def __init__(self, df, processor, root_dir, transforms=None, max_target_length=128):
        self.df = df
        self.processor = processor
        self.root_dir = root_dir
        self.transforms = transforms
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['image_path']
        text = row['text']
        image_path = self.root_dir / file_name
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logging.warning(f"Файл {image_path} не найден.")
        if self.transforms:  # Применяем аугментации, если они есть
            image_np = self.transforms(image=np.array(image))['image']
            image = Image.fromarray(image_np)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values  # Обработка изображения процессором
        labels = self.processor.tokenizer(  # Обработка текста процессором
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        # Для функции потерь заменяем padding-токены на -100
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

train_df = pd.read_csv(train_csv_path)  # Загружаем ГОТОВЫЙ DataFrame для train Dataset
eval_df = pd.read_csv(val_csv_path)  # Загружаем ГОТОВЫЙ DataFrame для val Dataset
train_dataset = CyrillicHandwrittenDataset(df=train_df, processor=processor, root_dir=images_dir_path, transforms=train_transforms)  # --- Создание экземпляров датасета ---
eval_dataset = CyrillicHandwrittenDataset(df=eval_df, processor=processor, root_dir=images_dir_path) # Валидация без аугментаций
print(f"\nДанные загружены. Обучение: {len(train_dataset)}, Валидация: {len(eval_dataset)}")
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
        per_device_train_batch_size=64,  # 64 для RTX4000ada, 48 для T4 и L4, 96 для А100 (VRAM 26 из 40)
        per_device_eval_batch_size=128,  # 96 для RTX4000ada
        fp16=True,  # Используем смешанную точность для ускорения

        # --- настройки логирования ---
        logging_dir=str(LOG_DIR),
        logging_strategy="steps",
        logging_steps=100,  # Частое логирование для плавных графиков
        eval_strategy="steps",
        eval_steps=500,    # Частая валидация для мониторинга
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to=["tensorboard"],

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
        dataloader_pin_memory=torch.cuda.is_available(),  # Ускорение загрузки данных
        dataloader_num_workers=8,    # Параллельная загрузка
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
    main()
