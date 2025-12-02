import albumentations as A
import os
from datetime import datetime
import torch
from transformers import (TrOCRProcessor,
                          TrainerCallback,
                          VisionEncoderDecoderModel,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          default_data_collator,
                          )
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
import evaluate
import pandas as pd
from pathlib import Path
from PIL import Image
from src.utils import init_status, save_status


class ProgressCallback(TrainerCallback):
    """Коллбэк для обновления статуса обучения"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        progress = (state.global_step / state.max_steps) * 100 if state.max_steps else 0
        data = {
            "status": "in_progress" if not state.is_world_process_zero or not state.is_finished else "completed",
            "epoch": state.epoch,
            "step": state.global_step,
            "progress_pct": round(progress, 2),
            "last_loss": logs.get("loss"),
        }
        save_status(data)


# Загрузка метрик CER и WER с помощью библиотеки evaluate
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(pred, tokenizer):
    """
    Вычисляет CER и WER на основе предсказаний модели.
    Args:
        pred (Seq2SeqPrediction): Объект, содержащий предсказанные ID и истинные ID меток.
        tokenizer: Токенизатор TrOCRProcessor.tokenizer.
    """
    # 1. Получение предсказаний и меток pred.predictions: предсказанные ID токенов, pred.label_ids: истинные ID токенов (метки)
    # 2. Игнорирование токенов -100 в метках (которые мы использовали для padding)
    label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)
    # 3. Декодирование ID в текст Декодируем предсказания. skip_special_tokens=True пропускает [CLS], [SEP], [PAD] и т.д.
    pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)  # Декодируем истинные метки
    # 4. Расчет метрик
    cer = cer_metric.compute(predictions=pred_str, references=label_str)  # CER (Character Error Rate)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)  # WER (Word Error Rate)
    return {"cer": cer, "wer": wer}


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
        if hasattr(idx, 'iloc'):  # ФИКС: Если пришел pandas Series, берем первое значение
            idx = idx.iloc[0]
        elif isinstance(idx, (list, tuple)):  # ФИКС: Если пришел список, берем первый элемент
            idx = idx[0]
        idx = int(idx)  # Конвертируем в int

        row = self.df.iloc[idx]
        file_name = row['filename']
        text = row['text']
        image_path = Path(self.root_dir) / file_name
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Внимание: Файл не найден {image_path}. Пропускаем.")
            return self.__getitem__((idx + 1) % len(self))  # Возвращаем следующий элемент, чтобы избежать ошибки
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
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels).unsqueeze(0)}


def train_trocr_model(config: dict):
    init_status()
    model_name = config.get("model", "microsoft/trocr-small-handwritten")
    output_dir = os.path.join("./models", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    # model
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id  # Устанавливаем токен BOS для генерации
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # model.config.vocab_size = model.config.decoder.vocab_size
    # model.config.decoder.dropout = 0.3  # Настраиваем dropout для регуляризации
    # model.config.decoder.attention_dropout = 0.3

    # dataset
    dataset_path = config.get("dataset_path", "./datasets")  # Предполагаемая структура: data_dir/images/ и data_dir/labels.csv
    IMAGES_DIR_PATH = Path(f"{dataset_path}/images")
    VALIDATION_SPLIT_SIZE, RANDOM_SEED = 0.05, 42
    df = pd.read_csv(os.path.join(dataset_path, "ramdisk_dataset.csv"))

    train_transforms = A.Compose([  # --- Пайплайн аугментаций --- Это сильный набор аугментаций для борьбы с переобучением
        A.Rotate(limit=5, p=0.7),
        A.Perspective(scale=(0.01, 0.05), p=0.3),
        A.OneOf([A.MotionBlur(blur_limit=5, p=1.0), A.GaussianBlur(blur_limit=5, p=1.0),], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.ImageCompression(quality_lower=80, quality_upper=99, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    ], p=1.0)

    train_df, eval_df = train_test_split(df, test_size=VALIDATION_SPLIT_SIZE, random_state=RANDOM_SEED)
    train_dataset = CyrillicHandwrittenDataset(df=train_df, processor=processor, root_dir=IMAGES_DIR_PATH, transforms=train_transforms)  # --- Создание экземпляров датасета ---
    eval_dataset = CyrillicHandwrittenDataset(df=eval_df, processor=processor, root_dir=IMAGES_DIR_PATH)  # Валидация без аугментаций

    # Инициализация метрик (CER/WER)
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")


    def compute_metrics_wrapper(pred):
        """Обертка для передачи токенизатора в основную функцию метрик."""
        # 1. Игнорирование токенов -100 в метках
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        # 2. Декодирование ID в текст
        pred_str = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # 3. Расчет метрик
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer, "wer": wer}


    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.get("epochs", 1)),
        remove_unused_columns=False,
        per_device_train_batch_size=int(config.get("batch_size", 2)),
        eval_strategy="epoch",
        metric_for_best_model="eval_cer",
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="epoch",
        load_best_model_at_end=True,  # Загрузить лучшую модель после обучения
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[ProgressCallback()],
        compute_metrics=compute_metrics_wrapper,
    )

    print(f"--- Запуск обучения --- {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    trainer.train()

    final_model_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_model_dir)
    processor.save_pretrained(final_model_dir)
    save_status({"status": "completed", "progress_pct": 100})
    print(f"--- Обучение завершено. Модель сохранена в: {final_model_dir} ---")
