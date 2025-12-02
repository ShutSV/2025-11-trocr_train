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
import numpy as np
import evaluate
import pandas as pd
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
    # 1. Получение предсказаний и меток
    # pred.predictions: предсказанные ID токенов
    # pred.label_ids: истинные ID токенов (метки)

    # 2. Игнорирование токенов -100 в метках
    # (которые мы использовали для padding)
    label_ids = np.where(pred.label_ids != -100, pred.label_ids, tokenizer.pad_token_id)

    # 3. Декодирование ID в текст

    # Декодируем предсказания
    # skip_special_tokens=True пропускает [CLS], [SEP], [PAD] и т.д.
    pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

    # Декодируем истинные метки
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # 4. Расчет метрик

    # CER (Character Error Rate)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    # WER (Word Error Rate)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}

# Ручная загрузка датасета из папки с изображениями
def load_custom_dataset(data_dir):
    data = []
    # Предполагаемая структура: data_dir/images/ и data_dir/labels.csv
    labels_df = pd.read_csv(os.path.join(data_dir, "ramdisk_dataset.csv"))

    for _, row in labels_df.iterrows():
        image_path = os.path.join(data_dir, "images", row["filename"])
        if os.path.exists(image_path):
            data.append({
                "image_path": image_path,
                "text": row["text"]
            })
    full_dataset = Dataset.from_list(data)
    # Рекомендуется сразу разделить на обучение и валидацию (например, 90/10)
    # Это возвращает DatasetDict с ключами 'train' и 'test'
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset


def train_trocr_model(config: dict):
    init_status()
    model_name = config.get("model", "microsoft/trocr-small-handwritten")
    dataset_path = config.get("dataset_path", "./datasets")
    output_dir = os.path.join("./models", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Устанавливаем токен BOS для генерации
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Загружаем и разделяем датасет
    split_dataset = load_custom_dataset(dataset_path)

    # Функция предобработки данных
    def preprocess_function(examples):
        images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)

        labels_batch = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=64,  # Длина должна соответствовать вашей задаче
            truncation=True,
            return_tensors="pt"
        )
        # Замена padding токенов на -100. Это предотвращает вычисление потерь для padding-токенов
        labels = labels_batch.input_ids.squeeze(0)  # [1, seq_len] -> [seq_len]
        labels = torch.where(labels != processor.tokenizer.pad_token_id, labels, torch.tensor(-100))

        # 5. Возвращаем как numpy массивы или списки
        return {
            "pixel_values": pixel_values.numpy(),  # Конвертируем в numpy
            "labels": labels.numpy()  # Конвертируем в numpy
        }

    # Применяем предобработку
    tokenized_dataset = split_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=4,
        remove_columns=split_dataset["train"].column_names,
        desc="Preprocessing dataset"
    )
    tokenized_dataset.set_format(type="torch")

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
        evaluation_strategy="epoch",
        metric_for_best_model="eval_cer",
        logging_dir=os.path.join(output_dir, "logs"),
        save_strategy="epoch",
        load_best_model_at_end=True,  # Загрузить лучшую модель после обучения
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
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
