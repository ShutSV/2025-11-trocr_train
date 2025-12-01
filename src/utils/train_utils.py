import os
from datetime import datetime
from transformers import (TrOCRProcessor,
                          TrainerCallback,
                          VisionEncoderDecoderModel,
                          Seq2SeqTrainer,
                          Seq2SeqTrainingArguments,
                          default_data_collator,
                          )
from datasets import load_dataset, Dataset
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
    return Dataset.from_list(data)


def train_trocr_model(config: dict):
    init_status()
    model_name = config.get("model", "microsoft/trocr-small-handwritten")
    dataset_path = config.get("dataset_path", "./datasets")
    output_dir = os.path.join("./models", datetime.now().strftime("%Y%m%d_%H%M%S"))

    os.makedirs(output_dir, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    dataset = load_custom_dataset(dataset_path)

    # Функция предобработки данных
    def preprocess_function(examples):
        images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values

        labels = processor.tokenizer(
            examples["text"],
            padding="max_length",
            max_length=64,
            truncation=True
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    # Применяем предобработку
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(config.get("EPOCHS", 1)),
        per_device_train_batch_size=int(config.get("BATCH_SIZE", 2)),
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer,
        callbacks=[ProgressCallback()]
    )

    trainer.train()
    save_status({"status": "completed", "progress_pct": 100})

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
