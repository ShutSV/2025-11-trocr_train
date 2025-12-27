

import io, re, gc, torch, logging, json
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
from PIL import Image
from pathlib import Path
from jiwer import cer, wer
from transformers import (VisionEncoderDecoderModel, TrOCRProcessor, TrainingArguments, Trainer, TrainerCallback)
from peft import (LoraConfig, get_peft_model, TaskType,)
from datetime import datetime

# =============================
# CONFIG
# =============================
MODEL_PATH = "input/trocr_cyr_ready"
PROCESSOR_PATH = "input/trocr_cyr_processor"
DATASET_DIR = Path(r"D:\datasets\rus\datasets\wds_format")
TRAIN_SHARDS = str(DATASET_DIR / "train-{000000..000044}.tar")
VAL_SHARDS   = str(DATASET_DIR / "val-{000000..000003}.tar")
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\trocr_cyr_lora\{TIMESTAMP}")
LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")

MAX_LEN = 128
BATCH_SIZE = 64
# GRAD_ACCUM = 4
LR = 1e-4
MAX_STEPS = 300_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================
def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)  # Создаем директорию для логов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"training_{timestamp}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_file

# =============================
# TEXT NORMALIZATION
# =============================
def normalize_text(text: str) -> str:
    # text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =============================
# DATASET TRANSFORM
# =============================
class TrOCRTransform:
    def __init__(self, processor, max_len=128):
        self.processor = processor
        self.max_len = max_len

    def __call__(self, sample):
        if sample is None:
            return None
        img_raw, text = sample
        try:
            if isinstance(img_raw, bytes):
                image = Image.open(io.BytesIO(img_raw)).convert("RGB")
            else:
                image = img_raw.convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]
            enc = self.processor.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len,)
            labels = [t if t != self.processor.tokenizer.pad_token_id else -100 for t in enc.input_ids]
            return {"pixel_values": pixel_values, "labels": torch.tensor(labels, dtype=torch.long),}
        except Exception:
            return None

def filter_none(x):
    return x is not None

def get_wds_dataset(shards, processor, max_len=128, shuffle=True):
    fixed = "file:" + shards.replace("\\", "/")
    transform = TrOCRTransform(processor, max_len)
    dataset = wds.WebDataset(fixed, shardshuffle=100 if shuffle else 0)
    if shuffle:
        dataset = dataset.shuffle(1_500_000)
    dataset = (dataset.decode().to_tuple("png", "txt").map(transform, handler=wds.warn_and_continue).select(filter_none))
    return dataset

# =======================
# DataLoader collate_fn
# =======================
def collate_fn_trOCR(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

# =============================
# КАСТОМНЫЙ CALLBACK ДЛЯ ЛОГИРОВАНИЯ
# =============================
class DetailedValidationLogger(TrainerCallback):
    def __init__(self, processor):
        self.processor = processor
        self.validation_logs = []
        self.best_cer = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Логируем в консоль с деталями
            logging.info("=" * 80)
            logging.info(f"VALIDATION - Step {state.global_step}")
            logging.info("-" * 80)
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.6f}")
            # Сохраняем метрики
            self.validation_logs.append({"step": state.global_step, "metrics": metrics, "timestamp": datetime.now().isoformat()})
            # Сохраняем в JSON файл
            with open("validation_metrics.json", "w") as f:
                json.dump(self.validation_logs, f, indent=2)
            # Логируем улучшение CER
            if "eval_cer" in metrics and metrics["eval_cer"] < self.best_cer:
                self.best_cer = metrics["eval_cer"]
                logging.info(f"🎉 {datetime.now().strftime('%Y-%m-%d_%H-%M')} NEW BEST CER: {self.best_cer:.6f} (improvement)")
                best_model_dir = Path(rf"{OUTPUT_DIR}\best_cer_model_cer-{self.best_cer}")
                best_model_dir.mkdir(parents=True, exist_ok=True)
                self.processor.save_pretrained(best_model_dir)
                logging.info(f"🎉 {datetime.now().strftime('%Y-%m-%d_%H-%M')} лучшая Модель сохранена в {best_model_dir}")

            logging.info("=" * 80)

# =============================
# КАСТОМНЫЙ CALLBACK ДЛЯ ОЧИСТКИ ВИДЕОПАМЯТИ
# =============================
class SmartMemoryCallback(TrainerCallback):
    def __init__(self, memory_threshold_gb=0.5):
        self.memory_threshold = memory_threshold_gb * 1024 ** 3  # Порог освобождения памяти в байтах
        self.last_memory = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        current_memory = torch.cuda.memory_allocated()
        if current_memory - self.last_memory > self.memory_threshold:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if logs is not None:
                logs["memory_cleaned"] = True
                logs["memory_before_cleanup_gb"] = current_memory / 1024 ** 3
            self.last_memory = torch.cuda.memory_allocated()

    def on_evaluate_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

# =============================
# LOAD PROCESSOR & MODEL
# =============================
processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)

# =============================
# LORA
# =============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"],
)
model.decoder = get_peft_model(model.decoder, lora_config)
for p in model.encoder.parameters():
    p.requires_grad = False
model.decoder.print_trainable_parameters()

# =============================
# DATASETS
# =============================
train_dataset = get_wds_dataset(TRAIN_SHARDS, processor, MAX_LEN, shuffle=True,)
val_dataset = get_wds_dataset(VAL_SHARDS, processor, MAX_LEN,shuffle=False,)

# =============================
# METRICS
# =============================
def decode_predictions(pred_ids):
    texts = processor.batch_decode(pred_ids, skip_special_tokens=True,)
    return [normalize_text(t) for t in texts]

def decode_labels(labels):
    labels = labels.clone()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    texts = processor.batch_decode(labels, skip_special_tokens=True,)
    return [normalize_text(t) for t in texts]

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = decode_predictions(pred_ids)
    label_str = decode_labels(labels_ids)
    #
    print(f"\nПримеры валидации (CER: {cer:.4f}):")  # Просто печатаем примеры сразу здесь
    for i in range(min(3, len(pred_str))):
        print(f"  True: '{label_str[i]}'")
        print(f"  Pred: '{pred_str[i]}'")
        print()
    # return {"cer": cer(label_str, pred_str), "wer": wer(label_str, pred_str)}

    metrics = {"cer": cer(label_str, pred_str), "wer": wer(label_str, pred_str)}  # Вычисляем метрики
    total_chars = sum(len(text) for text in label_str)  # Добавляем дополнительные метрики
    total_words = sum(len(text.split()) for text in label_str)
    metrics.update({"total_samples": len(label_str), "avg_chars_per_sample": total_chars / len(label_str), "avg_words_per_sample": total_words / len(label_str),})
    if len(pred_str) > 0:  # Логируем несколько примеров (первые 3)
        logging.info("Validation samples:")
        for i in range(min(3, len(pred_str))):
            logging.info(f"  Sample {i}:")
            logging.info(f"    True:  {label_str[i]}")
            logging.info(f"    Pred:  {pred_str[i]}")
            logging.info(f"    Match: {label_str[i] == pred_str[i]}")
    return metrics


if __name__ == "__main__":

    # Настраиваем логирование
    log_file = setup_logging()
    logging.info(f"Training started. Logs will be saved to: {log_file}")

    # =============================
    # TRAINING ARGS
    # =============================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        # warmup_steps=10_000,
        warmup_steps=200,
        # max_steps=MAX_STEPS,
        max_steps=2_000,
        fp16=True,

        # Стратегии логирования
        logging_strategy="steps",
        logging_steps=50,
        logging_dir="./logs/tensorboard",

        # Стратегии валидации
        eval_strategy="steps",
        # eval_steps=20_000,
        eval_steps=300,

        # Стратегии сохранения
        save_strategy="steps",
        # save_steps=20_000,
        save_steps=300,
        save_total_limit=3,

        # Детализация логирования
        log_level="info",
        log_level_replica="warning",
        logging_first_step=True,  # Логируем первый шаг
        logging_nan_inf_filter=False,  # Логируем все значения
        disable_tqdm=False,  # Видим прогресс-бар

        # Отчетность
        report_to="tensorboard",

        # Оптимизация
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # Технические настройки
        remove_unused_columns=False,
        eval_accumulation_steps=1,  # Для стабильности оценки
        dataloader_pin_memory=torch.cuda.is_available(),  # Ускорение загрузки данных
        dataloader_num_workers=4,
        dataloader_prefetch_factor=64,  # Количество батчей, загружаемых каждым worker'ом заранее - ОТКЛЮЧИТЬ для [i9 185H]
    )

    # =============================
    # TRAINER
    # =============================
    validation_logger = DetailedValidationLogger(processor)  # Создаем callback для логирования
    memory_callback = SmartMemoryCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_trOCR,
        tokenizer=None,
        compute_metrics=compute_metrics,
        callbacks=[validation_logger, memory_callback],
    )
    # =============================
    # TRAIN
    # =============================
    logging.info("Starting training...")
    trainer.train()

    # Финальная валидация
    logging.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logging.info(f"Final metrics: {final_metrics}")

    # Сохраняем модель
    FINAL_PATH = Path(rf"{OUTPUT_DIR}\final")
    FINAL_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_PATH))
    processor.save_pretrained(str(FINAL_PATH))
    logging.info(f"🎉 Модель сохранена в: {str(FINAL_PATH)}. Training completed!")
