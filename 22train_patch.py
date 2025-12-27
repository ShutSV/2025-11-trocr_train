

import io, re, torch
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
from PIL import Image
from pathlib import Path
from jiwer import cer, wer
from transformers import (VisionEncoderDecoderModel, TrOCRProcessor, TrainingArguments, Trainer,)
from peft import (LoraConfig, get_peft_model, TaskType,)

# =============================
# CONFIG
# =============================
MODEL_PATH = "input/trocr_cyr_ready"
PROCESSOR_PATH = "input/trocr_cyr_processor"
DATASET_DIR = Path(r"D:\datasets\rus\datasets\wds_format")
TRAIN_SHARDS = str(DATASET_DIR / "train-{000000..000044}.tar")
VAL_SHARDS   = str(DATASET_DIR / "val-{000000..000003}.tar")
OUTPUT_DIR = "trocr_cyr_lora"

MAX_LEN = 128
BATCH_SIZE = 64
# GRAD_ACCUM = 4
LR = 1e-4
MAX_STEPS = 300_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    # Просто печатаем примеры сразу здесь
    print(f"\nПримеры валидации (CER: {cer:.4f}):")
    for i in range(min(3, len(pred_str))):
        print(f"  True: '{label_str[i]}'")
        print(f"  Pred: '{pred_str[i]}'")
        print()
    return {"cer": cer(label_str, pred_str), "wer": wer(label_str, pred_str)}


if __name__ == "__main__":

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
        logging_steps=50,
        # save_steps=20_000,
        save_steps=300,
        # eval_steps=20_000,
        eval_steps=300,
        eval_strategy="steps",
        save_total_limit=3,
        remove_unused_columns=False,
        logging_first_step=True,  # Логируем первый шаг
        logging_nan_inf_filter=False,  # Логируем все значения
        eval_accumulation_steps=1,  # Для стабильности оценки
        dataloader_pin_memory=torch.cuda.is_available(),  # Ускорение загрузки данных
        dataloader_num_workers=4,
        dataloader_prefetch_factor=64,  # Количество батчей, загружаемых каждым worker'ом заранее - ОТКЛЮЧИТЬ для [i9 185H]
        report_to="none",
    )

    # =============================
    # TRAINER
    # =============================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_trOCR,
        # tokenizer=processor.tokenizer,
        tokenizer=None,
        compute_metrics=compute_metrics,
    )
    # =============================
    # TRAIN
    # =============================
    trainer.train()
