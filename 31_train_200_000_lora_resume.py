
import io, os, re, torch, logging, json, webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from pathlib import Path
from jiwer import cer, wer
from transformers import (VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)
from transformers.trainer_utils import get_last_checkpoint
from peft import (LoraConfig, get_peft_model, TaskType,)
from datetime import datetime




# =============================
# ДО main
# =============================
def setup_logging():
    log_file = LOG_DIR / f"training.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return log_file


class ValidationLogger(TrainerCallback):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.validation_logs = []
        self.best_cer = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Логируем в консоль с деталями
            logging.info(f"VALIDATION - Step {state.global_step}")
            # Сохраняем метрики
            self.validation_logs.append({"step": state.global_step, "metrics": metrics, "timestamp": datetime.now().isoformat()})
            # Сохраняем в JSON файл
            with open("validation_metrics.json", "w") as f:
                json.dump(self.validation_logs, f, indent=2)
            # Логируем улучшение CER
            if "eval_cer" in metrics and metrics["eval_cer"] < self.best_cer:
                self.best_cer = metrics["eval_cer"]
                logging.info(f"NEW BEST CER: {self.best_cer:.6f} (improvement)")
                best_model_dir = Path(rf"{OUTPUT_DIR}\best_cer_model_cer-{self.best_cer:.4f}")
                best_model_dir.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(best_model_dir))
                self.processor.save_pretrained(best_model_dir)
                logging.info(f"лучшая Модель сохранена в {best_model_dir}")
            logging.info("=" * 80)

class BeamOnCERCallback(TrainerCallback):
    def __init__(self, model, cer_threshold=0.6, num_beams=4):
        self.model = model
        self.cer_threshold = cer_threshold
        self.num_beams = num_beams
        self.enabled = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.enabled:
            return
        eval_cer = metrics.get("eval_cer")
        if eval_cer is None:
            return
        if eval_cer < self.cer_threshold:
            self.model.generation_config.num_beams = self.num_beams
            self.model.generation_config.length_penalty = 1.0
            self.enabled = True
            logging.info(f" Beam search ENABLED (num_beams={self.num_beams}, CER={eval_cer:.4f}, step={state.global_step})")




# датасеты =====================================================================
def normalize_text(text: str) -> str:
    # text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
            enc = self.processor.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt",)
            labels = enc.input_ids.squeeze(0)          # Tensor [seq_len]
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            return {"pixel_values": pixel_values, "labels": labels,}
        except Exception as e:
            logging.warning(f"Dataset error: {e}")
            return None


def filter_none(x):
    return x is not None


def get_wds_dataset(shards, processor, max_len=128, shuffle=True, limit=None):
    fixed = "file:" + shards.replace("\\", "/")
    transform = TrOCRTransform(processor, max_len)
    dataset = wds.WebDataset(fixed, shardshuffle=100 if shuffle else 0)
    if shuffle:
        dataset = dataset.shuffle(1_500_000)
    dataset = (dataset.decode().to_tuple("png", "txt").map(transform, handler=wds.warn_and_continue).select(filter_none))
    if limit is not None:
        dataset = dataset.with_epoch(limit)
    return dataset


def collate_fn_trOCR(batch):  # Коллатор, работающий с WebDataset
    pixel_values = torch.stack([item["pixel_values"] for item in batch])  # WebDataset формат (через TrOCRTransform)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    return {"pixel_values": pixel_values, "labels": labels}

# =============================
# main
# =============================
if __name__ == "__main__":
    ID_PROJECT = "2025-12-30_14-30_reduced_lora"  # Имя проекта

    MODEL_PATH = rf"D:\DOC\2025-11-trocr_train\input\trocr_cyr_ready"
    PROCESSOR_PATH = rf"D:\DOC\2025-11-trocr_train\input\trocr_cyr_processor"
    DATASET_DIR = Path(rf"D:\datasets\rus\datasets\wds_format")
    TRAIN_SHARDS = str(DATASET_DIR / "train-{000000..000044}.tar")
    VAL_SHARDS = str(DATASET_DIR / "val-{000000..000003}.tar")
    OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{ID_PROJECT}")  # Папка проекта
    TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
    LOG_DIR = Path(rf"{OUTPUT_DIR}\logs\{TIMESTAMP}")  # Директория для логов
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging()

    MAX_LEN = 128
    BATCH_SIZE = 64
    # GRAD_ACCUM = 2
    LR = 2e-4
    MAX_STEPS = 200_000
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # модель ======================================================================
    processor = TrOCRProcessor.from_pretrained(PROCESSOR_PATH, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.generation_config.num_beams = 1
    model.generation_config.length_penalty = 1.0
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.repetition_penalty = 1.2
    model.generation_config.do_sample = False
    model.generation_config.max_length = 64
    model.to(DEVICE)

    lora_config = LoraConfig(
        r=32,  # можно даже меньше
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            # self-attention
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
            # cross-attention (КРИТИЧНО)
            "encoder_attn.q_proj",
            "encoder_attn.k_proj",
            "encoder_attn.v_proj",
            "encoder_attn.out_proj",
        ],
    )
    model.decoder = get_peft_model(model.decoder, lora_config)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.decoder.print_trainable_parameters()

    # датасеты ======================================================================
    train_dataset = get_wds_dataset(TRAIN_SHARDS, processor, MAX_LEN, shuffle=True, )
    val_dataset = get_wds_dataset(VAL_SHARDS, processor, MAX_LEN, shuffle=False, limit=10000, )

    # метрики ======================================================================
    def decode_predictions(pred_ids):
            if not torch.is_tensor(pred_ids):
                pred_ids = torch.from_numpy(pred_ids)
            texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            return [normalize_text(t) for t in texts]

    def decode_labels(labels):
        if not torch.is_tensor(labels):
            labels = torch.from_numpy(labels)
        labels = labels.clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        texts = processor.batch_decode(labels, skip_special_tokens=True)
        return [normalize_text(t) for t in texts]

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        pred_texts = decode_predictions(pred_ids)
        gt_texts = decode_labels(label_ids)
       # Печатаем прямо в стандартный вывод ячейки
        print("\n" + "="*30)
        print(f"Validation samples at step:")
        for i in range(min(3, len(pred_texts))):
            msg = f"  Sample {i}: True: {gt_texts[i]}; Pred: {pred_texts[i]}; Match: {gt_texts[i] == pred_texts[i]}"
            print(msg)
            logging.info(msg) # Оставляем для записи в файл training.log
        print("="*30 + "\n")
        return {"cer": cer(gt_texts, pred_texts), "wer": wer(gt_texts, pred_texts), }

    # тренировка ===================================================================
    os.environ["WANDB_DISABLED"] = "true"

    training_args = Seq2SeqTrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            # gradient_accumulation_steps=GRAD_ACCUM,  # ОТКЛЮЧИТЬ
            learning_rate=LR,
            warmup_steps=6_000,
            max_steps=MAX_STEPS,
            bf16=True,  # bf16 [i9 185H с RTX 4000ada]
            # fp16_full_eval=True,  # для colab
            # gradient_checkpointing=True,  # Оптимизации памяти - ОТКЛЮЧИТЬ

            # Стратегии логирования
            logging_strategy="steps",
            logging_steps=300,
            logging_dir=str(LOG_DIR),

            # Стратегии сохранения
            save_strategy="steps",
            save_steps=3_000,
            save_total_limit=10,

            # стратегия валидации
            eval_strategy="steps",
            eval_steps=3_000,
            predict_with_generate=True,
            generation_max_length=MAX_LEN,
            eval_accumulation_steps=2,

            # Технические настройки
            disable_tqdm=False,
            remove_unused_columns=False,
            dataloader_pin_memory=torch.cuda.is_available(),
            dataloader_num_workers=4,  # УТОЧНИТЬ. было 4 для [i9 185H]
            dataloader_prefetch_factor=4,  # УТОЧНИТЬ. было 64 для [i9 185H]
        )

    validation_logger = ValidationLogger(model, processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_trOCR,
        tokenizer=None,
        compute_metrics=compute_metrics,
        callbacks=[validation_logger, BeamOnCERCallback(model, cer_threshold=0.6)],
    )

    logging.info("Starting training...")

    last_checkpoint = get_last_checkpoint(str(OUTPUT_DIR))
    if last_checkpoint:
        logging.info(f"Resume from {last_checkpoint}")
        print(f"Resume from {last_checkpoint}. Возобновление из Checkpoint займет от 30 минут (на A100)  до 1 часа (на L4)")
    else:
        logging.info(f"Training from scratch")
        print(f"Training from scratch")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Финальная валидация
    logging.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logging.info(f"Final metrics: {final_metrics}")

    # Сохраняем модель
    FINAL_PATH = Path(rf"{OUTPUT_DIR}\final")
    FINAL_PATH.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_PATH))
    processor.save_pretrained(str(FINAL_PATH))
    logging.info(f" Модель сохранена в: {str(FINAL_PATH)}. Training completed!")
