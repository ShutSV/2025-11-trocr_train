from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from transformers import (
    VisionEncoderDecoderModel,
    # TrOCRProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import evaluate

from datasets_rus import MODEL_NAME, processor, train_dataset, val_dataset



TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{TIMESTAMP}")
# LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    labels = np.where(
        labels != -100,
        labels,
        processor.tokenizer.pad_token_id,
    )

    pred_str = processor.batch_decode(
        predictions, skip_special_tokens=True
    )
    label_str = processor.batch_decode(
        labels, skip_special_tokens=True
    )

    return {
        "cer": cer_metric.compute(
            predictions=pred_str,
            references=label_str,
        ),
        "wer": wer_metric.compute(
            predictions=pred_str,
            references=label_str,
        ),
    }

def main():
    # ==============================
    # 4. TrainingArguments (i9 185H RTX4000ada)
    # ==============================

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=5e-5,
        warmup_steps=500,

        fp16=torch.cuda.is_available(),

        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        logging_steps=500,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",

        dataloader_num_workers=6,
        dataloader_persistent_workers=True,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
