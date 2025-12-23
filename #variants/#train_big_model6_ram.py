from pathlib import Path
from datetime import datetime
import random
import torch
import numpy as np
from transformers import (
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

from datasets_rus import MODEL_NAME, processor, train_dataset, val_dataset


TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{TIMESTAMP}")
LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id,)
    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    return {"cer": cer_metric.compute(predictions=pred_str, references=label_str,), "wer": wer_metric.compute(predictions=pred_str, references=label_str,),}

def main():
    # ==============================
    # 4. TrainingArguments (i9 185H RTX4000ada)
    # ==============================

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        predict_with_generate=True,
        per_device_train_batch_size=64,  # 64 для RTX4000ada, 48 для T4 и L4, 96 для А100 (VRAM 26 из 40)
        per_device_eval_batch_size=96,  # 96 для RTX4000ada
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
        # dataloader_num_workers=4,    # Параллельная загрузка - ОТКЛЮЧИТЬ для [i9 185H]
    )

    class EnhancedValidationCallback(TrainerCallback):
        def __init__(self, checkpoint_dir, processor, log_every=100, num_samples=5, early_stopping_patience=3):
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
        eval_dataset=val_dataset,
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
    main()
