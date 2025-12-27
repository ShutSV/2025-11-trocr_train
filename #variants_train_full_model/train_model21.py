"""
скрипт тренировки модели microsoft-small-handwritten на кириллическом датасете
"""

from pathlib import Path
from datetime import datetime
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate

from dataset_rus21 import MODEL_NAME, processor, train_dataset, val_dataset


TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
OUTPUT_DIR = Path(rf"D:\DOC\2025-11-trocr_train\output\{TIMESTAMP}")
LOG_DIR = Path(rf"{OUTPUT_DIR}\logs")

model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.decoder.resize_token_embeddings(len(processor.tokenizer))

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Настройки для предсказания (валидации)
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = evaluate.load("cer")

def main():

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)  # Декодируем
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        # Просто печатаем примеры сразу здесь
        print(f"\nПримеры валидации (CER: {cer:.4f}):")
        for i in range(min(3, len(pred_str))):
            print(f"  True: '{label_str[i]}'")
            print(f"  Pred: '{pred_str[i]}'")
            print()
        return {"cer": float(cer)}

    # ==============================
    # TrainingArguments (i9 185H RTX4000ada)
    # ==============================

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        predict_with_generate=True,
        per_device_train_batch_size=64,  # 64 для RTX4000ada, 48 для T4 и L4, 96 для А100 (VRAM 26 из 40)
        per_device_eval_batch_size=64,  # 96 для RTX4000ada
        fp16=True,  # Используем смешанную точность для ускорения

        # --- настройки логирования ---
        logging_dir=str(LOG_DIR),
        logging_strategy="steps",
        logging_steps=50,  # Частое логирование для плавных графиков
        eval_strategy="steps",
        eval_steps=500,    # Частая валидация для мониторинга
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        report_to=["tensorboard"],

        # --- Гиперпараметры ---
        # num_train_epochs=10,  # Так как датасет бесконечный/потоковый, лучше задавать шаги, а не эпохи
        max_steps=300_000,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=200,
        lr_scheduler_type="linear",
        # label_smoothing_factor=0.1,  # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: борется с самоуверенностью в пробелах

        # --- Управление лучшей моделью ---
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,

        # --- параметры для улучшенного логирования ---
        logging_first_step=True,  # Логируем первый шаг
        logging_nan_inf_filter=False,  # Логируем все значения
        eval_accumulation_steps=3,  # Для стабильности оценки
        dataloader_pin_memory=torch.cuda.is_available(),  # Ускорение загрузки данных
        dataloader_num_workers=4,    # Параллельная загрузка - ОТКЛЮЧИТЬ для [i9 185H]
        dataloader_prefetch_factor=64,  # Количество батчей, загружаемых каждым worker'ом заранее - ОТКЛЮЧИТЬ для [i9 185H]
        remove_unused_columns=False,  # Для IterableDataset нужно явно указать, что не используется длина

        # --- Дополнительные параметры
        eval_delay=0,  # - валидация начинается сразу
        dataloader_drop_last=False,  # не отбрасывать последний батч
        disable_tqdm=False,  # Включить прогресс-бары
    )


    class FreezeEncoderCallback(TrainerCallback):
        def __init__(self, unfreeze_step=180):
            self.unfreeze_step = unfreeze_step
            self.is_unfrozen = False

        def on_step_begin(self, args, state, control, **kwargs):
            model = kwargs['model']

            # На самом первом шаге замораживаем энкодер
            if state.global_step == 0:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                print(f"❄️ {datetime.now().strftime('%Y-%m-%d_%H-%M')} Энкодер заморожен на первые {self.unfreeze_step} шагов.")

            # При достижении нужного шага — размораживаем
            if state.global_step >= self.unfreeze_step and not self.is_unfrozen:
                for param in model.encoder.parameters():
                    param.requires_grad = True
                self.is_unfrozen = True
                print(f"🔥 {datetime.now().strftime('%Y-%m-%d_%H-%M')} Шаг {state.global_step}: Энкодер разморожен. Начинаем полное дообучение.")


    class MemoryOptimizationCallback(TrainerCallback):
        def __init__(self):
            self.last_log_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            # Периодическая очистка кэша
            current_time = time.time()
            if current_time - self.last_log_time > 1200:  # Каждые 5 минут
                torch.cuda.empty_cache()
                self.last_log_time = current_time
                print(f"🧹 {datetime.now().strftime('%Y-%m-%d_%H-%M')} Очищена память GPU после 20 минут")

        def on_evaluate_end(self, args, state, control, **kwargs):
            # Принудительная очистка после валидации
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(f"🧹 {datetime.now().strftime('%Y-%m-%d_%H-%M')} Очищена память GPU после валидации")

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
            # self.writer = None  # Будет инициализирован при первом вызове
            self.writer = SummaryWriter(log_dir=str(Path(checkpoint_dir) / "logs"))  # Явная инициализация
            print(f"📝 Логгер примеров инициализирован в: {checkpoint_dir}/logs")

        def init_writer(self, logs):
            pass

        def on_evaluate(self, args, state, control, **kwargs):
            """Вызывается после каждого этапа валидации"""
            metrics = kwargs.get('metrics', {})
            global_step = state.global_step

                        # Получаем метрики
            cer = metrics.get('eval_cer', float('inf'))
            predictions = metrics.get('eval_predictions', [])
            labels = metrics.get('eval_labels', [])

            # # --- БЛОК ПЕЧАТИ В КОНСОЛЬ ---
            # if len(predictions) > 0:
            #     print(f"\n--- ПРИМЕРЫ НА ШАГЕ {global_step} (CER: {cer:.4f}) ---")
            #     for i in range(min(3, len(predictions))):
            #         p_ids = predictions[i]
            #         l_ids = labels[i]
            #
            #         # Заменяем -100 на pad_token_id, чтобы декодер не выдал ошибку
            #         p_ids[p_ids == -100] = self.processor.tokenizer.pad_token_id
            #         l_ids[l_ids == -100] = self.processor.tokenizer.pad_token_id
            #
            #         p_text = self.processor.decode(p_ids, skip_special_tokens=True)
            #         l_text = self.processor.decode(l_ids, skip_special_tokens=True)
            #
            #         print(f"ОЖИДАНИЕ: '{l_text}'")
            #         print(f"РЕАЛЬНОСТЬ: '{p_text}'")
            #         print("-" * 20)

            # Логируем CER в TensorBoard
            if self.writer:
                self.writer.add_scalar("Val/cer", cer, global_step)

            # Логируем примеры в TensorBoard (вкладка TEXT)
            if len(predictions) > 0 and self.writer:
                n_samples = min(self.num_samples, len(predictions))
                # Выбираем случайные индексы для разнообразия
                indices = random.sample(range(len(predictions)), n_samples)

                for i, idx in enumerate(indices):
                    curr_p_ids = predictions[idx]
                    curr_l_ids = labels[idx]

                    curr_p_ids[curr_p_ids == -100] = self.processor.tokenizer.pad_token_id
                    curr_l_ids[curr_l_ids == -100] = self.processor.tokenizer.pad_token_id

                    pred_text = self.processor.decode(curr_p_ids, skip_special_tokens=True)
                    true_text = self.processor.decode(curr_l_ids, skip_special_tokens=True)

                    self.writer.add_text(
                        f"Val/sample_{i}",
                        f"**True:** `{true_text}`  \n**Pred:** `{pred_text}`",
                        global_step
                    )
                self.writer.flush()

            # --- ЛОГИКА СОХРАНЕНИЯ ЛУЧШЕЙ МОДЕЛИ ---
            if cer < self.best_cer:
                self.best_cer = cer
                self.epochs_no_improve = 0
                best_model_dir = self.checkpoint_dir / f"best_cer_model_cer-{cer}"
                best_model_dir.mkdir(parents=True, exist_ok=True)

                kwargs['model'].save_pretrained(best_model_dir)
                self.processor.save_pretrained(best_model_dir)

                if args.local_rank in [-1, 0]:
                    print(f"🎯 Новый лучший CER: {cer:.4f}. Модель сохранена.")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    print(f"⚠️ CER не улучшается {self.epochs_no_improve} проверок подряд")

            if args.local_rank in [-1, 0]:
                print(f"Validation @ step {global_step} - CER: {cer:.4f} | Best: {self.best_cer:.4f}")

        def on_train_end(self, args, state, control, **kwargs):
            """Закрываем ресурсы при завершении"""
            if self.writer:
                self.writer.close()
                print("✅ Логгер TensorBoard закрыт")


    # Создаем наш новый колбэк (например, разморозка на 3000 шаге)
    freeze_callback = FreezeEncoderCallback(unfreeze_step=180)

    memory_callback = MemoryOptimizationCallback()

    callback = EnhancedValidationCallback(
        checkpoint_dir=OUTPUT_DIR,
        processor=processor,
        log_every=500,  # Логировать каждые 200 шагов
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
        callbacks=[callback, freeze_callback, memory_callback],
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
