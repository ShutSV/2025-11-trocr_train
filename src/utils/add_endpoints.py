"""
FastAPI эндпойнты:

POST /start_training - запуск обучения
POST /stop_training - остановка обучения
GET /training_status - получение статуса обучения
GET /checkpoints - список чекпоинтов

"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import json
import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)
from datasets import Dataset, load_metric
from PIL import Image
import os
import configparser
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TrOCR Training API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для управления обучением
training_process = None
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "current_step": 0,
    "total_steps": 0,
    "loss": None,
    "learning_rate": None,
    "status": "idle",
    "last_update": None
}


class TrainingConfig(BaseModel):
    model_name: str = "microsoft/trocr-base-printed"
    dataset_path: str
    output_dir: str = "./trocr-model"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    resume_from_checkpoint: Optional[str] = None


class TrainingResponse(BaseModel):
    message: str
    training_id: Optional[str] = None


def load_training_config(config_path: str = "training_config.ini") -> Dict[str, Any]:
    """Загрузка конфигурации из INI файла"""
    config = configparser.ConfigParser()
    config.read(config_path)

    return {
        "model_name": config.get("training", "model_name", fallback="microsoft/trocr-base-printed"),
        "dataset_path": config.get("training", "dataset_path", fallback="./dataset"),
        "output_dir": config.get("training", "output_dir", fallback="./trocr-model"),
        "epochs": config.getint("training", "epochs", fallback=3),
        "batch_size": config.getint("training", "batch_size", fallback=4),
        "learning_rate": config.getfloat("training", "learning_rate", fallback=5e-5),
        "warmup_steps": config.getint("training", "warmup_steps", fallback=500),
        "logging_steps": config.getint("training", "logging_steps", fallback=10),
        "eval_steps": config.getint("training", "eval_steps", fallback=100),
        "save_steps": config.getint("training", "save_steps", fallback=200),
    }


def load_dataset(dataset_path: str):
    """Загрузка и подготовка датасета"""
    # Здесь должна быть ваша логика загрузки датасета
    # Для примера возвращаем пустой датасет
    return Dataset.from_dict({
        "pixel_values": [],
        "labels": []
    })


def compute_metrics(eval_pred):
    """Вычисление метрик для валидации"""
    metric = load_metric("cer")
    logits, labels = eval_pred
    predictions = logits.argmax(-1)

    # Декодирование предсказаний и меток
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")

    pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)

    cer = metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


def training_function(config: TrainingConfig):
    """Функция обучения модели"""
    global training_status

    try:
        logger.info("Начало обучения TrOCR модели")

        # Загрузка процессора и модели
        processor = TrOCRProcessor.from_pretrained(config.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(config.model_name)

        # Настройка специальных токенов
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        # Загрузка датасета
        dataset = load_dataset(config.dataset_path)

        # Разделение на train/validation
        train_dataset = dataset
        eval_dataset = dataset  # В реальности нужно разделить

        # Аргументы обучения
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            predict_with_generate=True,
            report_to=None,
            push_to_hub=False,
            resume_from_checkpoint=config.resume_from_checkpoint,
        )

        # Создание тренера
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
        )

        # Колбэк для отслеживания прогресса
        class ProgressCallback:
            def __init__(self):
                self.current_step = 0
                self.total_steps = len(train_dataset) // config.batch_size * config.epochs

            def __call__(self, args, state, control, **kwargs):
                self.current_step = state.global_step
                training_status.update({
                    "current_step": state.global_step,
                    "total_steps": self.total_steps,
                    "current_epoch": state.epoch,
                    "total_epochs": config.epochs,
                    "loss": state.log_history[-1].get("loss", None) if state.log_history else None,
                    "learning_rate": state.log_history[-1].get("learning_rate", None) if state.log_history else None,
                    "last_update": datetime.now().isoformat()
                })

        trainer.add_callback(ProgressCallback())

        # Запуск обучения
        training_status.update({
            "is_training": True,
            "status": "training",
            "last_update": datetime.now().isoformat()
        })

        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

        # Сохранение модели
        trainer.save_model()
        processor.save_pretrained(config.output_dir)

        training_status.update({
            "is_training": False,
            "status": "completed",
            "last_update": datetime.now().isoformat()
        })

        logger.info("Обучение завершено успешно")

    except Exception as e:
        logger.error(f"Ошибка при обучении: {str(e)}")
        training_status.update({
            "is_training": False,
            "status": f"error: {str(e)}",
            "last_update": datetime.now().isoformat()
        })


@app.post("/start_training", response_model=TrainingResponse)
async def start_training(background_tasks: BackgroundTasks, resume: bool = False):
    """Запуск обучения модели"""
    global training_process, training_status

    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение уже запущено")

    # Загрузка конфигурации
    config_dict = load_training_config()
    config = TrainingConfig(**config_dict)

    if resume:
        # Поиск последнего чекпоинта
        checkpoints = [d for d in os.listdir(config.output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            config.resume_from_checkpoint = os.path.join(config.output_dir, latest_checkpoint)

    # Сброс статуса
    training_status.update({
        "is_training": True,
        "current_epoch": 0,
        "total_epochs": config.epochs,
        "current_step": 0,
        "total_steps": 0,
        "loss": None,
        "learning_rate": None,
        "status": "starting",
        "last_update": datetime.now().isoformat()
    })

    # Запуск обучения в фоне
    background_tasks.add_task(training_function, config)

    return TrainingResponse(
        message="Обучение запущено" + (" с последнего чекпоинта" if resume else ""),
        training_id="training_001"
    )


@app.post("/stop_training")
async def stop_training():
    """Остановка обучения"""
    global training_status

    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Обучение не запущено")

    # В реальной реализации здесь должна быть логика остановки обучения
    training_status.update({
        "is_training": False,
        "status": "stopped",
        "last_update": datetime.now().isoformat()
    })

    return {"message": "Обучение остановлено"}


@app.get("/training_status")
async def get_training_status():
    """Получение статуса обучения"""
    return training_status


@app.get("/checkpoints")
async def get_checkpoints():
    """Получение списка доступных чекпоинтов"""
    config_dict = load_training_config()
    output_dir = config_dict["output_dir"]

    if not os.path.exists(output_dir):
        return {"checkpoints": []}

    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    return {"checkpoints": sorted(checkpoints)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
