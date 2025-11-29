from pydantic_settings import BaseSettings, SettingsConfigDict, Field
from typing import Optional, Dict, Any
import torch


class GlobalSettingsOCR(BaseSettings):
    """Глобальные настройки inference (только для чтения)"""
    # MODEL_PATH: str = "microsoft/trocr-small-handwritten"
    MODEL_PATH: str = Field(default="microsoft/trocr-small-handwritten", env="MODEL_PATH")
    DEVICE: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = SettingsConfigDict(
        env_file=".env_ocr",
        env_file_encoding="utf-8"
    )


class GlobalSettingsTrainOCR(BaseSettings):
    """Глобальные настройки обучения"""
    TRAIN_MODEL_PATH: str = "microsoft/trocr-small-handwritten"
    TRAIN_DEVICE: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_BATCH_SIZE: int = 4
    TRAIN_EPOCHS: int = 3
    TRAIN_LEARNING_RATE: float = 5e-5
    TRAIN_OUTPUT_DIR: str = "./models"
    TRAIN_DATASET_PATH: str = "./datasets"

    model_config = SettingsConfigDict(
        env_file=".env_train",
        env_file_encoding="utf-8",
        extra='ignore'
    )


class SessionSettingsOCR:
    """Настройки текущей сессии inference"""

    def __init__(self):
        self.model_path: Optional[str] = None
        self.device: Optional[str] = None

    def update(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device

    def get_model_path(self):
        return self.model_path or global_settings_ocr.model_path

    def get_device(self):
        return self.device or global_settings_ocr.device


class SessionSettingsTrain:
    """Настройки текущей сессии обучения"""

    def __init__(self):
        self.model_path: Optional[str] = None
        self.device: Optional[str] = None
        self.batch_size: Optional[int] = None
        self.epochs: Optional[int] = None
        self.learning_rate: Optional[float] = None
        self.output_dir: Optional[str] = None
        self.dataset_path: Optional[str] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_training_config(self) -> Dict[str, Any]:
        global_train = GlobalSettingsTrainOCR()
        return {
            "model_path": self.model_path or global_train.TRAIN_MODEL_PATH,
            "device": self.device or global_train.TRAIN_DEVICE,
            "batch_size": self.batch_size or global_train.TRAIN_BATCH_SIZE,
            "epochs": self.epochs or global_train.TRAIN_EPOCHS,
            "learning_rate": self.learning_rate or global_train.TRAIN_LEARNING_RATE,
            "output_dir": self.output_dir or global_train.TRAIN_OUTPUT_DIR,
            "dataset_path": self.dataset_path or global_train.TRAIN_DATASET_PATH,
        }


# Глобальные настройки
global_settings_ocr = GlobalSettingsOCR()
global_settings_train = GlobalSettingsTrainOCR()
