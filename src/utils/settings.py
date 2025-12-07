from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class SettingsOCR(BaseSettings):
    """Настройки inference"""
    model: str = "microsoft/trocr-small-handwritten"
    device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    is_custom: bool = False

    model_config = SettingsConfigDict(
        env_file=".env_ocr",
        env_file_encoding="utf-8",
        extra='ignore'
    )


class SettingsTrainOCR(BaseSettings):
    """Настройки обучения"""
    model: str = "microsoft/trocr-small-handwritten"
    device: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    max_length: int = 512
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    log_interval: int = 10
    num_workers: int = 2
    checkpoint_interval: int = 2
    output_dir: str = "./models"
    dataset_path: str = "./datasets"
    labels_filename: str = "dataset.csv"
    custom_loader_dataset: str = "CyrillicHandwrittenDataset"
    validation_split_size: float = 0.05
    random_seed: int = 42

    model_config = SettingsConfigDict(
        env_file=".env_train",
        env_file_encoding="utf-8",
        extra='ignore'
    )


settings_ocr = SettingsOCR()
settings_train = SettingsTrainOCR()
