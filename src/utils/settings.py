from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsOCR(BaseSettings):
    MODEL_PATH: str
    DEVICE: str

    model_config = SettingsConfigDict(
        env_file=".env_ocr",
        env_file_encoding="utf-8"
    )


class SettingsTrainOCR(BaseSettings):
    MODEL_PATH: str
    DEVICE: str

    model_config = SettingsConfigDict(
        env_file=".env_train",
        env_file_encoding="utf-8"
    )

settings_ocr = SettingsOCR()
settings_train = SettingsTrainOCR()
