from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str
    DEVICE: str


settings = Settings()
