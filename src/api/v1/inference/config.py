from fastapi import APIRouter, status, Depends
from fastapi.responses import ORJSONResponse
from src.utils import settings_ocr, SettingsOCR, ConfigOCRInference


router = APIRouter(
    prefix="/config",
    default_response_class=ORJSONResponse,
    tags=["Inference"]
)

@router.get(
    path='/',
    status_code=status.HTTP_200_OK,
    response_model=ConfigOCRInference,
    name='Получение конфигурации',
)
async def get_settings() -> ConfigOCRInference:
    return ConfigOCRInference(**settings_ocr.model_dump())


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=ConfigOCRInference,
    name="Установка конфигурации",
)
async def set_config(config: ConfigOCRInference) -> ConfigOCRInference:
    if config.model:
        settings_ocr.model = config.model
        settings_ocr.is_custom = True
    if config.device: settings_ocr.device = config.device
    return ConfigOCRInference(**settings_ocr.model_dump())


@router.post(
    path="/reload",
    status_code=status.HTTP_201_CREATED,
    response_model=ConfigOCRInference,
    name="Сброс конфигурации",
)
async def reload_config() -> ConfigOCRInference:
    global settings_ocr
    settings_ocr = SettingsOCR()
    return ConfigOCRInference(**settings_ocr.model_dump())