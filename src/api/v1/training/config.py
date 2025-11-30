from fastapi import APIRouter, status, Depends
from fastapi.responses import ORJSONResponse
from src.utils import settings_train, SettingsTrainOCR, ConfigOCRTraining


router = APIRouter(
    prefix="/config",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)


@router.get(
    path='/',
    status_code=status.HTTP_200_OK,
    response_model=ConfigOCRTraining,
    name='Получение конфигурации обучения',
)
async def get_train_settings() -> ConfigOCRTraining:
    return ConfigOCRTraining(**settings_train.model_dump())


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=ConfigOCRTraining,
    name="Установка конфигурации обучения",
)
async def set_train_config(config: ConfigOCRTraining) -> ConfigOCRTraining:
    update_data = config.model_dump(exclude_none=True)
    settings_train.model_validate(update_data)
    for field, value in update_data.items():
        setattr(settings_train, field, value)
    return ConfigOCRTraining(**settings_train.model_dump())


@router.post(
    path="/reload",
    status_code=status.HTTP_201_CREATED,
    response_model=ConfigOCRTraining,
    name="Сброс конфигурации обучения",
)
async def set_train_config() -> ConfigOCRTraining:
    global settings_train
    settings_train = SettingsTrainOCR()
    return ConfigOCRTraining(**settings_train.model_dump())
