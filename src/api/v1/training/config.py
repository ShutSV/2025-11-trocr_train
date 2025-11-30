from fastapi import APIRouter, status, Depends
from fastapi.responses import ORJSONResponse
from src.utils import settings_train, ConfigOCRTraining


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
    if config.model: settings_train.model = config.model
    settings_train.device = config.device
    settings_train.epochs = config.epochs
    return ConfigOCRTraining(**settings_train.model_dump())
