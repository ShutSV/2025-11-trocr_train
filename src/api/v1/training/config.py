from typing import Dict
from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse
from src.utils import settings


router = APIRouter(
    prefix="/config",
    default=ORJSONResponse,
    tags=["Training models"]
)


@router.get(
        path='/',
        status_code=status.HTTP_200_OK,
        response_model=Dict,
        name='Получение конфигурации для обучения # ЗАГЛУШКА #',
)
async def get_settings():
    return {
        "model": settings.MODEL_PATH, # ИЗМЕНИТЬ НА ПЕРЕМЕННЫЕ ОБУЧЕНИЯ
        "device": settings.DEVICE, # ИЗМЕНИТЬ НА ПЕРЕМЕННЫЕ ОБУЧЕНИЯ
    }

@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict,
    name="Установка конфигурации для обучения # ЗАГЛУШКА #",  # ДОПОЛНИТЬ НА ID ПЕРЕМЕННЫХ ОБУЧЕНИЯ
)

async def set_config(model_path: str, device: str = "cpu"):
    settings.MODEL_PATH = model_path
    settings.DEVICE = device
    return {
        "model": settings.MODEL_PATH,
        "device": settings.DEVICE,
        "message": "Конфигурация успешно обновлена",
    }
