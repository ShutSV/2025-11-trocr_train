from typing import Dict
from fastapi import APIRouter, status, Depends
from fastapi.responses import ORJSONResponse
from src.utils import SessionSettingsOCR
from src import get_session_settings

router = APIRouter(
    prefix="/config",
    default_response_class=ORJSONResponse,
    tags=["Inference"]
)

@router.get(
    path='/',
    status_code=status.HTTP_200_OK,
    response_model=Dict,
    name='Получение конфигурации',
)
async def get_settings(session_settings: SessionSettingsOCR = Depends(get_session_settings)):
    return {
        "model": session_settings.get_model_path(),
        "device": session_settings.get_device(),
        "is_custom": session_settings.model_path is not None
    }

@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict,
    name="Установка конфигурации",
)
async def set_config(
    model_path: str,
    device: str = "cpu",
    session_settings: SessionSettingsOCR = Depends(get_session_settings)
):
    session_settings.update(model_path, device)
    return {
        "model": session_settings.get_model_path(),
        "device": session_settings.get_device(),
        "message": "Конфигурация успешно обновлена для текущей сессии",
    }
