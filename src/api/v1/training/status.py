from typing import Dict, Any
from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse
from src.utils import load_status


router = APIRouter(
    prefix="/status",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)


@router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="Получение статуса обучения",
)
def get_training_status():
    """Возвращает текущий статус обучения"""
    return load_status()
