from typing import Dict
from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse


router = APIRouter(
    prefix="/health",
    default=ORJSONResponse,
    tags=["health"]
)


@router.get(
        path='/',
        status_code=status.HTTP_200_OK,
        response_model=Dict,
        name='Проверка активности',
)
async def health_check():
    """Проверка состояния сервера"""
    return {
        "message": "Train Server is running",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "train": "/train",
            "config": "/config",
            "status": "/status",
        },
        # "status": "healthy" if (processor is not None and model is not None) else "model_not_loaded",
    }