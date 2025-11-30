from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import ORJSONResponse

from src.utils import train_trocr_model, settings_train


router = APIRouter(
    prefix="/start",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)
@router.post(
    path="/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Dict[str, Any],
    name="Запуск обучения модели",
)
async def start_training(background_tasks: BackgroundTasks):
    """Асинхронный запуск обучения TrOCR"""
    config = settings_train.model_dump()
    background_tasks.add_task(train_trocr_model, config)
    return {"status": "started", "config": config}
