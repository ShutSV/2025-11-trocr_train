from typing import Dict, Any
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager


router = APIRouter(
    prefix="/pause",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.post(
    path="/{training_id}",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="Пауза обучения",
)
async def pause_training(training_id: str):
    """Пауза тренировки с возможностью возобновления"""
    success = training_manager.pause_training(training_id)

    if success:
        return {
            "training_id": training_id,
            "status": "paused",
            "message": "Обучение поставлено на паузу"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Активное обучение не найдено"
        )
