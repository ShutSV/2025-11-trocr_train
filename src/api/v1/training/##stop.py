from typing import Dict, Any
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager


router = APIRouter(
    prefix="/stop",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.post(
    path="/{training_id}",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="Остановка обучения",
)
async def stop_training(training_id: str):
    """Остановка конкретной тренировки"""
    success = training_manager.stop_training(training_id)

    if success:
        return {
            "training_id": training_id,
            "status": "stopped",
            "message": "Обучение остановлено"
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Активное обучение не найдено"
        )
