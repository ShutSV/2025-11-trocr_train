from typing import Dict, Any
from fastapi import APIRouter, status, Query
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager


router = APIRouter(
    prefix="/history",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.get(
    path="/",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="История тренировок",
)
async def get_training_history(
        limit: int = Query(10, description="Лимит записей")
):
    """Получение истории тренировок"""
    history = training_manager.state_manager.get_training_history(limit)

    return {
        "total_trainings": len(history),
        "trainings": [
            {
                "training_id": training.training_id,
                "status": training.status.value,
                "progress": training.progress,
                "current_epoch": training.current_epoch,
                "total_epochs": training.total_epochs,
                "start_time": training.start_time,
                "last_update": training.last_update,
                "message": training.message
            }
            for training in history
        ]
    }
