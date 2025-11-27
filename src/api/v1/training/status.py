from typing import Dict, Any
from fastapi import APIRouter, status, HTTPException, Query
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager



router = APIRouter(
    prefix="/train",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.get(
    path="/status",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="Получение статуса обучения",
)
async def get_training_status(
        training_id: str = Query(None, description="ID тренировки (опционально)")
):
    """Получение статуса конкретной или активной тренировки"""
    status_info = training_manager.get_status(training_id)

    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Тренировка не найдена"
        )

    return {
        "training_id": status_info.training_id,
        "status": status_info.status.value,
        "progress": status_info.progress,
        "current_epoch": status_info.current_epoch,
        "total_epochs": status_info.total_epochs,
        "current_loss": status_info.current_loss,
        "start_time": status_info.start_time,
        "last_update": status_info.last_update,
        "message": status_info.message,
        "checkpoint_available": status_info.checkpoint_path is not None
    }


@router.get(
    path="/status/all",
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name="Получение статуса всех тренировок",
)
async def get_all_training_status():
    """Получение статуса всех активных тренировок"""
    active_trainings = training_manager.state_manager.get_active_trainings()

    return {
        "active_trainings": len(active_trainings),
        "trainings": [
            {
                "training_id": training.training_id,
                "status": training.status.value,
                "progress": training.progress,
                "current_epoch": training.current_epoch,
                "message": training.message
            }
            for training in active_trainings
        ]
    }
