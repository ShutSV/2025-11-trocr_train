from typing import Dict, Any
from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager, settings_train


router = APIRouter(
    prefix="/start",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.post(
    path="/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Dict[str, Any],
    name="Запуск обучения модели",
)
async def start_training():
    """Запуск нового процесса обучения"""
    try:
        training_id = training_manager.start_training(**settings_train)
        return {
            "training_id": training_id,
            "status": "training_started",
            "message": "Обучение модели запущено",
            "config": {**settings_train}
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
