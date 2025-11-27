from typing import Dict, Any
from fastapi import APIRouter, status, Depends, BackgroundTasks, HTTPException
from fastapi.responses import ORJSONResponse
from src.utils import SessionSettingsTrain, RobustTrainingManager
from src import get_train_session_settings


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
async def start_training(
        train_session_settings: SessionSettingsTrain = Depends(get_train_session_settings)
):
    """Запуск нового процесса обучения"""
    try:
        training_config = train_session_settings.get_training_config()
        training_id = training_manager.start_training(training_config)

        return {
            "training_id": training_id,
            "status": "training_started",
            "message": "Обучение модели запущено",
            "config": training_config
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
