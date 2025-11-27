from typing import Dict, Any
from fastapi import APIRouter, status, HTTPException
from fastapi.responses import ORJSONResponse
from src.utils import RobustTrainingManager


router = APIRouter(
    prefix="/train",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)

training_manager = RobustTrainingManager()


@router.post(
    path="/resume/{training_id}",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Dict[str, Any],
    name="Возобновление обучения",
)
async def resume_training(training_id: str):
    """Возобновление прерванной тренировки"""
    try:
        success = training_manager.resume_training(training_id)

        if success:
            return {
                "training_id": training_id,
                "status": "training_resumed",
                "message": "Обучение возобновлено"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Не удалось возобновить тренировку"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
