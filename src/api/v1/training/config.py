from typing import Dict, Any
from fastapi import APIRouter, status, Depends
from fastapi.responses import ORJSONResponse
from src.utils import SessionSettingsTrain
from src import get_train_session_settings


router = APIRouter(
    prefix="/config",
    default_response_class=ORJSONResponse,
    tags=["Training"]
)


@router.get(
    path='/',
    status_code=status.HTTP_200_OK,
    response_model=Dict[str, Any],
    name='Получение конфигурации обучения',
)
async def get_train_settings(
        train_session_settings: SessionSettingsTrain = Depends(get_train_session_settings)
):
    return {
        "training_config": train_session_settings.get_training_config(),
        "is_custom": any([
            train_session_settings.model_path,
            train_session_settings.device,
            train_session_settings.batch_size,
            train_session_settings.epochs,
            train_session_settings.learning_rate,
            train_session_settings.output_dir,
            train_session_settings.dataset_path
        ]),
        "message": "Конфигурация обучения для текущей сессии",
    }


@router.post(
    path="/",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, Any],
    name="Установка конфигурации обучения",
)
async def set_train_config(
        model_path: str = None,
        device: str = None,
        batch_size: int = None,
        epochs: int = None,
        learning_rate: float = None,
        output_dir: str = None,
        dataset_path: str = None,
        train_session_settings: SessionSettingsTrain = Depends(get_train_session_settings)
):
    config_updates = {}
    if model_path: config_updates['model_path'] = model_path
    if device: config_updates['device'] = device
    if batch_size: config_updates['batch_size'] = batch_size
    if epochs: config_updates['epochs'] = epochs
    if learning_rate: config_updates['learning_rate'] = learning_rate
    if output_dir: config_updates['output_dir'] = output_dir
    if dataset_path: config_updates['dataset_path'] = dataset_path

    train_session_settings.update(**config_updates)

    return {
        "training_config": train_session_settings.get_training_config(),
        "message": "Конфигурация обучения успешно обновлена для текущей сессии",
    }
