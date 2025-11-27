import json
import os
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    IDLE = "idle"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class TrainingState:
    training_id: str
    status: TrainingStatus
    config: Dict[str, Any]
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    metrics: Dict[str, Any] = None
    start_time: str = None
    last_update: str = None
    checkpoint_path: str = None
    message: str = ""

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
        self.last_update = datetime.now().isoformat()


class PersistentTrainingManager:
    def __init__(self, state_dir: str = "./training_states"):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.current_training_id: Optional[str] = None

    def _get_state_file(self, training_id: str) -> str:
        return os.path.join(self.state_dir, f"{training_id}.json")

    def _get_checkpoint_dir(self, training_id: str) -> str:
        return os.path.join(self.state_dir, training_id, "checkpoints")

    def create_training(self, config: Dict[str, Any]) -> str:
        """Создание новой тренировки с уникальным ID"""
        training_id = str(uuid.uuid4())

        state = TrainingState(
            training_id=training_id,
            status=TrainingStatus.TRAINING,
            config=config
        )

        self.save_state(state)
        self.current_training_id = training_id
        return training_id

    def save_state(self, state: TrainingState):
        """Сохранение состояния тренировки"""
        state_file = self._get_state_file(state.training_id)

        # Обновляем время последнего изменения
        state.last_update = datetime.now().isoformat()

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(state), f, indent=2, ensure_ascii=False)

    def load_state(self, training_id: str) -> Optional[TrainingState]:
        """Загрузка состояния тренировки"""
        state_file = self._get_state_file(training_id)

        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Конвертируем строку статуса обратно в Enum
            data['status'] = TrainingStatus(data['status'])
            return TrainingState(**data)
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния {training_id}: {e}")
            return None

    def get_active_trainings(self) -> List[TrainingState]:
        """Получение всех активных тренировок"""
        active_trainings = []

        for file in os.listdir(self.state_dir):
            if file.endswith('.json'):
                training_id = file[:-5]  # убираем .json
                state = self.load_state(training_id)

                if state and state.status in [TrainingStatus.TRAINING, TrainingStatus.PAUSED]:
                    active_trainings.append(state)

        return active_trainings

    def get_training_history(self, limit: int = 10) -> List[TrainingState]:
        """Получение истории тренировок"""
        all_states = []

        for file in os.listdir(self.state_dir):
            if file.endswith('.json'):
                training_id = file[:-5]
                state = self.load_state(training_id)
                if state:
                    all_states.append(state)

        # Сортируем по времени последнего обновления
        all_states.sort(key=lambda x: x.last_update, reverse=True)
        return all_states[:limit]

    def cleanup_old_states(self, days: int = 30):
        """Очистка старых состояний"""
        cutoff_time = time.time() - (days * 24 * 60 * 60)

        for file in os.listdir(self.state_dir):
            if file.endswith('.json'):
                file_path = os.path.join(self.state_dir, file)
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Удален старый файл состояния: {file}")
