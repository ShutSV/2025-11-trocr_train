import json
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
from . import PersistentTrainingManager, TrainingState, TrainingStatus


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataset_path(dataset_path: str) -> bool:
    """Простая проверка существования датасета"""
    return os.path.exists(dataset_path)


class RobustTrainingManager:
    def __init__(self, state_dir: str = "./training_states"):
        self.state_manager = PersistentTrainingManager(state_dir)
        self._stop_requested = False
        self._current_training_thread = None

    def start_training(self, config: Dict[str, Any]) -> str:
        """Запуск новой тренировки"""
        # Проверяем активные тренировки
        active_trainings = self.state_manager.get_active_trainings()

        if active_trainings:
            # Можно либо запретить параллельные тренировки, либо разрешить
            # Сейчас запрещаем для простоты
            raise Exception("Обнаружены активные тренировки. Завершите их перед запуском новой.")

        training_id = self.state_manager.create_training(config)

        # Запускаем в отдельном потоке (в реальности лучше использовать celery/ray)
        import threading
        self._current_training_thread = threading.Thread(
            target=self._training_worker,
            args=(training_id, config),
            daemon=True
        )
        self._current_training_thread.start()

        return training_id

    def resume_training(self, training_id: str) -> bool:
        """Возобновление прерванной тренировки"""
        state = self.state_manager.load_state(training_id)

        if not state:
            raise Exception(f"Тренировка {training_id} не найдена")

        if state.status not in [TrainingStatus.PAUSED, TrainingStatus.ERROR]:
            raise Exception(f"Невозможно возобновить тренировку со статусом {state.status}")

        # Обновляем статус
        state.status = TrainingStatus.TRAINING
        state.message = "Возобновление обучения..."
        self.state_manager.save_state(state)

        # Запускаем продолжение обучения
        import threading
        self._current_training_thread = threading.Thread(
            target=self._training_worker,
            args=(training_id, state.config, state),
            daemon=True
        )
        self._current_training_thread.start()

        return True

    def _training_worker(self, training_id: str, config: Dict[str, Any],
                         resume_state: Optional[TrainingState] = None):
        """Рабочий процесс обучения с сохранением состояния"""
        try:
            if resume_state:
                # Продолжение с checkpoint
                state = resume_state
                start_epoch = state.current_epoch
                state.message = "Возобновление обучения с checkpoint"
            else:
                # Новая тренировка
                state = self.state_manager.load_state(training_id)
                # Простая проверка датасета
                if not validate_dataset_path(config['dataset_path']):
                    state.status = TrainingStatus.ERROR
                    state.message = f"Датасет не найден: {config['dataset_path']}"
                    self.state_manager.save_state(state)
                    return

                start_epoch = 0
                state.message = f"Начало обучения. Датасет: {config['dataset_path']}"

            self.state_manager.save_state(state)

            # Имитация процесса обучения с чекпоинтами
            for epoch in range(start_epoch, config['epochs']):
                if self._stop_requested:
                    state.status = TrainingStatus.STOPPED
                    state.message = "Остановлено пользователем"
                    self.state_manager.save_state(state)
                    break

                # Обновляем состояние
                state.current_epoch = epoch + 1
                state.progress = (epoch + 1) / config['epochs'] * 100
                state.current_loss = 1.0 / (epoch + 1)  # Имитация лосса
                state.status = TrainingStatus.TRAINING
                state.message = f"Эпоха {epoch + 1}/{config['epochs']}"

                # Сохраняем чекпоинт каждые 2 эпохи
                if (epoch + 1) % 2 == 0:
                    checkpoint_path = self._save_checkpoint(training_id, epoch + 1, config)
                    state.checkpoint_path = checkpoint_path
                    state.message = f"Эпоха {epoch + 1}/{config['epochs']} - чекпоинт сохранен"

                self.state_manager.save_state(state)

                # Имитация времени обучения
                time.sleep(2)

            if not self._stop_requested and state.current_epoch >= config['epochs']:
                state.status = TrainingStatus.COMPLETED
                state.progress = 100.0
                state.message = "Обучение завершено успешно"
                self.state_manager.save_state(state)

        except Exception as e:
            logger.error(f"Ошибка в тренировке {training_id}: {e}")
            state.status = TrainingStatus.ERROR
            state.message = f"Ошибка: {str(e)}"
            self.state_manager.save_state(state)
        finally:
            self._stop_requested = False

    def _save_checkpoint(self, training_id: str, epoch: int, config: Dict[str, Any]) -> str:
        """Сохранение чекпоинта модели"""
        checkpoint_dir = self.state_manager._get_checkpoint_dir(training_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}")

        # В реальности здесь сохраняем модель
        # torch.save(model.state_dict(), checkpoint_path)

        # Пока сохраняем метаинформацию
        checkpoint_info = {
            'epoch': epoch,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

        with open(f"{checkpoint_path}.json", 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

        return checkpoint_path

    def stop_training(self, training_id: str) -> bool:
        """Остановка конкретной тренировки"""
        state = self.state_manager.load_state(training_id)

        if state and state.status == TrainingStatus.TRAINING:
            self._stop_requested = True
            state.status = TrainingStatus.STOPPED
            state.message = "Остановлено пользователем"
            self.state_manager.save_state(state)
            return True

        return False

    def pause_training(self, training_id: str) -> bool:
        """Пауза тренировки (сохраняет состояние для возобновления)"""
        state = self.state_manager.load_state(training_id)

        if state and state.status == TrainingStatus.TRAINING:
            state.status = TrainingStatus.PAUSED
            state.message = "На паузе, можно возобновить"
            self.state_manager.save_state(state)
            self._stop_requested = True
            return True

        return False

    def get_status(self, training_id: str = None) -> Optional[TrainingState]:
        """Получение статуса тренировки"""
        if training_id:
            return self.state_manager.load_state(training_id)
        elif self.state_manager.current_training_id:
            return self.state_manager.load_state(self.state_manager.current_training_id)

        # Если ID не указан, возвращаем первую активную тренировку
        active_trainings = self.state_manager.get_active_trainings()
        return active_trainings[0] if active_trainings else None

