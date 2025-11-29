from fastapi import Request
from src.utils import SessionSettingsOCR, SessionSettingsTrain


def get_session_settings(request: Request) -> SessionSettingsOCR:
    """Получить или создать настройки inference сессии"""
    if not hasattr(request.state, 'session_settings'):
        request.state.session_settings = SessionSettingsOCR()
    return request.state.session_settings


def get_train_session_settings(request: Request) -> SessionSettingsTrain:
    """Получить или создать настройки тренировочной сессии"""
    if not hasattr(request.state, 'train_session_settings'):
        request.state.train_session_settings = SessionSettingsTrain()
    return request.state.train_session_settings
