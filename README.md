# 2025-11-trocr\_train



endpoints:
api/v1:
    /inference
        /health
        /ocr
        /config
    /training
        /start
        /stop
        /pause
        /resume
        /status
        /config
        /history
    /results
        /models
        /datasets

Решение сценариев:
1. Сессия прервалась - нужно подключиться позже
# После переподключения:
GET /api/v1/train/status/all
# Увидим все активные тренировки

GET /api/v1/train/status?training_id=UUID
# Получим детальный статус конкретной тренировки

2. Обучение было прервано - нужно возобновить
# Находим ID прерванной тренировки
GET /api/v1/train/history

# Возобновляем
POST /api/v1/train/resume/UUID

Ключевые особенности:

    🎯 Персистентность - состояния сохраняются на диск

    🔄 Возобновляемость - можно продолжить с последнего чекпоинта

    📊 Мультисессионность - несколько пользователей могут отслеживать тренировки

    ⏸️ Контроль паузы - можно ставить на паузу и возобновлять

    📈 История - полная история всех тренировок

    🔍 Поиск - возможность найти тренировку по ID


2025-11-trocr_train/
├── .env_ocr
├── .env_train
├── app.py
├── main.py
└── src/
    ├── __init__.py
    ├── dependencies.py
    ├── utils/
    │   ├── __init__.py
    │   ├── settings.py
    │   ├── ocr.py
    │   └── train_utils.py
    └── api/
        └── v1/
            └── endpoints/
                ├── __init__.py
                ├── config.py
                ├── ocr.py
                ├── train_config.py
                └── train.py
