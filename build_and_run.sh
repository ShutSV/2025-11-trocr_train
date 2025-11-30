#!/bin/bash

# Сборка Docker образа
# Создать multi-arch builder (если еще не сделано)
#docker buildx create --name multiarch-builder --use
#docker buildx inspect --bootstrap

echo "Сборка Docker образа для Mac..."
docker build -f Dockerfile.from_train-ocr -t train-ocr-server:blank --load .

echo "Сборка Docker образа для WIN..."
docker buildx build --platform linux/amd64 -f Dockerfile.cuda -t train-ocr-server:2025-11 --load .

# Посмотреть архитектуру образа
docker image inspect train-ocr-server:2025-11 | grep Architecture
# "Architecture": "arm64"

# Запуск контейнера
echo "Запуск контейнера для Mac с пробросом в контейнер папки проекта (для разработки)"
docker run -d -p 8000:8000 -v $(pwd):/app --name train-ocr-server-volume train-ocr-server:blank

# для Win перед запуском контейнера проверить доступность видеокарты (в тч можно 'nvcc --version' для rtx4000ada cuda 12.8
# для этого выключить Atomman, вынуть из него шнур питания на 5 сек, включить и загрузить, затем перезагрузить, и затем:
nvidia-smi
docker run -d --gpus all --name 2025-11-cuda-ocr-server -p 8000:8000 --restart unless-stopped 2025-11-win-train-ocr-server

echo "Сервер запущен на http://localhost:8000"
echo "Для проверки: curl http://localhost:8000/health"

# если надо запустить на хосте Win
uvicorn app:app --host 0.0.0.0 --port 8000 --reload