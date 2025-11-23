#!/bin/bash

# Сборка Docker образа
echo "Сборка Docker образа для Mac..."
docker build -t train-ocr-server .

echo "Сборка Docker образа для WIN..."
docker buildx build --platform linux/amd64 -t 2025-11-win-train-ocr-server .

# Запуск контейнера
echo "Запуск контейнера..."
docker run -d --name ocr-server -p 8000:8000 --restart unless-stopped 2025-11-win-train-ocr-server

echo "Сервер запущен на http://localhost:8000"
echo "Для проверки: curl http://localhost:8000/health"
