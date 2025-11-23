#!/bin/bash

# Сборка Docker образа
echo "Сборка Docker образа для Mac..."
docker build -t ocr-server .

echo "Сборка Docker образа для AMD..."
docker buildx build --platform linux/amd64 -t 2025-11-amd64-my-app-image .

# Запуск контейнера
echo "Запуск контейнера..."
docker run -d --name ocr-server -p 8000:8000 --restart unless-stopped ocr-server

echo "Сервер запущен на http://localhost:8000"
echo "Для проверки: curl http://localhost:8000/health"
