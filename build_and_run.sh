#!/bin/bash

# Сборка Docker образа
#echo "Сборка Docker образа для Mac..."
#docker build -t train-ocr-server .

echo "Сборка Docker образа для WIN..."
docker buildx build --platform linux/amd64 -t 2025-11-win-train-ocr-server .

# Запуск контейнера
echo "Запуск контейнера..."
# перед запуском контейнера проверить доступность видеокарты (в тч можно 'nvcc --version' для rtx4000ada cuda 12.8
# для этого выключить Atomman, вынуть из него шнур питания на 5 сек, включить и загрузить, затем перезагрузить, и затем:
nvidia-smi

docker run -d --name ocr-server -p 8000:8000 --restart unless-stopped 2025-11-win-train-ocr-server


echo "Сервер запущен на http://localhost:8000"
echo "Для проверки: curl http://localhost:8000/health"
