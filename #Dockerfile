FROM ubuntu:22.04
LABEL authors="sergio"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Установка Python и системных зависимостей
RUN apt-get update && apt-get install -y python3.10 python3-pip cmake build-essential && rm -rf /var/lib/apt/lists/*

# CUDA инструменты
RUN apt update && apt install -y --no-install-recommends cuda-toolkit-12-8 nvidia-cuda-toolkit && rm -rf /var/lib/apt/lists/*

# переменные окружения CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Затем устанавливаем тяжелые пакеты отдельно
RUN pip install --timeout 300 --retries 5 --no-cache-dir \
    kraken transformers fastapi uvicorn orjson pydantic_settings python-multipart \
    sentencepiece sacremoses \
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
#RUN pip uninstall torch torchvision torchaudio

# Копирование requirements и установка Python зависимостей
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt



# Копирование исходного кода
COPY . /app

# Создание директории для временных файлов
RUN mkdir -p /tmp/uploads

# Переменные
ENV MODEL_PATH=microsoft/trocr-small-handwritten
ENV DEVICE=cpu

# Открытие порта
EXPOSE 8000

# Запуск сервера
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]