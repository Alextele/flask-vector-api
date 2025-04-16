FROM python:3.9-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем модель (GGUF)
RUN mkdir -p /app/models && \
    wget -O /app/models/saiga_yandexgpt_8b.Q4_K_M.gguf \
    https://huggingface.co/IlyaGusev/saiga_yandexgpt_8b_gguf/resolve/main/saiga_yandexgpt_8b.Q4_K_M.gguf

# Копируем код приложения
COPY config.py .
COPY app.py .

CMD ["python", "app.py"]
