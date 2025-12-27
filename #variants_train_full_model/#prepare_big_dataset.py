
"""
# from gemini задача:
В системе w11 в папке D:\datasets\rus\info размещен файл текстовых меток gt.txt для обучения trocr,
а в папке D:\datasets\rus размещены десять файлов zip : от img-00.zip до img-09.zip.
Требуется просканировать все zip и txt, и написать на python скрипт, который создаст два датасета train и val,
содержащие по одному файлу csv и одному файлу zip, в пропорции 93% к 7%.
Итоговый путь к обоим датасетам - D:\datasets\rus\datasets. В csv файлах должны быть поля 'image_path' и 'text'
"""


import os
import zipfile
import csv
import random
import re
from collections import defaultdict  # Хотя defaultdict не используется, оставляем на всякий случай

# --- Константы ---
BASE_DIR = r"D:\datasets\rus"
INFO_DIR = os.path.join(BASE_DIR, "info")
GT_FILE_PATH = os.path.join(INFO_DIR, "gt.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets")
SPLIT_RATIO = 0.93  # 93% для train, 7% для val

# --- Шаг 1: Чтение меток из gt.txt с помощью регулярных выражений ---
print(f"1. Чтение меток из {GT_FILE_PATH} с использованием RegEx...")
labels = {}
try:
    with open(GT_FILE_PATH, 'r', encoding='utf-8') as f:
        # Читаем весь файл как одну строку, так как нет переводов строк между записями
        content = f.read()

    # Паттерн: (имя_файла.png),(текст_метки)(?=следующее_имя_файла.png|\Z)
    regex_pattern = r"(img/stackmix_hkr_\d+\.png),(.*?)(?=img/stackmix_hkr_|\Z)"

    # Находим все совпадения
    matches = re.findall(regex_pattern, content, re.DOTALL)

    for filename_with_path, text in matches:
        # Извлекаем только имя файла без пути 'img/'
        filename = os.path.basename(filename_with_path)
        labels[filename] = text.strip()

except FileNotFoundError:
    print(f"ОШИБКА: Файл меток не найден по пути: {GT_FILE_PATH}")
    exit()

print(f"Найдено {len(labels)} уникальных текстовых меток.")

# --- Шаг 2 & 3: Сбор данных и поиск соответствий (Остается без изменений) ---
print("2. Сканирование ZIP-архивов и сопоставление данных...")
file_to_zip_map = {}
full_dataset = []

zip_files = sorted([f for f in os.listdir(BASE_DIR) if f.startswith('img-') and f.endswith('.zip')])

if not zip_files:
    print(f"ОШИБКА: ZIP-файлы не найдены в {BASE_DIR}")
    exit()

for zip_name in zip_files:
    zip_path = os.path.join(BASE_DIR, zip_name)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if member.endswith('/'):
                    continue

                # Имя файла в ZIP-архиве должно быть без пути (например, 'stackmix_hkr_3701358.png')
                filename = os.path.basename(member)

                if filename in labels:
                    file_to_zip_map[filename] = zip_path
                    full_dataset.append({
                        'image_path': filename,
                        'text': labels[filename],
                        'source_zip': zip_path
                    })

    except zipfile.BadZipFile:
        print(f"Предупреждение: Файл {zip_name} поврежден или не является ZIP-архивом.")
        continue

print(f"Полный датасет содержит {len(full_dataset)} пар (изображение, метка).")

if not full_dataset:
    print("ОШИБКА: Не удалось найти ни одного изображения с соответствующей меткой.")
    exit()

# --- Шаг 4: Разделение на train/val (Остается без изменений) ---
print("3. Разделение датасета на обучающий (train) и валидационный (val)...")
random.seed(42)
random.shuffle(full_dataset)

split_point = int(len(full_dataset) * SPLIT_RATIO)
train_data = full_dataset[:split_point]
val_data = full_dataset[split_point:]

print(f"  - Train: {len(train_data)} ({len(train_data) / len(full_dataset):.2%})")
print(f"  - Val:   {len(val_data)} ({len(val_data) / len(full_dataset):.2%})")

# --- Шаг 5, 6, 7: Создание выходных файлов и папок (Остается без изменений) ---
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_dataset_files(data, name):
    """Создает CSV и ZIP файлы для заданного набора данных (train или val)."""
    print(f"4. Создание датасета '{name}'...")

    csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    zip_path = os.path.join(OUTPUT_DIR, f"{name}.zip")

    # 5. Создание CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_path', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # 6. Создание ZIP и одновременное заполнение CSV
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as new_zip:
            source_zip_cache = {}  # Кэш для открытых исходных ZIP-файлов

            for item in data:
                image_name = item['image_path']
                source_zip_path = item['source_zip']
                text = item['text']

                # Запись в CSV
                writer.writerow({'image_path': image_name, 'text': text})

                # Копирование файла в новый ZIP

                if source_zip_path not in source_zip_cache:
                    source_zip_cache[source_zip_path] = zipfile.ZipFile(source_zip_path, 'r')

                source_zip = source_zip_cache[source_zip_path]

                try:
                    # В ZIP-файле имя изображения, скорее всего, хранится как 'img/stackmix_hkr_...'
                    # Попробуем найти и по полному пути, и по базовому имени.
                    # Поскольку мы ищем по 'member' в Шаге 2, то member содержит полный путь,
                    # поэтому будем использовать его, если оно есть в namelist().

                    zip_member_path = f"img/{image_name}" if f"img/{image_name}" in source_zip.namelist() else image_name

                    image_data = source_zip.read(zip_member_path)

                    # Записываем данные в новый ZIP
                    new_zip.writestr(image_name, image_data)
                except KeyError:
                    # Это запасной вариант, если изображение не найдено ни по одному из вариантов пути
                    print(f"Предупреждение: Файл {image_name} не найден в {source_zip_path}. Пропускаем.")

            # Закрываем все кэшированные исходные ZIP-файлы
            for source_zip in source_zip_cache.values():
                source_zip.close()

    print(f"  - {name}.csv создан по пути: {csv_path}")
    print(f"  - {name}.zip создан по пути: {zip_path}")
    print(f"  - Общий размер {name}.zip: {os.path.getsize(zip_path) / (1024 * 1024):.2f} MB")


# Запуск функции для train и val
create_dataset_files(train_data, "train")
create_dataset_files(val_data, "val")

print("\n✅ Скрипт успешно завершен!")
print(f"Результаты находятся в папке: {OUTPUT_DIR}")
