# Два класса датасета: train - в RAM, val - в ZIP

import io
import zipfile
# import shutil
from pathlib import Path
import os
# import sys
# import subprocess
# import time
# import ctypes
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
)
import albumentations as A
# from tqdm import tqdm


# ==============================
# 0. Проверка RAM-disk
# ==============================
def check_ramdisk(drive_letter='R'):
    if os.path.exists(f"{drive_letter}:\\"):
        print(f"Диск {drive_letter}: существует")
        return True

check_ramdisk("R")

# ==============================
# 1. Пути
# ==============================

DATASET_DIR = Path(r"d:\datasets\rus\datasets")
RAMDISK_DIR = Path(r"R:\ramdisk")

TRAIN_CSV = DATASET_DIR / "train.csv"
VAL_CSV   = DATASET_DIR / "val.csv"

TRAIN_ZIP_SSD = DATASET_DIR / "train.zip"
VAL_ZIP_SSD   = DATASET_DIR / "val.zip"

TRAIN_RAM = RAMDISK_DIR

OUTPUT_DIR = "./trocr_rus_model"

MODEL_NAME = "microsoft/trocr-small-handwritten"

# ==============================
# 4. Dataset (RAM)
# ==============================

class CyrillicHandwrittenDatasetRam(Dataset):
    def __init__(self, df, processor, root_dir, transforms=None, max_target_length=128):
        self.df = df
        self.processor = processor
        self.root_dir = root_dir
        self.transforms = transforms
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if hasattr(idx, 'iloc'):  # Если пришел pandas Series, берем первое значение
            idx = idx.iloc[0]
        elif isinstance(idx, (list, tuple)):  # Если пришел список, берем первый элемент
            idx = idx[0]
        idx = int(idx)  # Конвертируем в int

        row = self.df.iloc[idx]
        file_name = row["image_path"]
        text = row['text']
        image_path = Path(self.root_dir) / file_name
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Внимание: Файл не найден {image_path}. Пропускаем.")
            return self.__getitem__((idx + 1) % len(self))  # Возвращаем следующий элемент, чтобы избежать ошибки
        if self.transforms:  # Применяем аугментации, если они есть
            image_np = self.transforms(image=np.array(image))['image']
            image = Image.fromarray(image_np)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values  # Обработка изображения процессором
        labels = self.processor.tokenizer(  # Обработка текста процессором
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]  # Для функции потерь заменяем padding-токены на -100
        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}


# ==============================
# 3. Аугментации
# ==============================

train_transforms = A.Compose([
    A.Rotate(limit=3, p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
])

# ==============================
# 4. Dataset (ZIP)
# ==============================

class CyrillicHandwrittenDatasetZip(Dataset):
    def __init__(
        self,
        df,
        processor,
        zip_path: Path,
        transforms=None,
        max_target_length=128,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.transforms = transforms
        self.max_target_length = max_target_length

        self.zip_file = zipfile.ZipFile(zip_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["image_path"]
        text = str(row["text"])

        with self.zip_file.open(image_name) as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        if self.transforms:
            image = Image.fromarray(
                self.transforms(image=np.array(image))["image"]
            )

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids

        labels = [
            l if l != self.processor.tokenizer.pad_token_id else -100
            for l in labels
        ]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# ==============================
# 5. CSV
# ==============================

train_df = pd.read_csv(
        TRAIN_CSV,
        engine='python',
        encoding='utf-8',
        on_bad_lines='warn',  # или 'skip'
        sep=',',  # явно указываем разделитель
    )
val_df = pd.read_csv(
        VAL_CSV,
        engine='python',
        encoding='utf-8',
        on_bad_lines='warn',  # или 'skip'
        sep=',',  # явно указываем разделитель
    )

# ==============================
# 6. Модель
# ==============================

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# ==============================
# 7. Datasets
# ==============================

train_dataset = CyrillicHandwrittenDatasetRam(
    df=train_df,
    processor=processor,
    root_dir=TRAIN_RAM,   # ✅ RAM
    transforms=train_transforms,
)

val_dataset = CyrillicHandwrittenDatasetZip(
    df=val_df,
    processor=processor,
    zip_path=VAL_ZIP_SSD,     # ✅ SSD
    transforms=None,
)

print(f"\nДатасеты загружены. Обучение: {len(train_dataset)}, Валидация: {len(val_dataset)}")

# ==============================
# 8. Проверка доступности данных
# ==============================
first_image_path = TRAIN_RAM / train_df.iloc[0]['image_path']  # Проверим, что путь к первому файлу корректен и существует
print(f"Проверочный путь: {first_image_path} - Изображение в TRAIN датасете существует: {os.path.exists(first_image_path)}")
print(train_df.head())

first_image_path = TRAIN_RAM / val_df.iloc[0]['image_path']  # Проверим, что путь к первому файлу корректен и существует
print(f"Проверочный путь: {first_image_path} - Изображение в VAL датасете существует: {os.path.exists(first_image_path)}")
print(val_df.head())

row = val_df.iloc[0]
image_name = row["image_path"]
zip_file = zipfile.ZipFile(VAL_ZIP_SSD, "r")
with zip_file.open(image_name) as f:
    image = Image.open(io.BytesIO(f.read())).convert("RGB")
    if image:
        print(f"Проверочный путь: {image_name} - Изображение в zip-файле VAL датасета существует")