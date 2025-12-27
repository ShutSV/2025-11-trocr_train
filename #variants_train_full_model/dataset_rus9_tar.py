import io
from pathlib import Path
from PIL import Image
import webdataset as wds
import torch
from transformers import TrOCRProcessor
import traceback


OUTPUT_DIR = Path(r"D:\datasets\rus\datasets\wds_format")
MODEL_NAME = "microsoft/trocr-small-handwritten"

processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)  # Инициализация процессора на верхнем уровне
cyrillic_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.,!?- "
num_added_toks = processor.tokenizer.add_tokens(list(cyrillic_alphabet))
print(f"Добавлено токенов: {num_added_toks}")


class TrOCRTransform:
    def __init__(self, proc, max_len=128):
        self.proc = proc
        self.max_len = max_len

    def __call__(self, sample):
        if sample is None: return None
        img_raw, text = sample
        try:
            if isinstance(img_raw, bytes):  # Обработка изображения (поддержка и байтов, и массивов)
                image = Image.open(io.BytesIO(img_raw)).convert("RGB")
            else:
                image = img_raw.convert("RGB")
            if image.width < 5 or image.height < 5:  # Валидация размера
                print(f"Image too small: {image.size}")
                return None
            pixel_values = self.proc(image, return_tensors="pt")
            labels = self.proc.tokenizer(text, padding="max_length", max_length=self.max_len, truncation=True).input_ids  # Токенизация текста
            labels = [l if l != self.proc.tokenizer.pad_token_id else -100 for l in labels]  # Замена pad на -100
            return {"pixel_values": pixel_values.pixel_values.squeeze(), "labels": torch.tensor(labels)}
        except Exception as e:
            print(f"Error processing sample: {e}")
            traceback.print_exc()
            return None


def filter_none(sample):
    return sample is not None


def get_wds_dataset(urls, proc, max_len=128, shuffle=True):
    fixed_urls = "file:" + str(urls).replace("\\", "/")  # Исправление пути для Windows
    transformer = TrOCRTransform(proc, max_len)
    pipeline = wds.WebDataset(fixed_urls, shardshuffle=100 if shuffle else 0)
    if shuffle:
        pipeline = pipeline.shuffle(500_000)
    pipeline = (pipeline.decode().to_tuple("png", "txt").map(transformer, handler=wds.warn_and_continue).select(filter_none))
    return pipeline

# Датасеты
train_shards = str(OUTPUT_DIR / "train-{000000..000044}.tar")
val_shards = str(OUTPUT_DIR / "val-{000000..000003}.tar")

train_dataset = get_wds_dataset(train_shards, processor)
val_dataset = get_wds_dataset(val_shards, processor, shuffle=False)
