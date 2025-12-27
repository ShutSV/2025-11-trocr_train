import lmdb
import pickle
import io
import psutil
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


# ==============================
# 1. Пути
# ==============================
TRAIN_DIR_LMDB = Path(r"d:\datasets\rus\datasets\train_lmdb\data.mdb")
VAL_DIR_LMDB = Path(r"d:\datasets\rus\datasets\val_lmdb\data.mdb")
MODEL_NAME = "microsoft/trocr-small-handwritten"

# ==============================
# 4. Dataset LMDB
# ==============================

class TrOCRLMDBDataset(Dataset):
    def __init__(self, db_path, processor, max_target_length=128):
        self.db_path = db_path
        self.processor = processor
        self.max_target_length = max_target_length

        # Открываем среду LMDB
        self.env = lmdb.open(
            db_path,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def get_ram_stats(self):
        vm = psutil.virtual_memory()
        cached = getattr(vm, 'cached', 0) / (1024 ** 3)
        available = vm.available / (1024 ** 3)
        return f"RAM: {vm.percent}% | Cached: {cached:.1f}GB | Avail: {available:.1f}GB"

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            key = f'sample_{idx:08d}'.encode('ascii')
            data = txn.get(key)

        if not data:
            return None

        # Извлекаем байты и текст
        sample = pickle.loads(data)
        image = Image.open(io.BytesIO(sample['image'])).convert("RGB")
        text = sample['label']

        # Обработка через процессор Hugging Face
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze()

        # Токенизация текста
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        # Важно: заменяем токен паддинга на -100, чтобы он игнорировался в Loss
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

# ==============================
# 3. Аугментации
# ==============================



# ==============================
# 5. Processor
# ==============================

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

# ==============================
# 7. Datasets
# ==============================

# 1. Инициализация наших LMDB датасетов
print("🚀 Загрузка LMDB датасетов...")
train_dataset = TrOCRLMDBDataset(
    db_path=str(TRAIN_DIR_LMDB),
    processor=processor
)
val_dataset = TrOCRLMDBDataset(
    db_path=str(VAL_DIR_LMDB),
    processor=processor
)

print(f"📊 {train_dataset.get_ram_stats()}")

print(f"\nДатасеты загружены. Обучение: {len(train_dataset)}, Валидация: {len(val_dataset)}")
