# test_dataset.py
import sys
sys.path.append('src/utils')
from train_utils import CyrillicHandwrittenDataset
import pandas as pd
from transformers import TrOCRProcessor
from pathlib import Path
from torch.utils.data import DataLoader

# Загрузите небольшую часть данных
df = pd.read_csv('datasets/ukr/ramdisk_dataset.csv').head(10)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
root_dir = Path("datasets/ukr/images")

# Создайте датасет
dataset = CyrillicHandwrittenDataset(df, processor, root_dir)

# Протестируйте одну запись
print("Тест одной записи:")
item = dataset[0]
print(f"Ключи: {item.keys()}")
print(f"pixel_values shape: {item['pixel_values'].shape}")
print(f"labels shape: {item['labels'].shape}")

# Протестируйте DataLoader
print("\nТест DataLoader:")
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
for i, batch in enumerate(dataloader):
    print(f"Батч {i}:")
    print(f"  pixel_values: {batch['pixel_values'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    if i >= 2:
        break
