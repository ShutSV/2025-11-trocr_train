import pandas as pd
import os
from pathlib import Path
from PIL import Image

# Загрузите ваш CSV/датасет
dataset_path = "./datasets/ukr"
data_path = os.path.join(dataset_path, "ramdisk_dataset.csv")
df = pd.read_csv(data_path)  # или другой формат

print("Первые 5 строк:")
print(df.head())

print("\nКолонки:")
print(df.columns)

print("\nТипы данных:")
print(df.dtypes)

print("\nПример строки в filename:")
sample = df.iloc[0]["filename"]  # или "filename"
print(f"Тип: {type(sample)}")
print(f"Значение: {repr(sample)}")

# img = Path(dataset_path + "/images")
# img2 = Path(f"{img}/{sample}")
# print(img2)
# # image = Image.open(img).convert("RGB")
#
# IMAGES_DIR_PATH = Path(os.path.join(dataset_path, "/images"))
# print(IMAGES_DIR_PATH)


# Проверьте, содержит ли строка символы новой строки
if isinstance(sample, str) and '\n' in sample:
    print("⚠️ Обнаружены символы новой строки в данных!")
    # Разделите строку
    parts = sample.strip().split('\n')
    print(f"Найдено {len(parts)} записей в одной строке")
    for i, part in enumerate(parts[:3]):
        print(f"  Часть {i}: {part}")