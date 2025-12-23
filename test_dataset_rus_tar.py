import webdataset as wds
from PIL import Image
import io
import numpy as np

url = "file:D:/datasets/rus/datasets/wds_format/train-000000.tar"
# Убираем "rgb", чтобы получить сырые байты
dataset = wds.WebDataset(url).decode().to_tuple("png", "txt")

for i, (img_data, txt) in enumerate(dataset):
    # img_data здесь может быть байтами или массивом
    if isinstance(img_data, bytes):
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        # Если это уже массив, превращаем в картинку
        img = Image.fromarray(img_data).convert("RGB")

    print(f"Sample {i}: Text='{txt}', Real Size={img.size}")
    img.show()
    if i == 0: break
