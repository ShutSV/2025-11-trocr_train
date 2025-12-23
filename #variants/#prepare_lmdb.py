import lmdb
import pandas as pd
import zipfile
import os
import pickle
import shutil
from tqdm import tqdm


def create_lmdb_dataset(zip_path, csv_path, lmdb_output_path, map_size=536870912000):  # 500 ГБ
    """
    Правильное создание LMDB датасета.
    """
    # Удаляем старую базу, если существует
    if os.path.exists(lmdb_output_path):
        shutil.rmtree(lmdb_output_path)
        print(f"Удалена старая база: {lmdb_output_path}")

    # Читаем CSV
    df = pd.read_csv(csv_path)
    print(f"Всего записей в CSV: {len(df)}")

    # Создаем LMDB окружение
    env = lmdb.open(
        lmdb_output_path,
        map_size=map_size,
        writemap=True,  # Важно для Windows!
        max_readers=1024,
        lock=False  # Меньше проблем с блокировками
    )

    count = 0
    batch_size = 100000

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Получаем список файлов в архиве
        archive_files = set(zf.namelist())
        print(f"Файлов в архиве: {len(archive_files)}")

        txn = env.begin(write=True)

        try:
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Создание LMDB"):
                img_name = row['image_path']
                label = str(row['text'])

                # Проверяем, есть ли файл в архиве
                if img_name not in archive_files:
                    print(f"Файл отсутствует в архиве: {img_name}")
                    continue

                try:
                    # Читаем изображение
                    with zf.open(img_name) as f:
                        img_bytes = f.read()

                    # Упаковываем данные
                    data = pickle.dumps({
                        'image': img_bytes,
                        'label': label,
                        'index': i
                    })

                    # Записываем в LMDB
                    txn.put(f'{i:010d}'.encode('ascii'), data)
                    count += 1

                    # Периодически коммитим
                    if count % batch_size == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                        # Периодически синхронизируем
                        if count % 10000 == 0:
                            env.sync()
                            print(f"Синхронизировано: {count}")

                except Exception as e:
                    print(f"Ошибка с файлом {img_name}: {e}")
                    continue

        except Exception as e:
            print(f"Критическая ошибка: {e}")
            txn.abort()
            raise

        finally:
            # Финальный коммит
            txn.commit()

    # Финальная синхронизация
    env.sync()
    env.close()

    print(f"Готово! Создано записей: {count}")
    print(f"LMDB база создана в: {lmdb_output_path}")

    # Проверяем размер
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(lmdb_output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    print(f"Размер LMDB базы: {total_size / 1024 ** 3:.2f} ГБ")


# Пути
base_path = r"D:\datasets\rus\datasets"
# create_lmdb_dataset(
#     os.path.join(base_path, "train.zip"),
#     os.path.join(base_path, "train.csv"),
#     os.path.join(base_path, "train_lmdb"),
#     map_size=644245094400  # 600 ГБ с запасом
# )
create_lmdb_dataset(
    os.path.join(base_path, "val.zip"),
    os.path.join(base_path, "val.csv"),
    os.path.join(base_path, "val_lmdb"),
    map_size=214748364800  # 200 ГБ с запасом
)
