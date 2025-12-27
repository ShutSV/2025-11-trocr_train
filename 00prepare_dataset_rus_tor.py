"""
WebDataset (WDS) — это отличная альтернатива LMDB, особенно если вы планируете масштабировать обучение или хранить данные в облаке.
В отличие от LMDB, который является бинарной базой с произвольным доступом, WebDataset — это последовательный формат (набор .tar архивов).

Для системы с 96 ГБ RAM WebDataset также эффективен: ОС будет кэшировать читаемые части tar-файлов в тот же самый Page Cache.
Подготовка данных (Конвертация в .tar). Для WebDataset данные нужно сначала "упаковать". Вот быстрый скрипт для создания шардов (частей):
"""


import os
import pandas as pd
import webdataset as wds
import zipfile
from tqdm import tqdm


def convert_to_wds(zip_path, csv_path, output_pattern, max_count):
    # Загружаем метаданные
    df = pd.read_csv(csv_path)
    # Создаем объект записи WebDataset
    sink = wds.ShardWriter(output_pattern, maxsize=1024**4, maxcount=max_count)

    count = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Проходим по строкам CSV
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {os.path.basename(zip_path)}"):
            img_name = row['image_path']
            label = str(row['text'])

            try:
                # Читаем байты изображения прямо из ZIP
                with zf.open(img_name) as f:
                    image_data = f.read()

                # Создаем пример (sample)
                # Ключ (.txt, .png) определяет расширение внутри tar
                sample = {
                    "__key__": img_name.split('.')[0],  # уникальный ключ без расширения
                    "png": image_data,  # сами байты картинки
                    "txt": label.encode('utf-8')  # текст метки
                }
                sink.write(sample)
                count += 1
            except KeyError:
                print(f"Файл {img_name} не найден в архиве, пропускаем...")

    sink.close()
    print(f"Готово! Обработано объектов: {count}")


# Настройки путей
base_path = r"D:\datasets\rus\datasets"
output_path = os.path.join(base_path, "wds_format")
os.makedirs(output_path, exist_ok=True)

# Функция для конвертации пути под WebDataset
def fix_path_for_wds(path):
    # Превращаем D:\path\to\file в file:D:/path/to/file
    return "file:" + path.replace("\\", "/")

# Запуск для Train и Val
if __name__ == "__main__":
    # Для тренировочного набора (создаст файлы train-000000.tar, train-000001.tar и т.д.)
    convert_to_wds(
        zip_path=os.path.join(base_path, "train.zip"),
        csv_path=os.path.join(base_path, "train.csv"),
        output_pattern=fix_path_for_wds(os.path.join(output_path, "train-%06d.tar")),
        max_count=250_000,  # Количество картинок в одном шарде
    )

    # Для валидационного набора
    # convert_to_wds(
    #     zip_path=os.path.join(base_path, "val.zip"),
    #     csv_path=os.path.join(base_path, "val.csv"),
    #     output_pattern=fix_path_for_wds(os.path.join(output_path, "val-%06d.tar")),
    #     max_count=1_000,  # Количество картинок в одном шарде
    # )



"""
Многопоточный вариант (не тестировался)

import os
import pandas as pd
import webdataset as wds
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time


class ParallelWDSCreator:
    """ '''Параллельный создатель WebDataset''' """
    
    def __init__(self, zip_path, csv_path, output_pattern, max_samples_per_shard=500000, num_workers=8):
        self.zip_path = zip_path
        self.csv_path = csv_path
        self.output_pattern = output_pattern
        self.max_samples_per_shard = max_samples_per_shard
        self.num_workers = num_workers
        
        self.df = pd.read_csv(csv_path)
        self.total_samples = len(self.df)
        
        # Блокировка для потокобезопасной записи
        self.lock = threading.Lock()
        self.sink = None
        self.current_count = 0
        self.sample_queue = Queue(maxsize=10000)  # Очередь для предобработки
        
    def process_single_sample(self, index):
        """ '''Обработка одного образца''' """
        row = self.df.iloc[index]
        img_name = row['image_path']
        label = str(row['text'])
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                with zf.open(img_name) as f:
                    image_data = f.read()
            
            if len(image_data) == 0:
                return None
            
            key = os.path.splitext(img_name)[0].replace('/', '_').replace('\\', '_')
            return {
                "index": index,
                "key": key,
                "png": image_data,
                "txt": label.encode('utf-8')
            }
        except Exception as e:
            return None
    
    def writer_thread(self):
        """ '''Поток для записи данных''' """
        while True:
            sample = self.sample_queue.get()
            if sample is None:  # Сигнал завершения
                break
            
            with self.lock:
                if self.sink is None:
                    self.sink = wds.ShardWriter(
                        self.output_pattern,
                        maxsize=100 * 1024**4,
                        maxcount=self.max_samples_per_shard
                    )
                
                wds_sample = {
                    "__key__": sample["key"],
                    "png": sample["png"],
                    "txt": sample["txt"]
                }
                self.sink.write(wds_sample)
                self.current_count += 1
                
                if self.current_count % 10000 == 0:
                    print(f"Записано: {self.current_count} образцов")
            
            self.sample_queue.task_done()
    
    def convert(self):
        """ '''Основной метод конвертации''' """
        print(f"Начало параллельной конвертации")
        print(f"Всего образцов: {self.total_samples}")
        print(f"Количество worker потоков: {self.num_workers}")
        print(f"Образцов в шарде: {self.max_samples_per_shard}")
        
        start_time = time.time()
        
        # Запускаем поток для записи
        import threading
        writer = threading.Thread(target=self.writer_thread, daemon=True)
        writer.start()
        
        # Параллельная обработка с использованием ThreadPoolExecutor
        processed = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Создаем future для каждого образца
            future_to_idx = {
                executor.submit(self.process_single_sample, idx): idx 
                for idx in range(self.total_samples)
            }
            
            # Обрабатываем результаты по мере готовности
            for future in tqdm(as_completed(future_to_idx), total=self.total_samples, desc="Обработка"):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=30)
                    if result is not None:
                        self.sample_queue.put(result)
                        processed += 1
                except Exception as e:
                    print(f"Ошибка в образце {idx}: {e}")
        
        # Сигнал завершения для writer потока
        self.sample_queue.put(None)
        writer.join(timeout=60)
        
        if self.sink:
            self.sink.close()
        
        end_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Конвертация завершена за {end_time - start_time:.2f} секунд")
        print(f"Успешно обработано: {processed}/{self.total_samples} образцов")
        print(f"Скорость: {processed / (end_time - start_time):.2f} образцов/сек")


def convert_to_wds_fast(zip_path, csv_path, output_pattern, max_samples_per_shard=500000, num_workers=8):
    """ '''Быстрая многопоточная конвертация''' """
    
    # Читаем данные
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    print(f"Всего образцов: {total_samples}")
    print(f"Используется {num_workers} worker потоков")
    
    # Открываем ZIP файл один раз (он потокобезопасен для чтения)
    zf = zipfile.ZipFile(zip_path, 'r')
    archive_files = set(zf.namelist())
    
    # Создаем writer
    sink = wds.ShardWriter(
        output_pattern,
        maxsize=100 * 1024**4,
        maxcount=max_samples_per_shard
    )
    
    # Функция для обработки одного batch
    def process_batch(batch_indices):
        results = []
        for idx in batch_indices:
            row = df.iloc[idx]
            img_name = row['image_path']
            label = str(row['text'])
            
            if img_name not in archive_files:
                continue
                
            try:
                with zf.open(img_name) as f:
                    image_data = f.read()
                
                if len(image_data) == 0:
                    continue
                
                key = os.path.splitext(img_name)[0].replace('/', '_').replace('\\', '_')
                results.append({
                    "key": key,
                    "png": image_data,
                    "txt": label.encode('utf-8')
                })
            except:
                continue
        
        return results
    
    # Разбиваем на batch'и для параллельной обработки
    batch_size = 1000
    batches = [list(range(i, min(i + batch_size, total_samples))) 
               for i in range(0, total_samples, batch_size)]
    
    start_time = time.time()
    processed = 0
    
    # Параллельная обработка batch'ей
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch in batches:
            future = executor.submit(process_batch, batch)
            futures.append(future)
        
        # Собираем и записываем результаты
        for future in tqdm(as_completed(futures), total=len(futures), desc="Обработка батчей"):
            batch_results = future.result()
            
            for result in batch_results:
                sample = {
                    "__key__": result["key"],
                    "png": result["png"],
                    "txt": result["txt"]
                }
                sink.write(sample)
                processed += 1
            
            if processed % 10000 == 0:
                print(f"Обработано: {processed} образцов")
    
    zf.close()
    sink.close()
    
    end_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Конвертация завершена за {end_time - start_time:.2f} секунд")
    print(f"Успешно обработано: {processed}/{total_samples} образцов")
    print(f"Скорость: {processed / (end_time - start_time):.2f} образцов/сек")


# Упрощенная многопоточная версия (рекомендуется)
def convert_to_wds_simple_parallel(zip_path, csv_path, output_pattern, max_samples_per_shard=500000, num_workers=4):
    """ '''Упрощенная многопоточная версия''' """
    
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    
    print(f"Всего образцов: {total_samples}")
    
    # Открываем ZIP один раз
    zf = zipfile.ZipFile(zip_path, 'r')
    archive_files = set(zf.namelist())
    
    # Создаем writer
    sink = wds.ShardWriter(
        output_pattern,
        maxsize=100 * 1024**4,
        maxcount=max_samples_per_shard
    )
    
    # Блокировка для потокобезопасной записи
    lock = threading.Lock()
    processed_counter = 0
    start_time = time.time()
    
    def process_and_write(idx):
        nonlocal processed_counter
        
        row = df.iloc[idx]
        img_name = row['image_path']
        label = str(row['text'])
        
        if img_name not in archive_files:
            return 0
        
        try:
            with zf.open(img_name) as f:
                image_data = f.read()
            
            if len(image_data) == 0:
                return 0
            
            key = os.path.splitext(img_name)[0].replace('/', '_').replace('\\', '_')
            sample = {
                "__key__": key,
                "png": image_data,
                "txt": label.encode('utf-8')
            }
            
            with lock:
                sink.write(sample)
                processed_counter += 1
                
                if processed_counter % 10000 == 0:
                    elapsed = time.time() - start_time
                    speed = processed_counter / elapsed
                    print(f"Обработано: {processed_counter}, скорость: {speed:.1f} образцов/сек")
            
            return 1
        except:
            return 0
    
    print(f"Запуск {num_workers} worker потоков...")
    
    # Параллельная обработка
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Запускаем задачи
        futures = [executor.submit(process_and_write, i) for i in range(total_samples)]
        
        # Собираем результаты
        successful = 0
        for future in tqdm(as_completed(futures), total=total_samples, desc="Конвертация"):
            successful += future.result()
    
    zf.close()
    sink.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"Конвертация завершена!")
    print(f"Время: {total_time:.2f} секунд")
    print(f"Успешно: {successful}/{total_samples} образцов")
    print(f"Скорость: {successful/total_time:.2f} образцов/сек")
    print(f"Среднее время на образец: {total_time/successful*1000:.2f} мс")


# Настройки путей
base_path = r"D:\datasets\rus\datasets"
output_path = os.path.join(base_path, "wds_format")
os.makedirs(output_path, exist_ok=True)

def fix_path_for_wds(path):
    return "file:" + path.replace("\\", "/")


if __name__ == "__main__":
    print("Выберите метод конвертации:")
    print("1. Однопоточный (оригинальный)")
    print("2. Многопоточный (упрощенный, рекомендуется)")
    print("3. Многопоточный с очередью (продвинутый)")
    
    method = 2  # Рекомендуемый вариант
    
    if method == 1:
        # Оригинальный однопоточный
        from your_original_script import convert_to_wds
        convert_to_wds(
            zip_path=os.path.join(base_path, "train.zip"),
            csv_path=os.path.join(base_path, "train.csv"),
            output_pattern=fix_path_for_wds(os.path.join(output_path, "train-%06d.tar")),
            max_samples_per_shard=500000,
        )
    elif method == 2:
        # Упрощенный многопоточный (рекомендуется)
        convert_to_wds_simple_parallel(
            zip_path=os.path.join(base_path, "train.zip"),
            csv_path=os.path.join(base_path, "train.csv"),
            output_pattern=fix_path_for_wds(os.path.join(output_path, "train-%06d.tar")),
            max_samples_per_shard=500000,
            num_workers=8  # Подберите оптимальное значение для вашей системы
        )
    elif method == 3:
        # Продвинутый многопоточный
        creator = ParallelWDSCreator(
            zip_path=os.path.join(base_path, "train.zip"),
            csv_path=os.path.join(base_path, "train.csv"),
            output_pattern=fix_path_for_wds(os.path.join(output_path, "train-%06d.tar")),
            max_samples_per_shard=500000,
            num_workers=8
        )
        creator.convert()

"""