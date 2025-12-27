import zipfile
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ТИПЫ ДЛЯ ЧИСТОТЫ КОДА ---
# Значением в словаре будет кортеж: (Полный путь к ZIP-файлу, Полный внутренний путь в ZIP)
ZipInfo = Tuple[Path, str]
FileToZipMap = Dict[str, ZipInfo]


def create_full_zip_index(gt_file_path: Path, zip_files_root_path: Path) -> pd.DataFrame:
    """
    Сканирует все ZIP-архивы, строит карту, используя БАЗОВОЕ ИМЯ ФАЙЛА как ключ,
    и сопоставляет записи gt.txt с полным путем к родительскому ZIP-архиву.

    :param gt_file_path: Путь к файлу gt.txt.
    :param zip_files_root_path: Корневая папка, содержащая все ZIP-архивы (img-01.zip, и т.д.).
    :return: DataFrame с колонками ['image_path', 'text', 'zip_path'].
    """
    # 1. Сканирование ZIP-архивов (Построение карты: Базовое_имя_файла -> ZipInfo)
    zip_archive_paths = list(zip_files_root_path.glob("img-*.zip"))
    if not zip_archive_paths:
        logging.error(f"Не найдено ZIP-архивов в папке: {zip_files_root_path}")
        return pd.DataFrame()

    file_to_zip_map: FileToZipMap = {}
    logging.info(f"Начинаем сканирование {len(zip_archive_paths)} ZIP-архивов...")

    for zip_path in zip_archive_paths:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for internal_name in zf.namelist():

                    # 1. Получаем КЛЮЧ для сопоставления: БАЗОВОЕ ИМЯ ФАЙЛА (нижний регистр)
                    # 'img/stackmix_hkr_00000.png' -> 'stackmix_hkr_00000.png'
                    key_for_map = Path(internal_name).name.lower()

                    # 2. Проверка, что это не пустая строка или папка
                    if not key_for_map or internal_name.endswith('/'):
                        continue

                    # 3. Добавление в карту (обработка дубликатов)
                    if key_for_map in file_to_zip_map:
                        # Логируем дубликат, но используем первый найденный
                        logging.warning(
                            f"Дубликат файла '{key_for_map}' найден в '{zip_path.name}' и '{file_to_zip_map[key_for_map][0].name}'. "
                            f"Используем первый найденный ZIP."
                        )
                        continue

                    # 4. Сохраняем полный путь к архиву (zip_path) и полный внутренний путь (internal_name)
                    # Стандартизируем внутренний путь для надежного чтения в Dataset
                    standardized_internal_name = internal_name.replace('\\', '/')
                    file_to_zip_map[key_for_map] = (zip_path, standardized_internal_name)

        except Exception as e:
            logging.error(f"Ошибка при чтении или кэшировании ZIP {zip_path}: {e}")

    logging.info(f"✅ Карта создана. Всего уникальных путей в ZIP-архивах: {len(file_to_zip_map)}")

    # 2. Загрузка gt.txt
    try:
        df_gt_raw = pd.read_csv(
            gt_file_path,
            sep=',',
            header=None,
            names=['image_path_raw', 'text'],
            encoding='utf-8'
        )
    except FileNotFoundError:
        logging.error(f"Файл gt.txt не найден по пути: {gt_file_path}")
        return pd.DataFrame()

    logging.info(f"Загружено {len(df_gt_raw)} записей из gt.txt.")

    # Создаем КЛЮЧ для поиска: БАЗОВОЕ ИМЯ ФАЙЛА из gt.txt (нижний регистр)
    df_gt_raw['search_key'] = df_gt_raw['image_path_raw'].apply(lambda x: Path(str(x).strip()).name.lower())

    # --- Отладочный блок (Удалите после успешного запуска) ---
    sample_key_gt = df_gt_raw.iloc[0]['search_key']

    if sample_key_gt in file_to_zip_map:
        logging.info(f"✅ Успешное сопоставление для 0-й строки. Ключ: {sample_key_gt}")
    else:
        print("\n--- СТРОГОЕ НЕСОВПАДЕНИЕ ---")
        print(f"Ключ из gt.txt (0-я строка, Базовое имя): '{sample_key_gt}' (Длина: {len(sample_key_gt)})")
        print(
            f"Пример ключа из ZIP-карты:   '{list(file_to_zip_map.keys())[0]}' (Длина: {len(list(file_to_zip_map.keys())[0])})")
        print("Проверьте, что имя файла не содержит лишних символов (пробелов, кодировки).")
        # Удалите этот 'else' блок после того, как сопоставление заработает!

    # -----------------------------------------------------------

    # 3. Сопоставление и создание финального DataFrame
    def get_zip_info(search_key: str) -> pd.Series:
        # Ищем по базовому имени файла
        info: Optional[ZipInfo] = file_to_zip_map.get(search_key)
        if info:
            # Возвращаем zip_path и полный внутренний путь
            return pd.Series(info, index=['zip_path', 'image_path'])
        else:
            return pd.Series((None, None), index=['zip_path', 'image_path'])

    # Применяем функцию, которая вернет две новые колонки
    df_gt_raw[['zip_path', 'image_path']] = df_gt_raw['search_key'].apply(get_zip_info)

    # 4. Финализация DataFrame
    df_final = df_gt_raw.dropna(subset=['zip_path']).reset_index(drop=True)

    # image_path теперь содержит полный путь внутри ZIP ('img/stackmix_hkr_00000.png'),
    # который мы извлекли из карты!

    # Оставляем только нужные колонки
    df_final = df_final[['image_path', 'text', 'zip_path']]

    logging.info(f"✅ Финальный DataFrame готов. Найдено соответствие для {len(df_final)} из {len(df_gt_raw)} записей.")

    return df_final


# --- ВАШ ОСНОВНОЙ КОД ДЛЯ ЗАПУСКА ---

GT_FILE_PATH = Path(r"D:\datasets\rus\info\gt.txt")
ZIP_ROOT_DIR = Path(r"D:\datasets\rus")

df_indexed = create_full_zip_index(GT_FILE_PATH, ZIP_ROOT_DIR)

if not df_indexed.empty:
    # Сохраняем индекс (рекомендуется)
    df_indexed.to_csv(ZIP_ROOT_DIR / "dataset_full_index.csv", index=False, encoding='utf-8')
    logging.info(f"Финальный индекс сохранен в: {ZIP_ROOT_DIR / 'dataset_full_index.csv'}")
    print("\nПример финального DataFrame:")
    print(df_indexed.head())
