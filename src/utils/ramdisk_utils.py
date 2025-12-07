from collections import defaultdict
import ctypes
import logging
import os
import psutil
import subprocess
import sys
import time
from tqdm import tqdm
import zipfile


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_disk_with_progress(root_path='R:\\'):
    """Сканирует диск с отображением прогресса"""
    stats = defaultdict(lambda: {'size': 0, 'files': 0})
    # Предварительный подсчет файлов
    print("Подсчет общего количества файлов...", file=sys.stderr)
    total_files = sum(len(files) for _, _, files in os.walk(root_path))
    # Настройка tqdm для работы при импорте
    with tqdm(
            total=total_files,
            unit='file',
            file=sys.stderr,
            dynamic_ncols=True,
            disable=None,
    ) as pbar:
        for dirpath, dirnames, filenames in os.walk(root_path):
            total_size = 0
            file_count = len(filenames)
            for f in filenames:
                try:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
                except (FileNotFoundError, PermissionError) as e:
                    pbar.write(f"Ошибка доступа к файлу: {fp} ({str(e)})")
                    continue
                finally:
                    pbar.update(1)
            stats[dirpath]['size'] = total_size
            stats[dirpath]['files'] = file_count
            parts = dirpath.split(os.sep)
            for i in range(1, len(parts)):
                parent = os.sep.join(parts[:i]) + os.sep
                if parent in stats:
                    stats[parent]['size'] += total_size
                    stats[parent]['files'] += file_count
    sys.stderr.flush()
    return dict(stats)


def kill_processes_using_drive(drive_letter):
    drive_path = f"{drive_letter}:\\"
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            if proc.info['open_files']:
                for file in proc.info['open_files']:
                    if file.path and file.path.startswith(drive_path):
                        print(f"Завершаем процесс {proc.info['name']} (PID: {proc.info['pid']})")
                        proc.kill()
                        time.sleep(0.5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue


def remove_ramdisk(drive_letter='R'):
    if not os.path.exists(f"{drive_letter}:\\"):
        print(f"Диск {drive_letter}: не существует")
        return True
    try:
        # Стандартное удаление
        subprocess.run(
            ['imdisk.exe', '-d', '-m', f'{drive_letter}:'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Диск {drive_letter}: успешно удален")
        return True
    except subprocess.CalledProcessError:
        print("Попытка принудительного удаления...")
        kill_processes_using_drive(drive_letter)
        time.sleep(1)
        subprocess.run(
            ['imdisk.exe', '-D', '-m', f'{drive_letter}:'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"Диск {drive_letter}: принудительно удален")
        return True
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False


def extract_zip(images_zip_path, target_images_path):
    with zipfile.ZipFile(images_zip_path, 'r') as zf:
        file_list = zf.infolist()  # Получаем список всех файлов для tqdm, чтобы знать общее количество
        with tqdm(  # Создаем прогресс-бар, итерируясь по списку файлов
            total=len(file_list),
            desc=f"Распаковка {images_zip_path}"
        ) as pbar:
            for file in file_list:
                zf.extract(file, target_images_path)  # Распаковываем каждый файл
                pbar.update(1)  # Обновляем прогресс-бар на 1 шаг
    # os.remove(os.path.join(target_images_path, "images.zip"))


def create_imdisk_ramdisk(size_mb, drive_letter):
    """Создает RAM-диск с помощью ImDisk"""
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    if not is_admin():
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()
    try:
        imdisk_path = 'imdisk.exe'
        cmd = [
            imdisk_path,
            '-a',
            '-s', f'{size_mb}M',
            '-m', f'{drive_letter}:',
            '-p', '/fs:ntfs /q /y'
        ]
        result = subprocess.run(cmd, check=True, text=True, stdout=None, stderr=None)
        print(result.stdout)  # Выводим результат команды
        print(f"RAM-диск {drive_letter}: создан, размер {size_mb}MB")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка создания ramdisk:\n{e.stderr}")
        raise
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        raise
    time.sleep(2)
    if not os.path.exists(f"{drive_letter}:\\"):
        raise Exception(f"Диск {drive_letter}: не доступен после создания")


if __name__ == "__main__":
    # Только при прямом запуске
    print(scan_disk_with_progress())

    if remove_ramdisk('R'):
        print("Операция завершена успешно")
    else:
        print("Не удалось удалить RAM-диск")
        input("Нажмите Enter для выхода...")



    source_folder = r"D:\datasets\ukr"  # Используйте сырые строки для путей
    ramdisk_letter = "R"
    ramdisk_size_mb = 30000
    try:
        create_imdisk_ramdisk(size_mb=ramdisk_size_mb, drive_letter=ramdisk_letter)  # Создаем RAM-диск
        extract_zip(r"D:\datasets\ukr\images.zip", r"R:")
    except Exception as e:
        print(f"\nОшибка в основном потоке: {e}")
        input("Нажмите Enter для выхода...")
