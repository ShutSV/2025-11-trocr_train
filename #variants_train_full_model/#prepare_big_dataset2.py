import zipfile
import csv
from pathlib import Path
from tqdm import tqdm


# ================= НАСТРОЙКИ =================
BASE_DIR = Path(r"D:\datasets\rus")
INFO_FILE = BASE_DIR / "info" / "gt.txt"
OUT_DIR = BASE_DIR / "datasets"

TRAIN_ZIPS = [f"img-{i:02d}.zip" for i in range(9)]
VAL_ZIP = "img-09.zip"

TRAIN_ZIP_PATH = OUT_DIR / "train.zip"
TRAIN_CSV_PATH = OUT_DIR / "train.csv"
VAL_CSV_PATH = OUT_DIR / "val.csv"
# =============================================


def load_gt(gt_path):
    """
    Загружает gt.txt в словарь:
    key   -> filename.png
    value -> text
    """
    gt = {}
    bad_lines = 0

    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Чтение gt.txt"):
            line = line.rstrip("\n").strip()
            if not line:
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                bad_lines += 1
                continue

            img_path, text = parts

            # img/stackmix_hkr_00000.png -> stackmix_hkr_00000.png
            filename = img_path.replace("\\", "/").split("/")[-1]

            gt[filename] = text

    print(f"\nЗагружено меток: {len(gt):,}")
    print(f"Пропущено битых строк: {bad_lines:,}\n")

    return gt


def merge_train_zips(gt_dict):
    """
    Объединяет img-00..img-08.zip в train.zip
    и формирует train.csv
    """
    with zipfile.ZipFile(TRAIN_ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as train_zip, \
         open(TRAIN_CSV_PATH, "w", newline="", encoding="utf-8") as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=["image_path", "text"])
        writer.writeheader()

        for zip_name in TRAIN_ZIPS:
            zip_path = BASE_DIR / zip_name
            print(f"\nОбработка {zip_name}")

            with zipfile.ZipFile(zip_path, "r") as z:
                for info in tqdm(z.infolist(), leave=False):
                    if info.is_dir():
                        continue

                    data = z.read(info.filename)
                    train_zip.writestr(info, data)

                    text = gt_dict.get(info.filename)
                    if text is not None:
                        writer.writerow({
                            "image_path": info.filename,
                            "text": text
                        })


def create_val_csv(gt_dict):
    """
    Создает val.csv для img-09.zip
    """
    zip_path = BASE_DIR / VAL_ZIP

    with zipfile.ZipFile(zip_path, "r") as z, \
         open(VAL_CSV_PATH, "w", newline="", encoding="utf-8") as csv_f:

        writer = csv.DictWriter(csv_f, fieldnames=["image_path", "text"])
        writer.writeheader()

        print("\nСоздание val.csv")

        for info in tqdm(z.infolist()):
            if info.is_dir():
                continue

            text = gt_dict.get(info.filename)
            if text is not None:
                writer.writerow({
                    "image_path": info.filename,
                    "text": text
                })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Загрузка gt.txt в RAM...")
    gt_dict = load_gt(INFO_FILE)

    print("Создание train.zip и train.csv...")
    merge_train_zips(gt_dict)

    print("\nСоздание val.csv...")
    create_val_csv(gt_dict)

    print("\n✅ ГОТОВО")
    print(f"Результаты в: {OUT_DIR}")


if __name__ == "__main__":
    main()
