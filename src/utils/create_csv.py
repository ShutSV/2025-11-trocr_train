import os
import pandas as pd
from pathlib import Path

def create_csv():
    images_dir_path = Path(r"D:\datasets\ukr\images")  # Исходные файлы
    labels_path = Path(r"D:\datasets\ukr\train.tsv_labels.txt")  # файл c метками
    final_csv_path = Path(r"D:\datasets\ukr\dataset.csv")  # Путь к итоговому файлу

    if os.path.exists(final_csv_path):  # если итоговый файл существует, то пропускаем
        pass
    else:  # Читаем и парсим файл с метками
        print(f"Чтение файла с метками: {labels_path}")
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = [line.strip().split('\t', 1) for line in f]
            df = pd.DataFrame(data, columns=['filename', 'text'])
            print(f"✅ Файл с метками успешно прочитан.")
        except Exception as e:
            print(f"❌ Произошла ошибка при чтении меток: {e}")
            df = pd.DataFrame()

        if not df.empty:  # Создаем финальный датасет с постоянными путями
            df['image_path'] = df['filename'].apply(lambda x: os.path.join(images_dir_path, x))
            df.to_csv(final_csv_path, index=False)  # Сохраняем DataFrame в CSV-файл на Google Drive
            print(f"--- ✅ Готовый датасет сохранен! ---\nИтоговый CSV файл находится здесь: {final_csv_path}\nИзображения находятся здесь: {images_dir_path}")
            df.head()
        else:
            print("\nНе удалось создать датасет из-за ошибок.")


if __name__ == "__main__":
    create_csv()  # создается файл меток