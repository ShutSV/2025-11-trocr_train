"""Tkinter приложение:

Загрузка/сохранение конфигурации из INI файла
Запуск/остановка обучения
Продолжение обучения с чекпоинта
Мониторинг прогресса обучения в реальном времени
Визуализация прогресса через progress bar

Особенности:

Поддержка прерывания и возобновления обучения
Конфигурация через INI файл
Реальное время отслеживание прогресса
Логирование всех операций


training_config.ini:

[training]
model_name = microsoft/trocr-base-printed
dataset_path = ./dataset
output_dir = ./trocr-model
epochs = 3
batch_size = 4
learning_rate = 5e-5
warmup_steps = 500
logging_steps = 10
eval_steps = 100
save_steps = 200
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import requests
import threading
import time
import configparser
import json
from datetime import datetime


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TrOCR Training Controller")
        self.root.geometry("800x600")

        self.api_url = "http://localhost:8000"
        self.is_monitoring = False
        self.monitor_thread = None

        self.load_config()
        self.create_widgets()
        self.load_config_to_ui()

    def load_config(self):
        """Загрузка конфигурации из INI файла"""
        self.config = configparser.ConfigParser()
        try:
            self.config.read('training_config.ini')
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию: {e}")

    def save_config(self):
        """Сохранение конфигурации в INI файл"""
        try:
            with open('training_config.ini', 'w') as configfile:
                self.config.write(configfile)
            messagebox.showinfo("Успех", "Конфигурация сохранена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию: {e}")

    def load_config_to_ui(self):
        """Загрузка конфигурации в UI"""
        try:
            if self.config.has_section('training'):
                self.model_name_var.set(
                    self.config.get('training', 'model_name', fallback='microsoft/trocr-base-printed'))
                self.dataset_path_var.set(self.config.get('training', 'dataset_path', fallback='./dataset'))
                self.output_dir_var.set(self.config.get('training', 'output_dir', fallback='./trocr-model'))
                self.epochs_var.set(self.config.get('training', 'epochs', fallback='3'))
                self.batch_size_var.set(self.config.get('training', 'batch_size', fallback='4'))
                self.learning_rate_var.set(self.config.get('training', 'learning_rate', fallback='5e-5'))
                self.warmup_steps_var.set(self.config.get('training', 'warmup_steps', fallback='500'))
                self.logging_steps_var.set(self.config.get('training', 'logging_steps', fallback='10'))
                self.eval_steps_var.set(self.config.get('training', 'eval_steps', fallback='100'))
                self.save_steps_var.set(self.config.get('training', 'save_steps', fallback='200'))
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации в UI: {e}")

    def save_config_from_ui(self):
        """Сохранение конфигурации из UI"""
        if not self.config.has_section('training'):
            self.config.add_section('training')

        self.config.set('training', 'model_name', self.model_name_var.get())
        self.config.set('training', 'dataset_path', self.dataset_path_var.get())
        self.config.set('training', 'output_dir', self.output_dir_var.get())
        self.config.set('training', 'epochs', self.epochs_var.get())
        self.config.set('training', 'batch_size', self.batch_size_var.get())
        self.config.set('training', 'learning_rate', self.learning_rate_var.get())
        self.config.set('training', 'warmup_steps', self.warmup_steps_var.get())
        self.config.set('training', 'logging_steps', self.logging_steps_var.get())
        self.config.set('training', 'eval_steps', self.eval_steps_var.get())
        self.config.set('training', 'save_steps', self.save_steps_var.get())

        self.save_config()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Конфигурация колонок и строк
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Переменные для полей ввода
        self.model_name_var = tk.StringVar(value="microsoft/trocr-base-printed")
        self.dataset_path_var = tk.StringVar(value="./dataset")
        self.output_dir_var = tk.StringVar(value="./trocr-model")
        self.epochs_var = tk.StringVar(value="3")
        self.batch_size_var = tk.StringVar(value="4")
        self.learning_rate_var = tk.StringVar(value="5e-5")
        self.warmup_steps_var = tk.StringVar(value="500")
        self.logging_steps_var = tk.StringVar(value="10")
        self.eval_steps_var = tk.StringVar(value="100")
        self.save_steps_var = tk.StringVar(value="200")

        # Создание вкладок
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Вкладка конфигурации
        config_frame = ttk.Frame(notebook, padding="10")
        notebook.add(config_frame, text="Конфигурация")

        # Вкладка обучения
        training_frame = ttk.Frame(notebook, padding="10")
        notebook.add(training_frame, text="Обучение")

        # Вкладка логов
        logs_frame = ttk.Frame(notebook, padding="10")
        notebook.add(logs_frame, text="Логи")

        # Настройка конфигурации
        self.create_config_tab(config_frame)

        # Настройка вкладки обучения
        self.create_training_tab(training_frame)

        # Настройка вкладки логов
        self.create_logs_tab(logs_frame)

        # Кнопки управления
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Сохранить конфигурацию",
                   command=self.save_config_from_ui).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Загрузить конфигурацию",
                   command=self.load_config_to_ui).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Обновить статус",
                   command=self.check_status).pack(side=tk.LEFT, padx=5)

    def create_config_tab(self, parent):
        """Создание вкладки конфигурации"""
        row = 0

        # Модель
        ttk.Label(parent, text="Модель:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.model_name_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E),
                                                                           pady=2)
        row += 1

        # Путь к датасету
        ttk.Label(parent, text="Путь к датасету:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.dataset_path_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E),
                                                                             pady=2)
        row += 1

        # Выходная директория
        ttk.Label(parent, text="Выходная директория:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.output_dir_var, width=50).grid(row=row, column=1, sticky=(tk.W, tk.E),
                                                                           pady=2)
        row += 1

        # Эпохи
        ttk.Label(parent, text="Количество эпох:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.epochs_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Размер батча
        ttk.Label(parent, text="Размер батча:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.batch_size_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Learning rate
        ttk.Label(parent, text="Learning rate:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.learning_rate_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Warmup steps
        ttk.Label(parent, text="Warmup steps:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.warmup_steps_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Logging steps
        ttk.Label(parent, text="Logging steps:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.logging_steps_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Eval steps
        ttk.Label(parent, text="Eval steps:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.eval_steps_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # Save steps
        ttk.Label(parent, text="Save steps:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self.save_steps_var, width=20).grid(row=row, column=1, sticky=tk.W, pady=2)

        parent.columnconfigure(1, weight=1)

    def create_training_tab(self, parent):
        """Создание вкладки обучения"""
        # Статус обучения
        status_frame = ttk.LabelFrame(parent, text="Статус обучения", padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=8, width=70)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(parent, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # Метка прогресса
        self.progress_label = ttk.Label(parent, text="Готов к обучению")
        self.progress_label.grid(row=2, column=0, sticky=tk.W)

        # Кнопки управления обучением
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, pady=10)

        ttk.Button(button_frame, text="Начать обучение",
                   command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Продолжить обучение",
                   command=self.resume_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Остановить обучение",
                   command=self.stop_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Начать мониторинг",
                   command=self.start_monitoring).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Остановить мониторинг",
                   command=self.stop_monitoring).pack(side=tk.LEFT, padx=5)

        parent.columnconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

    def create_logs_tab(self, parent):
        """Создание вкладки логов"""
        self.logs_text = scrolledtext.ScrolledText(parent, height=15, width=80)
        self.logs_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Button(parent, text="Очистить логи",
                   command=self.clear_logs).grid(row=1, column=0, pady=5)

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def log_message(self, message):
        """Добавление сообщения в логи"""
        self.logs_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        self.logs_text.config(state=tk.DISABLED)

    def clear_logs(self):
        """Очистка логов"""
        self.logs_text.config(state=tk.NORMAL)
        self.logs_text.delete(1.0, tk.END)
        self.logs_text.config(state=tk.DISABLED)

    def update_status_display(self, status_data):
        """Обновление отображения статуса"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)

        status_info = f"""Статус: {status_data.get('status', 'unknown')}
Обучение активно: {status_data.get('is_training', False)}
Эпоха: {status_data.get('current_epoch', 0)}/{status_data.get('total_epochs', 0)}
Шаг: {status_data.get('current_step', 0)}/{status_data.get('total_steps', 0)}
Loss: {status_data.get('loss', 'N/A')}
Learning Rate: {status_data.get('learning_rate', 'N/A')}
Последнее обновление: {status_data.get('last_update', 'N/A')}
"""

        self.status_text.insert(tk.END, status_info)
        self.status_text.config(state=tk.DISABLED)

        # Обновление прогресс бара
        if status_data.get('total_steps', 0) > 0:
            progress = (status_data.get('current_step', 0) / status_data.get('total_steps', 1)) * 100
            self.progress_var.set(progress)
            self.progress_label.config(
                text=f"Прогресс: {status_data.get('current_step', 0)}/{status_data.get('total_steps', 0)} "
                     f"({progress:.1f}%)"
            )

    def start_training(self):
        """Запуск обучения"""

        def train_thread():
            try:
                response = requests.post(f"{self.api_url}/start_training")
                if response.status_code == 200:
                    self.log_message("Обучение успешно запущено")
                else:
                    self.log_message(f"Ошибка при запуске обучения: {response.text}")
            except Exception as e:
                self.log_message(f"Ошибка подключения к серверу: {e}")

        threading.Thread(target=train_thread, daemon=True).start()

    def resume_training(self):
        """Продолжение обучения с чекпоинта"""

        def resume_thread():
            try:
                response = requests.post(f"{self.api_url}/start_training?resume=true")
                if response.status_code == 200:
                    self.log_message("Обучение продолжено с чекпоинта")
                else:
                    self.log_message(f"Ошибка при продолжении обучения: {response.text}")
            except Exception as e:
                self.log_message(f"Ошибка подключения к серверу: {e}")

        threading.Thread(target=resume_thread, daemon=True).start()

    def stop_training(self):
        """Остановка обучения"""

        def stop_thread():
            try:
                response = requests.post(f"{self.api_url}/stop_training")
                if response.status_code == 200:
                    self.log_message("Обучение остановлено")
                else:
                    self.log_message(f"Ошибка при остановке обучения: {response.text}")
            except Exception as e:
                self.log_message(f"Ошибка подключения к серверу: {e}")

        threading.Thread(target=stop_thread, daemon=True).start()

    def check_status(self):
        """Проверка статуса обучения"""

        def status_thread():
            try:
                response = requests.get(f"{self.api_url}/training_status")
                if response.status_code == 200:
                    status_data = response.json()
                    self.root.after(0, lambda: self.update_status_display(status_data))
                    self.log_message("Статус обновлен")
                else:
                    self.log_message(f"Ошибка при получении статуса: {response.text}")
            except Exception as e:
                self.log_message(f"Ошибка подключения к серверу: {e}")

        threading.Thread(target=status_thread, daemon=True).start()

    def start_monitoring(self):
        """Запуск мониторинга статуса"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.log_message("Мониторинг запущен")

        def monitor():
            while self.is_monitoring:
                self.check_status()
                time.sleep(2)  # Проверка каждые 2 секунды

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Остановка мониторинга"""
        self.is_monitoring = False
        self.log_message("Мониторинг остановлен")


def main():
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
