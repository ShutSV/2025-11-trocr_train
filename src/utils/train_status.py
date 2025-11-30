import json
import os

STATUS_FILE = "training_status.json"

def init_status():
    """Создаёт пустой файл состояния"""
    status = {"status": "idle", "epoch": 0, "step": 0, "progress_pct": 0.0, "last_loss": None}
    save_status(status)

def save_status(data: dict):
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f)

def load_status() -> dict:
    if not os.path.exists(STATUS_FILE):
        init_status()
    with open(STATUS_FILE, "r") as f:
        return json.load(f)
