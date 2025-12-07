import os
import subprocess
import sys
import time
import shutil

# === Настройки ===
LOG_DIR = r"D:\DOC\2025-11-trocr_train\logs"
TENSORBOARD_PORT = 6006

def kill_process_by_name(name):
    """Завершить все процессы по имени"""
    try:
        if os.name == "nt":  # Windows
            subprocess.call(f"taskkill /F /IM {name}", shell=True)
        else:  # Linux / macOS
            subprocess.call(f"pkill -f {name}", shell=True)
    except Exception as e:
        print(f"[WARN] Не удалось завершить процессы {name}: {e}")

def start_tensorboard():
    """Запуск TensorBoard"""
    print("[INFO] Останавливаю старые процессы TensorBoard...")
    kill_process_by_name("tensorboard.exe")
    kill_process_by_name("tensorboard")

    print("[INFO] Запускаю TensorBoard...")
    cmd = [
        sys.executable, "-m", "tensorboard",
        f"--logdir={LOG_DIR}",
        f"--port={TENSORBOARD_PORT}",
        "--reload_multifile", "true",
        "--host=0.0.0.0"
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    print(f"[READY] TensorBoard локально: http://localhost:{TENSORBOARD_PORT}")

def start_cloudflare_tunnel():
    """Запуск Cloudflare Tunnel"""
    print("[INFO] Запускаю Cloudflare Tunnel...")
    # if not shutil.which("cloudflared"):
    #     print("[INSTALL] Устанавливаю cloudflared...")
    #     subprocess.check_call("winget install --id Cloudflare.cloudflared -e --source winget", shell=True)

    cmd = ["cloudflared", "tunnel", "--url", f"http://localhost:{TENSORBOARD_PORT}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    public_url = None
    for line in proc.stdout:
        if "trycloudflare.com" in line:
            public_url = line.strip().split(" ")[-1]
            break

    if public_url:
        print(f"[READY] TensorBoard доступен извне: {public_url}")
    else:
        print("[ERROR] Не удалось получить ссылку Cloudflare Tunnel")

if __name__ == "__main__":
    start_tensorboard()  # 1. Запускаем TensorBoard
    start_cloudflare_tunnel()  # 2. Запускаем туннель для внешнего доступа

