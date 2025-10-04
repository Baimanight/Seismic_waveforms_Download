from pathlib import Path
import time
import subprocess
import os
import sys


" Just remind the Completion"

def start_timer():
    return time.time()

def end_timer(t_start, task_name=None):
    # 如果没有传 task_name，就取当前脚本文件名
    if task_name is None:
        task_name = os.path.basename(sys.argv[0]) or "脚本"

    elapsed = time.time() - t_start
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    parts = []
    if h > 0: parts.append(f"{h}h")
    if m > 0: parts.append(f"{m}m")
    if s > 0: parts.append(f"{s}s")
    elapsed_str = " ".join(parts[:2]) if parts else "0s"

    message = f"{task_name} 完成，用时: {elapsed_str}"
    print(message)

    # 桌面通知
    try:
        subprocess.run(["notify-send", f"{task_name} 完成", message])
    except FileNotFoundError:
        pass
