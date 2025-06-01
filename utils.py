import os
from datetime import datetime

def log_message(msg, log_file=None):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = f"[{now}] {msg}"
    print(log)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log + '\n') 