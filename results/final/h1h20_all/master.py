# coding: utf-8
import os
import pandas as pd
import numpy as np
import time
import math
import subprocess

# Настройки лимита очереди (поправьте под лимиты вашего кластера)
# Если в очереди уже MAX_QUEUE_SIZE ваших задач, скрипт будет ждать.
MAX_QUEUE_SIZE = 50


def get_active_jobs_count():
    """Возвращает количество задач текущего пользователя в очереди Slurm."""
    try:
        user = "ikvasilev"
        # Запрашиваем только задачи пользователя, считаем строки
        result = subprocess.check_output(f"squeue -u {user} -h | wc -l", shell=True)
        return int(result.strip())
    except Exception:
        return 0


def active_sleep(duration_seconds):
    """
    Имитирует активность, чтобы планировщик Slurm не убил процесс за простой.
    Создает пульсирующую нагрузку: считает факториал, затем спит 10 секунд.
    """
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < duration_seconds:
        # 1. Фоновая нагрузка: вычисление факториала 50 000
        # (длинная арифметика в Python нагружает CPU на долю секунды)
        _ = math.factorial(50000)

        # 2. Короткий сон, чтобы не сжигать 100% CPU впустую
        time.sleep(0.1)

        # 3. Раз в час пишем в лог, что мы живы (полезно для долгих задач)
        current_time = time.time()
        if current_time - last_log_time >= 3600:
            elapsed_hours = (current_time - start_time) / 3600
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ждем... Прошло {elapsed_hours:.1f} ч. до следующего батча.")
            last_log_time = current_time


# --- ИСХОДНЫЕ ДАННЫЕ ИЗ ВАШЕГО MASTER.PY ---
horizons = [1, 20]
exp2_donors = [27.9, 28.0001, 28.01, 28.1]
exp3_donors = [27.9, 27.99, 28.0001, 28.01, 28.1, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]

# 1. СОБИРАЕМ СПИСОК УЖЕ ВЫПОЛНЕННЫХ ЗАДАЧ
completed_tasks = set()

for h in horizons:
    # Ищем файлы в текущей директории или по пути из eval_worker.py
    for path in [f"all_experiments_h{h}.csv", f"../assets/results/all_experiments_h{h}.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    exp = row['Experiment']
                    rec_size = int(row['Recipient_Size'])
                    d_size = int(row['Donor_Size'])
                    d_r = row['Donor_R']

                    # Обработка None для EXP1_CLEAR
                    d_r_float = 0.0 if str(d_r).strip() in ['None', 'nan', 'NaN'] else float(d_r)

                    # Уникальный ключ: (Эксперимент, Размер реципиента, r донора, Размер донора, горизонт)
                    task_key = (exp, rec_size, round(d_r_float, 4), d_size, h)
                    completed_tasks.add(task_key)
            except Exception as e:
                print(f"Ошибка чтения {path}: {e}")

print(f"Найдено уже выполненных задач в CSV: {len(completed_tasks)}")

# 2. ГЕНЕРИРУЕМ КОМАНДЫ ТОЛЬКО ДЛЯ НЕВЫПОЛНЕННЫХ ЗАДАЧ
commands_to_run = []

for h in horizons:
    # EXP1_CLEAR
    sizes_exp1 = np.linspace(10000, 40000, 24).astype(int)
    for size in sizes_exp1:
        task_key = ("EXP1_CLEAR", size, 0.0, 0, h)
        if task_key not in completed_tasks:
            cmd = f"sbatch worker.slurm EXP1_CLEAR {size},0.0,0 {h}"
            commands_to_run.append(cmd)

    # EXP2_MIX
    rec_sizes_exp2 = np.linspace(0, 10000, 12).astype(int)
    for r in exp2_donors:
        for rec_size in rec_sizes_exp2:
            donor_size = 10000 - rec_size
            task_key = ("EXP2_MIX", rec_size, round(r, 4), donor_size, h)
            if task_key not in completed_tasks:
                cmd = f"sbatch worker.slurm EXP2_MIX {rec_size},{r},{donor_size} {h}"
                commands_to_run.append(cmd)

    # EXP3_HYBRID
    total_sizes_exp3 = np.linspace(10000, 40000, 24).astype(int)
    for r in exp3_donors:
        for total_size in total_sizes_exp3:
            rec_size = 10000
            donor_size = total_size - rec_size
            task_key = ("EXP3_HYBRID", rec_size, round(r, 4), donor_size, h)
            if task_key not in completed_tasks:
                cmd = f"sbatch worker.slurm EXP3_HYBRID {rec_size},{r},{donor_size} {h}"
                commands_to_run.append(cmd)

print(f"Осталось запустить задач: {len(commands_to_run)}")

# 3. УМНАЯ ОТПРАВКА С ПРОВЕРКОЙ ОЧЕРЕДИ
batch_size = 20  # Отправляем небольшими порциями

for i in range(0, len(commands_to_run), batch_size):
    # Ждем, пока освободится место в очереди
    while get_active_jobs_count() >= MAX_QUEUE_SIZE:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Очередь заполнена (лимит {MAX_QUEUE_SIZE}). Ждем 2 минуты...")
        # Используем ВАШУ функцию "активного" сна
        active_sleep(1200)

    batch = commands_to_run[i:i + batch_size]
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Отправка батча (задачи {i + 1}-{min(i + batch_size, len(commands_to_run))})...")

    for cmd in batch:
        os.system(cmd)

print("Все оставшиеся задачи успешно отправлены!")