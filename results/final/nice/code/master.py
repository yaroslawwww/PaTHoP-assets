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
        result = subprocess.check_output(f"squeue -u {user} -h | wc -l", shell=True)
        return int(result.strip())
    except Exception:
        return 0


def active_sleep(duration_seconds):
    """
    Имитирует активность, чтобы планировщик Slurm не убил процесс за простой.
    """
    start_time = time.time()
    last_log_time = start_time

    while time.time() - start_time < duration_seconds:
        _ = math.factorial(50000)
        time.sleep(0.1)
        current_time = time.time()
        if current_time - last_log_time >= 3600:
            elapsed_hours = (current_time - start_time) / 3600
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ждем... Прошло {elapsed_hours:.1f} ч.")
            last_log_time = current_time


# --- ИСХОДНЫЕ ДАННЫЕ И ПАРАМЕТРЫ ЭКСПЕРИМЕНТОВ ---
horizons = [1, 10, 20]
exp2_donors = [27.9, 28.0001, 28.01, 28.1]
exp3_donors = [27.9, 27.99, 28.0001, 28.01, 28.1, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]

# Параметры r для Эксперимента 4 (~Fig 5 в статье):
# Основная сетка (15-75) + дополнительная мелкая сетка 20 точек близ r = 28 +- 0.1
exp4_main = np.linspace(15, 75, 121)
exp4_fine = np.linspace(27.9, 28.1, 20)
exp4_donors = np.unique(np.concatenate((exp4_main, exp4_fine)))  # unique исключит дубликаты

# 1. СОБИРАЕМ СПИСОК УЖЕ ВЫПОЛНЕННЫХ ЗАДАЧ
completed_tasks = set()

for h in horizons:
    for path in [f"all_experiments_h{h}.csv", f"../assets/results/all_experiments_h{h}.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    exp = row['Experiment']
                    rec_size = int(row['Recipient_Size'])
                    d_size = int(row['Donor_Size'])
                    d_r = row['Donor_R']

                    d_r_float = 0.0 if str(d_r).strip() in ['None', 'nan', 'NaN'] else float(d_r)
                    task_key = (exp, rec_size, round(d_r_float, 4), d_size, h)
                    completed_tasks.add(task_key)
            except Exception as e:
                print(f"Ошибка чтения {path}: {e}")

print(f"Найдено уже выполненных задач в CSV: {len(completed_tasks)}")

# 2. ГЕНЕРИРУЕМ КОМАНДЫ ТОЛЬКО ДЛЯ НЕВЫПОЛНЕННЫХ ЗАДАЧ
commands_to_run = []

for h in horizons:
    # --- EXP1_CLEAR ---
    sizes_exp1 = np.linspace(10000, 80000, 24).astype(int)
    for size in sizes_exp1:
        if ("EXP1_CLEAR", size, 0.0, 0, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP1_CLEAR {size},0.0,0 {h}")

    # --- EXP2_MIX ---
    rec_sizes_exp2 = np.linspace(0, 10000, 12).astype(int)
    for r in exp2_donors:
        for rec_size in rec_sizes_exp2:
            donor_size = 10000 - rec_size

            # Приводим r к 0.0, если донора нет
            r_key = 0.0 if donor_size == 0 else round(r, 4)

            if ("EXP2_MIX", rec_size, r_key, donor_size, h) not in completed_tasks:
                commands_to_run.append(f"sbatch worker.slurm EXP2_MIX {rec_size},{r_key},{donor_size} {h}")
                # Добавляем в сет, чтобы не закинуть несколько одинаковых бейзлайнов за один проход
                completed_tasks.add(("EXP2_MIX", rec_size, r_key, donor_size, h))

    # --- EXP3_HYBRID ---
    total_sizes_exp3 = np.linspace(10000, 40000, 24).astype(int)
    for r in exp3_donors:
        for total_size in total_sizes_exp3:
            rec_size = 10000
            donor_size = total_size - rec_size

            # Приводим r к 0.0, если донора нет
            r_key = 0.0 if donor_size == 0 else round(r, 4)

            if ("EXP3_HYBRID", rec_size, r_key, donor_size, h) not in completed_tasks:
                commands_to_run.append(f"sbatch worker.slurm EXP3_HYBRID {rec_size},{r_key},{donor_size} {h}")
                completed_tasks.add(("EXP3_HYBRID", rec_size, r_key, donor_size, h))
    # --- EXP4: Сравнение 5000, 5000+5000, 10000 (вкл. мелкую сетку) ---
    for r in exp4_donors:
        if ("EXP4_FIXED", 5000, round(r, 4), 5000, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP4_FIXED 5000,{r},5000 {h}")

    # Бейзлайны для EXP4
    if ("EXP4_BASE_5K", 5000, 0.0, 0, h) not in completed_tasks:
        commands_to_run.append(f"sbatch worker.slurm EXP4_BASE_5K 5000,0.0,0 {h}")
    if ("EXP4_BASE_10K", 10000, 0.0, 0, h) not in completed_tasks:
        commands_to_run.append(f"sbatch worker.slurm EXP4_BASE_10K 10000,0.0,0 {h}")

# --- EXP5: Априорный метод (ТОЛЬКО ДЛЯ h=10) ---
total_sizes_exp5 = np.linspace(10000, 40000, 24).astype(int)
for r in exp3_donors:
    for total_size in total_sizes_exp5:
        rec_size = 10000
        donor_size = total_size - rec_size

        # Приводим r к 0.0, если донора нет
        r_key = 0.0 if donor_size == 0 else round(r, 4)

        if ("EXP5_APRIORI", rec_size, r_key, donor_size, 10) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP5_APRIORI {rec_size},{r_key},{donor_size} 10")
            completed_tasks.add(("EXP5_APRIORI", rec_size, r_key, donor_size, 10))

# --- ПОДГОТОВКА И ЗАПУСК ФАЙЛА МЕТРИК ДЛЯ EXP6 ---
if not os.path.exists("selected_donors.csv"):
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Файл 'selected_donors.csv' не найден. Запускаем заранее созданный metrics_worker.slurm (8 ядер)...")

    # Отправляем в очередь (предполагаем, что файл metrics_worker.slurm уже создан)
    os.system("sbatch metrics_worker.slurm")

    # Ждем появления файла
    while not os.path.exists("selected_donors.csv"):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ожидаем завершения отбора доноров (создания selected_donors.csv)...")
        active_sleep(120)  # спим по 2 минуты

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Расчет метрик завершен! Файл selected_donors.csv успешно создан.")

# --- EXP6: Сравнение отобранных Best-48 и Random-48 ---
donors_df = pd.read_csv("selected_donors.csv")
best_donors = donors_df[donors_df['Group'] == 'Best']['r'].values
rand_donors = donors_df[donors_df['Group'] == 'Random']['r'].values

for h in horizons:
    for r in best_donors:
        if ("EXP6_BEST", 10000, round(r, 4), 20000, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP6_BEST 10000,{r},20000 {h}")
    for r in rand_donors:
        if ("EXP6_RANDOM", 10000, round(r, 4), 20000, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP6_RANDOM 10000,{r},20000 {h}")

    # Бейзлайны для EXP6 (оригинальный ряд без донора и с увеличенным объемом)
    if ("EXP6_BASE_10K", 10000, 0.0, 0, h) not in completed_tasks:
        commands_to_run.append(f"sbatch worker.slurm EXP6_BASE_10K 10000,0.0,0 {h}")
    if ("EXP6_BASE_30K", 30000, 0.0, 0, h) not in completed_tasks:
        commands_to_run.append(f"sbatch worker.slurm EXP6_BASE_30K 30000,0.0,0 {h}")
tpu_metrics_file = "tpu_ensemble_metrics_final.csv"
if os.path.exists(tpu_metrics_file):
    tpu_donors_df = pd.read_csv(tpu_metrics_file)
    tpu_selected_donors = tpu_donors_df[tpu_donors_df['Group'] == 'Selected']['r'].values
    tpu_rand_donors = tpu_donors_df[tpu_donors_df['Group'] == 'Random']['r'].values

    for h in horizons:
        for r in tpu_selected_donors:
            if ("EXP7_SELECTED", 10000, round(r, 4), 20000, h) not in completed_tasks:
                commands_to_run.append(f"sbatch worker.slurm EXP7_SELECTED 10000,{r},20000 {h}")
        for r in tpu_rand_donors:
            if ("EXP7_RANDOM", 10000, round(r, 4), 20000, h) not in completed_tasks:
                commands_to_run.append(f"sbatch worker.slurm EXP7_RANDOM 10000,{r},20000 {h}")

        # Добавляем бейзлайны для 7-го эксперимента, чтобы было удобно собирать статистику
        if ("EXP7_BASE_10K", 10000, 0.0, 0, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP7_BASE_10K 10000,0.0,0 {h}")
        if ("EXP7_BASE_30K", 30000, 0.0, 0, h) not in completed_tasks:
            commands_to_run.append(f"sbatch worker.slurm EXP7_BASE_30K 30000,0.0,0 {h}")
else:
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Файл '{tpu_metrics_file}' не найден. Пропускаем Эксперимент 7.")

print(f"Осталось запустить задач: {len(commands_to_run)}")

# 3. УМНАЯ ОТПРАВКА С ПРОВЕРКОЙ ОЧЕРЕДИ
batch_size = 20

for i in range(0, len(commands_to_run), batch_size):
    # Ждем, пока освободится место в очереди
    while get_active_jobs_count() >= MAX_QUEUE_SIZE:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Очередь заполнена (лимит {MAX_QUEUE_SIZE}). Ждем 2 минуты...")
        active_sleep(120)

    batch = commands_to_run[i:i + batch_size]
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Отправка батча (задачи {i + 1}-{min(i + batch_size, len(commands_to_run))})...")

    for cmd in batch:
        os.system(cmd)

print("Все оставшиеся задачи успешно отправлены в Slurm!")