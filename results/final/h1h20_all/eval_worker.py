# coding: utf-8
import sys
import os
from evaluation import evaluation

if len(sys.argv) < 4:
    print("Использование: python eval_worker.py <group_name> <param_string> <horizon>")
    sys.exit(1)

experiment_name = sys.argv[1]
param_string = sys.argv[2]
horizon = int(sys.argv[3])

# Общий файл для результатов
output_path = f"../assets/results/all_experiments_h{horizon}.csv"

# Парсим строку параметров
rec_size_str, donor_r_str, donor_size_str = param_string.split(',')
rec_size = int(rec_size_str)
donor_r = float(donor_r_str)
donor_size = int(donor_size_str)

# Безопасное создание заголовка
if not os.path.exists(output_path):
    try:
        with open(output_path, "x") as f:
            f.write("Experiment,Recipient_Size,Donor_R,Donor_Size,rmse_01,np_01,mape_01,rmse_03,np_03,mape_03\n")
    except FileExistsError:
        pass

print(f"[{experiment_name} h={horizon}] Rec={rec_size}, D_r={donor_r}, D_size={donor_size}")

# Формируем списки для evaluation (функция evaluation сама добавит тестовый ряд в начало)
r_vals = []
ts_sizes = []

if rec_size > 0:
    r_vals.append(28.0) # Оригинальный ряд (Реципиент)
    ts_sizes.append(rec_size)

if donor_size > 0:
    r_vals.append(donor_r) # Чужой ряд (Донор)
    ts_sizes.append(donor_size)

# Запуск алгоритма
res = evaluation(r_values=r_vals, ts_sizes=ts_sizes, prediction_size=horizon)

# Форматируем вывод (если донора нет, запишем None для удобства аналитики)
out_donor_r = f"{donor_r:.4f}" if donor_size > 0 else "None"

line = f"{experiment_name},{rec_size},{out_donor_r},{donor_size},{res[0]:.6f},{res[1]:.6f},{res[2]:.6f},{res[3]:.6f},{res[4]:.6f},{res[5]:.6f}\n"

with open(output_path, "a") as f:
    f.write(line)

print("Done.")