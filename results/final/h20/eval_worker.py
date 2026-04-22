# coding: utf-8
import sys
import os
from evaluation import evaluation

group_name = sys.argv[1]
r_value = float(sys.argv[2])
output_path = "../assets/results/final_results_h20.csv"

# Пишем заголовок, если файла нет
if not os.path.exists(output_path):
    with open(output_path, "w") as f:
        f.write("Group,r,rmse_01,np_01,mape_01,rmse_03,np_03,mape_03\n")

print("Processing Group: {0}, r: {1}".format(group_name, r_value))

if group_name == "BASELINE_10K":
    # 28.0, размер 10000
    res = evaluation(r_values=[28.0], ts_sizes=[10000], prediction_size=20)
elif group_name == "BASELINE_30K":
    # 28.0, размер 30000
    res = evaluation(r_values=[28.0], ts_sizes=[30000], prediction_size=20)
else:
    # Доноры: целевой 28.0 (10к) + донор r (20к)
    res = evaluation(r_values=[28.0, r_value], ts_sizes=[10000, 20000], prediction_size=20)

# res — это кортеж из 6 чисел: (rmse1, np1, mape1, rmse2, np2, mape2)
line = "{0},{1:.6f},{2},{3},{4},{5},{6},{7}\n".format(
    group_name, r_value, res[0], res[1], res[2], res[3], res[4], res[5]
)

with open(output_path, "a") as f:
    f.write(line)

print("Done.")
