import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd

def load_col(filename, col_index):
    with open(filename, 'r') as f:
        return np.array([float(line.strip().split(',')[col_index]) for line in f if line.strip()])


metrics = {
    0: "RMSE",
    1: "Непрогнозируемые точки",
    2: "MAPE"
}

best_data = {i: load_col('best_480.txt', i) for i in metrics}
perron = {i: load_col('best_48.txt', i) for i in metrics}
shamfer = {i: load_col('best_48_shamfer_only.txt', i)[:48] for i in metrics}
alpha = 0.05
print("среднее для лучших по двум метрикам:",best_data[0].mean())
print("среднее для лучших по одной метрике Perron:", perron[0].mean())
print("среднее для лучших по одной метрике shamfer:", shamfer[0].mean())
print("дисперсия для лучших по двум метрикам:",best_data[0].var())
print("дисперсия для лучших по одной метрике Perron:", perron[0].var())
print("дисперсия для лучших по одной метрике shamfer:", shamfer[0].var())

print("медиана для лучших по двум метрикам:",pd.Series(best_data[0]).median())
print("медиана для лучших по одной метрике Perron:", pd.Series(perron[0]).median())
print("медиана для лучших по одной метрике shamfer:", pd.Series(shamfer[0]).median())
print("максимальное для лучших по двум метрикам:",best_data[0].max())
print("максимальное для лучших по одной метрике Perron:", perron[0].max())
print("максимальное для лучших по одной метрике shamfer:", shamfer[0].max())



print("дисперсия для лучших по двум метрикам:",best_data[0].var()/best_data[0].var())
print("дисперсия для лучших по одной метрике Perron:", perron[0].var()/best_data[0].var())
print("дисперсия для лучших по одной метрике shamfer:", shamfer[0].var()/best_data[0].var())

# for idx, name in metrics.items():
#     best = best_data[idx]
#     random = random_data[idx]
#
#     stat, p = mannwhitneyu(best, random, alternative='less')
#
#     print(f"\n--- Метрика: {name} ---")
#     print(f"Размер выборки (best): {len(best)}")
#     print(f"Размер выборки (random): {len(random)}")
#     print(f"U-статистика: {stat}")
#     print(f"p-значение: {p:.10f}")
#
#     if p < alpha:
#         print("Результат: Отобранные ряды-кандидаты показал статистически значимо лучший результат (меньше значения).")
#     else:
#         print("Результат: разница между best.txt и random.txt не является статистически значимой.")