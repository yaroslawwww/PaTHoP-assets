import numpy as np
from scipy.stats import mannwhitneyu


def load_col(filename, col_index):
    with open(filename, 'r') as f:
        return np.array([float(line.strip().split(',')[col_index]) for line in f if line.strip()])


metrics = {
    0: "RMSE",
    1: "Непрогнозируемые точки",
    2: "MAPE"
}

best_data = {i: load_col('best.txt', i) for i in metrics}
random_data = {i: load_col('random.txt', i) for i in metrics}

alpha = 0.05

for idx, name in metrics.items():
    best = best_data[idx]
    random = random_data[idx]

    stat, p = mannwhitneyu(best, random, alternative='less')

    print(f"\n--- Метрика: {name} ---")
    print(f"Размер выборки (best): {len(best)}")
    print(f"Размер выборки (random): {len(random)}")
    print(f"U-статистика: {stat:.4f}")
    print(f"p-значение: {p:.4f}")

    if p < alpha:
        print("Результат: Отобранные ряды-кандидаты показал статистически значимо лучший результат (меньше значения).")
    else:
        print("Результат: разница между best.txt и random.txt не является статистически значимой.")