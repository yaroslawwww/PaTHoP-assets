import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde, wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon, cdist
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Классы Lorentz и TimeSeries остаются без изменений ---
class Lorentz:
    def __init__(self, s=10, b=8 / 3):
        self.s = s
        self.b = b
        self.r = None

    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return -x * z + r * x - y

    def Z(self, x, y, z, b):
        return x * y - b * z

    def RK4(self, x, y, z, s, r, b, dt):
        k_1 = self.X(x, y, s)
        l_1 = self.Y(x, y, z, r)
        m_1 = self.Z(x, y, z, b)
        k_2 = self.X((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), s)
        l_2 = self.Y((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), r)
        m_2 = self.Z((x + k_1 * dt * 0.5), (y + l_1 * dt * 0.5), (z + m_1 * dt * 0.5), b)
        k_3 = self.X((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), s)
        l_3 = self.Y((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), r)
        m_3 = self.Z((x + k_2 * dt * 0.5), (y + l_2 * dt * 0.5), (z + m_2 * dt * 0.5), b)
        k_4 = self.X((x + k_3 * dt), (y + l_3 * dt), s)
        l_4 = self.Y((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), r)
        m_4 = self.Z((x + k_3 * dt), (y + l_3 * dt), (z + m_3 * dt), b)
        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1 / 6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1 / 6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1 / 6)
        return x, y, z

    def generate(self, dt, steps, r=28):
        x_0, y_0, z_0 = 1, 1, 1
        x_list, y_list, z_list = [x_0], [y_0], [z_0]
        self.r = r
        for _ in range(steps):
            x, y, z = self.RK4(x_list[-1], y_list[-1], z_list[-1], self.s, self.r, self.b, dt)
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        return np.array(x_list), np.array(y_list), np.array(z_list)


class TimeSeries:
    def __init__(self, series_type="Lorentz", size=0, r=28, dt=0.01, array=None):
        if series_type == "Lorentz":
            divisor = int(0.1 / dt)
            x, _, _ = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            self.values = x[::divisor]
        else:
            self.values = np.array(array)

from tslearn.metrics import dtw


def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    """
    Реконструирует фазовое пространство из временного ряда методом задержек.

    Args:
        x: 1D временной ряд.
        dim: Размерность вложения.
        delay: Временная задержка.

    Returns:
        2D массив, где каждая строка - это точка на аттракторе.
    """
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий для заданных параметров dim и delay.")

    # Создаем массив точек аттрактора
    # np.column_stack эффективно собирает столбцы в матрицу
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor


def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий для заданных параметров dim и delay.")
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor




def chamfer_distance_metric(x1: np.ndarray, x2: np.ndarray, dim: int = 3, delay: int = 10) -> float:
    y1 = reconstruct_attractor(x1, dim, delay)
    y2 = reconstruct_attractor(x2, dim, delay)

    tree1 = KDTree(y1)
    tree2 = KDTree(y2)

    dist_y1_to_y2, _ = tree2.query(y1, k=1, workers=-1)
    dist_y2_to_y1, _ = tree1.query(y2, k=1, workers=-1)

    chamfer_dist = np.mean(dist_y1_to_y2 ** 2) + np.mean(dist_y2_to_y1 ** 2)

    return chamfer_dist


def candidate_score(r1, size1, r2, size2):

    ts1 = TimeSeries(series_type="Lorentz", size=size1, r=r1)
    ts2 = TimeSeries(series_type="Lorentz", size=size2, r=r2)
    x1, x2 = ts1.values, ts2.values
    return chamfer_distance_metric(x1, x2)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Подготовка данных
metric_names_orig = ['RMSE', 'Unpredictable Points', 'MAPE']
metric_names_display = ['RMSE', 'Unpredictable Points, %', 'MAPE']
col_names = metric_names_orig + ['Parameter r']

# Считываем только необходимые данные
df_best = pd.read_csv('best.txt', header=None, names=col_names)
df_random = pd.read_csv('random.txt', header=None, names=col_names)
# df_best = pd.concat([df_best, df_random], axis=0)
df_baselines = pd.read_csv('baselines.txt', header=None, names=metric_names_orig)
rs = df_best['Parameter r']
scores = []
for r in tqdm(rs):
    scores.append(candidate_score(28,10000,r,20000))
# print(scores)
sorted_indices = np.argsort(scores)
sorted_indices = list(sorted_indices)
best_indices = sorted_indices[:10]
print(np.array(scores)[best_indices])
print(np.array(rs)[best_indices])

df_best = df_best.iloc[best_indices]
# Преобразуем 'Unpredictable Points' в проценты
df_best['Unpredictable Points'] = df_best['Unpredictable Points'] * 100
df_baselines['Unpredictable Points'] = df_baselines['Unpredictable Points'] * 100

# Добавляем столбец с названием метода
df_best['Method'] = 'Best Augmentation'

# Переименовываем столбцы для корректного отображения
df_best.rename(columns={'Unpredictable Points': 'Unpredictable Points, %'}, inplace=True)
df_baselines.rename(columns={'Unpredictable Points': 'Unpredictable Points, %'}, inplace=True)

# Преобразуем датафрейм в "длинный" формат для удобства визуализации
df_long = df_best.melt(
    id_vars=['Method'],
    value_vars=metric_names_display,
    var_name='Metric',
    value_name='Value'
)

# 2. Визуализация
sns.set_theme(style="whitegrid")
color_best = "#3498db"

# Создаем сетку графиков 1x3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Comparison of Best Augmentation Results with Baselines', fontsize=18, y=1.0)

# Проходим по каждой метрике и строим график
for i, metric_name in enumerate(metric_names_display):
    ax = axes[i]

    # Выбираем данные для текущей метрики
    metric_data = df_long[df_long['Metric'] == metric_name]

    # Строим stripplot
    sns.stripplot(
        data=metric_data, y='Value',
        color=color_best, ax=ax,
        alpha=0.6, jitter=0.25, size=6
    )

    # Добавляем горизонтальные линии базовых значений
    baseline_10k = df_baselines.loc[0, metric_name]
    baseline_30k = df_baselines.loc[1, metric_name]
    ax.axhline(baseline_10k, ls='--', color='gray', lw=2)
    ax.axhline(baseline_30k, ls=':', color='black', lw=2)

    # Настраиваем внешний вид графика
    ax.set_title(metric_name, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])

    # Добавляем подпись оси Y только для первого графика
    if i == 0:
        ax.set_ylabel('Best Augmentation', fontsize=14, labelpad=15)

# 3. Создание общей легенды
handles = [
    Line2D([0], [0], color='gray', linestyle='--', lw=2),
    Line2D([0], [0], color='black', linestyle=':', lw=2)
]
labels = [
    'Original (10k)',
    'Original (30k)'
]
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Best_Augmentation_benefit_visualisation.png')
plt.show()
















