import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.linalg import eig

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


def candidate_score(r1, size1, r2, size2, n_bins=32):
    """
    Сравнивает два временных ряда Лоренца на основе их матриц перехода.

    Аргументы:
    r1 (float): Параметр r для первого ряда Лоренца.
    size1 (int): Размер первого временного ряда.
    r2 (float): Параметр r для второго ряда Лоренца.
    size2 (int): Размер второго временного ряда.
    n_bins (int): Количество ячеек (состояний) для дискретизации.

    Возвращает:
    float: Норма Фробениуса разности матриц перехода.
    """
    # 1. Генерация временных рядов
    ts1 = TimeSeries(series_type="Lorentz", size=size1, r=r1)
    ts2 = TimeSeries(series_type="Lorentz", size=size2, r=r2)

    series1 = ts1.values
    series2 = ts2.values

    # 2. Дискретизация пространства на общей сетке
    # Находим общие границы для обоих рядов
    global_min = min(series1.min(), series2.min())
    global_max = max(series1.max(), series2.max())

    # Создаем общую сетку (bins)
    bins = np.linspace(global_min, global_max, n_bins + 1)

    # Присваиваем каждому значению в рядах номер ячейки
    s1_discrete = np.digitize(series1, bins) - 1
    s2_discrete = np.digitize(series2, bins) - 1

    # Убедимся, что максимальное значение не выходит за пределы индекса N-1
    s1_discrete[s1_discrete == n_bins] = n_bins - 1
    s2_discrete[s2_discrete == n_bins] = n_bins - 1

    def build_transition_matrix(discrete_series, n_states):
        # 3. Построение матрицы переходов
        T = np.zeros((n_states, n_states))
        for i in range(len(discrete_series) - 1):
            from_state = discrete_series[i]
            to_state = discrete_series[i + 1]
            T[from_state, to_state] += 1

        # 4. Нормировка матрицы
        row_sums = T.sum(axis=1, keepdims=True)
        # Избегаем деления на ноль для состояний, из которых не было переходов
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) > 0:
            T[non_zero_rows] /= row_sums[non_zero_rows]

        return T
    print()
    T1 = build_transition_matrix(s1_discrete, n_bins)
    print(T1)
    T2 = build_transition_matrix(s2_discrete, n_bins)

    # 5. Вычисление метрики (Норма Фробениуса)
    frobenius_norm = np.linalg.norm(T1 - T2, 'fro')

    return frobenius_norm


# --- НОВАЯ ФУНКЦИЯ ---
def find_optimal_bins_by_entropy_rate(time_series, n_bins_range, show_plot=True):
    """
    Определяет оптимальное количество бинов путем нахождения максимума
    скорости энтропии (entropy rate) марковской цепи, аппроксимирующей временной ряд.

    Аргументы:
    time_series (np.array): Временной ряд для анализа.
    n_bins_range (range or list): Диапазон количеств бинов для проверки.
    show_plot (bool): Если True, строит график зависимости.

    Возвращает:
    int: Оптимальное количество бинов.
    """
    entropy_rate_values = []

    for n_bins in tqdm(n_bins_range, desc="Расчет скорости энтропии для разного числа бинов"):
        # 1. Дискретизация
        global_min = time_series.min()
        global_max = time_series.max()
        bins = np.linspace(global_min, global_max, n_bins + 1)
        discrete_series = np.digitize(time_series, bins) - 1
        discrete_series[discrete_series == n_bins] = n_bins - 1

        # 2. Построение матрицы переходов
        T = np.zeros((n_bins, n_bins))
        for i in range(len(discrete_series) - 1):
            from_state = discrete_series[i]
            to_state = discrete_series[i + 1]
            T[from_state, to_state] += 1

        row_sums = T.sum(axis=1, keepdims=True)
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) > 0:
            T[non_zero_rows] /= row_sums[non_zero_rows]

        # 3. Вычисление стационарного распределения (pi)
        # pi - это левый собственный вектор матрицы T для собственного значения 1
        eigenvalues, eigenvectors = eig(T.T)
        one_idx = np.argmin(np.abs(eigenvalues - 1))
        stationary_distribution = np.real(eigenvectors[:, one_idx])
        stationary_distribution /= stationary_distribution.sum()

        # 4. Вычисление скорости энтропии (entropy rate)
        # H(X) = - sum_i(pi_i * sum_j(T_ij * log2(T_ij)))
        with np.errstate(divide='ignore', invalid='ignore'):
            # p * log(p) = 0 если p = 0
            log_T = np.log2(T)
            log_T[T == 0] = 0
            H_i = -np.sum(T * log_T, axis=1)

        entropy_rate = np.sum(stationary_distribution * H_i)
        entropy_rate_values.append(entropy_rate)

    # Находим оптимальное количество бинов
    optimal_bins_index = np.argmax(entropy_rate_values)
    optimal_n_bins = n_bins_range[optimal_bins_index]

    if show_plot:
        plt.figure(figsize=(12, 7))
        plt.plot(n_bins_range, entropy_rate_values, marker='o', linestyle='-')
        plt.title('Зависимость скорости энтропии от количества бинов', fontsize=16)
        plt.xlabel('Количество бинов (n_bins)', fontsize=12)
        plt.ylabel('Скорость энтропии (биты/шаг)', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Выделяем максимум на графике
        max_entropy_rate = entropy_rate_values[optimal_bins_index]
        plt.axvline(x=optimal_n_bins, color='r', linestyle='--',
                    label=f'Оптимум: {optimal_n_bins} бинов\nЭнтропия: {max_entropy_rate:.4f} бит/шаг')
        plt.scatter(optimal_n_bins, max_entropy_rate, color='r', s=100, zorder=5)
        plt.legend()
        plt.show()

    return optimal_n_bins


# --- Пример использования ---
# Генерируем ряд Лоренца с заданными параметрами
print("Генерация временного ряда Лоренца...")
lorentz_series = TimeSeries(series_type="Lorentz", size=20000, r=28).values
print("Генерация завершена.")

# Определяем диапазон для поиска оптимального количества бинов
# Начинаем с 10 и идем до, например, 250 с шагом 5
n_bins_to_test = range(10, 1500, 20)

# Находим и выводим оптимальное количество бинов
optimal_bins = find_optimal_bins_by_entropy_rate(lorentz_series, n_bins_to_test)

print(f"\nОптимальное количество бинов для ряда Лоренца (r=28, size=10000): {optimal_bins}")