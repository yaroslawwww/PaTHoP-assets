
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr, entropy

# --- ИСХОДНЫЕ ПАРАМЕТРЫ (БЕЗ ИЗМЕНЕНИЙ) ---
DIM = 3
DELAY = 10
N_BINS = 32


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


# --- МЕТРИКА 1: PERRON-FROBENIUS (ИСХОДНАЯ) ---
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

    T1 = build_transition_matrix(s1_discrete, n_bins)
    T2 = build_transition_matrix(s2_discrete, n_bins)

    # 5. Вычисление метрики (Норма Фробениуса)
    frobenius_norm = np.linalg.norm(T1 - T2, 'fro')

    return frobenius_norm


# --- МЕТРИКА 3: CHAMFER DISTANCE ---
def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий.")
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor


def chamfer_distance_metric(x1: np.ndarray, x2: np.ndarray) -> float:
    y1 = reconstruct_attractor(x1, DIM, DELAY)
    y2 = reconstruct_attractor(x2, DIM, DELAY)
    tree1 = KDTree(y1)
    tree2 = KDTree(y2)
    d1, _ = tree2.query(y1, k=1, workers=-1)
    d2, _ = tree1.query(y2, k=1, workers=-1)
    return np.mean(d1 ** 2) + np.mean(d2 ** 2)

