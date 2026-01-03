!pip
install
tslearn
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr

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
    ts1 = TimeSeries(series_type="Lorentz", size=size1, r=r1)
    ts2 = TimeSeries(series_type="Lorentz", size=size2, r=r2)
    series1 = ts1.values
    series2 = ts2.values
    global_min = min(series1.min(), series2.min())
    global_max = max(series1.max(), series2.max())
    bins = np.linspace(global_min, global_max, n_bins + 1)
    s1_discrete = np.digitize(series1, bins) - 1
    s2_discrete = np.digitize(series2, bins) - 1
    s1_discrete[s1_discrete == n_bins] = n_bins - 1
    s2_discrete[s2_discrete == n_bins] = n_bins - 1

    def build_transition_matrix(discrete_series, n_states):
        T = np.zeros((n_states, n_states))
        for i in range(len(discrete_series) - 1):
            from_state = discrete_series[i]
            to_state = discrete_series[i + 1]
            T[from_state, to_state] += 1
        row_sums = T.sum(axis=1, keepdims=True)
        non_zero_rows = np.where(row_sums > 0)[0]
        if len(non_zero_rows) > 0:
            T[non_zero_rows] /= row_sums[non_zero_rows]
        return T

    T1 = build_transition_matrix(s1_discrete, n_bins)
    T2 = build_transition_matrix(s2_discrete, n_bins)
    return np.linalg.norm(T1 - T2, 'fro')


# --- МЕТРИКА 2: DTW (АЛГОРИТМ ИЗ ИНТЕРНЕТА) ---
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            last_min = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]


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


import numpy as np
from scipy.signal import argrelextrema
from tslearn.metrics import dtw as tslearn_dtw  # Обычный DTW


def extract_local_extrema(series: np.ndarray):
    """
    Извлекает локальные экстремумы из временного ряда.
    Возвращает два массива значений: для минимумов и максимумов (в порядке индексов).
    """
    # Локальные максимумы
    max_idx = argrelextrema(series, np.greater)[0]
    maxima_values = series[max_idx] if len(max_idx) > 0 else np.array([])

    # Локальные минимумы
    min_idx = argrelextrema(series, np.less)[0]
    minima_values = series[min_idx] if len(min_idx) > 0 else np.array([])

    # Сортируем по индексам (на всякий случай, хотя argrelextrema уже возвращает отсортированные)
    maxima_values = maxima_values[np.argsort(max_idx)] if len(max_idx) > 0 else np.array([])
    minima_values = minima_values[np.argsort(min_idx)] if len(min_idx) > 0 else np.array([])

    return minima_values, maxima_values


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Обёртка над tslearn DTW для удобства"""
    return tslearn_dtw(a, b)


def ledtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    LE-DTW расстояние между двумя временными рядами.
    """
    minima1, maxima1 = extract_local_extrema(s1)
    minima2, maxima2 = extract_local_extrema(s2)

    # Проверка на достаточное количество экстремумов
    if (len(minima1) < 2 or len(minima2) < 2 or
            len(maxima1) < 2 or len(maxima2) < 2):
        return dtw_distance(s1, s2)

    # DTW отдельно для минимумов и максимумов
    dtw_minima = dtw_distance(minima1, minima2)
    dtw_maxima = dtw_distance(maxima1, maxima2)

    return dtw_minima + dtw_maxima


# --- МЕТРИКИ 4, 5, 6: CORRELATIONS & RMSE & LE-DTW ---
def compute_all_metrics(r_ref, r_val, steps):
    # Генерация данных (используем те же параметры, что в исходном коде)
    ts_ref = TimeSeries(series_type="Lorentz", size=steps, r=r_ref)
    ts_test = TimeSeries(series_type="Lorentz", size=steps, r=r_val)
    x1, x2 = ts_ref.values, ts_test.values

    # 1. Perron-Frobenius
    m1 = candidate_score(r_ref, steps, r_val, steps)

    # 2. DTW (ограничение 1000 точек для производительности)
    m2 = dtw_distance(x1, x2)

    # 3. Chamfer
    m3 = chamfer_distance_metric(x1, x2)

    # 4. Pearson
    m4, _ = pearsonr(x1, x2)

    # 5. Spearman
    m5, _ = spearmanr(x1, x2)

    # 6. LE-DTW
    m6 = ledtw_distance(x1, x2)

    return [m1, m2, m3, m4, m5, m6]


if __name__ == '__main__':
    steps = 10000
    r_values = [
        28.0001, 28.01, 27.99, 27.9, 28.1, 30.0, 31.0, 32.0,
        33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0
    ]

    header = r"\begin{tabular}{|c|c|c|c|c|c|c|}"
    hline = r"\hline"
    columns = r"$r$ & P-Frob & DTW & Chamfer & Pearson & Spearman & LE-DTW \\"
    rows = []

    for r in r_values:
        m = compute_all_metrics(28, r, steps)
        row = f"{r:8.4f} & {m[0]:8.3f} & {m[1]:9.2f} & {m[2]:8.3f} & {m[3]:8.3f} & {m[4]:8.3f} & {m[5]:9.2f} \\\\"
        rows.append(row)

    print(header)
    print(hline)
    print(columns)
    print(hline)
    for row in rows:
        print(row)
        print(hline)
    print(r"\end{tabular}")