import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing.shared_memory as shm
from WishartClusterizationAlgorithm import Wishart

class Lorentz:
    def __init__(self, s=10, b=8/3):
        self.s = s
        self.b = b
        self.r = None

    def X(self, x, y, s):
        return s * (y - x)

    def Y(self, x, y, z, r):
        return (-x) * z + r * x - y

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

        x += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dt * (1/6)
        y += (l_1 + 2 * l_2 + 2 * l_3 + l_4) * dt * (1/6)
        z += (m_1 + 2 * m_2 + 2 * m_3 + m_4) * dt * (1/6)

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
            x, y, z = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
            x = (x - x.min()) / (x.max() - x.min())  # нормализация чисел
            self.values = list(x)[::divisor]
        else:
            x = np.array(array)
            x = (x - x.min()) / (x.max() - x.min())
            self.values = list(x)
        self.train = None
        self.after_test_train = None
        self.test = None
        self.val = []
        self.time = [i for i in range(len(self.values))]

    def split_train_val_test(self, window_index, test_size=100):
        if window_index + test_size > len(self.values):
            raise ValueError("test index out of range")

        self.train = self.values[:window_index]
        self.test = self.values[window_index:window_index + test_size]

    def close_shm(self):
        if self.shm_block:
            self.shm_block.close()
            self.shm_block.unlink()
            self.shm_block = None

class Templates:
    def __init__(self, template_length, max_template_spread):
        self.train_set = None
        self.affiliation_matrix = None
        self.template_length = template_length
        self.max_template_spread = max_template_spread
        templates_quantity = max_template_spread ** (template_length - 1)
        templates = np.zeros((templates_quantity, template_length), dtype=int)
        for i in range(1, template_length):
            step_size = max_template_spread ** (template_length - i - 1)
            repeat_count = max_template_spread ** i
            block = np.repeat(np.arange(1, max_template_spread + 1), step_size)
            templates[:, i] = np.tile(block, repeat_count // max_template_spread) + templates[:, i - 1]
        self.templates = templates
        shapes = np.diff(templates, axis=1)
        self.observation_indexes = shapes[:, ::-1].cumsum(axis=1)[:, ::-1] * -1

    def add_data_to_train_set(self, data, all_train_sets):
        if len(data) == 0:
            return
        x_dim = self.templates.shape[0]
        y_dim = max(len(data) - self.templates[i][-1] for i in range(x_dim))
        if y_dim <= 0:
            return
        z_dim = self.templates.shape[1]
        individual_train_set = np.full((x_dim, y_dim, z_dim), np.inf, dtype=float)
        for i in range(len(self.templates)):
            template_window = self.templates[i][-1]
            n_windows = len(data) - template_window
            if n_windows > 0:
                time_series_indexes = self.templates[i] + np.arange(n_windows)[:, None]
                time_series_vectors = data[time_series_indexes]
                individual_train_set[i, :n_windows] = time_series_vectors
        all_train_sets.append(individual_train_set)

    def add_data_to_affiliation_matrix(self, data, affiliation_matrix, index):
        x_dim = self.templates.shape[0]
        y_dim = max(len(data) - self.templates[i][-1] for i in range(x_dim))
        z_dim = self.templates.shape[1]
        affiliation_matrix.append(np.full((x_dim, y_dim, z_dim), index, dtype=int))

    def create_train_set(self, time_series_list):
        all_train_sets = []
        affiliation_matrix = []
        for i, time_series in enumerate(time_series_list):
            data = np.array(time_series.train if time_series.train is not None else time_series.values)
            self.add_data_to_train_set(data, all_train_sets)
            self.add_data_to_affiliation_matrix(data, affiliation_matrix, i)
        if all_train_sets:
            self.train_set = np.concatenate(all_train_sets, axis=1)
            self.affiliation_matrix = np.concatenate(affiliation_matrix, axis=1)

def calc_distance_matrix(test_vectors, train_vectors):
    return np.squeeze(cdist(test_vectors, train_vectors, 'euclidean'), axis=0)

class TSProcessor:
    def __init__(self, k=16, mu=0.45):
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu
        self.motifs = None

    def fit(self, time_series_list, template_length, max_template_spread):
        print("Fitting\n")
        self.templates_ = Templates(template_length, max_template_spread)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = {}
        self.templates_.create_train_set(time_series_list)
        z_vectors = self.templates_.train_set
        for template in tqdm(range(z_vectors.shape[0])):
            inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
            temp_z_v = z_vectors[template][inf_mask]
            wishart.fit(temp_z_v)
            cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
            motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
            self.motifs[template] = np.array(motifs).reshape(-1, len(motifs[0]))
        self.templates_.train_set = None  # Free memory

    def predict(self, time_series, window_index, test_size, eps):
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(window_index, test_size)
        steps = len(self.time_series_.test)
        values = np.array(self.time_series_.train + self.time_series_.val + [np.nan] * steps)
        forecast_trajectories = np.full((steps, 1), np.nan)
        observation_indexes = self.templates_.observation_indexes
        for step in range(steps):
            test_vectors = values[:len(self.time_series_.train) + step][observation_indexes]
            motifs_pool = []
            for template in self.motifs.keys():
                train_truncated_vectors_template = self.motifs[template][:, :-1]
                distance_matrix = calc_distance_matrix([test_vectors[template]], train_truncated_vectors_template)
                distance_mask = distance_matrix < eps
                best_motifs = self.motifs[template][distance_mask]
                motifs_pool.extend(best_motifs)
            motifs_pool = np.array(motifs_pool)
            forecast_point = self.freeze_point(motifs_pool)
            forecast_trajectories[step, 0] = forecast_point
            values[len(self.time_series_.train) + step] = forecast_point
        return values

    def freeze_point(self, motifs_pool):
        if motifs_pool.size == 0:
            return np.nan
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(0.01, min_samples=4)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
        if cluster_labels.size > 0 and np.count_nonzero((cluster_sizes / cluster_sizes.max()).round(2) > 0.3) == 1:
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            return points_pool[mask].mean()
        return np.nan

def rmse(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    return np.sqrt(np.mean((y_true_masked - y_pred_masked) ** 2)) if len(y_true_masked) > 0 else np.nan

def mape(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    if len(y_true_masked) == 0:
        return 0
    zero_mask = y_true_masked != 0
    if not np.any(zero_mask):
        return np.nan
    y_true_non_zero = y_true_masked[zero_mask]
    y_pred_non_zero = y_pred_masked[zero_mask]
    return np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero))

def predict_handler(gap, test_size_constant, epsilon, ts_shm_name, ts_size, tsproc):
    # Reconstruct TimeSeries with shared memory
    ts = TimeSeries()
    ts.shm_name = ts_shm_name
    ts.shm_block = shm.SharedMemory(name=ts_shm_name)
    ts.values = np.ndarray((ts_size,), dtype=np.float64, buffer=ts.shm_block.buf)
    window_index = ts_size - (gap + 1) - test_size_constant
    if window_index < 0 or window_index >= ts_size:
        return None, None, None
    values = tsproc.predict(ts, window_index, test_size_constant, epsilon)
    real_values = np.array(ts.values[window_index:window_index + test_size_constant])
    pred_values = np.array(values[-test_size_constant:])
    is_np_point = 1 if np.isnan(pred_values[-1]) else 0
    return pred_values[-1], is_np_point, real_values[-1]

def process_batch(gaps, test_size_constant, epsilon, ts_shm_name, ts_size, tsproc):
    results = []
    for gap in gaps:
        pred_point, is_np_point, real_point = predict_handler(
            gap, test_size_constant, epsilon, ts_shm_name, ts_size, tsproc
        )
        results.append((pred_point, is_np_point, real_point))
    return results

def parallel_research(r_values, ts_size, how_many_gaps, test_size_constant, dt=0.01, epsilon=0.01,
                      template_length_constant=4, template_spread_constant=10):
    batch_size = 50
    list_ts = [TimeSeries("Lorentz", size=size, r=r, dt=dt) for size, r in zip(ts_size, r_values) if size > 0]
    tsproc = TSProcessor()
    tsproc.fit(list_ts[1:], template_length_constant, template_spread_constant)
    pred_points_values = []
    is_np_points = []
    real_points_values = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_batch,
                range(batch_start, min(batch_start + batch_size, how_many_gaps)),
                test_size_constant,
                epsilon,
                list_ts[0].shm_name,
                len(list_ts[0].values),
                tsproc
            )
            for batch_start in range(0, how_many_gaps, batch_size)
        ]
        for future in futures:
            batch_result = future.result()
            for pred_point, is_np_point, real_point in batch_result:
                pred_points_values.append(pred_point)
                is_np_points.append(is_np_point)
                real_points_values.append(real_point)
    for ts in list_ts:
        ts.close_shm()
    return rmse(pred_points_values, real_points_values), np.mean(is_np_points), mape(pred_points_values, real_points_values)
