# coding: utf-8
import sys
import numpy as np
import os
from WishartClusterizationAlgorithm import Wishart
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from tqdm import tqdm


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
            x = (x - x.min()) / (x.max() - x.min())
            self.values = x[::divisor]
        else:
            self.values = np.array(array)
            self.values = (self.values - self.values.min()) / (self.values.max() - self.values.min())
        self.train = None
        self.test = None

    def split_train_val_test(self, window_index, test_size=100):
        if window_index + test_size > len(self.values):
            raise ValueError("test index out of range")
        self.train = self.values[:window_index]
        self.test = self.values[window_index:window_index + test_size]


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


class TSProcessor:
    def __init__(self, k=11, mu=0.2):
        self.templates_ = None
        self.time_series_ = None
        self.k, self.mu = k, mu
        self.motifs = None

    def fit(self, time_series_list, template_length, max_template_spread):
        print("fitting")
        self.templates_ = Templates(template_length, max_template_spread)
        self.templates_.create_train_set(time_series_list)
        wishart = Wishart(k=self.k, mu=self.mu)
        self.motifs = dict()
        clusters_number = []
        save_labels = []
        z_vectors = self.templates_.train_set
        for template in tqdm(range(z_vectors.shape[0])):
            inf_mask = ~np.isinf(z_vectors[template]).any(axis=1)
            temp_z_v = z_vectors[template][inf_mask]
            wishart.fit(temp_z_v)
            cluster_labels, cluster_sizes = np.unique(wishart.labels_[wishart.labels_ > -1], return_counts=True)
            clusters_number.append(len(cluster_labels))
            save_labels.append(wishart.labels_)
            motifs = [temp_z_v[wishart.labels_ == i].mean(axis=0) for i in cluster_labels]
            if template in self.motifs:
                self.motifs[template] += list(np.array(motifs).reshape(-1, len(motifs[0])))
            else:
                self.motifs[template] = list(np.array(motifs).reshape(-1, len(motifs[0])))
        for template in self.motifs.keys():
            self.motifs[template] = np.array(self.motifs[template])
        avg_clusters = sum(clusters_number) / len(clusters_number)
        return avg_clusters

    def predict(self, time_series, window_index, test_size, eps, np_t, method="dbscan"):
        self.time_series_ = time_series
        self.time_series_.split_train_val_test(window_index, test_size)
        steps = len(self.time_series_.test)
        values = np.array(list(self.time_series_.train) + [np.nan] * steps)
        forecast_trajectories = np.full((steps, 1), np.nan)
        observation_indexes = self.templates_.observation_indexes

        for step in range(steps):
            test_vectors = values[:len(self.time_series_.train) + step][observation_indexes]
            all_motifs = []
            for template in self.motifs.keys():
                train_truncated = self.motifs[template][:, :-1]
                distance_matrix = calc_distance_matrix([test_vectors[template]], train_truncated)
                distance_mask = distance_matrix < eps
                matched_motifs = self.motifs[template][distance_mask.ravel()]

                if matched_motifs.size > 0:
                    all_motifs.append(matched_motifs)

            motifs_pool = np.vstack(all_motifs) if all_motifs else np.empty((0, 4))

            # ПЕРЕДАЕМ method СЮДА
            forecast_point = self.freeze_point(motifs_pool, np_t, method)
            forecast_trajectories[step, 0] = forecast_point
            values[len(self.time_series_.train) + step] = forecast_point

        return values

    def freeze_point(self, motifs_pool, np_threshold, method="dbscan"):
        if motifs_pool.size == 0:
            return np.nan
        points_pool = motifs_pool[:, -1].reshape(-1, 1)
        dbs = DBSCAN(0.01, min_samples=4)
        dbs.fit(points_pool)
        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

        if cluster_labels.size > 0:
            mask = (dbs.labels_ == cluster_labels[cluster_sizes.argmax()])
            upv = points_pool[mask].mean()

            # Если метод априорный, игнорируем кластерное соотношение (np_threshold)
            # и возвращаем лучшее предположение. Проверка ошибки будет в predict_handler.
            if method == "apriori":
                return upv
            else:
                # Классический DBSCAN: проверяем доминацию кластера
                if np.count_nonzero((cluster_sizes / cluster_sizes.max()).round(2) > np_threshold) == 1:
                    return upv
        return np.nan
def calc_distance_matrix(test_vectors, train_vectors):
    return np.squeeze(cdist(test_vectors, train_vectors, 'euclidean'), axis=0)


def predict_handler(gap, test_size_constant, epsilon, ts, tsproc, np_t, method="dbscan"):
    ts_size = len(ts.values)
    window_index = ts_size - (gap + 1) - test_size_constant
    if window_index < 0 or window_index >= ts_size:
        return None, None, None

    values = tsproc.predict(ts, window_index, test_size_constant, epsilon, np_t, method)
    real_values = np.array(ts.values[window_index:window_index + test_size_constant])
    pred_values = np.array(values[-test_size_constant:])

    final_pred = pred_values[-1]
    final_real = real_values[-1]

    # Логика априорного метода
    if method == "apriori":
        # Если прогноз не состоялся вообще ИЛИ отклонение больше 0.05
        if np.isnan(final_pred) or abs(final_pred - final_real) > 0.05:
            is_np_point = 1
            final_pred = np.nan  # Исключаем из подсчета RMSE
        else:
            is_np_point = 0
    else:
        # Стандартная логика
        is_np_point = 1 if np.isnan(final_pred) else 0

    return final_pred, is_np_point, final_real


def research(r_values, ts_size, how_many_gaps, test_size_constant, dt=0.001, epsilon=0.01,
             template_length_constant=4, template_spread_constant=10, method="dbscan"):
    list_ts = [TimeSeries("Lorentz", size=size, r=r, dt=dt) for size, r in zip(ts_size, r_values) if size > 0]
    tsproc = TSProcessor()
    avg_clusters = tsproc.fit(list_ts[1:], template_length_constant, template_spread_constant)

    pred_points_values_first = []
    is_np_points_first = []
    real_points_values = []

    pred_points_values_second = []
    is_np_points_second = []

    ts = list_ts[0]
    np_first_thres = 0.1
    np_second_thres = 0.3

    for gap in range(how_many_gaps):
        pred_point, is_np_point, real_point = predict_handler(
            gap, test_size_constant, epsilon, ts, tsproc, np_first_thres, method
        )
        if pred_point is not None:
            pred_points_values_first.append(pred_point)
            is_np_points_first.append(is_np_point)
            real_points_values.append(real_point)

    for gap in range(how_many_gaps):
        pred_point, is_np_point, real_point = predict_handler(
            gap, test_size_constant, epsilon, ts, tsproc, np_second_thres, method
        )
        if pred_point is not None:
            pred_points_values_second.append(pred_point)
            is_np_points_second.append(is_np_point)

    rmse1, np1, mape1 = rmse(pred_points_values_first, real_points_values), np.mean(is_np_points_first), mape(
        pred_points_values_first, real_points_values)
    rmse2, np2, mape2 = rmse(pred_points_values_second, real_points_values), np.mean(is_np_points_second), mape(
        pred_points_values_second, real_points_values)

    return rmse1, np1, mape1, rmse2, np2, mape2


def evaluation(r_values, ts_sizes, prediction_size=20, method="dbscan"):
    how_many_gaps = 1000

    rmse1, np1, mape1, rmse2, np2, mape2 = research(
        r_values=[28] + r_values,
        ts_size=np.array([how_many_gaps + 100 + ts_sizes[0]] + list(ts_sizes)),
        how_many_gaps=how_many_gaps,
        test_size_constant=prediction_size,
        method=method
    )
    return rmse1, np1, mape1, rmse2, np2, mape2
