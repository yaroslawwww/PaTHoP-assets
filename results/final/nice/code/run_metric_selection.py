# coding: utf-8
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm
from scipy.spatial import KDTree
import time

from tqdm import tqdm

# --- НАСТРОЙКИ ---
DIM = 3
DELAY = 10
N_BINS_PER_DIM = 6

class Lorentz:
    def __init__(self, s=10, b=8 / 3):
        self.s = s
        self.b = b
        self.r = None

    def X(self, x, y, s): return s * (y - x)

    def Y(self, x, y, z, r): return -x * z + r * x - y

    def Z(self, x, y, z, b): return x * y - b * z

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


# --- РЕКОНСТРУКЦИЯ И МЕТРИКИ ---
def reconstruct_attractor(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
    n = len(x)
    max_idx = n - (dim - 1) * delay
    if max_idx <= 0:
        raise ValueError("Ряд слишком короткий.")
    attractor = np.column_stack([x[i:i + max_idx] for i in range(0, dim * delay, delay)])
    return attractor


def candidate_score(r1, size1, r2, size2, n_bins_per_dim=6):
    ts1 = TimeSeries(series_type="Lorentz", size=size1, r=r1)
    ts2 = TimeSeries(series_type="Lorentz", size=size2, r=r2)

    y1 = reconstruct_attractor(ts1.values, DIM, DELAY)
    y2 = reconstruct_attractor(ts2.values, DIM, DELAY)

    global_min = min(y1.min(), y2.min())
    global_max = max(y1.max(), y2.max())
    bins = np.linspace(global_min, global_max, n_bins_per_dim + 1)

    y1_discrete = np.digitize(y1, bins) - 1
    y2_discrete = np.digitize(y2, bins) - 1
    y1_discrete[y1_discrete >= n_bins_per_dim] = n_bins_per_dim - 1
    y2_discrete[y2_discrete >= n_bins_per_dim] = n_bins_per_dim - 1

    multiplier = np.array([n_bins_per_dim ** i for i in range(DIM)])
    s1_states = np.sum(y1_discrete * multiplier, axis=1)
    s2_states = np.sum(y2_discrete * multiplier, axis=1)

    n_states = n_bins_per_dim ** DIM

    def build_sparse_transition_matrix(state_sequence, n_states):
        from_states = state_sequence[:-1]
        to_states = state_sequence[1:]
        data = np.ones(len(from_states))

        T_counts = coo_matrix((data, (from_states, to_states)), shape=(n_states, n_states)).tocsr()
        row_sums = np.array(T_counts.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0

        Row_Diag = diags(1.0 / row_sums)
        T_prob = Row_Diag @ T_counts
        return T_prob

    T1 = build_sparse_transition_matrix(s1_states, n_states)
    T2 = build_sparse_transition_matrix(s2_states, n_states)

    return sparse_norm(T1 - T2, 'fro')


def chamfer_distance_metric(x1: np.ndarray, x2: np.ndarray) -> float:
    y1 = reconstruct_attractor(x1, DIM, DELAY)
    y2 = reconstruct_attractor(x2, DIM, DELAY)
    tree1 = KDTree(y1)
    tree2 = KDTree(y2)
    d1, _ = tree2.query(y1, k=1, workers=-1)
    d2, _ = tree1.query(y2, k=1, workers=-1)
    return np.mean(d1 ** 2) + np.mean(d2 ** 2)


# --- ПАРАЛЛЕЛЬНЫЕ ВОРКЕРЫ ---
def eval_p_frob(r):
    score = candidate_score(28.0, 10000, r, 10000, n_bins_per_dim=N_BINS_PER_DIM)
    return r, score


def eval_chamfer(r):
    ts1 = TimeSeries(series_type="Lorentz", size=10000, r=28.0).values
    ts2 = TimeSeries(series_type="Lorentz", size=10000, r=r).values
    score = chamfer_distance_metric(ts1, ts2)
    return r, score


# --- ???????? ?????? ---
if __name__ == '__main__':
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ?????? ???????? ?????? ??????? (2 ??????)...")

    # 1. ????????? 10,000 ?????????? r ~ U[23, 33]
    np.random.seed(666)
    candidates_r = np.random.uniform(23, 33, 10000)

    # ????? ???????? 48 ????????? ??? ???????? ("?????? ????????? ???????")
    random_48 = np.random.choice(candidates_r, 48, replace=False)

    # 2. ???? 1: ?????????? Perron-Frobenius (????? 480 ?????? ?? 10,000)
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ????????? Perron-Frobenius ??? {len(candidates_r)} ?????????? (8 ????)...")
    start_time = time.time()

    with Pool(8) as p:
        results_pf = list(tqdm(p.imap(eval_p_frob, candidates_r), total=len(candidates_r), desc="Perron-Frobenius"))

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ?????? P-Frob ?????? {time.time() - start_time:.2f} ???.")

    # ????????? ? ????? 480 ??????
    df_pf = pd.DataFrame(results_pf, columns=['r', 'p_frob'])
    df_pf = df_pf.sort_values('p_frob')
    top_480_r = df_pf.head(480)['r'].values

    # 3. ???? 2: ?????????? Chamfer (????? 48 ?????? ?? 480)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ????????? Chamfer distance ??? ???-480 ?????????? (8 ????)...")
    start_time = time.time()

    with Pool(8) as p:
        results_chamfer = list(tqdm(p.imap(eval_chamfer, top_480_r), total=len(top_480_r), desc="Chamfer"))

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ?????? Chamfer ?????? {time.time() - start_time:.2f} ???.")

    # ????????? ? ????? 48 ??????
    df_ch = pd.DataFrame(results_chamfer, columns=['r', 'chamfer'])
    df_ch = df_ch.sort_values('chamfer')
    best_48_r = df_ch.head(48)['r'].values

    # 4. ????????? ???????? ??????????
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ????????? ?????????? ? 'selected_donors.csv'...")

    df_out = pd.DataFrame({
        'r': np.concatenate([best_48_r, random_48]),
        'Group': ['Best'] * 48 + ['Random'] * 48
    })

    df_out.to_csv("selected_donors.csv", index=False)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ??????! ???? 'selected_donors.csv' ????????.")