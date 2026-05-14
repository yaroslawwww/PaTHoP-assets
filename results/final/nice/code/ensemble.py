import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm

# Форсируем максимальную точность для TPU, чтобы убрать шум 0.01 в Шамфере
jax.config.update("jax_default_matmul_precision", "highest")

# Настройка TPU
if 'TPU_NAME' in os.environ:
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()

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
    def __init__(self, size=0, r=28, dt=0.01):
        divisor = int(0.1 / dt)
        x, _, _ = Lorentz().generate(dt=dt, steps=size * divisor, r=r)
        # Добавляем 1e-10 для стабильности нормализации
        x = (x - x.min()) / (x.max() - x.min() + 1e-10)
        self.values = x[::divisor]

# --- КОНСТАНТЫ ---
STEPS = 10000
DIM = 4
K_BINS = 5
template_grid = np.array(list(itertools.product(range(1, 11), repeat=3)))
template_offsets = jnp.array(np.hstack([np.zeros((1000, 1)), np.cumsum(template_grid, axis=1)]).astype(jnp.int32))
MAX_LAG = int(template_offsets.max())

N_REF = STEPS - MAX_LAG - 1
N_DON = (2 * STEPS) - MAX_LAG - 1

@jax.jit(static_argnums=(2,))
def get_t_matrix(series, offset, n_points):
    num_states = K_BINS ** DIM
    bins = jnp.linspace(0.0, 1.0, K_BINS + 1)
    v0 = jax.lax.dynamic_slice_in_dim(series, offset[0], n_points + 1)
    v1 = jax.lax.dynamic_slice_in_dim(series, offset[1], n_points + 1)
    v2 = jax.lax.dynamic_slice_in_dim(series, offset[2], n_points + 1)
    v3 = jax.lax.dynamic_slice_in_dim(series, offset[3], n_points + 1)
    d0 = jnp.clip(jnp.digitize(v0, bins) - 1, 0, K_BINS - 1)
    d1 = jnp.clip(jnp.digitize(v1, bins) - 1, 0, K_BINS - 1)
    d2 = jnp.clip(jnp.digitize(v2, bins) - 1, 0, K_BINS - 1)
    d3 = jnp.clip(jnp.digitize(v3, bins) - 1, 0, K_BINS - 1)
    states = d0 + d1 * K_BINS + d2 * (K_BINS ** 2) + d3 * (K_BINS ** 3)
    from_s = states[:-1]
    to_s = states[1:]
    flat_idx = from_s * num_states + to_s
    counts = jnp.bincount(flat_idx, length=num_states ** 2).reshape((num_states, num_states))
    row_sums = jnp.sum(counts, axis=1, keepdims=True)
    T = counts / jnp.where(row_sums > 0, row_sums, 1.0)
    return T


@jax.jit(static_argnums=(2, 3))
def pf_kernel(s1, s2, n1, n2, offsets):
    def body(_, off):
        T1 = get_t_matrix(s1, off, n1)
        T2 = get_t_matrix(s2, off, n2)
        dist = jnp.sqrt(jnp.sum((T1 - T2) ** 2))
        return None, dist

    return jax.lax.scan(body, None, offsets)[1]


@jax.jit(static_argnums=(2, 3))
def chamfer_kernel(s1, s2, n1, n2, offsets):
    def body(_, off):
        y1 = jnp.stack([jax.lax.dynamic_slice_in_dim(s1, off[i], n1 + 1) for i in range(DIM)], -1)
        y2 = jnp.stack([jax.lax.dynamic_slice_in_dim(s2, off[i], n2 + 1) for i in range(DIM)], -1)
        # Прямое вычисление разности для максимальной точности на идентичных данных
        # Используем бродкастинг для получения матрицы (N1, N2)
        diff = y1[:, None, :] - y2[None, :, :]
        d_mat = jnp.sum(diff ** 2, axis=-1)
        return None, jnp.mean(jnp.sqrt(jnp.min(d_mat, axis=1))) + jnp.mean(jnp.sqrt(jnp.min(d_mat, axis=0)))

    return jax.lax.scan(body, None, offsets)[1]


# --- ИСПОЛНЕНИЕ ---
ref_ts = TimeSeries(size=STEPS, r=28.0)
ref_data = jnp.array(ref_ts.values)

print(f"{'Donor Type/R':<12} | {'Mean PF':<10} | {'PF 5-95% CI':<18} | {'Mean CH':<10} | {'CH 5-95% CI':<18}")
print("-" * 85)

# 1. Бейзлайны
# 10k vs 10k (полная идентичность)
b1_pf_dist = np.array(pf_kernel(ref_data, ref_data, N_REF, N_REF, template_offsets))
b1_ch_dist = np.array(chamfer_kernel(ref_data, ref_data, N_REF, N_REF, template_offsets))
print(f"{'B1 10/10':<12} | {np.mean(b1_pf_dist):.6f} | {np.percentile(b1_pf_dist,5):.3f}-{np.percentile(b1_pf_dist,95):.3f} | {np.mean(b1_ch_dist):.6f} | {np.percentile(b1_ch_dist,5):.3f}-{np.percentile(b1_ch_dist,95):.3f}")

# 10k vs 30k (одна и та же система, разная длина)
ref_30k = TimeSeries(size=3*STEPS, r=28.0)
data_30k = jnp.array(ref_30k.values)
n30k = (3*STEPS) - MAX_LAG - 1
b2_pf_dist = np.array(pf_kernel(ref_data, data_30k, N_REF, n30k, template_offsets))
b2_ch_dist = np.array(chamfer_kernel(ref_data, data_30k, N_REF, n30k, template_offsets))
print(f"{'B2 10/30':<12} | {np.mean(b2_pf_dist):.6f} | {np.percentile(b2_pf_dist,5):.3f}-{np.percentile(b2_pf_dist,95):.3f} | {np.mean(b2_ch_dist):.6f} | {np.percentile(b2_ch_dist,5):.3f}-{np.percentile(b2_ch_dist,95):.3f}")

# 2. Список r_vals
r_vals = [27, 27.9, 27.99, 28.0, 28.0001, 28.01, 28.1, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]
for r in r_vals:
    don_ts = TimeSeries(size=2*STEPS, r=r)
    don_data = jnp.array(don_ts.values)
    pfs = np.array(pf_kernel(ref_data, don_data, N_REF, N_DON, template_offsets))
    chs = np.array(chamfer_kernel(ref_data, don_data, N_REF, N_DON, template_offsets))
    print(f"r={r:<10} | {np.mean(pfs):.6f} | {np.percentile(pfs,5):.3f}-{np.percentile(pfs,95):.3f} | {np.mean(chs):.6f} | {np.percentile(chs,5):.3f}-{np.percentile(chs,95):.3f}")

# 3. Массовый отбор (10000 -> 500 -> 50)
print("\n--- ЗАПУСК МАССОВОГО ОТБОРА (Seed 666) ---")
np.random.seed(666)
candidate_rs = np.random.uniform(23, 33, 10000)

pf_results = []
for r in tqdm(candidate_rs, desc="PF Selection"):
    don_data = jnp.array(TimeSeries(size=2*STEPS, r=r).values)
    pf_results.append(float(np.mean(pf_kernel(ref_data, don_data, N_REF, N_DON, template_offsets))))

df = pd.DataFrame({'r': candidate_rs, 'pf': pf_results})
top_500 = df.nsmallest(500, 'pf').copy()

ch_results = []
for r in tqdm(top_500['r'].values, desc="Chamfer Selection"):
    don_data = jnp.array(TimeSeries(size=2*STEPS, r=r).values)
    ch_results.append(float(np.mean(chamfer_kernel(ref_data, don_data, N_REF, N_DON, template_offsets))))

top_500['chamfer'] = ch_results
best_50 = top_500.nsmallest(50, 'chamfer')

# Сборка финального CSV
out = pd.DataFrame({
    'r': np.concatenate([best_50['r'].values, np.random.choice(candidate_rs, 50, replace=False)]),
    'Group': ['Selected'] * 50 + ['Random'] * 50
})
out.to_csv("tpu_ensemble_metrics_final.csv", index=False)
print("\nГотово. Результаты сохранены в tpu_ensemble_metrics_final.csv")