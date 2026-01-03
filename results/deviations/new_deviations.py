import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

# === Подготовка данных (оригинальная логика) ===
df = pd.read_csv('deviation_results.txt', header=None)
df[0] = df[0] + 28
df[4] = df[4] * 100
df[5] = df[5] * 100

special_row_5k = df[df[6] == 5000]
df_without_special = df[df[6] != 5000]
# Удаление дубликатов X необходимо для корректной работы математики сплайна
df_sorted = df_without_special.sort_values(by=0).drop_duplicates(subset=[0])

baseline_10k_row = df_sorted[df_sorted[0] == 28]


# === Универсальная функция сплайна ===
def add_spline(ax, x, y, multiplier, label='Spline'):
    x_v = x.values
    y_v = y.values
    # Определение коэффициента сглаживания s
    s_val = len(y_v) * np.var(y_v) * multiplier
    spl = UnivariateSpline(x_v, y_v, s=s_val, k=3)

    xs = np.linspace(x_v.min(), x_v.max(), 1000)
    ax.plot(xs, spl(xs), color='orange', linewidth=2.5, label=label, zorder=12)

    # Защита от "вылетов" сплайна за границы данных
    y_min, y_max = y_v.min(), y_v.max()
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 20),sharex=True)
fig.suptitle('Comparison of Augmentation Efficiency with Baseline Series', fontsize=16)

# === 1. RMSE Plot ===
ax1.plot(df_sorted[0], df_sorted[3], 'b.', marker='.', markersize=4, label='Hybrid Series (5000+5000)', zorder=10,
         alpha=0.4)
# Сглаженный сплайн для RMSE (multiplier=0.3)
add_spline(ax1, df_sorted[0], df_sorted[3], 0.4, label='RMSE Spline')

if not special_row_5k.empty:
    val_5k = special_row_5k[3].values[0]
    ax1.axhline(y=val_5k, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Baseline Level (5000 points)')
    ax1.scatter(special_row_5k[0].iloc[0], val_5k, color='red', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (5000)')

if not baseline_10k_row.empty:
    val_10k = baseline_10k_row[3].values[0]
    ax1.axhline(y=val_10k, color='green', linestyle=':', alpha=0.9, linewidth=2, label='Baseline Level (10000 points)')
    ax1.scatter(baseline_10k_row[0].iloc[0], val_10k, color='green', marker='s', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (10000)')

# ax1.set_xlabel('Parameter r value in the auxiliary series')
ax1.set_ylabel('RMSE')
ax1.set_title('RMSE Dependence on Parameter r')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend()

# === 2. NP (Non-Predicted points share) Plot ===
ax2.plot(df_sorted[0], df_sorted[4], 'b.', marker='.', markersize=4, label='Hybrid Series (5000+5000)', zorder=10,
         alpha=0.4)
# Более точный сплайн для NP, чтобы поймать "два горба" (multiplier=0.1)
add_spline(ax2, df_sorted[0], df_sorted[4], 0.1, label='NP Spline (More accurate)')

if not special_row_5k.empty:
    val_5k = special_row_5k[4].values[0]
    ax2.axhline(y=val_5k, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Baseline Level (5000 points)')
    ax2.scatter(special_row_5k[0].iloc[0], val_5k, color='red', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (5000)')

if not baseline_10k_row.empty:
    val_10k = baseline_10k_row[4].values[0]
    ax2.axhline(y=val_10k, color='green', linestyle=':', alpha=0.9, linewidth=2, label='Baseline Level (10000 points)')
    ax2.scatter(baseline_10k_row[0].iloc[0], val_10k, color='green', marker='s', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (10000)')

# ax2.set_xlabel('Parameter r value in the auxiliary series')
ax2.set_ylabel('Share of NP points (%)')
ax2.set_title('Dependence of the Share of Non-Predicted Points on r')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend()

# === 3. MAPE Plot ===
ax3.plot(df_sorted[0], df_sorted[5], 'b.', marker='.', markersize=4, label='Hybrid Series (5000+5000)', zorder=10,
         alpha=0.4)
# Средняя точность для MAPE (multiplier=0.2)
add_spline(ax3, df_sorted[0], df_sorted[5], 0.5, label='MAPE Spline')

if not special_row_5k.empty:
    val_5k = special_row_5k[5].values[0]
    ax3.axhline(y=val_5k, color='red', linestyle='--', alpha=0.9, linewidth=1.5, label='Baseline Level (5000 points)')
    ax3.scatter(special_row_5k[0].iloc[0], val_5k, color='red', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (5000)')

if not baseline_10k_row.empty:
    val_10k = baseline_10k_row[5].values[0]
    ax3.axhline(y=val_10k, color='green', linestyle=':', alpha=0.9, linewidth=2, label='Baseline Level (10000 points)')
    ax3.scatter(baseline_10k_row[0].iloc[0], val_10k, color='green', marker='s', s=80, edgecolors='black', zorder=15,
                label='Point r=28 (10000)')

ax3.set_xlabel('Parameter r value in the auxiliary series')
ax3.set_ylabel('MAPE (%)')
ax3.set_title('MAPE Dependence on r')
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Different_r_comparison_with_points.png', dpi=300)
plt.show()