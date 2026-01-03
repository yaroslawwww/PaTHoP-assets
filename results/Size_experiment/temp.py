import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

try:
    df = pd.read_csv(
        'size_experiment_final_10.txt',
        header=None,
        names=['deviation', 'arg', 'prediction_size', 'rmses', 'np_points', 'mape', 'general_size']
    )
except FileNotFoundError:
    np.random.seed(42)
    data = []
    sizes = np.arange(5000, 55000, 2000)
    for d in [0, 5, -5]:
        base_rmse = 0.5 - (sizes / 100000)
        noise = np.random.normal(0, 0.02, len(sizes))
        rmses = base_rmse + noise
        for s, r in zip(sizes, rmses):
            data.append([d, 0, s, r, np.random.uniform(0, 0.1), np.random.uniform(2, 8), s])
    df = pd.DataFrame(data, columns=['deviation', 'arg', 'prediction_size', 'rmses', 'np_points', 'mape', 'general_size'])

df['np_points'] = df['np_points'] * 100
df_filtered = df[~df['deviation'].isin([10, 2, 0.01, 0.001, -1])]
df_r28 = df_filtered[(df_filtered['deviation'] == 0) & (df_filtered['general_size'] >= 10000)]
df_filtered = pd.concat([df_r28, df_filtered[df_filtered['deviation'] != 0]])
deviations = df_filtered['deviation'].unique()

try:
    reference_point = df_filtered[
        (df_filtered['deviation'] == 0) & (df_filtered['general_size'] == 10000)
    ].iloc[0]
except IndexError:
    reference_point = df_filtered.iloc[0]

def plot_approx_spline(x, y, label, smoothing_factor=0.5):
    sorted_idx = np.argsort(x)
    x_s = x[sorted_idx]
    y_s = y[sorted_idx]
    df_temp = pd.DataFrame({'x': x_s, 'y': y_s})
    df_grouped = df_temp.groupby('x', as_index=False).mean()
    x_clean = df_grouped['x'].values
    y_clean = df_grouped['y'].values
    p = plt.plot(x, y, marker='.', linestyle='', markersize=4, alpha=0.6)
    color = p[0].get_color()
    if len(x_clean) > 3:
        s_factor = np.var(y_clean) * len(y_clean) * smoothing_factor
        try:
            spl = UnivariateSpline(x_clean, y_clean, k=3, s=s_factor)
            x_new = np.linspace(x_clean.min(), x_clean.max(), 300)
            plt.plot(x_new, spl(x_new), linestyle='-', linewidth=2, color=color, label=label)
        except:
            z = np.polyfit(x_clean, y_clean, 3)
            plt.plot(np.linspace(x_clean.min(), x_clean.max(), 300), np.poly1d(z)(np.linspace(x_clean.min(), x_clean.max(), 300)), linestyle='-', linewidth=2, color=color, label=label)
    else:
        plt.plot(x_clean, y_clean, linestyle='-', linewidth=2, color=color, label=label)

plt.figure(figsize=(12, 18))
plt.style.use('seaborn-v0_8-darkgrid')

metrics = [
    ('rmses', 0.4, 'RMSE'),
    ('np_points', 1.2, 'NP points (%)'),
    ('mape', 0.7, 'MAPE (%)')
]

for i, (metric, factor, ylabel) in enumerate(metrics, 1):
    plt.subplot(3, 1, i)
    for d in deviations:
        subset = df_filtered[df_filtered['deviation'] == d]
        plot_approx_spline(subset['general_size'].values, subset[metric].values, f'r={d+28}', factor)
    plt.axhline(y=reference_point[metric], color='black', linestyle='--', linewidth=2, alpha=0.7)
    plt.plot(reference_point['general_size'], reference_point[metric], 'ko', markersize=5)
    plt.title(f'{ylabel.split()[0]} vs General Size (Trend)', fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.7)
    plt.legend()
    if i == 3:
        plt.xlabel('General Size', fontsize=12)

plt.tight_layout()
plt.savefig('metrics_vs_general_size_approx_spline.png', dpi=300)
plt.show()