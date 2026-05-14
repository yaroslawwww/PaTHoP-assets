import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

os.makedirs('exp5_plots', exist_ok=True)

# Exp 5 logic: only h=10
horizons = [10]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

for suffix, alpha in alphas.items():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab20.colors

    h = 10
    df = pd.read_csv(f'h{h}_all.txt', skiprows=1)

    # Filter for Apriori experiment
    df_apriori = df[df['Experiment'] == 'EXP5_APRIORI'].copy()
    df_apriori['Donor_R'] = df_apriori['Donor_R'].fillna('None').astype(str).str.strip()

    donors = sorted([d for d in df_apriori['Donor_R'].unique() if d != 'None'], key=float)
    df_base = df_apriori[df_apriori['Donor_R'] == 'None']

    for c_idx, donor in enumerate(donors):
        df_d = df_apriori[df_apriori['Donor_R'] == donor]
        df_combined = pd.concat([df_d, df_base])

        # Aggregate and sort by added data volume
        df_c = df_combined.groupby('Donor_Size').mean(numeric_only=True).reset_index().sort_values('Donor_Size')

        # Сдвигаем ось X на 10000
        x = df_c['Donor_Size'].values + 10000

        for col_idx, metric in enumerate(metrics):
            y = df_c[f'{metric}_{suffix}'].values

            # Force spline through start point with weights
            w = np.ones_like(y, dtype=float)
            w[0] = 1000.0
            spline = UnivariateSpline(x, y, w=w)
            spline.set_smoothing_factor(len(x) * np.var(y) * 0.5 + 1e-7)

            xs = np.linspace(x.min(), x.max(), 300)
            ys = spline(xs)
            ys[0] = y[0]  # Exact match for left edge

            ax = axes[col_idx]
            # Стандартная отрисовка для всех линий
            ax.plot(x, y, 'o', ms=4, alpha=0.3, color=colors[c_idx])
            ax.plot(xs, ys, '-', lw=2, color=colors[c_idx], label=f'r={float(donor):.4f}')

        # Final formatting and gold star baseline
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]

        # Highlight the starting point (сдвинута на 10000)
        y_start = df_base[f'{metric}_{suffix}'].mean()
        ax.plot(10000, y_start, '*', ms=15, color='gold', markeredgecolor='black', zorder=10, label='Baseline (10k)')

        ax.set_title(f'APRIORI: {metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('General Size (10000 pure series + x donor)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
        ax.margins(y=0.1)

    plt.suptitle(f'Experiment 5: Apriori Predictability Method (h=10, α={alpha})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'exp5_plots/h10_apriori_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.show()

print("Done. Files saved in 'exp5_plots'.")