import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

os.makedirs('exp4_plots', exist_ok=True)

horizons = [1, 10, 20]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

for suffix, alpha in alphas.items():
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(22, 15))

    for row_idx, h in enumerate(horizons):
        df = pd.read_csv(f'h{h}_all.txt', skiprows=1)

        # 1. Hybrid Data (fixed 5k rec + 5k donor)
        df_fixed = df[df['Experiment'] == 'EXP4_FIXED'].copy()
        df_fixed['Donor_R'] = pd.to_numeric(df_fixed['Donor_R'], errors='coerce')
        df_fixed = df_fixed.dropna(subset=['Donor_R'])
        df_fixed = df_fixed.groupby('Donor_R').mean(numeric_only=True).reset_index().sort_values('Donor_R')

        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(20, 6))

        x = df_fixed['Donor_R'].values

        for col_idx, metric in enumerate(metrics):
            y = df_fixed[f'{metric}_{suffix}'].values

            # Extract baseline values
            val_5k = df[df['Experiment'] == 'EXP4_BASE_5K'][f'{metric}_{suffix}'].mean()
            val_10k = df[df['Experiment'] == 'EXP4_BASE_10K'][f'{metric}_{suffix}'].mean()

            # Smoothing spline
            spline = UnivariateSpline(x, y)
            spline.set_smoothing_factor(len(x) * np.var(y) * 0.8)
            x_smooth = np.linspace(x.min(), x.max(), 500)
            y_smooth = spline(x_smooth)

            for ax in [axes_ind[col_idx], axes_sum[row_idx, col_idx]]:
                # Trend and points
                ax.plot(x, y, 'o', ms=3, alpha=0.3, color='royalblue')
                ax.plot(x_smooth, y_smooth, '-', lw=2.5, color='darkorange', label='Hybrid Trend (5k+5k)')

                # Horizontal baselines
                ax.axhline(val_5k, color='red', linestyle='--', lw=1.5, label='5k Recipient Baseline')
                ax.axhline(val_10k, color='green', linestyle=':', lw=2, label='10k Recipient Baseline')

                # Target Rayleigh highlight
                ax.plot(28.0, val_10k, 's', color='black', ms=8, label='Target r=28.0', zorder=10)

                ax.grid(True, linestyle='--', alpha=0.5)  # FIXED: changed set_grid to grid
                ax.margins(x=0.02, y=0.1)

        # Formatting Individual
        for i, metric in enumerate(metrics):
            axes_ind[i].set_title(metric.upper(), fontsize=14, fontweight='bold')
            axes_ind[i].set_xlabel('Donor Rayleigh Parameter (r)', fontsize=12)
            axes_ind[i].legend(fontsize=9, loc='upper right')

        fig_ind.suptitle(f'Experiment 4: Horizon h={h}, α={alpha}', fontsize=16)
        fig_ind.tight_layout()
        fig_ind.savefig(f'exp4_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        # Formatting Summary
        for i, metric in enumerate(metrics):
            ax_s = axes_sum[row_idx, i]
            if row_idx == 0: ax_s.set_title(f'SUMMARY: {metric.upper()}', fontsize=14)
            if row_idx == 2: ax_s.set_xlabel('Donor Rayleigh (r)')
            ax_s.set_ylabel(f'h={h}', fontweight='bold')

    fig_sum.suptitle(f'Experiment 4: Rayleigh Sensitivity Summary (α={alpha})', fontsize=18)
    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_sum.savefig(f'exp4_plots/summary_exp4_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.close(fig_sum)

print("Done. Check 'exp4_plots' folder.")