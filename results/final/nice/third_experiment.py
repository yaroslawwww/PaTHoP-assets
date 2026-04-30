import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

os.makedirs('exp3_plots', exist_ok=True)

horizons = [1, 10, 20]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

for suffix, alpha in alphas.items():
    # Large 3x3 grid for the summary plot
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(20, 15))

    # tab20 palette to handle up to 20 distinct donors (Exp3 has ~14 donors)
    colors = plt.cm.tab20.colors

    for row_idx, h in enumerate(horizons):
        df = pd.read_csv(f'h{h}_all.txt', skiprows=1)

        # --- PREPARE HYBRID DATA ---
        df_mix = df[df['Experiment'] == 'EXP3_HYBRID'].copy()
        df_mix['Donor_R'] = df_mix['Donor_R'].fillna('None').astype(str).str.strip()

        df_base = df_mix[df_mix['Donor_R'] == 'None']
        donors = sorted([d for d in df_mix['Donor_R'].unique() if d != 'None'], key=float)

        # --- PREPARE PURE DATA BASELINE (EXP1_CLEAR) ---
        # We align EXP1 to EXP3 by calculating how much data was "added" past the base 10k
        df_clear = df[df['Experiment'] == 'EXP1_CLEAR'].copy()
        df_clear['Donor_Size'] = df_clear['Recipient_Size'] - 10000
        df_clear = df_clear[(df_clear['Donor_Size'] >= 0) & (df_clear['Donor_Size'] <= 30000)]
        df_clear_c = df_clear.groupby('Donor_Size').mean(numeric_only=True).reset_index().sort_values('Donor_Size')

        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(20, 6))

        # Plot the EXP1_CLEAR Pure Expansion line first (Black Dashed)
        x_clear = df_clear_c['Donor_Size'].values
        for col_idx, metric in enumerate(metrics):
            y_clear = df_clear_c[f'{metric}_{suffix}'].values
            spline_c = UnivariateSpline(x_clear, y_clear)
            spline_c.set_smoothing_factor(len(x_clear) * np.var(y_clear) * 0.5 + 1e-6)

            xs_c = np.linspace(x_clear.min(), x_clear.max(), 300)
            ys_c = spline_c(xs_c)

            axes_ind[col_idx].plot(xs_c, ys_c, '--', color='black', lw=3, label='Pure Recipient Expansion', zorder=10)
            axes_sum[row_idx, col_idx].plot(xs_c, ys_c, '--', color='black', lw=3, label='Pure Recipient Exp',
                                            zorder=10)

        # Loop through each donor
        for c_idx, donor in enumerate(donors):
            df_d = df_mix[df_mix['Donor_R'] == donor]
            df_combined = pd.concat([df_d, df_base])

            # Grouping by Donor_Size (Added volume)
            df_c = df_combined.groupby('Donor_Size').mean(numeric_only=True).reset_index()
            df_c = df_c.sort_values('Donor_Size')
            x = df_c['Donor_Size'].values

            label_text = f'Donor r={float(donor):.4f}'

            for col_idx, metric in enumerate(metrics):
                y = df_c[f'{metric}_{suffix}'].values

                # Weight array to force the spline precisely through the starting point
                w = np.ones_like(y, dtype=float)
                w[0] = 1000.0

                spline = UnivariateSpline(x, y, w=w)
                spline.set_smoothing_factor(len(x) * np.var(y) * 0.5 + 1e-6)

                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = spline(x_smooth)
                y_smooth[0] = y[0]

                # --- INDIVIDUAL PLOT ---
                axes_ind[col_idx].plot(x, y, 'o', ms=4, alpha=0.3, color=colors[c_idx])
                axes_ind[col_idx].plot(x_smooth, y_smooth, '-', lw=2, color=colors[c_idx], label=label_text)

                # --- SUMMARY PLOT ---
                ax_sum = axes_sum[row_idx, col_idx]
                ax_sum.plot(x, y, 'o', ms=2, alpha=0.3, color=colors[c_idx])
                ax_sum.plot(x_smooth, y_smooth, '-', lw=2, color=colors[c_idx], label=label_text)

                # --- BASELINE HIGHLIGHT (x == 0) ---
                zero_mask = x == 0
                if zero_mask.any():
                    axes_ind[col_idx].plot(x[zero_mask], y[zero_mask], '*', ms=14,
                                           color='gold', markeredgecolor='black', alpha=0.9, zorder=15)
                    ax_sum.plot(x[zero_mask], y[zero_mask], '*', ms=12,
                                color='gold', markeredgecolor='black', alpha=0.9, zorder=15)

        # --- Format Individual Plot ---
        for col_idx, metric in enumerate(metrics):
            axes_ind[col_idx].plot([], [], '*', ms=12, markeredgecolor='black', color='gold',
                                   label='Baseline (0 Donor)')
            axes_ind[col_idx].set_title(metric.upper(), fontsize=14, fontweight='bold')
            axes_ind[col_idx].set_xlabel('Added Donor Size', fontsize=12)
            axes_ind[col_idx].grid(True, linestyle='--', alpha=0.6)
            axes_ind[col_idx].margins(y=0.1)
            # Use 2 columns for legend because 14 donors + baseline = too tall
            axes_ind[col_idx].legend(fontsize=8, ncol=2)

        fig_ind.suptitle(f'Experiment 3: Horizon h={h}, α={alpha} (Base Recipient=10000)', fontsize=16)
        fig_ind.tight_layout()
        fig_ind.savefig(f'exp3_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        # --- Format Summary Plot ---
        for col_idx, metric in enumerate(metrics):
            ax_sum = axes_sum[row_idx, col_idx]
            if row_idx == 0:
                ax_sum.set_title(f'SUMMARY: {metric.upper()}', fontsize=14, fontweight='bold')
            if row_idx == len(horizons) - 1:
                ax_sum.set_xlabel('Added Donor Size', fontsize=12)

            ax_sum.set_ylabel(f'Horizon h={h}', fontsize=12, fontweight='bold')
            ax_sum.grid(True, linestyle='--', alpha=0.5)
            ax_sum.margins(y=0.1)

            if col_idx == 0:
                ax_sum.plot([], [], '*', ms=10, markeredgecolor='black', color='gold', label='Baseline')
                ax_sum.legend(fontsize=8, ncol=2)

    fig_sum.suptitle(f'Experiment 3: Hybrid Expansion Comparison (α={alpha}, Base Recipient=10000)', fontsize=18)
    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_sum.savefig(f'exp3_plots/summary_exp3_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.close(fig_sum)

print("Done. Check 'exp3_plots' folder.")