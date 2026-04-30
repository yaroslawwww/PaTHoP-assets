import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

os.makedirs('exp2_plots', exist_ok=True)

horizons = [1, 10, 20]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

for suffix, alpha in alphas.items():
    # Significantly increased figure size for more vertical space (volume)
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(20, 15))
    colors = plt.cm.tab10.colors

    for row_idx, h in enumerate(horizons):
        df = pd.read_csv(f'h{h}_all.txt', skiprows=1)
        df_mix = df[df['Experiment'] == 'EXP2_MIX'].copy()
        df_mix['Donor_R'] = df_mix['Donor_R'].fillna('None').astype(str).str.strip()

        df_base = df_mix[df_mix['Donor_R'] == 'None']
        donors = sorted([d for d in df_mix['Donor_R'].unique() if d != 'None'])

        # Increased individual figure size
        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(20, 6))

        for c_idx, donor in enumerate(donors):
            df_d = df_mix[df_mix['Donor_R'] == donor]
            df_combined = pd.concat([df_d, df_base])

            df_c = df_combined.groupby('Recipient_Size').mean(numeric_only=True).reset_index()
            df_c = df_c.sort_values('Recipient_Size')
            x = df_c['Recipient_Size'].values

            label_text = f'Donor r={float(donor):.4f}'

            for col_idx, metric in enumerate(metrics):
                y = df_c[f'{metric}_{suffix}'].values

                w = np.ones_like(y, dtype=float)
                w[0] = 1000.0
                w[-1] = 1000.0

                spline = UnivariateSpline(x, y, w=w)
                spline.set_smoothing_factor(len(x) * np.var(y) * 0.5 + 1e-6)

                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = spline(x_smooth)

                y_smooth[0] = y[0]
                y_smooth[-1] = y[-1]

                # --- INDIVIDUAL PLOT ---
                axes_ind[col_idx].plot(x, y, 'o', ms=4, alpha=0.3, color=colors[c_idx])
                axes_ind[col_idx].plot(x_smooth, y_smooth, '-', lw=2.5, color=colors[c_idx], label=label_text)

                # --- SUMMARY PLOT ---
                ax_sum = axes_sum[row_idx, col_idx]
                ax_sum.plot(x, y, 'o', ms=2, alpha=0.3, color=colors[c_idx])
                ax_sum.plot(x_smooth, y_smooth, '-', lw=2.5, color=colors[c_idx], label=label_text)

                # --- ZERO-SHOT HIGHLIGHTS (x == 0) ---
                zero_mask = x == 0
                if zero_mask.any():
                    # alpha=0.8 allows us to see overlapping stars slightly better
                    axes_ind[col_idx].plot(x[zero_mask], y[zero_mask], '*', ms=14,
                                           color=colors[c_idx], markeredgecolor='black', alpha=0.8, zorder=5)
                    ax_sum.plot(x[zero_mask], y[zero_mask], '*', ms=12,
                                color=colors[c_idx], markeredgecolor='black', alpha=0.8, zorder=5)

        # --- Format Individual Plot ---
        for col_idx, metric in enumerate(metrics):
            axes_ind[col_idx].plot([], [], '*', ms=12, markeredgecolor='black', color='gray', label='Zero-shot (Rec=0)')
            axes_ind[col_idx].set_title(metric.upper(), fontsize=14, fontweight='bold')
            axes_ind[col_idx].set_xlabel('Recipient Size', fontsize=12)
            axes_ind[col_idx].grid(True, linestyle='--', alpha=0.6)
            axes_ind[col_idx].margins(y=0.1)  # Adds 10% padding to the Y axis to separate clustered points
            axes_ind[col_idx].legend(fontsize=10)

        fig_ind.suptitle(f'Experiment 2: Horizon h={h}, α={alpha} (Total Size=10000)', fontsize=16)
        fig_ind.tight_layout()
        fig_ind.savefig(f'exp2_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        # --- Format Summary Plot ---
        for col_idx, metric in enumerate(metrics):
            ax_sum = axes_sum[row_idx, col_idx]
            if row_idx == 0:
                ax_sum.set_title(f'SUMMARY: {metric.upper()}', fontsize=14, fontweight='bold')
            if row_idx == len(horizons) - 1:
                ax_sum.set_xlabel('Recipient Size', fontsize=12)

            ax_sum.set_ylabel(f'Horizon h={h}', fontsize=12, fontweight='bold')
            ax_sum.grid(True, linestyle='--', alpha=0.5)
            ax_sum.margins(y=0.1)  # Expand Y scale dynamically

            if col_idx == 0:
                ax_sum.plot([], [], '*', ms=10, markeredgecolor='black', color='gray', label='Zero-shot')
                ax_sum.legend(fontsize=10)

    fig_sum.suptitle(f'Experiment 2: All Horizons Trend Comparison (α={alpha}, Total Size=10000)', fontsize=18)
    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_sum.savefig(f'exp2_plots/summary_exp2_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.close(fig_sum)

print("Done. Check 'exp2_plots' folder.")