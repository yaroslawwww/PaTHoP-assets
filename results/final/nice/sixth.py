import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('exp6_plots', exist_ok=True)

horizons = [1, 10, 20]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

for suffix, alpha in alphas.items():
    # Large figure for all horizons
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(20, 15))

    for row_idx, h in enumerate(horizons):
        df = pd.read_csv(f'h{h}_all.txt', skiprows=1)

        # 1. Filter Groups
        best_data = df[df['Experiment'] == 'EXP6_BEST']
        rand_data = df[df['Experiment'] == 'EXP6_RANDOM']

        # 2. Extract Baselines (Horizontal lines)
        base_10k = df[df['Experiment'] == 'EXP6_BASE_10K'][f'rmse_{suffix}'].mean()  # placeholders
        base_30k = df[df['Experiment'] == 'EXP6_BASE_30K'][f'rmse_{suffix}'].mean()

        # Individual Figure for this specific (h, alpha)
        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(20, 6))

        for col_idx, metric in enumerate(metrics):
            m_col = f'{metric}_{suffix}'

            best_vals = best_data[m_col].values
            rand_vals = rand_data[m_col].values

            # Baseline values for this specific metric
            b10 = df[df['Experiment'].str.contains('BASE_10K', na=False)][m_col].mean()
            b30 = df[df['Experiment'].str.contains('BASE_30K', na=False)][m_col].mean()

            plot_targets = [axes_ind[col_idx], axes_sum[row_idx, col_idx]]

            for ax in plot_targets:
                # Plot individual points for density
                ax.plot(np.random.normal(1, 0.04, len(best_vals)), best_vals, 'o', ms=5, color='royalblue', alpha=0.5,
                        label='Best-48 Series')
                ax.plot(np.random.normal(2, 0.04, len(rand_vals)), rand_vals, 'o', ms=5, color='orange', alpha=0.5,
                        label='Random-48 Series')

                # Boxplot for distribution summary
                ax.boxplot([best_vals, rand_vals], positions=[1, 2], widths=0.4,
                           patch_artist=True, boxprops=dict(alpha=0.3), medianprops=dict(color='black', lw=2))

                # Baselines
                ax.axhline(b10, color='black', linestyle='--', lw=1.5, label='Baseline (10k Pure)')
                ax.axhline(b30, color='green', linestyle=':', lw=2, label='Ideal (30k Pure)')

                ax.set_xticks([1, 2])
                ax.set_xticklabels(['Best-48', 'Random-48'])
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax.margins(y=0.15)

        # Formatting Individual
        for i, metric in enumerate(metrics):
            axes_ind[i].set_title(f'{metric.upper()} DISTRIBUTION', fontsize=14, fontweight='bold')
            if i == 0: axes_ind[i].legend(fontsize=9)

        fig_ind.suptitle(f'Experiment 6: Metric Validation (Horizon h={h}, α={alpha})', fontsize=16)
        fig_ind.tight_layout()
        fig_ind.savefig(f'exp6_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        # Formatting Summary
        for i, metric in enumerate(metrics):
            ax_s = axes_sum[row_idx, i]
            if row_idx == 0: ax_s.set_title(f'SUMMARY: {metric.upper()}', fontsize=14)
            ax_s.set_ylabel(f'Horizon h={h}', fontweight='bold')

    fig_sum.suptitle(f'Experiment 6: Selected vs Random Donor Performance (α={alpha})', fontsize=18)
    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_sum.savefig(f'exp6_plots/summary_exp6_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.close(fig_sum)

print("Done. Validation results in 'exp6_plots'.")