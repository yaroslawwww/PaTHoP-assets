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
    # Создаем сетку 3x3 для сводного графика текущего alpha
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(16, 10))
    colors = plt.cm.tab10.colors

    for row_idx, h in enumerate(horizons):
        df = pd.read_csv(f'h{h}_all.txt', skiprows=1)
        df_mix = df[df['Experiment'] == 'EXP2_MIX'].copy()

        # Исправление ошибки m=0: надежно заполняем NaN и приводим к строкам
        df_mix['Donor_R'] = df_mix['Donor_R'].fillna('None').astype(str).str.strip()

        # Извлекаем "чистый" реципиент (Donor_R = 'None') и список доноров
        df_base = df_mix[df_mix['Donor_R'] == 'None']
        donors = sorted([d for d in df_mix['Donor_R'].unique() if d != 'None'])

        # Индивидуальный график для текущего (h, alpha)
        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(16, 4))

        for c_idx, donor in enumerate(donors):
            # Объединяем данные донора и добавляем точку 100% реципиента
            df_d = df_mix[df_mix['Donor_R'] == donor]
            df_combined = pd.concat([df_d, df_base])

            # Усредняем дубликаты и сортируем для строго возрастающего X (важно для сплайна)
            df_c = df_combined.groupby('Recipient_Size').mean(numeric_only=True).reset_index()
            x = df_c['Recipient_Size'].values

            label_text = f'Donor r={float(donor):.4f}'

            for col_idx, metric in enumerate(metrics):
                y = df_c[f'{metric}_{suffix}'].values

                # Сглаживающий сплайн
                spline = UnivariateSpline(x, y)
                spline.set_smoothing_factor(len(x) * np.var(y) * 0.5 + 1e-6)
                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = spline(x_smooth)

                # Рисуем на индивидуальном графике (точки + линия)
                axes_ind[col_idx].plot(x, y, 'o', alpha=0.3, color=colors[c_idx])
                axes_ind[col_idx].plot(x_smooth, y_smooth, '-', lw=2, color=colors[c_idx], label=label_text)

                # Рисуем только линию на сводном графике
                ax_sum = axes_sum[row_idx, col_idx]
                ax_sum.plot(x_smooth, y_smooth, '-', lw=2, color=colors[c_idx], label=label_text)

        # --- Оформление индивидуального графика ---
        for col_idx, metric in enumerate(metrics):
            axes_ind[col_idx].set_title(metric.upper())
            axes_ind[col_idx].set_xlabel('Доля реципиента (Recipient_Size)')
            axes_ind[col_idx].grid(True, linestyle='--', alpha=0.6)
            axes_ind[col_idx].legend(fontsize='small')

        fig_ind.suptitle(f'Experiment 2: Horizon h={h}, α={alpha} (N_total=10000)', fontsize=14)
        fig_ind.tight_layout()
        fig_ind.savefig(f'exp2_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        # --- Оформление сводного графика (текущая строка) ---
        for col_idx, metric in enumerate(metrics):
            ax_sum = axes_sum[row_idx, col_idx]
            if row_idx == 0:
                ax_sum.set_title(f'SUMMARY: {metric.upper()}', fontsize=12, fontweight='bold')
            if row_idx == len(horizons) - 1:
                ax_sum.set_xlabel('Recipient_Size')

            ax_sum.set_ylabel(f'h={h}', fontweight='bold')
            ax_sum.grid(True, linestyle='--', alpha=0.5)
            if col_idx == 0:  # Легенду делаем только в первом столбце
                ax_sum.legend(fontsize='8')

    # Сохраняем сводный график для текущей альфы
    fig_sum.suptitle(f'Experiment 2: All Horizons Trend Comparison (α={alpha}, N_total=10000)', fontsize=16)
    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_sum.savefig(f'exp2_plots/summary_exp2_alpha_{alpha.replace("/", "_")}.png', dpi=200)
    plt.close(fig_sum)

print("Done. Check 'exp2_plots' folder.")