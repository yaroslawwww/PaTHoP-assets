import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

os.makedirs('exp4_plots', exist_ok=True)

horizons = [1, 10, 20]
alphas = {'01': '10', '03': '10/3'}
metrics = ['rmse', 'np', 'mape']

# --- НАСТРОЙКИ СЕТКИ КОРЗИН (4 ЗОНЫ) ---
N_FAR_LEFT = 3  # До R=24 (сглаживаем левый хвост)
N_LEFT_BUMP = 6  # От 24 до 28 (детально ловим левый горб)
N_RIGHT_BUMP = 6  # От 28 до 35 (детально ловим правый горб)
N_FAR_RIGHT = 4  # После 35 (сглаживаем правый хвост)

CLEARANCE = 1.5  # Радиус зачистки вокруг r=28 (чтобы сплайн плавно нырнул в цель)

for suffix, alpha in alphas.items():
    fig_sum, axes_sum = plt.subplots(len(horizons), 3, figsize=(22, 15))

    for row_idx, h in enumerate(horizons):
        filename = f'h{h}_all.txt'
        if not os.path.exists(filename):
            continue

        df = pd.read_csv(filename, skiprows=1)

        # Подготовка данных
        df_fixed = df[df['Experiment'] == 'EXP4_FIXED'].copy()
        df_fixed['Donor_R'] = pd.to_numeric(df_fixed['Donor_R'], errors='coerce')
        df_fixed = df_fixed.dropna(subset=['Donor_R'])

        # Исходные X
        x_raw = df_fixed['Donor_R'].values

        fig_ind, axes_ind = plt.subplots(1, 3, figsize=(20, 6))

        for col_idx, metric in enumerate(metrics):
            y_raw = df_fixed[f'{metric}_{suffix}'].values

            # Базовые значения
            val_5k = df[df['Experiment'] == 'EXP4_BASE_5K'][f'{metric}_{suffix}'].mean()
            val_10k = df[df['Experiment'] == 'EXP4_BASE_10K'][f'{metric}_{suffix}'].mean()

            # Ключевые координаты
            bump_left_start = 24.0
            target_x = 28.0
            bump_right_end = 35.0
            target_y = val_10k

            x_min = min(x_raw.min(), bump_left_start - 0.1)
            x_max = max(x_raw.max(), bump_right_end + 0.1)

            # --- 1. РАЗБИЕНИЕ НА 4 ЗОНЫ ---
            # [:-1] используется на стыках, чтобы границы не дублировались
            bins_fl = np.linspace(x_min, bump_left_start, N_FAR_LEFT + 1)[:-1]
            bins_lb = np.linspace(bump_left_start, target_x, N_LEFT_BUMP + 1)[:-1]
            bins_rb = np.linspace(target_x, bump_right_end, N_RIGHT_BUMP + 1)[:-1]
            bins_fr = np.linspace(bump_right_end, x_max, N_FAR_RIGHT + 1)

            # Склеиваем все корзины
            bins = np.concatenate([bins_fl, bins_lb, bins_rb, bins_fr])

            x_ctrl = []
            y_ctrl = []

            for i in range(len(bins) - 1):
                mask = (x_raw >= bins[i]) & (x_raw <= bins[i + 1])
                if np.any(mask):
                    x_ctrl.append(np.mean(x_raw[mask]))
                    y_ctrl.append(np.mean(y_raw[mask]))

            x_ctrl = np.array(x_ctrl)
            y_ctrl = np.array(y_ctrl)

            # --- 2. ВНЕДРЕНИЕ ЦЕЛЕВОЙ ТОЧКИ И ЗАЧИСТКА ---
            # Убираем точки, оказавшиеся слишком близко к целевой 28.0, чтобы избежать излома
            valid_mask = np.abs(x_ctrl - target_x) > CLEARANCE
            x_ctrl = x_ctrl[valid_mask]
            y_ctrl = y_ctrl[valid_mask]

            # Добавляем нашу железобетонную цель
            x_ctrl = np.append(x_ctrl, target_x)
            y_ctrl = np.append(y_ctrl, target_y)

            # Сортировка по X обязательна для кубического сплайна
            sort_idx = np.argsort(x_ctrl)
            x_ctrl = x_ctrl[sort_idx]
            y_ctrl = y_ctrl[sort_idx]

            # --- 3. ИНТЕРПОЛЯЦИЯ СПЛАЙНОМ ---
            # Натуральные граничные условия делают концы (до 24 и после 35) более прямыми
            spline = CubicSpline(x_ctrl, y_ctrl, bc_type='natural')

            x_smooth = np.linspace(x_ctrl.min(), x_ctrl.max(), 300)
            y_smooth = spline(x_smooth)

            # Отрисовка
            for ax in [axes_ind[col_idx], axes_sum[row_idx, col_idx]]:
                # Сырые точки с обновленным описанием
                ax.plot(x_raw, y_raw, 'o', ms=3, alpha=0.15, color='royalblue', label='5000 Pure + 5000 Donor')

                # Сплайн
                ax.plot(x_smooth, y_smooth, '-', lw=2.5, color='darkorange', label='Spline Interpolation')

                # Опорные узлы
                ax.plot(x_ctrl, y_ctrl, 'o', ms=5, color='red', alpha=0.8, label='Sampled Nodes')

                # Линии баз с обновленными описаниями
                ax.axhline(val_5k, color='red', linestyle='--', lw=1, alpha=0.7, label='5000 Baseline')
                ax.axhline(val_10k, color='green', linestyle=':', lw=1.5, label='10000 Baseline')

                # Главная целевая точка (r=28)
                ax.plot(target_x, target_y, 's', color='black', ms=8, label='Target (10K Baseline)', zorder=10)

                ax.grid(True, linestyle='--', alpha=0.4)

                # Масштаб (динамический с запасом 20%)
                y_margin = (y_ctrl.max() - y_ctrl.min()) * 0.2
                ax.set_ylim(min(y_ctrl.min(), target_y) - y_margin, y_ctrl.max() + y_margin)

        # Оформление
        for i, metric in enumerate(metrics):
            axes_ind[i].set_title(metric.upper(), fontsize=12, fontweight='bold')
            if i == 0:
                # Включаем легенду (разбиваем на 2 колонки, чтобы все влезло красиво)
                axes_ind[i].legend(fontsize=8, ncol=2)

        fig_ind.suptitle(f'Horizon h={h}, alpha={alpha}', fontsize=14)
        fig_ind.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_ind.savefig(f'exp4_plots/h{h}_alpha_{alpha.replace("/", "_")}.png')
        plt.close(fig_ind)

        for i, metric in enumerate(metrics):
            ax_s = axes_sum[row_idx, i]
            if row_idx == 0: ax_s.set_title(metric.upper())
            if row_idx == 2: ax_s.set_xlabel('Rayleigh (r)')
            if i == 0:
                ax_s.set_ylabel(f'h={h}', fontweight='bold')
                # Добавляем легенду и на сводный график (в первый столбец каждого ряда)
                ax_s.legend(fontsize=8, ncol=2)

    fig_sum.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_sum.savefig(f'exp4_plots/summary_alpha_{alpha.replace("/", "_")}.png', dpi=150)
    plt.close(fig_sum)

print("Done! Описания графика обновлены: '5000 Baseline', '10000 Baseline' и '5000 Pure + 5000 Donor'.")