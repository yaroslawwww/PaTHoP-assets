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
    colors = plt.cm.tab20.colors  # Палитра для линий

    for h in horizons:
        filename = f'h{h}_all.txt'
        if not os.path.exists(filename):
            continue

        df = pd.read_csv(filename, skiprows=1)

        # --- ПОДГОТОВКА ДАННЫХ HYBRID ---
        df_mix = df[df['Experiment'] == 'EXP3_HYBRID'].copy()
        df_mix['Donor_R'] = df_mix['Donor_R'].fillna('None').astype(str).str.strip()

        # Базовая точка (0 добавленных доноров)
        df_base = df_mix[df_mix['Donor_R'] == 'None']

        # Получаем всех доноров и разделяем на 2 группы
        donors = [d for d in df_mix['Donor_R'].unique() if d != 'None']
        donors_lt_29 = sorted([d for d in donors if float(d) < 29.0], key=float)
        donors_ge_29 = sorted([d for d in donors if float(d) >= 29.0], key=float)

        # --- ПОДГОТОВКА БАЗОВОЙ ЛИНИИ (ЧИСТОЕ РАСШИРЕНИЕ EXP1_CLEAR) ---
        df_clear = df[df['Experiment'] == 'EXP1_CLEAR'].copy()
        # Ограничиваем ось X от 10000 до 40000
        df_clear = df_clear[(df_clear['Recipient_Size'] >= 10000) & (df_clear['Recipient_Size'] <= 40000)]
        df_clear_c = df_clear.groupby('Recipient_Size').mean(numeric_only=True).reset_index().sort_values(
            'Recipient_Size')

        # Создаем сетку 2x3: 2 ряда (группы доноров), 3 колонки (метрики)
        fig, axes = plt.subplots(2, 3, figsize=(22, 14))

        donor_groups = [
            (donors_lt_29, 'Donors R < 29', axes[0]),
            (donors_ge_29, 'Donors R >= 29', axes[1])
        ]

        for row_idx, (current_donors, group_title, ax_row) in enumerate(donor_groups):

            # 1. Отрисовка базовой линии чистого расширения (Черный пунктир)
            x_clear = df_clear_c['Recipient_Size'].values
            for col_idx, metric in enumerate(metrics):
                y_clear = df_clear_c[f'{metric}_{suffix}'].values

                # Фиксируем сплайн в начальной точке
                w_c = np.ones_like(y_clear, dtype=float)
                w_c[0] = 1000.0

                spline_c = UnivariateSpline(x_clear, y_clear, w=w_c)
                spline_c.set_smoothing_factor(len(x_clear) * np.var(y_clear) * 0.5 + 1e-6)

                xs_c = np.linspace(x_clear.min(), x_clear.max(), 300)
                ys_c = spline_c(xs_c)
                ys_c[0] = y_clear[0]

                ax_row[col_idx].plot(xs_c, ys_c, '--', color='black', lw=2.5, label='Pure Recipient Expansion',
                                     zorder=10)

            # 2. Отрисовка каждого донора из текущей группы
            for c_idx, donor in enumerate(current_donors):
                color = colors[c_idx % len(colors)]
                df_d = df_mix[df_mix['Donor_R'] == donor]
                df_combined = pd.concat([df_d, df_base])

                # Усреднение и сортировка
                df_c = df_combined.groupby('Donor_Size').mean(numeric_only=True).reset_index().sort_values('Donor_Size')

                # Прибавляем 10000 к размеру донора, чтобы получить общий размер
                x = df_c['Donor_Size'].values + 10000

                # Отсекаем точки, ушедшие за 40000
                mask_x = x <= 40000
                x = x[mask_x]

                # Если в вашем полном файле есть донор точно "28.0" или "28.0000",
                # и вы хотите его выделить жирным, раскомментируйте код ниже:
                # is_target = (donor == '28.0' or donor == '28.0000')

                for col_idx, metric in enumerate(metrics):
                    y = df_c[f'{metric}_{suffix}'].values[mask_x]

                    # Веса для жесткой фиксации сплайна в точке x=10000
                    w = np.ones_like(y, dtype=float)
                    w[0] = 1000.0
                    spline = UnivariateSpline(x, y, w=w)
                    spline.set_smoothing_factor(len(x) * np.var(y) * 0.5 + 1e-6)

                    xs = np.linspace(x.min(), x.max(), 300)
                    ys = spline(xs)
                    ys[0] = y[0]

                    # Отрисовываем саму линию
                    ax_row[col_idx].plot(xs, ys, '-', lw=2, color=color, label=f'r={float(donor):.4f}')

                    # Отрисовываем ТОЧКИ ярко и без прозрачности (alpha=0.9, ms=5)
                    ax_row[col_idx].plot(x, y, 'o', ms=5, alpha=0.9, color=color, markeredgecolor='white',
                                         markeredgewidth=0.5, zorder=12)

            # 3. Оформление графиков ряда
            for col_idx, metric in enumerate(metrics):
                ax = ax_row[col_idx]

                # Золотая звезда Baseline (10k)
                y_start = df_base[f'{metric}_{suffix}'].mean()
                ax.plot(10000, y_start, '*', ms=15, color='gold', markeredgecolor='black', zorder=25,
                        label='Baseline (10k)')

                ax.set_title(f'{metric.upper()} ({group_title})', fontsize=14, fontweight='bold')

                # Подписываем ось X только у нижнего ряда
                if row_idx == 1:
                    ax.set_xlabel('General Size (10000 pure series + x donor)', fontsize=12)

                ax.grid(True, linestyle='--', alpha=0.5)

                # ЖЕСТКАЯ ФИКСАЦИЯ ОСИ X (от 9500 до 40500 для отступов по краям)
                ax.set_xlim(9000, 41000)

                # Компактная легенда в 2 колонки
                ax.legend(fontsize=8, ncol=2)
                ax.margins(y=0.1)

        fig.suptitle(f'Experiment 3: Hybrid Expansion (h={h}, alpha={alpha})', fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        fig.savefig(f'exp3_plots/h{h}_alpha_{alpha.replace("/", "_")}.png', dpi=200)
        plt.close(fig)

print("Done! Теперь точки (кружочки) на всех линиях сделаны яркими и непрозрачными.")