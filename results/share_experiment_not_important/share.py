import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Чтение данных
df = pd.read_csv('share.txt', header=None, names=['RMSE', 'NP', 'MAPE', 'r', 'recipient_size'])

# Преобразуем r в число с плавающей точкой (на случай, если там строки)
df['r'] = pd.to_numeric(df['r'], errors='coerce')

# Получаем уникальные значения параметра r
unique_r = sorted(df['r'].unique())

# Создаем фигуру с тремя подграфиками
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
metrics = ['RMSE', 'NP', 'MAPE']
ylabels = ['RMSE', 'NP (доля)', 'MAPE']

# Для каждого r строим сглаженные кривые
for r_val in unique_r:
    subset = df[df['r'] == r_val].copy()
    subset = subset.sort_values('recipient_size')
    x = subset['recipient_size'].values
    if len(x) < 4:  # слишком мало точек для сплайна
        continue
    for i, metric in enumerate(metrics):
        y = subset[metric].values
        # Строим сплайн с небольшим сглаживанием (s - параметр, можно подобрать)
        # Используем s = len(x) * (стандартное отклонение y)^2 * 0.1, например
        # Но для простоты возьмем s = None, что даст интерполяционный сплайн,
        # а затем уменьшим число точек для сглаживания
        # Лучше использовать make_interp_spline и затем сгладить, или UnivariateSpline с s>0
        spline = UnivariateSpline(x, y, s=len(x)*np.var(y)*0.5, k=3)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = spline(x_smooth)
        axes[i].plot(x_smooth, y_smooth, label=f'r={r_val:.4f}'.rstrip('0').rstrip('.'))

# Оформление
for i, ax in enumerate(axes):
    ax.set_ylabel(ylabels[i])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='best', fontsize=8, ncol=2)

axes[-1].set_xlabel('recipient sample size')
plt.suptitle('The impact of the donor series on the quality of the forecast', fontsize=14)
plt.tight_layout()
plt.savefig('donor_effect_smoothed.png', dpi=300)
plt.show()