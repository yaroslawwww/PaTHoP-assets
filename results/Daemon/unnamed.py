#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv(
    'daemons_size_experiment_basic_dbscan_valid.txt',
    header=None,
    names=[
        'deviation',
        'arg',
        'prediction_size',
        'rmses',
        'np_points',
        'mape',
        'avg_cl',
        'general_size'
    ]
)
# Уникальные значения отклонений
deviations = df['deviation'].unique()
exclude_deviations = [31,32,33,34,35,36,37,38,39,40]
df_filtered = df[~df['deviation'].isin(exclude_deviations)]

# Для r=28 (deviation=0) удаляем точки с size < 10000
df_r28 = df_filtered[df_filtered['deviation'] == 0]
df_r28 = df_r28[df_r28['general_size'] >= 10000]
df_others = df_filtered[df_filtered['deviation'] != 0]
df_filtered = pd.concat([df_r28, df_others])

# Уникальные значения отклонений после фильтрации
deviations = df_filtered['deviation'].unique()

# Находим точку с r=28 и size=10000
reference_point = df_filtered[
    (df_filtered['deviation'] == 0) &
    (df_filtered['general_size'] == 10000)
].iloc[0]

# Создание фигуры с тремя субграфиками
plt.figure(figsize=(12, 18))
plt.style.use('seaborn-v0_8-darkgrid')

# График для RMSE
plt.subplot(3, 1, 1)
for d in deviations:
    subset = df_filtered[df_filtered['deviation'] == d]
    subset = subset.sort_values('general_size')
    plt.plot(
        subset['general_size'],
        subset['rmses'],
        'o-',
        linewidth=2,
        markersize=6,
        label=f'r={d+28}'
    )

# Добавляем горизонтальную линию и выделяем точку для RMSE
plt.axhline(y=reference_point['rmses'], color='black', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(reference_point['general_size'], reference_point['rmses'], 'ko', markersize=10)

plt.title('RMSE vs General Size', fontsize=14)
plt.ylabel('RMSE', fontsize=12)
plt.grid(True, alpha=0.7)
plt.legend()

# График для np_points
plt.subplot(3, 1, 2)
for d in deviations:
    subset = df_filtered[df_filtered['deviation'] == d]
    subset = subset.sort_values('general_size')
    plt.plot(
        subset['general_size'],
        subset['np_points'],
        's--',
        linewidth=2,
        markersize=6,
        label=f'r={d+28}'
    )

# Добавляем горизонтальную линию и выделяем точку для np_points
plt.axhline(y=reference_point['np_points'], color='black', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(reference_point['general_size'], reference_point['np_points'], 'ko', markersize=10)

plt.title('Number of Points vs General Size', fontsize=14)
plt.ylabel('np_points', fontsize=12)
plt.grid(True, alpha=0.7)
plt.legend()

# График для MAPE
plt.subplot(3, 1, 3)
for d in deviations:
    subset = df_filtered[df_filtered['deviation'] == d]
    subset = subset.sort_values('general_size')
    plt.plot(
        subset['general_size'],
        subset['mape'],
        '^-.',
        linewidth=2,
        markersize=6,
        label=f'r={d+28}'
    )

# Добавляем горизонтальную линию и выделяем точку для MAPE
plt.axhline(y=reference_point['mape'], color='black', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(reference_point['general_size'], reference_point['mape'], 'ko', markersize=10)

plt.title('MAPE vs General Size', fontsize=14)
plt.xlabel('General Size', fontsize=12)
plt.ylabel('MAPE (%)', fontsize=12)
plt.grid(True, alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('daempon_apr.png', dpi=300)
plt.show()
#%%
