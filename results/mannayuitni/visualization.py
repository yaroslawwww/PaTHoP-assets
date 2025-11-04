import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. Подготовка данных
metric_names_orig = ['RMSE', 'Непрогнозируемые точки', 'MAPE']
metric_names_display = ['RMSE', 'Непрогн. точки, %', 'MAPE']
col_names = metric_names_orig + ['Параметр r']

df_best = pd.read_csv('best.txt', header=None, names=col_names)
df_random = pd.read_csv('random.txt', header=None, names=col_names)
df_baselines = pd.read_csv('baselines.txt', header=None, names=metric_names_orig)

for df in [df_best, df_random, df_baselines]:
    df['Непрогнозируемые точки'] = df['Непрогнозируемые точки'] * 100

df_best['Метод'] = 'Лучшая аугментация'
df_random['Метод'] = 'Случайная аугментация'

df_combined = pd.concat([df_best, df_random])

df_combined.rename(columns={'Непрогнозируемые точки': 'Непрогн. точки, %'}, inplace=True)
df_baselines.rename(columns={'Непрогнозируемые точки': 'Непрогн. точки, %'}, inplace=True)

df_long = df_combined.melt(
    id_vars=['Метод'],
    value_vars=metric_names_display,
    var_name='Метрика',
    value_name='Значение'
)

# 2. Визуализация
sns.set_theme(style="whitegrid")
palette = {"Лучшая аугментация": "#3498db", "Случайная аугментация": "#f1c40f"}
groups = ['Лучшая аугментация', 'Случайная аугментация']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Сравнение результатов аугментации с эталонными значениями', fontsize=18, y=1.0)

for i, group_name in enumerate(groups):
    for j, metric_name in enumerate(metric_names_display):
        ax = axes[i, j]

        group_data = df_long[(df_long['Метод'] == group_name) & (df_long['Метрика'] == metric_name)]

        # Изменение здесь: убран параметр x
        sns.stripplot(
            data=group_data, y='Значение',
            color=palette[group_name], ax=ax,
            alpha=0.6, jitter=0.25, size=6
        )

        baseline_10k = df_baselines.loc[0, metric_name]
        baseline_30k = df_baselines.loc[1, metric_name]
        ax.axhline(baseline_10k, ls='--', color='gray', lw=2)
        ax.axhline(baseline_30k, ls=':', color='black', lw=2)

        if i == 0:
            ax.set_title(metric_name, fontsize=14)

        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        # ax.set_yscale('log')

        if j == 0:
            ax.set_ylabel(group_name, fontsize=14, labelpad=15)

# 3. Создание общей легенды
handles = [
    Line2D([0], [0], color='gray', linestyle='--', lw=2),
    Line2D([0], [0], color='black', linestyle=':', lw=2)
]
labels = [
    'Оригинал (10k)',
    'Оригинал (30k)'
]
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('Augmentations_benefit_visualisation.png')
plt.show()