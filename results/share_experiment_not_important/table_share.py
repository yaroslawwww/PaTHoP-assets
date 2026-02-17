import pandas as pd
import numpy as np

# Загрузка данных
df = pd.read_csv('share.txt', header=None, sep=',',
                 names=['RMSE', 'NP', 'MAPE', 'r', 'recipient_size'])

# Удаление дубликатов
df = df.drop_duplicates()

# Перевод NP в проценты и округление
df['NP'] = (df['NP'] * 100).round(2)
df['RMSE'] = df['RMSE'].round(4)
df['MAPE'] = df['MAPE'].round(4)

# Отбор нужных значений r (исключаем 27.9)
selected_r = [27.99, 28.0001, 28.01, 28.1]
df = df[df['r'].isin(selected_r)]


# Создаем функцию для формирования широкой таблицы по списку r
def make_wide_table(df, r_list, recipient_sizes):
    # Фильтруем по r и размерам
    subset = df[df['r'].isin(r_list) & df['recipient_size'].isin(recipient_sizes)].copy()
    subset = subset.sort_values(['r', 'recipient_size'])

    # Сводные таблицы по метрикам
    pivot_rmse = subset.pivot(index='recipient_size', columns='r', values='RMSE')
    pivot_np = subset.pivot(index='recipient_size', columns='r', values='NP')
    pivot_mape = subset.pivot(index='recipient_size', columns='r', values='MAPE')

    # Объединяем столбцы в нужном порядке: для каждого r: RMSE, NP%, MAPE
    wide = pd.DataFrame(index=pivot_rmse.index)
    for r in r_list:
        wide[f'RMSE (r={r})'] = pivot_rmse[r]
        wide[f'NP% (r={r})'] = pivot_np[r]
        wide[f'MAPE (r={r})'] = pivot_mape[r]

    wide = wide.reset_index().rename(columns={'recipient_size': 'Recipient size'})
    return wide


# Получаем все уникальные размеры реципиента
all_sizes = sorted(df['recipient_size'].unique())
# Сокращаем до 12 значений (каждый второй)
reduced_sizes = all_sizes[::2]  # берём каждый второй, можно настроить

# Группы доноров
group1 = [27.99, 28.0001]  # левая часть
group2 = [28.01, 28.1]  # правая часть

# Генерируем две таблицы
table1 = make_wide_table(df, group1, reduced_sizes)
table2 = make_wide_table(df, group2, reduced_sizes)


# Функция для вывода LaTeX с нужным форматированием
def latex_table(df, caption, label):
    return df.to_latex(
        index=False,
        float_format="%.4f",
        caption=caption,
        label=label,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1)  # первый столбец влево, остальные по центру
    )


# Генерируем код для двух таблиц
latex1 = latex_table(table1,
                     "Metrics for donors $r=27.99$ and $r=28.0001$ (left part).",
                     "tab:donors_left")
latex2 = latex_table(table2,
                     "Metrics for donors $r=28.01$ and $r=28.1$ (right part).",
                     "tab:donors_right")

print(latex1)
print(latex2)