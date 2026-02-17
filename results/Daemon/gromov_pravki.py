import pandas as pd

def generate_latex_table(dbscan_file, apriori_file):
    """
    Читает два текстовых файла (результаты экспериментов),
    формирует объединённую таблицу и возвращает строку с LaTeX-кодом.
    """
    # 1. Загружаем данные без заголовков
    df_db = pd.read_csv(dbscan_file, header=None)
    df_ap = pd.read_csv(apriori_file, header=None)

    # 2. Оставляем только строки, где первый столбец равен 0
    df_db = df_db[df_db[0] == 0]
    df_ap = df_ap[df_ap[0] == 0]

    # 3. Удаляем первые три столбца (индексы 0,1,2)
    df_db = df_db.drop(columns=[0, 1, 2])
    df_ap = df_ap.drop(columns=[0, 1, 2])

    # 4. Присваиваем понятные имена столбцам
    # После удаления остаются столбцы с исходными индексами 3,4,5,6
    df_db.columns = ['rmse_db', 'np_db', 'mape_db', 'size']
    df_ap.columns = ['rmse_ap', 'np_ap', 'mape_ap', 'size']

    # 5. Удаляем дубликаты по размеру выборки (чтобы для каждого size была одна строка)
    df_db = df_db.drop_duplicates(subset='size')
    df_ap = df_ap.drop_duplicates(subset='size')

    # 6. Объединяем таблицы по размеру выборки (inner join)
    merged = pd.merge(df_db, df_ap, on='size', how='inner')

    # 7. Сортируем по размеру выборки
    merged = merged.sort_values('size')

    # 8. Переставляем столбец size в начало
    cols = ['size', 'rmse_db', 'np_db', 'mape_db', 'rmse_ap', 'np_ap', 'mape_ap']
    merged = merged[cols]

    # 9. Преобразуем долю непрогнозируемых точек в проценты
    merged['np_db'] = merged['np_db'] * 100
    merged['np_ap'] = merged['np_ap'] * 100

    # 10. Генерируем LaTeX-таблицу с использованием booktabs и siunitx
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(r'\caption{Comparison of prediction quality for DBSCAN and Apriori methods.}')
    latex.append(r'\label{tab:comparison}')
    latex.append(r'\begin{tabular}{')
    latex.append(r'  S[table-format=5.0]   % Training Sample Size')
    latex.append(r'  S[table-format=1.4]   % RMSE(DBSCAN)')
    latex.append(r'  S[table-format=2.2]   % NP (DBSCAN)')
    latex.append(r'  S[table-format=1.4]   % MAPE (DBSCAN)')
    latex.append(r'  S[table-format=1.4]   % RMSE (Apriori)')
    latex.append(r'  S[table-format=2.2]   % NP (Apriori)')
    latex.append(r'  S[table-format=1.4]   % MAPE (Apriori)')
    latex.append(r'}')
    latex.append(r'\toprule')
    latex.append(r'{Size} & {RMSE (DBSCAN)} & {NP (\%) (DBSCAN)} & {MAPE (DBSCAN)} & {RMSE (Apriori)} & {NP (\%) (Apriori)} & {MAPE (Apriori)} \\')
    latex.append(r'\midrule')

    # Заполняем строки данными
    for _, row in merged.iterrows():
        line = (f"{row['size']:.0f} & {row['rmse_db']:.4f} & {row['np_db']:.2f} & "
                f"{row['mape_db']:.4f} & {row['rmse_ap']:.4f} & {row['np_ap']:.2f} & "
                f"{row['mape_ap']:.4f} \\\\")
        latex.append(line)

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)

if __name__ == '__main__':
    # Укажите пути к вашим файлам
    file1 = 'daemons_size_experiment_basic_dbscan.txt'
    file2 = 'daemons_size_experiment_apriori.txt'

    try:
        latex_output = generate_latex_table(file1, file2)
        print(latex_output)
    except Exception as e:
        print(f'Ошибка при обработке: {e}')