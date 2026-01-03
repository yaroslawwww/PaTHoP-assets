import pandas as pd
import numpy as np
from scipy import stats
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore")

# --- КОНФИГУРАЦИЯ ФАЙЛОВ ---
FILES_CONFIG = {
    1: {
        'baseline': 'baseline_480_1.txt',
        'best': 'best_480_1.txt',
        'random': 'random_480_1.txt'
    },
    10: {
        'baseline': 'baselines.txt',
        'best': 'best_480.txt',
        'random': 'random_480.txt'
    },
    20: {
        'baseline': 'baseline_480_20.txt',
        'best': 'best_480_20.txt',
        'random': 'random_480_20.txt'
    }
}


def parse_line(line):
    parts = [p.strip() for p in line.strip().split(',')]
    values = []
    for p in parts:
        if p:
            try:
                values.append(float(p))
            except ValueError:
                continue
    if len(values) >= 3:
        return {
            'RMSE': values[0],
            'NP': values[1],
            'MAPE': values[2],
            'r': values[3] if len(values) > 3 else np.nan
        }
    return None


def load_data(filepath, is_baseline=False):
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parsed = parse_line(line)
                if parsed:
                    data.append(parsed)
    except FileNotFoundError:
        return None

    if is_baseline:
        if len(data) < 2:
            return None
        return {'10k': data[0], '30k': data[1]}

    return pd.DataFrame(data)


def calculate_stats_v3(series, b10k, b30k):
    stats_res = {}
    stats_res['Median'] = series.median()

    # Thresholds
    thresh_easy = max(b10k, b30k)  # Worse baseline
    thresh_hard = min(b10k, b30k)  # Better baseline

    # Winner (Better than Worst Baseline)
    winners = series[series < thresh_easy]
    stats_res['Win_Med'] = winners.median() if not winners.empty else np.nan

    # Absolute Winner (Better than Best Baseline)
    abs_winners = series[series < thresh_hard]
    stats_res['AbsWin_Med'] = abs_winners.median() if not abs_winners.empty else np.nan

    # Loser (Worse than Worst Baseline)
    losers = series[series > thresh_easy]
    stats_res['Lose_Med'] = losers.median() if not losers.empty else np.nan

    # Defect (> 1.5 * Base 10k)
    defect_thresh = b10k * 1.5
    defect_count = (series > defect_thresh).sum()
    stats_res['Defect_%'] = (defect_count / len(series)) * 100

    return stats_res


def format_val(val, decimals=5):
    """Форматирование числа для LaTeX (обработка NaN)."""
    if pd.isna(val):
        return "-"  # или "n/a"
    return f"{val:.{decimals}f}"


def generate_latex_table(df):
    """Генерация строки LaTeX таблицы с подробным описанием."""

    # Шапка таблицы
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of forecast quality metrics across prediction horizons ($H$). The proposed selection method is compared against short ($B_{10k}$) and extended ($B_{30k}$) baselines, as well as random selection.}",
        r"\label{tab:forecast_results}",
        # Используем resizebox если таблица слишком широкая, или просто tabular
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{|c|c|cc|c|ccc|cc|c|}",
        r"\toprule",
        r"\hline",
        # Группировка заголовков для красоты
        r"\multirow{2}{*}{$H$} & \multirow{2}{*}{Metric} & \multicolumn{2}{c|}{Baselines} & \multicolumn{4}{c|}{Proposed Selection (Best-48)} & \multicolumn{2}{c|}{Defect Rate ($D\%$)} & \multirow{2}{*}{$p$-val} \\",
        r"\cline{3-10}",
        # Подзаголовки
        r" & & $B_{10k}$ & $B_{30k}$ & Median & Win $Q_{0.5}$ & AbsWin $Q_{0.5}$ & Lose $Q_{0.5}$ & Best & Rand & \\",
        r"\hline",
        r"\midrule"
    ]

    # Данные
    for idx, row in df.iterrows():
        h_val = int(row['H'])
        metric = row['Metric']
        b10k = format_val(row['Base_10k'])
        b30k = format_val(row['Base_30k'])
        med = format_val(row['Best_Med'])
        win = format_val(row['Win_Q50'])
        abs_win = format_val(row['AbsWin_Q50'])
        lose = format_val(row['Lose_Q50'])
        d_best = format_val(row['Best_Def(%)'], 2)
        d_rand = format_val(row['Rand_Def(%)'], 2)

        # Форматирование p-value
        if row['P-Val'] < 0.001:
            pval = r"$<10^{-4}$"
        else:
            pval = format_val(row['P-Val'], 3)

        # Выделяем жирным, если AbsWin существует (лучше оригинала)
        if abs_win != "-":
            abs_win = r"\textbf{" + abs_win + "}"

        line = f"{h_val} & {metric} & {b10k} & {b30k} & {med} & {win} & {abs_win} & {lose} & {d_best} & {d_rand} & {pval} \\\\"
        latex.append(line)
        latex.append(r"\hline")

    # Подвал таблицы с детальным описанием
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",  # Закрываем resizebox
        r"\vspace{1mm}",
        r"\begin{justify}",  # Требуется пакет ragged2e, или просто убрать environment
        r"\footnotesize",
        r"\textbf{Note:} ",
        r"\textbf{$H$}: Forecasting horizon steps. ",
        r"\textbf{$B_{10k}, B_{30k}$}: Performance of the target series trained on the original limited sample ($10^4$ points) and the extended sample ($3 \times 10^4$ points), respectively. ",
        r"\textbf{Median}: The median metric value across the selected donor group. ",
        r"\textbf{Win $Q_{0.5}$}: Median performance of the donor subset that outperformed the \textit{weakest} baseline (values $< \max(B_{10k}, B_{30k})$). ",
        r"\textbf{AbsWin $Q_{0.5}$ (Absolute Winner)}: Median performance of the elite donor subset that outperformed the \textit{strongest} baseline (values $< \min(B_{10k}, B_{30k})$), representing a quality gain unattainable by the target series alone. ",
        r"\textbf{Lose $Q_{0.5}$}: Median performance of donors performing worse than both baselines. ",
        r"\textbf{$D (\%)$ (Defect Rate)}: The proportion of donors resulting in negative transfer, defined as an error increase $> 1.5 \times B_{10k}$. ",
        r"\textbf{$p$-val}: Statistical significance of the improvement over random selection (Mann-Whitney U test, one-sided).",
        r"\end{justify}",
        r"\end{table*}"
    ])

    return "\n".join(latex)


def main():
    table_rows = []

    for h in [1, 10, 20]:
        files = FILES_CONFIG[h]
        baselines = load_data(files['baseline'], is_baseline=True)
        df_best = load_data(files['best'])
        df_random = load_data(files['random'])

        if not baselines or df_best is None or df_best.empty or df_random is None:
            continue

        for metric in ['RMSE', 'NP', 'MAPE']:
            b10k = baselines['10k'][metric]
            b30k = baselines['30k'][metric]

            best_stats = calculate_stats_v3(df_best[metric], b10k, b30k)
            rand_stats = calculate_stats_v3(df_random[metric], b10k, b30k)

            try:
                _, p_val = stats.mannwhitneyu(df_best[metric], df_random[metric], alternative='less')
            except ValueError:
                p_val = 1.0

            row = {
                'H': h,
                'Metric': metric,
                'Base_10k': b10k,
                'Base_30k': b30k,
                'Best_Med': best_stats['Median'],
                'Win_Q50': best_stats['Win_Med'],
                'AbsWin_Q50': best_stats['AbsWin_Med'],
                'Lose_Q50': best_stats['Lose_Med'],
                'Best_Def(%)': best_stats['Defect_%'],
                'Rand_Def(%)': rand_stats['Defect_%'],
                'P-Val': p_val
            }
            table_rows.append(row)

    df = pd.DataFrame(table_rows)

    # Генерация LaTeX
    latex_code = generate_latex_table(df)

    print("-" * 20)
    print(latex_code)
    print("-" * 20)

    with open('eswa_table.tex', 'w') as f:
        f.write(latex_code)
    print("LaTeX table saved to 'eswa_table.tex'")


if __name__ == "__main__":
    main()