import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
import io

# -------------------------------------------------------------------
# EMBEDDED TABLE DATA FROM THE PAPER (h = 10)
# -------------------------------------------------------------------
def get_table_exp1():
    data = {
        'Recipient_Size': [
            10000, 11304, 12608, 13913, 15217, 16521, 17826, 19130,
            20434, 21739, 23043, 24347, 25652, 26956, 28260, 29565,
            30869, 32173, 33478, 34782, 36086, 37391, 38695, 40000
        ],
        'rmse_01': [
            0.0421, 0.0276, 0.0365, 0.0328, 0.0474, 0.0428, 0.0435, 0.0370,
            0.0461, 0.0370, 0.0284, 0.0220, 0.0299, 0.0275, 0.0287, 0.0428,
            0.0313, 0.0337, 0.0284, 0.0287, 0.0263, 0.0227, 0.0311, 0.0382
        ],
        'np_01': [
            12.80, 9.93, 10.47, 9.93, 8.80, 9.60, 8.80, 8.13,
            6.53, 6.53, 5.53, 3.47, 4.53, 3.40, 4.53, 2.67,
            4.73, 4.87, 4.00, 2.73, 4.13, 2.67, 2.27, 3.00
        ]
    }
    df = pd.DataFrame(data)
    df['Experiment'] = 'EXP1_CLEAR'
    return df

def get_table_exp2():
    table2_data = {
        'Recipient_Size': [
            0, 870, 1739, 2609, 3478, 4348, 5217, 6087, 6957, 7826, 8696, 10000
        ],
        '27.9_rmse':    [0.1567, 0.1169, 0.0793, 0.0839, 0.0497, 0.0720, 0.0440, 0.0531, 0.0395, 0.0479, 0.0389, 0.0374],
        '27.9_np':      [24.73, 31.13, 17.67, 18.47, 13.80, 13.40, 14.20, 12.80, 14.60, 11.47, 12.07, 11.27],
        '28.0001_rmse': [0.1902, 0.0806, 0.0469, 0.0644, 0.0907, 0.0938, 0.0675, 0.0435, 0.0655, 0.0837, 0.0462, 0.0374],
        '28.0001_np':   [32.87, 15.80, 11.13, 10.33, 20.47, 18.67, 17.20, 18.73, 15.53, 15.13, 14.93, 11.27],
        '28.01_rmse':   [0.2014, 0.2004, 0.1243, 0.0377, 0.0896, 0.0751, 0.0546, 0.0690, 0.0674, 0.0683, 0.0346, 0.0374],
        '28.01_np':     [32.60, 38.20, 31.07, 13.60, 18.87, 18.87, 15.87, 15.40, 18.27, 18.07, 18.27, 11.27],
        '28.1_rmse':    [0.1784, 0.1602, 0.1141, 0.1251, 0.0743, 0.0650, 0.0467, 0.0433, 0.0511, 0.0587, 0.0413, 0.0374],
        '28.1_np':      [31.00, 41.07, 26.33, 24.87, 15.13, 16.00, 14.07, 13.87, 14.67, 12.07, 20.27, 11.27]
    }
    df_t2 = pd.DataFrame(table2_data)
    exp2_records = []
    donors = [27.9, 28.0001, 28.01, 28.1]
    for r in donors:
        r_str = str(r)
        for _, row in df_t2.iterrows():
            exp2_records.append({
                'Experiment': 'EXP2_MIX',
                'Recipient_Size': row['Recipient_Size'],
                'rmse_01': row[f'{r_str}_rmse'],
                'np_01': row[f'{r_str}_np'],
                'Donor_R_Val': r
            })
    return pd.DataFrame(exp2_records)

# -------------------------------------------------------------------
# LOAD DATA FROM FILES (NO SAFETY CHECKS)
# -------------------------------------------------------------------
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if ',' in line]
    df = pd.read_csv(io.StringIO("".join(data_lines)))
    df['Donor_R_Val'] = pd.to_numeric(df['Donor_R'], errors='coerce')
    return df

# Load h=1 and h=20 data from files (contains all experiments)
df_h1 = load_data('h1_all.txt')
df_h20 = load_data('h20_all.txt')

# For h=10 use embedded tables
df_h10_exp1 = get_table_exp1()
df_h10_exp2 = get_table_exp2()
# No h=10 data for experiment 3 in the paper, skip it
df_h1['np_01'] = df_h1['np_01'] * 100
df_h20['np_01'] = df_h20['np_01'] * 100

# -------------------------------------------------------------------
# UNIVERSAL SPLINE PLOTTING FUNCTION
# -------------------------------------------------------------------
def plot_spline_safe(ax, x_raw, y_raw, color, marker='o', label=None,
                     s=0.5, linewidth=2, linestyle='-',
                     highlight_zero=False, zero_marker='D', zero_size=80):
    df = pd.DataFrame({'x': x_raw, 'y': y_raw})
    df = df.sort_values('x').reset_index(drop=True)
    df = df.groupby('x', as_index=False)['y'].mean()
    x = df['x'].values
    y = df['y'].values

    mask_zero = (x == 0) if highlight_zero else np.zeros_like(x, dtype=bool)
    ax.scatter(x[~mask_zero], y[~mask_zero], color=color, alpha=0.5, s=25,
               marker=marker, edgecolors='none')
    if highlight_zero and np.any(mask_zero):
        ax.scatter(x[mask_zero], y[mask_zero], color=color, alpha=1.0, s=zero_size,
                   marker=zero_marker, edgecolors='black', linewidths=1.5,
                   label=None, zorder=5)

    if len(x) > 3:
        spl = UnivariateSpline(x, y, s=s)
        x_smooth = np.linspace(min(x), max(x), 300)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)
    else:
        ax.plot(x, y, linestyle=linestyle, color=color, linewidth=linewidth, label=label)

# -------------------------------------------------------------------
# EXPERIMENT 1: Pure sample expansion (3 lines on two subplots)
# -------------------------------------------------------------------
def plot_exp1_clear(df_h1, df_h10, df_h20, save_path='exp1_clear.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Experiment 1: Effect of pure training sample size', fontsize=16)

    # RMSE
    plot_spline_safe(axes[0], df_h1['Recipient_Size'], df_h1['rmse_01'],
                     color='blue', marker='o', label='h=1', s=0.5)
    plot_spline_safe(axes[0], df_h10['Recipient_Size'], df_h10['rmse_01'],
                     color='green', marker='^', label='h=10', s=0.5)
    plot_spline_safe(axes[0], df_h20['Recipient_Size'], df_h20['rmse_01'],
                     color='red', marker='s', label='h=20', s=0.5)
    axes[0].set_xlabel('Sample size (N)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Square Error (RMSE)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # NP
    plot_spline_safe(axes[1], df_h1['Recipient_Size'], df_h1['np_01'],
                     color='blue', marker='o', label='h=1', s=0.5)
    plot_spline_safe(axes[1], df_h10['Recipient_Size'], df_h10['np_01'],
                     color='green', marker='^', label='h=10', s=0.5)
    plot_spline_safe(axes[1], df_h20['Recipient_Size'], df_h20['np_01'],
                     color='red', marker='s', label='h=20', s=0.5)
    axes[1].set_xlabel('Sample size (N)')
    axes[1].set_ylabel('NP fraction')
    axes[1].set_title('Non-predictable points (NP)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# -------------------------------------------------------------------
# EXPERIMENT 2: Data replacement (2x3 grid of subplots)
# -------------------------------------------------------------------
def plot_exp2_mix(df_h1, df_h10, df_h20, save_path='exp2_mix.png'):
    donors = sorted([r for r in df_h1['Donor_R_Val'].unique() if pd.notna(r)])
    color_map = {27.9: 'blue', 28.0001: 'purple', 28.01: 'green', 28.1: 'orange'}
    markers = ['o', 's', '^', 'd']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Experiment 2: Replacing recipient data with donor data (Total N = 10,000)', fontsize=18)

    # First row: RMSE for h=1, h=10, h=20
    for col, (df, h_label) in enumerate([(df_h1, 'h=1'), (df_h10, 'h=10'), (df_h20, 'h=20')]):
        ax = axes[0, col]
        baseline = df[df['Recipient_Size'] == 10000].iloc[0]
        for idx, r in enumerate(donors):
            subset = df[df['Donor_R_Val'] == r].sort_values('Recipient_Size')
            if subset.empty:
                continue
            x_raw = np.array(subset['Recipient_Size'].tolist() + [10000])
            y_raw = np.array(subset['rmse_01'].tolist() + [baseline['rmse_01']])
            color = color_map.get(r, f'C{idx}')
            marker = markers[idx % len(markers)]
            plot_spline_safe(ax, x_raw, y_raw, color=color, marker=marker,
                             label=f'Donor r={r}', s=0.0, linewidth=2,
                             highlight_zero=True, zero_marker='D', zero_size=80)
        ax.set_title(f'RMSE ({h_label})')
        ax.set_xlabel('Number of recipient points')
        ax.set_ylabel('RMSE')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')

    # Second row: NP for h=1, h=10, h=20
    for col, (df, h_label) in enumerate([(df_h1, 'h=1'), (df_h10, 'h=10'), (df_h20, 'h=20')]):
        ax = axes[1, col]
        baseline = df[df['Recipient_Size'] == 10000].iloc[0]
        for idx, r in enumerate(donors):
            subset = df[df['Donor_R_Val'] == r].sort_values('Recipient_Size')
            if subset.empty:
                continue
            x_raw = np.array(subset['Recipient_Size'].tolist() + [10000])
            y_raw = np.array(subset['np_01'].tolist() + [baseline['np_01']])
            color = color_map.get(r, f'C{idx}')
            marker = markers[idx % len(markers)]
            plot_spline_safe(ax, x_raw, y_raw, color=color, marker=marker,
                             label=f'Donor r={r}', s=0.0, linewidth=2,
                             highlight_zero=True, zero_marker='D', zero_size=80)
        ax.set_title(f'Non-predictable Points Ratio ({h_label})')
        ax.set_xlabel('Number of recipient points')
        ax.set_ylabel('NP ratio')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# -------------------------------------------------------------------
# EXPERIMENT 3: Hybrid expansion (2x2 subplots, only h=1 and h=20)
# -------------------------------------------------------------------
def plot_exp3_hybrid(df_h1, df_h20, save_path='exp3_hybrid.png'):
    target_donors = [27.9, 28.0001, 28.01, 28.1, 31.0, 35.0, 38.0]
    normal_colors = {27.9: 'blue', 28.0001: 'green', 28.1: 'orange'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment 3: Expanding training set with donor observations', fontsize=18)

    for i, (df_full, h_label) in enumerate([(df_h1, 'h=1'), (df_h20, 'h=20')]):
        exp1 = df_full[df_full['Experiment'] == 'EXP1_CLEAR'].sort_values('Recipient_Size')
        exp3 = df_full[df_full['Experiment'] == 'EXP3_HYBRID'].copy()
        exp3['Total_Size'] = exp3['Recipient_Size'] + exp3['Donor_Size']

        ax_rmse = axes[i, 0]
        ax_np = axes[i, 1]

        if not exp1.empty:
            plot_spline_safe(ax_rmse, exp1['Recipient_Size'], exp1['rmse_01'],
                             color='black', marker='', label='Ideal (Clear Data)', s=0.5,
                             linewidth=2, linestyle='--')
            plot_spline_safe(ax_np, exp1['Recipient_Size'], exp1['np_01'],
                             color='black', marker='', label='Ideal (Clear Data)', s=0.5,
                             linewidth=2, linestyle='--')

        added_gt30_label = False
        added_28_01_label = False

        for r in target_donors:
            subset = exp3[np.isclose(exp3['Donor_R_Val'], r)].sort_values('Total_Size')
            if subset.empty:
                continue

            if r > 30.0:
                color = 'gray'
                lw = 1.5
                ls = ':'
                marker = 'x'
                label = 'Donor r > 30' if not added_gt30_label else ""
                added_gt30_label = True
            elif np.isclose(r, 28.01):
                color = 'red'
                lw = 3
                ls = '-'
                marker = 'o'
                label = 'Donor r = 28.01'
                added_28_01_label = True
            else:
                color = normal_colors.get(r, 'purple')
                lw = 2
                ls = '-'
                marker = 'o'
                label = f'Donor r={r}'

            plot_spline_safe(ax_rmse, subset['Total_Size'], subset['rmse_01'],
                             color=color, marker=marker, label=label,
                             s=2.0 if h_label == 'h=20' else 0.5,
                             linewidth=lw, linestyle=ls)

            # Исправлено: ax_rmse -> ax_np, rmse_01 -> np_01
            plot_spline_safe(ax_np, subset['Total_Size'], subset['np_01'],
                             color=color, marker=marker, label=label,
                             s=2.0 if h_label == 'h=20' else 0.5,
                             linewidth=lw, linestyle=ls)

        ax_rmse.set_title(f'RMSE Growth ({h_label})')
        ax_rmse.set_xlabel('Total sample size')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.grid(True, alpha=0.2)
        ax_rmse.legend(fontsize='small', loc='upper right')

        ax_np.set_title(f'NP % Growth ({h_label})')
        ax_np.set_xlabel('Total sample size')
        ax_np.set_ylabel('NP fraction')
        ax_np.grid(True, alpha=0.2)
        ax_np.legend(fontsize='small', loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Extract Experiment 1 data from loaded files and embedded table
    exp1_h1 = df_h1[df_h1['Experiment'] == 'EXP1_CLEAR'].sort_values('Recipient_Size')
    exp1_h20 = df_h20[df_h20['Experiment'] == 'EXP1_CLEAR'].sort_values('Recipient_Size')
    exp1_h10 = df_h10_exp1  # already ready DataFrame

    plot_exp1_clear(exp1_h1, exp1_h10, exp1_h20, 'exp1_clear.png')

    # Extract Experiment 2 data
    exp2_h1 = df_h1[df_h1['Experiment'] == 'EXP2_MIX'].copy()
    exp2_h1['Donor_R_Val'] = pd.to_numeric(exp2_h1['Donor_R'], errors='coerce')
    exp2_h20 = df_h20[df_h20['Experiment'] == 'EXP2_MIX'].copy()
    exp2_h20['Donor_R_Val'] = pd.to_numeric(exp2_h20['Donor_R'], errors='coerce')
    exp2_h10 = df_h10_exp2

    plot_exp2_mix(exp2_h1, exp2_h10, exp2_h20, 'exp2_mix.png')

    # Experiment 3 uses full data from files (already loaded)
    plot_exp3_hybrid(df_h1, df_h20, 'exp3_hybrid.png')

    print("All figures saved successfully.")