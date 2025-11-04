import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from statsmodels.tsa.stattools import acf, pacf

sigma, beta = 10, 8/3
main_rho = 28.0

# Создаем расширенный набор значений ρ
rho_base = np.concatenate([
    np.arange(27.8, 27.9, 0.02),
    np.arange(27.9, 28.1, 0.005),
    np.arange(28.1, 28.2, 0.02),np.array([28.0])
])
rho_values_extended = np.unique(np.concatenate([rho_base, [28,28.0001, 27.99, 27.9, 28.01,28.1,30,38]]))

# Для корреляции/ковариации исключаем эталонное значение
rho_values_for_comparison = rho_values_extended
highlight_rhos = [28,28.0001, 27.99, 27.9, 28.01,28.1,30,38]

num_points = 20000
warmup = 0
grid_points = 1000
x_range = [-20, 20]

def lorenz_system(t, xyz, rho):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

print("Generating trajectories...")
trajectories_dict = {}
for rho in rho_values_extended:
    t_span = (0, (num_points + warmup) * 0.01)
    t_eval = np.arange(0, (num_points + warmup) * 0.01, 0.01)
    x0 = [1.0, 1.0, 1.0]
    sol = solve_ivp(lorenz_system, t_span, x0, args=(rho,), t_eval=t_eval, method='RK45')
    trajectory = sol.y.T[warmup:]
    x_trajectory = trajectory[:, 0]
    trajectories_dict[rho] = x_trajectory

print("Computing invariant measures with KDE...")
x_grid = np.linspace(x_range[0], x_range[1], grid_points)
measures_dict = {}
for rho in rho_values_extended:
    kde = gaussian_kde(trajectories_dict[rho])
    density = kde(x_grid)
    density /= np.sum(density) * (x_grid[1] - x_grid[0])
    measures_dict[rho] = density

main_measure = measures_dict[28.0]

print("Computing correlations and covariances...")
correlations = []
covariances = []

for rho in rho_values_for_comparison:
    measure = measures_dict[rho]
    covariance = np.cov([main_measure, measure])[0, 1]
    correlation = np.corrcoef([main_measure, measure])[0, 1]
    covariances.append(covariance)
    correlations.append(correlation)

covariances = np.array(covariances)
correlations = np.array(correlations)

print("Creating correlation and covariance plots...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
mask_highlight = np.isin(rho_values_for_comparison, highlight_rhos)
mask_other = ~mask_highlight

plt.plot(rho_values_for_comparison[mask_other], covariances[mask_other], 'o',
         markersize=4, alpha=0.6, color='gray', label='Other ρ values')
plt.plot(rho_values_for_comparison[mask_highlight], covariances[mask_highlight], 'ro',
         markersize=8, label='Highlighted ρ values')
plt.axvline(x=28.0, color='k', linestyle='--', alpha=0.5, label='ρ=28.0 (reference)')
plt.xlabel('ρ parameter')
plt.ylabel('Covariance with ρ=28.0 measure')
plt.title('Covariance with Reference Measure (ρ=28.0)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(rho_values_for_comparison[mask_other], correlations[mask_other], 'o',
         markersize=4, alpha=0.6, color='gray', label='Other ρ values')
plt.plot(rho_values_for_comparison[mask_highlight], correlations[mask_highlight], 'ro',
         markersize=8, label='Highlighted ρ values')
plt.axvline(x=28.0, color='k', linestyle='--', alpha=0.5, label='ρ=28.0 (reference)')
plt.xlabel('ρ parameter')
plt.ylabel('Correlation with ρ=28.0 measure')
plt.title('Correlation with Reference Measure (ρ=28.0)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_covariance_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Creating ACF and PACF comparison plots...")
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
for rho in [28.0] + highlight_rhos:
    acf_vals = acf(trajectories_dict[rho], nlags=200, fft=True)
    plt.plot(acf_vals, label=f'ρ={rho}', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('ACF Comparison')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for rho in [28.0] + highlight_rhos:
    pacf_vals = pacf(trajectories_dict[rho], nlags=30, method='ywm')  # Уменьшил до 30 лагов, метод Yule-Walker с поправкой
    plt.plot(pacf_vals, label=f'ρ={rho}', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
conf_int = 1.96 / np.sqrt(len(trajectories_dict[28.0]))
plt.axhline(y=conf_int, color='r', linestyle='--', alpha=0.7, label='95% CI')
plt.axhline(y=-conf_int, color='r', linestyle='--', alpha=0.7)
plt.title('PACF Comparison (30 lags)')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('acf_pacf_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Computing ACF decay times...")
acf_decay_times = []
for rho in [28.0] + highlight_rhos:
    acf_vals = acf(trajectories_dict[rho], nlags=1000, fft=True)
    decay_time = np.argmax(acf_vals < 0.1)
    acf_decay_times.append(decay_time)

plt.figure(figsize=(8, 6))
plt.bar(range(len([28.0] + highlight_rhos)), acf_decay_times)
plt.xticks(range(len([28.0] + highlight_rhos)), [f'ρ={r}' for r in [28.0] + highlight_rhos], rotation=45)
plt.ylabel('ACF Decay Time (lag)')
plt.title('ACF Decay Time to 0.1')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('acf_decay_times.png', dpi=300, bbox_inches='tight')
plt.close()

print("Creating KDE measures comparison plot...")
plt.figure(figsize=(12, 6))
for rho in [28.0] + highlight_rhos:
    plt.plot(x_grid, measures_dict[rho], label=f'ρ={rho}', linewidth=2)
plt.title('Invariant Measures (KDE)')
plt.xlabel('x coordinate')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kde_measures_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSummary for comparison with ρ=28.0:")
print("ρ\t\tCovariance\t\tCorrelation\t\tACF Decay")
for rho in highlight_rhos:
    idx = np.where(rho_values_for_comparison == rho)[0][0]
    cov = covariances[idx]
    corr = correlations[idx]
    decay_idx = ([28.0] + highlight_rhos).index(rho)
    decay = acf_decay_times[decay_idx]
    print(f"{rho}\t\t{cov:.8f}\t{corr:.6f}\t\t{decay}")

print(f"\nReference measure: ρ=28.0")
print("All plots have been saved as PNG files.")