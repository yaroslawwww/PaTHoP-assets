import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde

# Параметры системы Лоренца
sigma, beta = 10, 8/3
main_rho = 28.0

# Только интересующие нас значения rho
highlight_rhos = [28.0, 28.0001, 27.99, 27.9, 28.01, 28.1, 30, 38]
rho_values_extended = np.array(highlight_rhos)

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
variances = {}  # <-- новое: словарь для дисперсий

dx = x_grid[1] - x_grid[0]  # шаг сетки

for rho in rho_values_extended:
    kde = gaussian_kde(trajectories_dict[rho])
    density = kde(x_grid)
    # Нормировка как плотности вероятности: ∫ p(x) dx ≈ 1
    density /= np.sum(density) * dx
    measures_dict[rho] = density

    # Расчёт среднего и дисперсии
    mean = np.sum(x_grid * density) * dx
    mean_sq = np.sum(x_grid**2 * density) * dx
    variance = mean_sq - mean**2
    variances[rho] = variance

main_measure = measures_dict[28.0]
main_variance = variances[28.0]

print("Computing correlations, covariances, and variances...")
results = []
for rho in highlight_rhos:
    measure = measures_dict[rho]
    covariance = np.cov([main_measure, measure])[0, 1]
    correlation = np.corrcoef([main_measure, measure])[0, 1]
    var_rho = variances[rho]
    results.append((rho, covariance, correlation, var_rho))

# Сохранение в файл
with open("cov_corr_var_results.txt", "w") as f:
    f.write("rho\t\tCovariance\t\tCorrelation\t\tVariance\n")
    f.write("-" * 70 + "\n")
    for rho, cov, corr, var in results:
        f.write(f"{rho:.6f}\t{cov:.10f}\t{corr:.8f}\t{var:.6f}\n")

print("Results saved to 'cov_corr_var_results.txt'")