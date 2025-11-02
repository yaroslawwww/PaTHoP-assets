import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Параметры Лоренца
sigma = 10.0
beta = 8.0 / 3.0


def lorenz_deriv(state, r):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def integrate_lorenz(r, initial, dt=0.01, t_max=500):
    n_steps = int(t_max / dt)
    state = np.array(initial, dtype=np.float64)
    traj = np.empty((n_steps, 3))
    for i in range(n_steps):
        k1 = lorenz_deriv(state, r)
        k2 = lorenz_deriv(state + dt * k1 / 2, r)
        k3 = lorenz_deriv(state + dt * k2 / 2, r)
        k4 = lorenz_deriv(state + dt * k3, r)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        traj[i] = state
    return traj


def correlation_dimension(traj, eps_range=(0.001, 10.0), n_eps=50, max_n_points=8000, min_pairs=10):
    """
    Вычисляет корреляционную размерность D2 по траектории.

    Параметры:
        traj: массив (N, 3) — траектория в фазовом пространстве
        eps_range: диапазон масштабов ε (min, max)
        n_eps: количество точек в логарифмической сетке
        max_n_points: максимальное число точек для ускорения
        min_pairs: минимальное число пар для надёжной оценки C(ε)
    """
    # Отбрасываем переходный процесс (первые 20%)
    n_trans = int(0.2 * len(traj))
    traj = traj[n_trans:]

    # Подвыборка, если слишком много точек
    if len(traj) > max_n_points:
        indices = np.random.choice(len(traj), size=max_n_points, replace=False)
        traj = traj[indices]

    N = len(traj)
    tree = cKDTree(traj)

    # Создаем логарифмическую сетку для ε
    epsilons = np.logspace(np.log10(eps_range[0]), np.log10(eps_range[1]), n_eps)

    C_eps = []
    valid_eps = []

    for eps in epsilons:
        # Подсчёт числа пар с расстоянием < eps
        count = tree.query_pairs(eps, output_type='ndarray')
        n_pairs = len(count)

        if n_pairs < min_pairs:
            continue  # пропускаем слишком малые ε

        C = 2 * n_pairs / (N * (N - 1))
        if C > 0:
            C_eps.append(C)
            valid_eps.append(eps)

    if len(C_eps) < 5:
        return None, None, None

    log_eps = np.log(valid_eps)
    log_C = np.log(C_eps)

    # Найдём линейный участок: попробуем несколько окон и выберем с max R²
    best_r2 = -np.inf
    best_slope = None
    best_window = None

    n = len(log_eps)
    window_size = max(5, n // 3)

    for i in range(n - window_size):
        x = log_eps[i:i + window_size]
        y = log_C[i:i + window_size]
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

        if r2 > best_r2:
            best_r2 = r2
            best_slope = slope
            best_window = (i, i + window_size)

    if best_slope is None:
        return None, log_eps, log_C

    D2 = best_slope  # C(ε) ~ ε^D2 → log C = D2 * log ε + const
    return D2, log_eps, log_C


# === Параметры ===
initial = [1.0, 1.0, 1.0]
r_values = [28.01]

print("Вычисление корреляционной размерности D2...")
results = {}

# Создаем отдельную фигуру для каждого r
figures = []

for r in r_values:
    print(f"  r = {r}")

    # Создаем новую фигуру для каждого r
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    traj = integrate_lorenz(r, initial, dt=0.001, t_max=2000)

    # Используем расширенный диапазон ε для лучшего охвата
    D2, log_eps, log_C = correlation_dimension(traj, eps_range=(0.001, 10.0),
                                               n_eps=60, max_n_points=15000, min_pairs=5)

    results[r] = D2
    if D2 is not None:
        print(f"    D2 ≈ {D2:.3f}")
        ax.plot(log_eps, log_C, 'o-', markersize=4, label=f'r = {r}, D2 = {D2:.3f}')
        ax.set_title(f'Корреляционный интеграл для r = {r}, D2 = {D2:.3f}')
    else:
        print("    Не удалось оценить D2")
        ax.plot(log_eps, log_C, 'o-', markersize=4, label=f'r = {r} (ошибка)')
        ax.set_title(f'Корреляционный интеграл для r = {r} (не удалось оценить D2)')

    ax.set_xlabel('log ε')
    ax.set_ylabel('log C(ε)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Настраиваем отображение для лучшей видимости всех точек
    if len(log_eps) > 0:
        x_margin = (max(log_eps) - min(log_eps)) * 0.1
        y_margin = (max(log_C) - min(log_C)) * 0.1
        ax.set_xlim(min(log_eps) - x_margin, max(log_eps) + x_margin)
        ax.set_ylim(min(log_C) - y_margin, max(log_C) + y_margin)

    plt.tight_layout()
    fig.savefig(f'lorenz_correlation_dimension_r_{r}.png', dpi=150, bbox_inches='tight')
    figures.append(fig)
    plt.close(fig)  # Закрываем фигуру чтобы освободить память

print("\nГрафики сохранены как 'lorenz_correlation_dimension_r_*.png'")

# Создаем также сводный график на одной фигуре (но с разными осями)
print("\nСоздание сводного графика...")
fig_summary, axes_summary = plt.subplots(len(r_values), 1, figsize=(10, 4 * len(r_values)))

if len(r_values) == 1:
    axes_summary = [axes_summary]

for i, (ax, r) in enumerate(zip(axes_summary, r_values)):
    print(f"  r = {r}")
    traj = integrate_lorenz(r, initial, dt=0.001, t_max=2000)
    D2, log_eps, log_C = correlation_dimension(traj, eps_range=(0.001, 10.0),
                                               n_eps=60, max_n_points=15000, min_pairs=5)

    results[r] = D2
    if D2 is not None:
        ax.plot(log_eps, log_C, 'o-', markersize=4, label=f'r = {r}, D2 = {D2:.3f}')
    else:
        ax.plot(log_eps, log_C, 'o-', markersize=4, label=f'r = {r} (ошибка)')

    ax.set_xlabel('log ε')
    ax.set_ylabel('log C(ε)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Настраиваем отображение для каждого подграфика отдельно
    if len(log_eps) > 0:
        x_margin = (max(log_eps) - min(log_eps)) * 0.1
        y_margin = (max(log_C) - min(log_C)) * 0.1
        ax.set_xlim(min(log_eps) - x_margin, max(log_eps) + x_margin)
        ax.set_ylim(min(log_C) - y_margin, max(log_C) + y_margin)

fig_summary.suptitle('Корреляционный интеграл и оценка D2 для различных r', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('lorenz_correlation_dimension_summary.png', dpi=150, bbox_inches='tight')
print("Сводный график сохранён как 'lorenz_correlation_dimension_summary.png'")

# Вывод результатов
print("\nРезультаты:")
for r in r_values:
    D2 = results[r]
    if D2 is not None:
        print(f"  r = {r:6.4f} → D2 = {D2:.3f}")
    else:
        print(f"  r = {r:6.4f} → D2 = не определён")