import numpy as np
from scipy.linalg import qr

# Параметры Лоренца
sigma = 10.0
beta = 8.0 / 3.0


def lorenz_jacobian(state, r):
    """Якобиан системы Лоренца."""
    x, y, z = state
    J = np.array([
        [-sigma, sigma, 0.0],
        [r - z, -1.0, -x],
        [y, x, -beta]
    ])
    return J


def lorenz_deriv(state, r):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (r - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])


def compute_lyapunov_spectrum(r, initial, dt=0.01, t_max=1000, t_trans=200, N_orth=100):
    """
    Вычисляет спектр из 3 ляпуновских показателей для системы Лоренца.

    Параметры:
        r: параметр системы
        initial: начальное состояние [x, y, z]
        dt: шаг интегрирования
        t_max: общее время интегрирования
        t_trans: время переходного процесса (не учитывается)
        N_orth: интервал ортонормировки (в шагах)

    Возвращает:
        lyapunov_exponents: массив из 3 значений (λ1 ≥ λ2 ≥ λ3)
    """
    n_steps = int(t_max / dt)
    n_trans = int(t_trans / dt)
    n_total = n_steps - n_trans

    # Начальное состояние
    state = np.array(initial, dtype=np.float64)

    # Инициализация вариаций: единичная матрица (3 ортогональных возмущения)
    Y = np.eye(3)  # столбцы — векторы вариаций

    # Накопители для логарифмов роста
    cum_log = np.zeros(3)

    for i in range(n_steps):
        # Интегрируем основную траекторию (RK4)
        k1 = lorenz_deriv(state, r)
        k2 = lorenz_deriv(state + dt * k1 / 2, r)
        k3 = lorenz_deriv(state + dt * k2 / 2, r)
        k4 = lorenz_deriv(state + dt * k3, r)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Интегрируем вариации: dY/dt = J * Y
        J = lorenz_jacobian(state, r)
        k1Y = J @ Y
        k2Y = (lorenz_jacobian(state + dt * k1 / 2, r)) @ (Y + dt * k1Y / 2)
        k3Y = (lorenz_jacobian(state + dt * k2 / 2, r)) @ (Y + dt * k2Y / 2)
        k4Y = (lorenz_jacobian(state + dt * k3, r)) @ (Y + dt * k3Y)
        Y = Y + dt * (k1Y + 2 * k2Y + 2 * k3Y + k4Y) / 6

        # Ортонормировка каждые N_orth шагов
        if (i + 1) % N_orth == 0:
            Q, R = qr(Y)
            Y = Q  # сохраняем ортонормированные векторы
            # Диагональ R содержит коэффициенты роста
            diag_R = np.abs(np.diag(R))
            cum_log += np.log(diag_R)

    # Нормируем по времени (только после переходного процесса!)
    n_orth_steps = (n_total) // N_orth
    if n_orth_steps == 0:
        n_orth_steps = 1
    lyap = cum_log / (n_orth_steps * N_orth * dt)

    # Сортируем по убыванию
    return np.sort(lyap)[::-1]


# === Запуск для ваших значений r ===
initial = [1.0, 1.0, 1.0]
r_values = [28, 28.0001, 27.99, 27.9, 28.1,30,32,38,28.01]
with open("lyapunov_spectrum.txt", "w") as f:
    f.write("Вычисление спектров Ляпунова...\n")
    f.write(f"{'r':>8} | {'λ1':>8}\n")
    f.write("-" * 45 + "\n")

    for r in r_values:
        le = compute_lyapunov_spectrum(
            r, initial,
            dt=0.01,
            t_max=1000,
            t_trans=200,
            N_orth=100
        )
        f.write(f"{r:8.4f} | {le[0]:8.4f}\n")