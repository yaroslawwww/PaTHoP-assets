import numpy as np
import matplotlib.pyplot as plt

class Lorentz:
    def __init__(self, s=10, b=8/3):
        self.s = s
        self.b = b
        self.r = None

    def _derivatives(self, x, y, z, s, r, b):
        dx_dt = s * (y - x)
        dy_dt = x * (r - z) - y
        dz_dt = x * y - b * z
        return dx_dt, dy_dt, dz_dt

    def RK4(self, x, y, z, s, r, b, dt):
        k1_x, k1_y, k1_z = self._derivatives(x, y, z, s, r, b)
        k2_x, k2_y, k2_z = self._derivatives(x + 0.5*dt*k1_x, y + 0.5*dt*k1_y, z + 0.5*dt*k1_z, s, r, b)
        k3_x, k3_y, k3_z = self._derivatives(x + 0.5*dt*k2_x, y + 0.5*dt*k2_y, z + 0.5*dt*k2_z, s, r, b)
        k4_x, k4_y, k4_z = self._derivatives(x + dt*k3_x, y + dt*k3_y, z + dt*k3_z, s, r, b)
        x_new = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_new = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        z_new = z + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
        return x_new, y_new, z_new

    def generate(self, dt, steps, r=28, initial_state=(1.0, 1.0, 1.0)):
        self.r = r
        trajectory = np.zeros((steps + 1, 3))
        trajectory[0] = initial_state
        for i in range(steps):
            x, y, z = trajectory[i]
            trajectory[i+1] = self.RK4(x, y, z, self.s, self.r, self.b, dt)
        return trajectory

# --- CHANGES START HERE ---

# Simulation parameters
dt = 0.001
steps = 50000
# Sort values for logical arrangement on the plot
r_values_to_compare = sorted([28.0001, 28.1, 27.9, 27.99, 30, 28.01])
initial_state = (1.0, 1.0, 1.0)

# Create Lorentz system instance
lorentz = Lorentz()

# Generate reference trajectory for r=28
traj_ref = lorentz.generate(dt, steps, r=28, initial_state=initial_state)

# 1. Create figure and subplot grid (3 rows, 2 columns)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18))

# Flatten 2D axes array to 1D for easy iteration
axes = axes.flatten()

# 2. Loop through r values and corresponding axes
for i, r in enumerate(r_values_to_compare):
    ax = axes[i] # Select current plotting window

    # Generate trajectory for current r
    traj_comp = lorentz.generate(dt, steps, r=r, initial_state=initial_state)

    # 3. Plot on specific axis `ax`
    # English: 'reference' instead of 'эталон'
    ax.plot(traj_ref[5000:40000:10, 0], label='r=28 (reference)', linewidth=1.5)
    ax.plot(traj_comp[5000:40000:10, 0], label=f'r={r}', alpha=0.8, linestyle='--')

    # English labels and titles
    ax.set_xlabel('Steps')
    ax.set_ylabel('x')
    ax.set_title(f'Trajectory Comparison: r=28 vs r={r}')
    ax.legend()
    ax.grid(True)

# 4. Automatically adjust layout
plt.tight_layout()

# 5. Save the figure
plt.savefig('rows_english.png', dpi=150)
plt.show()
#%%
