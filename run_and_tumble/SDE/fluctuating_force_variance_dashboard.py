import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
N = 8000
L = 60.0
dt = 0.05
steps_per_frame = 15  # Accelerates statistical convergence per visual update

# Physical constants
sigma = 0.5
mu_active = 5.0
D_bath = 1.0

# Binning setup
n_bins = 80
bin_edges = np.linspace(-L/2, L/2, n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Isolate the right side for the log-log tail (X > 0.5)
tail_mask = bin_centers > 0.5
tail_x = bin_centers[tail_mask]

# --- State Variables ---
# Initialize bath uniformly
x = np.random.uniform(-L/2, L/2, N)

# Statistical accumulators
sum_rho = np.zeros(n_bins)
sum_rho2 = np.zeros(n_bins)
samples = 0

# --- Dashboard Setup ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title('Fluctuation-Induced Variance Scaling')

# Top Chart: Linear Total Variance
line_total, = ax1.plot(bin_centers, np.zeros(n_bins), color='darkblue', lw=2)
ax1.set_title('Total Density Variance (Linear Scale)')
ax1.set_xlim(-L/2, L/2)
ax1.set_ylim(0, 10) # Will auto-scale
ax1.set_ylabel('Variance')
ax1.grid(True, linestyle='--', alpha=0.6)

# Bottom Chart: Log-Log Active Variance Tail
# Initialize with a tiny positive value instead of zeros to prevent log(0) warnings
line_active, = ax2.plot(tail_x, np.ones(len(tail_x)) * 1e-3, color='red', lw=2, label='Simulated Active Variance')
line_ref, = ax2.plot(tail_x, np.ones(len(tail_x)) * 1e-3, 'k--', lw=2, alpha=0.7, label='Theoretical 1/x² Scaling')
ax2.set_title('Active Variance Tail (Log-Log Scale)')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(min(tail_x), max(tail_x))
ax2.set_ylim(1e-3, 10) # Will auto-scale
ax2.set_xlabel('Distance from Origin (x)')
ax2.set_ylabel('Active Variance')
ax2.legend()
ax2.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()

def update(frame):
    global x, sum_rho, sum_rho2, samples
    
    # 1. Execute multiple SDE steps to build statistics faster
    for _ in range(steps_per_frame):
        # Shared active noise increment for the localized force
        dW_active = np.random.normal(0, np.sqrt(dt))
        
        # Spatial profile evaluation
        f_x = np.exp(-0.5 * (x / sigma)**2)
        
        # Independent thermal noise for all particles
        dW_thermal = np.random.normal(0, np.sqrt(dt), N)
        
        # Euler-Maruyama Ito update
        x += mu_active * f_x * dW_active + np.sqrt(2.0 * D_bath) * dW_thermal
        
        # Periodic boundary wrapping
        x = (x + L/2) % L - L/2
        
    # 2. Accumulate Statistics
    counts, _ = np.histogram(x, bins=bin_edges)
    sum_rho += counts
    sum_rho2 += counts**2
    samples += 1
    
    # Calculate running variance
    var_total = (sum_rho2 / samples) - (sum_rho / samples)**2
    
    # 3. Dynamic Thermal Offset Subtraction
    # Average the variance at the extreme left and right edges
    thermal_offset = np.mean(np.concatenate((var_total[:5], var_total[-5:])))
    
    # Active variance (clamped to a tiny value to prevent log(0) errors)
    var_active = np.maximum(var_total - thermal_offset, 1e-5)
    
    # 4. Update Visuals
    # Update Top Chart
    line_total.set_ydata(var_total)
    ax1.set_ylim(0, max(var_total) * 1.2 + 1)
    
    # Update Bottom Chart
    tail_active = var_active[tail_mask]
    line_active.set_ydata(tail_active)
    
    # Anchor the reference line to the first point of the active tail for visual alignment
    if tail_active[0] > 1e-4:
        anchor_constant = tail_active[0] * (tail_x[0]**2)
        line_ref.set_ydata(anchor_constant / (tail_x**2))
    
    # Auto-scale log-Y axis gracefully
    max_active = max(tail_active)
    if max_active > 1e-3:
        ax2.set_ylim(1e-3, max_active * 2.0)
        
    return line_total, line_active, line_ref

# Run the animation loop
ani = animation.FuncAnimation(fig, update, frames=2000, interval=30, blit=False)
plt.show()
