import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

# --- Simulation Parameters & Performance ---
N = 60000          
L = 40.0           
dt = 0.05

# SPEEDUP: Run 200 physics steps per visual frame. 
# This prevents Matplotlib from choking the CPU and builds statistics extremely fast.
steps_per_frame = 200 

# Warmup: Let the active signal fully diffuse across L=40 before collecting data
warmup_frames = 50   # 50 frames * 200 steps = 10,000 SDE steps (Plenty for diffusion)

# Physical constants
sigma = 0.5
mu_active = 6.0
D_bath = 1.0

# --- Fast 2D Binning Setup ---
n_bins = 60
dx = L / n_bins
dy = L / n_bins
bin_centers = np.linspace(-L/2 + dx/2, L/2 - dx/2, n_bins)
X, Y = np.meshgrid(bin_centers, bin_centers, indexing='ij')

R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)

# Radial bins for calculation
r_edges = np.logspace(np.log10(0.8), np.log10(L/2.2), 20)
r_centers = (r_edges[:-1] + r_edges[1:]) / 2

# Restriction: Only PLOT the tail from r >= 1.4
plot_mask = r_centers >= 1.4
plot_r = r_centers[plot_mask]

ring_mask = (R > 1.5) & (R < 3.5)
ring_theta = Theta[ring_mask]

# --- State Variables ---
x = np.random.uniform(-L/2, L/2, N)
y = np.random.uniform(-L/2, L/2, N)

sum_rho = np.zeros(n_bins * n_bins)
sum_rho2 = np.zeros(n_bins * n_bins)
samples = 0
frame_count = 0

# --- Dashboard Setup ---
fig = plt.figure(figsize=(15, 8))
fig.canvas.manager.set_window_title('Accelerated 2D Fluctuation-Induced Variance')
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.2])

# 1. 2D Heatmap
ax_heat = fig.add_subplot(gs[0, 0])
im = ax_heat.imshow(np.zeros((n_bins, n_bins)), extent=[-L/2, L/2, -L/2, L/2], 
                    origin='lower', cmap='inferno', vmin=0, vmax=2)
ax_heat.set_title('Active Variance Heatmap')
ax_heat.set_ylabel('Y')
plt.colorbar(im, ax=ax_heat, label='Variance')

status_text = ax_heat.text(0.05, 0.95, 'Status: Initializing...', transform=ax_heat.transAxes, 
                           fontsize=12, color='white', weight='bold', verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

# 2. Angular Dependence
ax_angle = fig.add_subplot(gs[1, 0])
scatter_angle = ax_angle.scatter(ring_theta, np.zeros_like(ring_theta), c='teal', alpha=0.6, s=15)
ax_angle.set_title('Angular Dependence (1.5 < r < 3.5)')
ax_angle.set_xlim(-np.pi, np.pi)
ax_angle.set_ylim(0, 1) 
ax_angle.set_xlabel('Angle θ (radians)')
ax_angle.set_ylabel('Active Variance')
ax_angle.grid(True, linestyle='--', alpha=0.5)

# 3. Radial Scaling (Log-Log)
ax_radial = fig.add_subplot(gs[:, 1])

line_radial, = ax_radial.plot(plot_r, np.ones_like(plot_r)*1e-3, 'o-', color='red', lw=2, label='Simulated Active Variance (r ≥ 1.4)')
line_ref, = ax_radial.plot(plot_r, np.ones_like(plot_r)*1e-3, 'k--', lw=2, alpha=0.7, label='Theoretical 1/r⁴ Scaling')

offset_text = ax_radial.text(0.05, 0.05, 'Subtracted Offset: 0.0', transform=ax_radial.transAxes, 
                             fontsize=11, verticalalignment='bottom', 
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax_radial.set_title('Radial Variance Tail (Log-Log Scale)')
ax_radial.set_xscale('log')
ax_radial.set_yscale('log')
ax_radial.set_xlim(1.3, max(plot_r) * 1.1)
ax_radial.set_xlabel('Distance from Origin (r)')
ax_radial.set_ylabel('Active Variance')
ax_radial.legend(loc='upper right')
ax_radial.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()

def update(frame):
    global x, y, sum_rho, sum_rho2, samples, frame_count
    frame_count += 1
    
    # 1. Fast SDE Integration (Runs purely in background arrays)
    for _ in range(steps_per_frame):
        dW_active_x = np.random.normal(0, np.sqrt(dt))
        dW_active_y = np.random.normal(0, np.sqrt(dt))
        
        r2 = x**2 + y**2
        f = np.exp(-0.5 * r2 / sigma**2)
        
        dW_therm_x = np.random.normal(0, np.sqrt(dt), N)
        dW_therm_y = np.random.normal(0, np.sqrt(dt), N)
        
        x += mu_active * f * dW_active_x + np.sqrt(2.0 * D_bath) * dW_therm_x
        y += mu_active * f * dW_active_y + np.sqrt(2.0 * D_bath) * dW_therm_y
        
        x = (x + L/2) % L - L/2
        y = (y + L/2) % L - L/2
        
    # 2. Fast Binning
    ix = np.clip(np.floor((x + L/2) / dx).astype(int), 0, n_bins - 1)
    iy = np.clip(np.floor((y + L/2) / dy).astype(int), 0, n_bins - 1)
    flat_idx = ix * n_bins + iy
    counts = np.bincount(flat_idx, minlength=n_bins*n_bins)
    
    # Accumulate
    sum_rho += counts
    sum_rho2 += counts**2
    samples += 1
    
    # --- WARMUP FLUSH LOGIC ---
    if frame_count < warmup_frames:
        status_text.set_text(f"Status: Warmup Transient ({frame_count}/{warmup_frames})")
    elif frame_count == warmup_frames:
        status_text.set_text("Status: Steady State - FLUSHING STATISTICS")
        sum_rho.fill(0)
        sum_rho2.fill(0)
        samples = 0
        return im, scatter_angle, line_radial, line_ref, offset_text, status_text
    else:
        status_text.set_text(f"Status: Steady State (Clean Samples: {samples})")

    # 3. Variance Calculation
    if samples > 0:
        var_total_flat = (sum_rho2 / samples) - (sum_rho / samples)**2
    else:
        var_total_flat = np.zeros(n_bins * n_bins)
        
    flat_R = R.flatten()
    
    radial_var_total = np.zeros_like(r_centers)
    for i in range(len(r_centers)):
        mask = (flat_R >= r_edges[i]) & (flat_R < r_edges[i+1])
        if np.any(mask):
            radial_var_total[i] = np.mean(var_total_flat[mask])
        else:
            radial_var_total[i] = 1e-5
            
    # Offset strictly at max boundary radius
    thermal_offset = radial_var_total[-1]
    
    # Active Variance Arrays
    radial_var_active = np.maximum(radial_var_total - thermal_offset, 1e-6)
    var_active_flat = np.maximum(var_total_flat - thermal_offset, 1e-6)
    var_active_2d = var_active_flat.reshape(n_bins, n_bins)
    
    # 4. Visual Updates
    im.set_data(var_active_2d.T) 
    vmax = np.percentile(var_active_flat, 99.5) if samples > 0 else 0.1
    im.set_clim(vmin=0, vmax=max(vmax, 0.1))
    
    ring_var = var_active_2d[ring_mask]
    scatter_angle.set_offsets(np.c_[ring_theta, ring_var])
    if len(ring_var) > 0:
        ax_angle.set_ylim(0, max(ring_var) * 1.5 + 1e-3)
        
    # --- PLOTTING STRICTLY FOR r >= 1.4 ---
    plot_var_active = radial_var_active[plot_mask]
    
    # Update simulated line
    line_radial.set_data(plot_r, plot_var_active)
    
    # Dynamic Anchor for the theoretical 1/r^4 line (Anchors to the peak of the restricted domain)
    if len(plot_var_active) > 0:
        peak_idx = np.argmax(plot_var_active)
        anchor_r = plot_r[peak_idx]
        anchor_var = plot_var_active[peak_idx]
        
        if anchor_var > 1e-6:
            anchor_const = anchor_var * (anchor_r**4)
            line_ref.set_data(plot_r, anchor_const / (plot_r**4))
            
        ax_radial.set_ylim(1e-6, max(plot_var_active) * 5.0 + 1e-5)
    
    offset_text.set_text(f"Subtracted Offset: {thermal_offset:.5f}\n(Measured strictly at r = {r_centers[-1]:.1f})")

    return im, scatter_angle, line_radial, line_ref, offset_text, status_text

# interval=200 pauses the UI to let the CPU chew through the steps_per_frame
ani = animation.FuncAnimation(fig, update, frames=2000, interval=200, blit=False)
plt.show()