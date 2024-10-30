using Random
using Plots
using Statistics
using ProgressMeter
using LinearAlgebra
using FFTW

mutable struct Particle
    position::Tuple{Float64, Float64}
    direction::Float64  # angle in radians
end

function initialize_particles(num_particles, box_size)
    particles = [Particle((rand() * box_size, rand() * box_size), 2 * π * rand()) for _ in 1:num_particles]
    return particles
end

function potential_field(box_size, grid_size)
    # Example potential: a simple quadratic well centered at the middle of the box
    potential = zeros(Float64, grid_size, grid_size)
    cell_size = box_size / grid_size
    for i in 1:grid_size, j in 1:grid_size
        x = (i - 0.5) * cell_size
        y = (j - 0.5) * cell_size
        potential[i, j] = 0.01 * ((x - box_size / 2)^2 + (y - box_size / 2)^2)  # Quadratic potential
    end
    return potential
end

function run_and_tumble_with_potential!(particles, box_size, speed, dt, tumble_prob, potential, grid_size, use_potential)
    cell_size = box_size / grid_size
    for p in particles
        if use_potential
            # Update direction considering the potential gradient
            x_idx = clamp(Int(floor(p.position[1] / cell_size)) + 1, 1, grid_size)
            y_idx = clamp(Int(floor(p.position[2] / cell_size)) + 1, 1, grid_size)

            # Calculate gradient of the potential (finite difference)
            grad_x = 0.0
            grad_y = 0.0
            if x_idx > 1 && x_idx < grid_size
                grad_x = (potential[y_idx, x_idx + 1] - potential[y_idx, x_idx - 1]) / (2 * cell_size)
            end
            if y_idx > 1 && y_idx < grid_size
                grad_y = (potential[y_idx + 1, x_idx] - potential[y_idx - 1, x_idx]) / (2 * cell_size)
            end

            # Adjust direction based on the potential gradient
            p.direction -= 0.1 * (grad_x * cos(p.direction) + grad_y * sin(p.direction)) * dt
        end

        # Update position based on direction and speed
        dx = speed * cos(p.direction) * dt
        dy = speed * sin(p.direction) * dt
        new_x = mod(p.position[1] + dx, box_size)
        new_y = mod(p.position[2] + dy, box_size)
        p.position = (new_x, new_y)

        # Randomly change direction (tumble)
        if rand() < tumble_prob * dt
            p.direction = 2 * π * rand()  # New random direction
        end
    end
end

function calculate_density(particles, box_size, grid_size)
    density = zeros(Float64, grid_size, grid_size)
    cell_size = box_size / grid_size
    for p in particles
        x_idx = clamp(Int(floor(p.position[1] / cell_size)) + 1, 1, grid_size)
        y_idx = clamp(Int(floor(p.position[2] / cell_size)) + 1, 1, grid_size)
        density[y_idx, x_idx] += 1  # Correct indexing to match matrix row-major order
    end
    density /= (cell_size^2)  # Normalize by cell area to get density
    return density
end

function plot_particles_and_density(particles, box_size, density, potential)
    # Plot particles
    scatter([p.position[1] for p in particles], [p.position[2] for p in particles],
        xlim=(0, box_size), ylim=(0, box_size), aspect_ratio=:equal, legend=false, markersize=2, label="Particles")
    # Overlay density as a transparent heatmap
    heatmap!(LinRange(0, box_size, size(density, 2) + 1), LinRange(0, box_size, size(density, 1) + 1), density, alpha=0.5, c=:viridis, legend=false)
    # Overlay potential as contour lines
    contour!(LinRange(0, box_size, size(potential, 2)), LinRange(0, box_size, size(potential, 1)), potential, levels=10, c=:black, linewidth=1, legend=false)
end

function calculate_spatial_correlation_fft(density)
    # Subtract mean density
    mean_density = mean(density)
    delta_density = density .- mean_density

    # Perform FFT
    fft_density = fft(delta_density)
    power_spectrum = abs2.(fft_density)

    # Inverse FFT to get the correlation
    correlation = ifft(power_spectrum)
    correlation = real(correlation) / (size(density, 1) * size(density, 2))

    return correlation
end

function calculate_temporal_correlation_fft(densities)
    num_steps = length(densities)
    # Calculate average density for each time step
    mean_density = mean([mean(d) for d in densities])
    delta_densities = [mean(d) - mean_density for d in densities]

    # Perform FFT on the time series of mean densities
    fft_densities = fft(delta_densities)
    power_spectrum = abs2.(fft_densities)

    # Inverse FFT to get the correlation
    correlation = ifft(power_spectrum)
    correlation = real(correlation) / num_steps

    return correlation
end

# Parameters
num_particles = 100
box_size = 20.0
speed = 0.1
dt = 0.1
tumble_prob = 0.1
num_steps = 100
grid_size = 20  # Grid size for density calculation
use_potential = true  # Set to true or false to enable or disable potential

# Initialize particles
particles = initialize_particles(num_particles, box_size)

# Create potential field
potential = potential_field(box_size, grid_size)

# Prepare to store densities over time
densities = Vector{Matrix{Float64}}(undef, num_steps)

# Run simulation with progress bar
progress = Progress(num_steps, 1, "Running simulation")
anim = @animate for step in 1:num_steps
    run_and_tumble_with_potential!(particles, box_size, speed, dt, tumble_prob, potential, grid_size, use_potential)
    density = calculate_density(particles, box_size, grid_size)
    densities[step] = density
    plot_particles_and_density(particles, box_size, density, potential)
    next!(progress)
end

# Save animation
gif(anim, "run_and_tumble_with_potential.gif", fps = 10)

# Save density data for later analysis
using JLD2
@save "density_data_with_potential.jld2" densities

# Calculate spatial and temporal correlations using FFT
spatial_correlation = calculate_spatial_correlation_fft(densities[end])
temporal_correlation = calculate_temporal_correlation_fft(densities)

# Save correlations for analysis
@save "correlation_data_with_potential.jld2" spatial_correlation temporal_correlation

# Plot spatial correlation
function plot_spatial_correlation(spatial_correlation)
    grid_size = size(spatial_correlation, 1)
    delta_range = collect(-div(grid_size, 2):div(grid_size, 2) - 1)
    heatmap(delta_range, delta_range, spatial_correlation, c=:viridis, xlabel="ΔX", ylabel="ΔY", title="Spatial Correlation", aspect_ratio=:equal, xlims=(-div(grid_size, 2), div(grid_size, 2) - 1), ylims=(-div(grid_size, 2), div(grid_size, 2) - 1))
end

# Plot temporal correlation
function plot_temporal_correlation(temporal_correlation)
    plot(temporal_correlation, xlabel="Time Lag", ylabel="Correlation", title="Temporal Correlation", legend=false)
end

# Generate and save correlation plots
plot_spatial_correlation(spatial_correlation)
savefig("spatial_correlation.png")

gui()  # Show plot in a window

plot_temporal_correlation(temporal_correlation)
savefig("temporal_correlation.png")

gui()  # Show plot in a window