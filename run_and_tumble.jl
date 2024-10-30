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

function run_and_tumble!(particles, box_size, speed, dt, tumble_prob)
    for p in particles
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

function plot_particles_and_density(particles, box_size, density)
    # Plot particles
    scatter([p.position[1] for p in particles], [p.position[2] for p in particles],
        xlim=(0, box_size), ylim=(0, box_size), aspect_ratio=:equal, legend=false, markersize=2, label="Particles")
    # Overlay density as a transparent heatmap
    heatmap!(LinRange(0, box_size, size(density, 2) + 1), LinRange(0, box_size, size(density, 1) + 1), density, alpha=0.5, c=:viridis, legend=false)
end

function compute_spatial_correlation(ρ)
    F = fft(ρ)
    power_spectrum = F .* conj(F)
    corr = real(ifft(power_spectrum))
    return fftshift(corr) / (size(ρ, 1) * size(ρ, 2))
end

function compute_time_correlation(densities)
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
num_particles = 300
box_size = 20.0
speed = 0.3
dt = 0.1
tumble_prob = 0.3
num_steps = 100
grid_size = 20  # Grid size for density calculation

# Initialize particles
particles = initialize_particles(num_particles, box_size)

# Prepare to store densities over time
densities = Vector{Matrix{Float64}}(undef, num_steps)

# Run simulation with progress bar
progress = Progress(num_steps, 1, "Running simulation")
anim = @animate for step in 1:num_steps
    run_and_tumble!(particles, box_size, speed, dt, tumble_prob)
    density = calculate_density(particles, box_size, grid_size)
    densities[step] = density
    plot_particles_and_density(particles, box_size, density)
    next!(progress)
end

# Save animation
gif(anim, "run_and_tumble_with_density.gif", fps = 10)

# Save density data for later analysis
using JLD2
@save "density_data.jld2" densities

# Calculate spatial and temporal correlations using FFT
spatial_correlation = compute_spatial_correlation(densities[end])
temporal_correlation = compute_time_correlation(densities)

# Save correlations for analysis
@save "correlation_data.jld2" spatial_correlation temporal_correlation

