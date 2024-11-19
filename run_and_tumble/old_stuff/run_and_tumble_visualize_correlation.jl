using Plots
using JLD2

# Load correlation data
spatial_correlation = nothing
temporal_correlation = nothing
@load "correlation_data.jld2" spatial_correlation temporal_correlation

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

# Generate plots
plot_spatial_correlation(spatial_correlation)
savefig("spatial_correlation.png")

gui()  # Show plot in a window

plot_temporal_correlation(temporal_correlation)
savefig("temporal_correlation.png")

gui()  # Show plot in a window