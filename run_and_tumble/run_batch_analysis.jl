# run_batch_analysis.jl
# Simplified batch analysis script for slopes vs system size

include("density_slope_analyzer.jl")
using Plots
using Printf

# Directory containing the files to analyze
analysis_dir = "dummy_states/for_current_analysis"

println("DENSITY SLOPE SCALING ANALYSIS")
println("="^80)
println("Analyzing files in: $analysis_dir")

# Get all 1D .jld2 files
all_files = readdir(analysis_dir)
jld2_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), all_files)

println("Found $(length(jld2_files)) 1D files")

# Initialize result arrays
system_sizes = Int[]
combined_slopes = Float64[]
left_slopes = Float64[]
right_slopes = Float64[]
gamma_values = Float64[]

# Analyze each file
for (i, file) in enumerate(jld2_files)
    full_path = joinpath(analysis_dir, file)
    
    try
        println("[$i/$(length(jld2_files))] Analyzing: $(basename(file))")
        
        # Analyze the file
        results = analyze_slopes_from_file(full_path)
        
        # Extract data
        L = results["system_length"]
        combined_slope = results["combined_slope"]
        left_slope = results["left_slope"]
        right_slope = results["right_slope"]
        gamma = results["gamma"]
        
        # Store results
        push!(system_sizes, L)
        push!(combined_slopes, combined_slope)
        push!(left_slopes, left_slope)
        push!(right_slopes, right_slope)
        push!(gamma_values, gamma)
        
        println("  L=$L, γ=$gamma, Combined slope=$(round(combined_slope, sigdigits=5))")
        
    catch e
        println("  ERROR: Failed to analyze $file: $e")
    end
end

println("\nAnalysis complete! Processed $(length(system_sizes)) files successfully.")

# Sort by system size
sorted_indices = sortperm(system_sizes)
sorted_sizes = system_sizes[sorted_indices]
sorted_combined = combined_slopes[sorted_indices]
sorted_left = left_slopes[sorted_indices]
sorted_right = right_slopes[sorted_indices]
sorted_gamma = gamma_values[sorted_indices]

# Print summary table
println("\n" * "="^100)
println("SUMMARY TABLE: SLOPES VS SYSTEM SIZE")
println("="^100)
println(@sprintf("%-8s %-10s %-15s %-15s %-15s %-10s", 
                 "L", "γ", "Combined", "Left", "Right", "Asymmetry"))
println(@sprintf("%-8s %-10s %-15s %-15s %-15s %-10s", 
                 "", "", "Slope", "Slope", "Slope", "|L-R|"))
println("-"^100)

for i in eachindex(sorted_sizes)
    L = sorted_sizes[i]
    gamma = sorted_gamma[i]
    combined = sorted_combined[i]
    left = sorted_left[i]
    right = sorted_right[i]
    asymmetry = abs(left - right)
    
    println(@sprintf("%-8d %-10.2f %-15.6e %-15.6e %-15.6e %-10.6e", 
                    L, gamma, combined, left, right, asymmetry))
end
println("="^100)

# Create the plot
println("\nCreating slope vs system size plot...")

p = plot(xlabel="System Size (L)", 
         ylabel="Slope", 
         title="Density Slopes vs System Size",
         legend=:topright,
         size=(800, 600),
         dpi=300)

# Plot combined periodic fit slopes
scatter!(p, sorted_sizes, sorted_combined,
         label="Combined Periodic Fit",
         color=:green,
         markersize=8,
         markerstrokewidth=2,
         markerstrokecolor=:darkgreen)

# Plot left and right slopes for comparison
scatter!(p, sorted_sizes, sorted_left,
         label="Left Region",
         color=:red,
         markersize=6,
         markerstrokewidth=1,
         markerstrokecolor=:darkred,
         alpha=0.7)

scatter!(p, sorted_sizes, sorted_right,
         label="Right Region", 
         color=:blue,
         markersize=6,
         markerstrokewidth=1,
         markerstrokecolor=:darkblue,
         alpha=0.7)

# Add connecting lines
plot!(p, sorted_sizes, sorted_combined,
      color=:green,
      linewidth=2,
      linestyle=:solid,
      alpha=0.6,
      label="")

plot!(p, sorted_sizes, sorted_left,
      color=:red,
      linewidth=1,
      linestyle=:dash,
      alpha=0.5,
      label="")

plot!(p, sorted_sizes, sorted_right,
      color=:blue,
      linewidth=1,
      linestyle=:dash,
      alpha=0.5,
      label="")

# Add grid
plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)

# Save plot
filename = "slope_vs_system_size.png"
full_path = abspath(filename)
savefig(p, filename)
println("Plot saved as: $full_path")

# Additional analysis
println("\nADDITIONAL ANALYSIS:")
println("-"^50)

println("Combined periodic fit slopes:")
for i in eachindex(sorted_sizes)
    L = sorted_sizes[i]
    slope = sorted_combined[i]
    println("  L=$L: slope = $(round(slope, sigdigits=6))")
end

# Check if slopes scale with 1/L
println("\nTesting 1/L scaling:")
for i in eachindex(sorted_sizes)
    L = sorted_sizes[i]
    slope = sorted_combined[i]
    scaled_slope = slope * L
    println("  L=$L: slope × L = $(round(scaled_slope, sigdigits=6))")
end

println("\nAnalysis completed successfully!")
