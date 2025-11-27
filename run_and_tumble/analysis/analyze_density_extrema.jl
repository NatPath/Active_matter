# analyze_density_extrema.jl
# Analysis script to study density peaks and minima vs system size

using JLD2
using Statistics
using Plots
using Printf
include("../modules_run_and_tumble.jl")
using .FP

"""
    extract_density_extrema(density::Vector)

Extract peak and minimum values from a 1D density profile, including second extrema.

# Arguments
- `density::Vector`: The 1D density array

# Returns
- `peak_value::Float64`: Maximum density value
- `min_value::Float64`: Minimum density value  
- `peak_position::Int`: Index of maximum density
- `min_position::Int`: Index of minimum density
- `second_peak_value::Float64`: Second highest density value
- `second_min_value::Float64`: Second lowest density value
- `second_peak_position::Int`: Index of second maximum density
- `second_min_position::Int`: Index of second minimum density
- `peak_min_ratio::Float64`: Ratio of peak to minimum
"""
function extract_density_extrema(density::Vector)
    # Find global extrema
    peak_value = maximum(density)
    min_value = minimum(density)
    peak_position = argmax(density)
    min_position = argmin(density)
    
    # Sort density values to find second extrema
    sorted_density = sort(density, rev=true)  # High to low
    second_peak_value = sorted_density[2]     # Second highest
    second_min_value = sort(density)[2]       # Second lowest (sort low to high, take 2nd)
    
    # Find positions of second extrema
    second_peak_position = findfirst(x -> x == second_peak_value && x != peak_value, density)
    if second_peak_position === nothing
        # If second peak has same value as first peak, find second occurrence
        indices = findall(x -> x == second_peak_value, density)
        second_peak_position = length(indices) > 1 ? indices[2] : indices[1]
    end
    
    second_min_position = findfirst(x -> x == second_min_value && x != min_value, density)
    if second_min_position === nothing
        # If second min has same value as first min, find second occurrence
        indices = findall(x -> x == second_min_value, density)
        second_min_position = length(indices) > 1 ? indices[2] : indices[1]
    end
    
    peak_min_ratio = peak_value / min_value
    
    return peak_value, min_value, peak_position, min_position, 
           second_peak_value, second_min_value, second_peak_position, second_min_position, 
           peak_min_ratio
end

"""
    analyze_extrema_from_file(filename::String)

Analyze density extrema from a saved state file.

# Arguments
- `filename::String`: Path to the .jld2 file

# Returns
- `results::NamedTuple`: Contains L, γ, peak_value, min_value, peak_position, min_position, peak_min_ratio
"""
function analyze_extrema_from_file(filename::String)
    try
        # Load the saved state
        println("Loading state from: $filename")
        @load filename state param potential
        
        # Extract parameters
        L = param.dims[1]
        γ = hasproperty(param, :γ) ? param.γ : (hasproperty(param, :gammap) ? param.gammap/param.N : 0.0)
        
        # Calculate density extrema
        peak_value, min_value, peak_position, min_position, second_peak_value, second_min_value, second_peak_position, second_min_position, peak_min_ratio = extract_density_extrema(state.ρ_avg)
        
        # Print results for this file
        println("  L=$L, γ=$γ")
        println("  Peak: $(peak_value) at position $(peak_position)")
        println("  2nd Peak: $(second_peak_value) at position $(second_peak_position)")
        println("  Min:  $(min_value) at position $(min_position)")
        println("  2nd Min: $(second_min_value) at position $(second_min_position)")
        println("  Peak/Min ratio: $(peak_min_ratio)")
        
        return (
            L = L,
            γ = γ,
            peak_value = peak_value,
            min_value = min_value,
            peak_position = peak_position,
            min_position = min_position,
            second_peak_value = second_peak_value,
            second_min_value = second_min_value,
            second_peak_position = second_peak_position,
            second_min_position = second_min_position,
            peak_min_ratio = peak_min_ratio,
            filename = basename(filename)
        )
        
    catch e
        println("Error processing file $filename: $e")
        return nothing
    end
end

"""
    plot_extrema_vs_size(results::Vector)

Create plots showing density extrema vs system size with various scaling fits.
"""
function plot_extrema_vs_size(results::Vector)
    # Extract data
    L_values = [r.L for r in results]
    peak_values = [r.peak_value for r in results]
    min_values = [r.min_value for r in results]
    second_peak_values = [r.second_peak_value for r in results]
    second_min_values = [r.second_min_value for r in results]
    ratios = [r.peak_min_ratio for r in results]
    
    # Sort by system size
    sort_indices = sortperm(L_values)
    L_sorted = L_values[sort_indices]
    peaks_sorted = peak_values[sort_indices]
    mins_sorted = min_values[sort_indices]
    second_peaks_sorted = second_peak_values[sort_indices]
    second_mins_sorted = second_min_values[sort_indices]
    ratios_sorted = ratios[sort_indices]
    
    println("\n" * "="^80)
    println("DENSITY EXTREMA SUMMARY")
    println("="^80)
    @printf("%-8s %-8s %-12s %-12s %-12s %-12s %-12s\n", "L", "γ", "Peak", "2nd Peak", "Min", "2nd Min", "Peak/Min")
    println("-"^88)
    for (i, idx) in enumerate(sort_indices)
        r = results[idx]
        @printf("%-8d %-8.2f %-12.6f %-12.6f %-12.6f %-12.6f %-12.3f\n", 
               r.L, r.γ, r.peak_value, r.second_peak_value, r.min_value, r.second_min_value, r.peak_min_ratio)
    end
    
    # Calculate statistics for each quantity
    println("\n" * "="^80)
    println("EXTREMA STATISTICS")
    println("="^80)
    
    # Peak statistics
    peak_max = maximum(peaks_sorted)
    peak_min = minimum(peaks_sorted)
    
    println("PEAK VALUES:")
    @printf("  Maximum: %.6f\n", peak_max)
    @printf("  Minimum: %.6f\n", peak_min)
    @printf("  Range (max-min): %.6f\n", peak_max - peak_min)
    
    # Second peak statistics  
    second_peak_max = maximum(second_peaks_sorted)
    second_peak_min = minimum(second_peaks_sorted)
    
    println("\nSECOND PEAK VALUES:")
    @printf("  Maximum: %.6f\n", second_peak_max)
    @printf("  Minimum: %.6f\n", second_peak_min)
    @printf("  Range (max-min): %.6f\n", second_peak_max - second_peak_min)
    
    # Min statistics
    min_max = maximum(mins_sorted)
    min_min = minimum(mins_sorted)
    
    println("\nMINIMUM VALUES:")
    @printf("  Maximum: %.6f\n", min_max)
    @printf("  Minimum: %.6f\n", min_min)
    @printf("  Range (max-min): %.6f\n", min_max - min_min)
    
    # Second min statistics
    second_min_max = maximum(second_mins_sorted)
    second_min_min = minimum(second_mins_sorted)
    
    println("\nSECOND MINIMUM VALUES:")
    @printf("  Maximum: %.6f\n", second_min_max)
    @printf("  Minimum: %.6f\n", second_min_min)
    @printf("  Range (max-min): %.6f\n", second_min_max - second_min_min)
    
    # Ratio statistics
    ratio_max = maximum(ratios_sorted)
    ratio_min = minimum(ratios_sorted)
    
    println("\nPEAK/MIN RATIO:")
    @printf("  Maximum: %.6f\n", ratio_max)
    @printf("  Minimum: %.6f\n", ratio_min)
    @printf("  Range (max-min): %.6f\n", ratio_max - ratio_min)
    
    # Calculate ranges for each quantity
    peak_max = maximum(peaks_sorted)
    peak_min = minimum(peaks_sorted)
    peak_diff = peak_max - peak_min
    
    min_max = maximum(mins_sorted)
    min_min = minimum(mins_sorted)
    min_diff = min_max - min_min
    
    ratio_max = maximum(ratios_sorted)
    ratio_min = minimum(ratios_sorted)
    ratio_diff = ratio_max - ratio_min
    
    println("\nRANGE ANALYSIS:")
    println("="^50)
    println("Peak Density:")
    println("  Maximum: $(round(peak_max, digits=6))")
    println("  Minimum: $(round(peak_min, digits=6))")
    println("  Difference: $(round(peak_diff, digits=6))")
    println("\nMin Density:")
    println("  Maximum: $(round(min_max, digits=6))")
    println("  Minimum: $(round(min_min, digits=6))")
    println("  Difference: $(round(min_diff, digits=6))")
    println("\nPeak/Min Ratio:")
    println("  Maximum: $(round(ratio_max, digits=6))")
    println("  Minimum: $(round(ratio_min, digits=6))")
    println("  Difference: $(round(ratio_diff, digits=6))")
    
    # Create plots
    p1 = plot(title="Density Peak vs System Size", xlabel="L", ylabel="Peak Density",
              legend=:topright, size=(800, 600))
    scatter!(p1, L_sorted, peaks_sorted, label="Peak values", color=:red, markersize=6)
    scatter!(p1, L_sorted, second_peaks_sorted, label="2nd Peak values", color=:orange, markersize=4, alpha=0.7)
    # Add horizontal lines for extrema
    hline!(p1, [peak_max], label="Max: $(round(peak_max, digits=3))", color=:red, linestyle=:dash, alpha=0.7)
    hline!(p1, [peak_min], label="Min: $(round(peak_min, digits=3))", color=:red, linestyle=:dash, alpha=0.7)
    
    p2 = plot(title="Density Minimum vs System Size", xlabel="L", ylabel="Min Density", 
              legend=:topright, size=(800, 600))
    scatter!(p2, L_sorted, mins_sorted, label="Min values", color=:blue, markersize=6)
    scatter!(p2, L_sorted, second_mins_sorted, label="2nd Min values", color=:lightblue, markersize=4, alpha=0.7)
    # Add horizontal lines for extrema
    hline!(p2, [min_max], label="Max: $(round(min_max, digits=3))", color=:blue, linestyle=:dash, alpha=0.7)
    hline!(p2, [min_min], label="Min: $(round(min_min, digits=3))", color=:blue, linestyle=:dash, alpha=0.7)
    
    p3 = plot(title="Peak/Min Ratio vs System Size", xlabel="L", ylabel="Peak/Min Ratio",
              legend=:topright, size=(800, 600))
    scatter!(p3, L_sorted, ratios_sorted, label="Peak/Min ratio", color=:purple, markersize=6)
    # Add horizontal lines for extrema
    hline!(p3, [ratio_max], label="Max: $(round(ratio_max, digits=3))", color=:purple, linestyle=:dash, alpha=0.7)
    hline!(p3, [ratio_min], label="Min: $(round(ratio_min, digits=3))", color=:purple, linestyle=:dash, alpha=0.7)
    
    # Add a fourth plot for second extrema comparison
    p4 = plot(title="Second Extrema vs System Size", xlabel="L", ylabel="Density",
              legend=:topright, size=(800, 600))
    scatter!(p4, L_sorted, second_peaks_sorted, label="2nd Peak values", color=:orange, markersize=6)
    scatter!(p4, L_sorted, second_mins_sorted, label="2nd Min values", color=:lightblue, markersize=6)
    # Add horizontal lines for second extrema
    hline!(p4, [second_peak_max], label="2nd Peak Max: $(round(second_peak_max, digits=3))", color=:orange, linestyle=:dash, alpha=0.7)
    hline!(p4, [second_peak_min], label="2nd Peak Min: $(round(second_peak_min, digits=3))", color=:orange, linestyle=:dash, alpha=0.7)
    hline!(p4, [second_min_max], label="2nd Min Max: $(round(second_min_max, digits=3))", color=:lightblue, linestyle=:dash, alpha=0.7)
    hline!(p4, [second_min_min], label="2nd Min Min: $(round(second_min_min, digits=3))", color=:lightblue, linestyle=:dash, alpha=0.7)
    
    # Combined plot
    p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1600, 1200))
    
    # Log-log plots
    p_log = plot(title="Log-Log: Density Extrema vs System Size", 
                xlabel="log(L)", ylabel="log(Density)", 
                legend=:topright, size=(800, 600))
    plot!(p_log, log.(L_sorted), log.(peaks_sorted), 
          label="log(Peak)", color=:red, lw=2, marker=:circle)
    plot!(p_log, log.(L_sorted), log.(second_peaks_sorted), 
          label="log(2nd Peak)", color=:orange, lw=2, marker=:circle, alpha=0.7)
    plot!(p_log, log.(L_sorted), log.(mins_sorted), 
          label="log(Min)", color=:blue, lw=2, marker=:square)
    plot!(p_log, log.(L_sorted), log.(second_mins_sorted), 
          label="log(2nd Min)", color=:lightblue, lw=2, marker=:square, alpha=0.7)
    
    # Save plots
    combined_path = abspath("density_extrema_vs_system_size.png")
    log_path = abspath("density_extrema_log_log.png")
    savefig(p_combined, combined_path)
    savefig(p_log, log_path)
    
    println("\nPlots saved:")
    println("  - $combined_path")
    println("  - $log_path")
    
    return p_combined, p_log
end

# Main analysis function
function main()
    println("DENSITY EXTREMA ANALYSIS")
    println("="^80)
    
    # Directory containing the analysis files
    data_dir = "../dummy_states/for_current_analysis"
    
    if !isdir(data_dir)
        println("ERROR: Directory $data_dir not found!")
        return
    end
    
    # Find all 1D .jld2 files
    files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), 
                  readdir(data_dir))
    
    if isempty(files)
        println("No 1D files found in $data_dir")
        return
    end
    
    println("Found $(length(files)) 1D files:")
    for (i, file) in enumerate(files)
        println("$i. $file")
    end
    println()
    
    # Analyze each file
    results = []
    for file in files
        full_path = joinpath(data_dir, file)
        result = analyze_extrema_from_file(full_path)
        if result !== nothing
            push!(results, result)
        end
        println()
    end
    
    if isempty(results)
        println("No files were successfully analyzed!")
        return
    end
    
    println("="^80)
    println("Analysis complete! Processed $(length(results)) files successfully.")
    
    # Create plots
    p_combined, p_log = plot_extrema_vs_size(results)
    
    # Display the plots
    display(p_combined)
    display(p_log)
    
    return results
end

# Run the analysis
results = main()
