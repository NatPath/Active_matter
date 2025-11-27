# density_slope_analyzer.jl
using JLD2
using LsqFit
using Statistics
using LinearAlgebra
using ArgParse
using Plots
include("../modules_run_and_tumble.jl")
using .FP

"""
    calculate_density_slope(density::Vector, region_indices::Union{UnitRange, Vector{Int}})

Calculate the slope of the density in a specific region using linear regression.

# Arguments
- `density::Vector`: The 1D density array
- `region_indices::Union{UnitRange, Vector{Int}}`: The range or vector of indices to analyze

# Returns
- `slope::Float64`: The slope of the linear fit
- `intercept::Float64`: The y-intercept of the linear fit
- `r_squared::Float64`: The R-squared value of the fit
- `fit_error::Float64`: The standard error of the slope
"""
function calculate_density_slope(density::Vector, region_indices::Union{UnitRange, Vector{Int}})
    # Extract the density values and positions for the specified region
    x_values = collect(region_indices)
    y_values = density[region_indices]
    
    # Perform linear regression: y = mx + b
    n = length(x_values)
    if n < 2
        throw(ArgumentError("Need at least 2 points for linear regression"))
    end
    
    # Calculate slope and intercept using least squares
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    
    # Calculate slope
    numerator = sum((x_values .- x_mean) .* (y_values .- y_mean))
    denominator = sum((x_values .- x_mean).^2)
    
    if denominator ≈ 0
        slope = 0.0
        intercept = y_mean
        r_squared = 0.0
        fit_error = Inf
    else
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_predicted = slope .* x_values .+ intercept
        ss_res = sum((y_values .- y_predicted).^2)
        ss_tot = sum((y_values .- y_mean).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard error of slope
        residuals = y_values .- y_predicted
        mse = sum(residuals.^2) / (n - 2)
        fit_error = sqrt(mse / denominator)
    end
    
    return slope, intercept, r_squared, fit_error
end

"""
    calculate_density_slope_xy(x_values::Vector, y_values::Vector)

Calculate the slope of the density using x,y coordinates directly.

# Arguments
- `x_values::Vector`: The x-coordinates (can include negative values for periodic boundary)
- `y_values::Vector`: The corresponding y-values (density values)

# Returns
- `slope::Float64`: The slope of the linear fit
- `intercept::Float64`: The y-intercept of the linear fit
- `r_squared::Float64`: The R-squared value of the fit
- `fit_error::Float64`: The standard error of the slope
"""
function calculate_density_slope_xy(x_values::Vector, y_values::Vector)
    # Perform linear regression: y = mx + b
    n = length(x_values)
    if n < 2
        throw(ArgumentError("Need at least 2 points for linear regression"))
    end
    
    if length(x_values) != length(y_values)
        throw(ArgumentError("x_values and y_values must have the same length"))
    end
    
    # Calculate slope and intercept using least squares
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    
    # Calculate slope
    numerator = sum((x_values .- x_mean) .* (y_values .- y_mean))
    denominator = sum((x_values .- x_mean).^2)
    
    if denominator ≈ 0
        slope = 0.0
        intercept = y_mean
        r_squared = 0.0
        fit_error = Inf
    else
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_predicted = slope .* x_values .+ intercept
        ss_res = sum((y_values .- y_predicted).^2)
        ss_tot = sum((y_values .- y_mean).^2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate standard error of slope
        residuals = y_values .- y_predicted
        mse = sum(residuals.^2) / (n - 2)
        fit_error = sqrt(mse / denominator)
    end
    
    return slope, intercept, r_squared, fit_error
end

"""
    analyze_density_slopes_1D(state, param)

Analyze the slopes of the average density in two regions for a 1D simulation:
- Left region: from start to middle-2
- Right region: from middle+2 to end
- Always excludes middle-1, middle, and middle+1 points

# Arguments
- `state`: The simulation state containing ρ_avg
- `param`: The simulation parameters

# Returns
- `results::Dict`: Dictionary containing slope analysis results
"""
function analyze_density_slopes_1D(state, param)
    if length(param.dims) != 1
        throw(ArgumentError("This function is only for 1D simulations"))
    end
    
    L = param.dims[1]
    # normalized_density = state.ρ_avg / sum(state.ρ_avg)
    println("sum_rho = $(sum(state.ρ_avg)) when L=$L ")
    normalized_density = state.ρ_avg*L/sum(state.ρ_avg)
    
    # Calculate middle point (for L=32, middle=16; for L=16, middle=8)
    middle = div(L, 2)
    
    # Always exclude middle-1, middle, middle+1
    # For L=32: middle=16, so exclude 15, 16, 17 -> left: 1:14, right: 18:32
    # For L=16: middle=8, so exclude 7, 8, 9 -> left: 1:6, right: 10:16
    left_region = 1:(middle - 3)
    right_region = (middle + 2):L
    excluded_points = [middle - 2,middle - 1, middle, middle + 1]
    
    # Calculate slopes for both regions
    left_slope, left_intercept, left_r2, left_error = calculate_density_slope(normalized_density, left_region)
    right_slope, right_intercept, right_r2, right_error = calculate_density_slope(normalized_density, right_region)
    
    # Calculate combined fit excluding middle points
    # For periodic system: treat right region as connected to left through boundary
    # Right indices are shifted by -L to make them continuous with left region
    left_indices = collect(left_region)
    right_indices = collect(right_region)
    
    # Create periodic-aware combined fit
    # Left region keeps original indices, right region gets shifted by -L for x-coordinates only
    periodic_left_x = left_indices  # x-coordinates for left region
    periodic_right_x = right_indices .- L  # x-coordinates for right region (shifted)
    
    # Combine x-coordinates and density values
    periodic_combined_x = vcat(periodic_left_x, periodic_right_x)
    periodic_combined_density = vcat(normalized_density[left_indices], normalized_density[right_indices])
    
    # Calculate slope using the shifted x-coordinates but original density values
    combined_slope, combined_intercept, combined_r2, combined_error = calculate_density_slope_xy(periodic_combined_x, periodic_combined_density)
    
    # Package results
    results = Dict(
        "system_length" => L,
        "middle_point" => middle,
        "excluded_points" => excluded_points,
        "left_region" => (first(left_region), last(left_region)),
        "right_region" => (first(right_region), last(right_region)),
        "combined_region" => (first(periodic_combined_x), last(periodic_combined_x)),
        "left_slope" => left_slope,
        "left_intercept" => left_intercept,
        "left_r_squared" => left_r2,
        "left_fit_error" => left_error,
        "right_slope" => right_slope,
        "right_intercept" => right_intercept,
        "right_r_squared" => right_r2,
        "right_fit_error" => right_error,
        "combined_slope" => combined_slope,
        "combined_intercept" => combined_intercept,
        "combined_r_squared" => combined_r2,
        "combined_fit_error" => combined_error,
        "slope_difference" => right_slope - left_slope,
        "slope_ratio" => abs(right_slope) > 1e-10 ? left_slope / right_slope : Inf,
        "normalized_density" => normalized_density
    )
    
    return results
end

"""
    analyze_slopes_from_file(filename::String)

Load a saved state from a .jld2 file and analyze density slopes.
Always excludes middle-1, middle, and middle+1 points.

# Arguments
- `filename::String`: Path to the .jld2 file containing saved state

# Returns
- `results::Dict`: Dictionary containing slope analysis results
"""
function analyze_slopes_from_file(filename::String)
    println("Loading state from: $filename")
    @load filename state param potential
    
    results = analyze_density_slopes_1D(state, param)
    
    # Add gamma (potential fluctuation rate) to results
    results["gamma"] = param.γ
    
    return results
end

"""
    print_slope_analysis(results::Dict)

Print a formatted report of the slope analysis results.
"""
function print_slope_analysis(results::Dict)
    println("\n" * "="^60)
    println("DENSITY SLOPE ANALYSIS RESULTS")
    println("="^60)
    println("System length: $(results["system_length"])")
    println("Middle point: $(results["middle_point"])")
    println("Excluded points: $(results["excluded_points"])")
    println()
    
    println("LEFT REGION ($(results["left_region"][1]):$(results["left_region"][2])):")
    println("  Slope: $(round(results["left_slope"], sigdigits=6))")
    println("  Intercept: $(round(results["left_intercept"], sigdigits=6))")
    println("  R²: $(round(results["left_r_squared"], sigdigits=4))")
    println("  Fit error: $(round(results["left_fit_error"], sigdigits=4))")
    println()
    
    println("RIGHT REGION ($(results["right_region"][1]):$(results["right_region"][2])):")
    println("  Slope: $(round(results["right_slope"], sigdigits=6))")
    println("  Intercept: $(round(results["right_intercept"], sigdigits=6))")
    println("  R²: $(round(results["right_r_squared"], sigdigits=4))")
    println("  Fit error: $(round(results["right_fit_error"], sigdigits=4))")
    println()
    
    println("COMBINED PERIODIC FIT (excluding middle):")
    println("  Slope: $(round(results["combined_slope"], sigdigits=6))")
    println("  Intercept: $(round(results["combined_intercept"], sigdigits=6))")
    println("  R²: $(round(results["combined_r_squared"], sigdigits=4))")
    println("  Fit error: $(round(results["combined_fit_error"], sigdigits=4))")
    println("  Note: Right region treated as continuous with left through periodic boundary")
    println()
    
    println("COMPARISON:")
    println("  Slope difference (right - left): $(round(results["slope_difference"], sigdigits=6))")
    if isfinite(results["slope_ratio"])
        println("  Slope ratio (left/right): $(round(results["slope_ratio"], sigdigits=6))")
    else
        println("  Slope ratio (left/right): Infinite (right slope ≈ 0)")
    end
    println("="^60)
end

"""
    plot_density_with_slopes(results::Dict; save_plot::Bool=true, filename::String="", auto_filename::Bool=true)

Create a plot showing the density profile with fitted lines for both regions.
If auto_filename is true and filename is empty, automatically generates filename with system length.
"""
function plot_density_with_slopes(results::Dict; save_plot::Bool=true, filename::String="", auto_filename::Bool=true)
    
    L = results["system_length"]
    
    # Auto-generate filename if not provided
    if auto_filename && isempty(filename)
        gamma = get(results, "gamma", 0.0)  # Default to 0.0 if gamma not found
        filename = "density_slope_analysis_L_$(L)_g_$(gamma).png"
    elseif isempty(filename)
        gamma = get(results, "gamma", 0.0)
        filename = "density_slope_analysis_L_$(L)_g_$(gamma).png"
    end
    
    density = results["normalized_density"]
    
    # Create main plot
    x_range = 1:L
    gamma = get(results, "gamma", 0.0)
    p = plot(x_range, density, 
             label="Normalized Density", 
             xlabel="Position", 
             ylabel="Normalized Density",
             title="1D Density Profile with Slope Analysis\nL=$L, γ=$gamma, Left slope=$(round(results["left_slope"], sigdigits=4)), Right slope=$(round(results["right_slope"], sigdigits=4))",
             lw=3, 
             marker=:circle,
             markersize=4,
             legend=:topright,
             size=(800, 600),
             dpi=300)
    
    # Add fitted lines for left region
    left_start, left_end = results["left_region"]
    left_x = left_start:left_end
    left_fit = results["left_slope"] .* left_x .+ results["left_intercept"]
    plot!(p, left_x, left_fit, 
          label="Left fit (slope=$(round(results["left_slope"], sigdigits=4)))",
          color=:red, lw=2, linestyle=:dash)
    
    # Add fitted lines for right region
    right_start, right_end = results["right_region"]
    right_x = right_start:right_end
    right_fit = results["right_slope"] .* right_x .+ results["right_intercept"]
    plot!(p, right_x, right_fit, 
          label="Right fit (slope=$(round(results["right_slope"], sigdigits=4)))",
          color=:blue, lw=2, linestyle=:dash)
    
    # Add combined fit line for all non-excluded points (periodic-aware)
    left_start, left_end = results["left_region"]
    right_start, right_end = results["right_region"]
    
    # For the left region, use original indices
    left_x = collect(left_start:left_end)
    left_combined_fit = results["combined_slope"] .* left_x .+ results["combined_intercept"]
    
    # For the right region, we need to account for the periodic boundary
    # The fit was calculated with right indices shifted by -L, so we need to shift them back
    right_x = collect(right_start:right_end)
    right_combined_fit = results["combined_slope"] .* (right_x .- L) .+ results["combined_intercept"]
    
    # Plot the combined fit as two segments
    plot!(p, left_x, left_combined_fit, 
          label="Combined periodic fit (slope=$(round(results["combined_slope"], sigdigits=4)))",
          color=:green, lw=3, linestyle=:solid, alpha=0.8)
    plot!(p, right_x, right_combined_fit, 
          label="", # No label for the second segment to avoid duplicate legend entries
          color=:green, lw=3, linestyle=:solid, alpha=0.8)
    
    # Mark the excluded region  
    if haskey(results, "excluded_points") && !isempty(results["excluded_points"])
        excluded_indices = results["excluded_points"]
        scatter!(p, excluded_indices, density[excluded_indices],
                label="Excluded middle points",
                color=:gray, 
                marker=:x,
                markersize=8,
                markerstrokewidth=2)
    end
    
    # Add vertical line at middle
    middle = results["middle_point"]
    vline!(p, [middle], 
           label="Middle", 
           color=:black, 
           linestyle=:dot,
           alpha=0.7)
    
    if save_plot
        savefig(p, filename)
        full_path = abspath(filename)
        println("Plot saved as: $full_path")
    end
    
    return p
end

"""
    plot_density_simple(results::Dict; save_plot::Bool=false, filename::String="", auto_filename::Bool=true)

Create a simple plot showing just the density profile without slope analysis.
If auto_filename is true and filename is empty, automatically generates filename with system length.
"""
function plot_density_simple(results::Dict; save_plot::Bool=false, filename::String="", auto_filename::Bool=true)
    L = results["system_length"]
    
    # Auto-generate filename if not provided
    if auto_filename && isempty(filename)
        gamma = get(results, "gamma", 0.0)  # Default to 0.0 if gamma not found
        filename = "density_profile_L_$(L)_g_$(gamma).png"
    elseif isempty(filename)
        gamma = get(results, "gamma", 0.0)
        filename = "density_profile_L_$(L)_g_$(gamma).png"
    end
    
    density = results["normalized_density"]
    
    # Create simple density plot
    x_range = 1:L
    gamma = get(results, "gamma", 0.0)
    p = plot(x_range, density, 
             label="Normalized Density", 
             xlabel="Position", 
             ylabel="Normalized Density",
             title="1D Density Profile (L=$L, γ=$gamma)",
             lw=3, 
             marker=:circle,
             markersize=4,
             legend=:topright,
             size=(800, 500),
             dpi=300,
             color=:blue)
    
    # Add vertical line at middle
    middle = results["middle_point"]
    vline!(p, [middle], 
           label="Middle", 
           color=:black, 
           linestyle=:dot,
           alpha=0.7)
    
    if save_plot
        savefig(p, filename)
        full_path = abspath(filename)
        println("Simple density plot saved as: $full_path")
    end
    
    return p
end

# Command line interface
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_state"
            help = "Path to the saved state file (.jld2)"
            required = true
        "--save_plot"
            help = "Save the plot to file"
            action = :store_true
        "--plot_filename"
            help = "Filename for the plot (if not provided, auto-generated with system length and gamma)"
            default = ""
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Analyze slopes from the file
    results = analyze_slopes_from_file(args["saved_state"])
    
    # Print results
    print_slope_analysis(results)
    
    # Create and optionally save plot
    if args["save_plot"]
        plot_density_with_slopes(results; 
                               save_plot=true, 
                               filename=args["plot_filename"],
                               auto_filename=isempty(args["plot_filename"]))
    end
    
    return results
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
