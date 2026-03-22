# analyze_size_scaling.jl
# Analyze all files in dummy_states/for_current_analysis and plot slope vs system size

include("density_slope_analyzer.jl")
using Plots
using Printf
using LsqFit

function analyze_all_files_in_directory(directory_path::String)
    """
    Analyze all 1D files in the specified directory and extract slope data.
    
    Returns:
    - system_sizes::Vector{Int}: System sizes (L values)
    - combined_slopes::Vector{Float64}: Combined periodic fit slopes
    - left_slopes::Vector{Float64}: Left region slopes
    - right_slopes::Vector{Float64}: Right region slopes
    - gamma_values::Vector{Float64}: Gamma values
    - file_names::Vector{String}: Full file paths
    """
    
    println("Analyzing files in: $directory_path")
    println("="^80)
    
    # Get all 1D .jld2 files
    all_files = readdir(directory_path)
    jld2_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), all_files)
    
    println("Found $(length(jld2_files)) 1D files:")
    for (i, file) in enumerate(jld2_files)
        println("$i. $file")
    end
    println()
    
    # Initialize result arrays
    system_sizes = Int[]
    combined_slopes = Float64[]
    left_slopes = Float64[]
    right_slopes = Float64[]
    gamma_values = Float64[]
    file_names = String[]
    
    # Analyze each file
    for (i, file) in enumerate(jld2_files)
        full_path = joinpath(directory_path, file)
        
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
            push!(file_names, full_path)
            
            println("  L=$L, γ=$gamma, Combined slope=$(round(combined_slope, sigdigits=5))")
            
        catch e
            println("  ERROR: Failed to analyze $file: $e")
        end
    end
    
    println("\n" * "="^80)
    println("Analysis complete! Processed $(length(system_sizes)) files successfully.")
    
    return system_sizes, combined_slopes, left_slopes, right_slopes, gamma_values, file_names
end

function plot_slope_vs_size(system_sizes, combined_slopes, fit_results=nothing; 
                           save_plot=true, filename="slope_vs_system_size_all_fits.png")
    """
    Create a plot of combined periodic slopes vs system size with all fitting models.
    """
    
    # Sort by system size for better visualization
    sorted_indices = sortperm(system_sizes)
    sorted_sizes = system_sizes[sorted_indices]
    sorted_combined = combined_slopes[sorted_indices]
    
    # Create the plot
    p = plot(xlabel="System Size (L)", 
             ylabel="Density Slope", 
             title="Combined Periodic Density Slopes vs System Size\nwith Multiple Model Fits",
             legend=:topright,
             size=(1000, 700),
             dpi=300)
    
    # Plot combined periodic fit slopes (data points)
    scatter!(p, sorted_sizes, sorted_combined,
             label="Combined Periodic Fit Data",
             color=:black,
             markersize=10,
             markerstrokewidth=2,
             markerstrokecolor=:black,
             markershape=:circle)
    
    # Add fitted curves if provided
    if fit_results !== nothing
        L_range = range(minimum(system_sizes) * 0.8, maximum(system_sizes) * 1.2, length=200)
        
        # Colors and styles for different models
        colors = [:red, :blue, :green, :purple, :orange]
        linestyles = [:dash, :dot, :solid, :dashdot, :dashdotdot]
        linewidths = [2, 2, 3, 2, 2]
        
        model_order = ["model1", "model2", "model3", "model4", "model5"]  # All 5 models
        
        for (i, model_key) in enumerate(model_order)
            if haskey(fit_results, model_key) && fit_results[model_key] !== nothing
                model_result = fit_results[model_key]
                fit_func = model_result["function"]
                model_name = model_result["name"]
                r2 = model_result["r_squared"]
                formula = model_result["formula"]
                
                fitted_slopes = [fit_func(L) for L in L_range]
                
                plot!(p, L_range, fitted_slopes,
                      color=colors[i],
                      linewidth=linewidths[i],
                      linestyle=linestyles[i],
                      label="$model_name (R²=$(round(r2, digits=4)))",
                      alpha=0.8)
            end
        end
    end
    
    # Add grid for better readability
    plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
    
    # Set axis limits with some padding
    xlims!(p, (minimum(system_sizes) * 0.9, maximum(system_sizes) * 1.1))
    
    if save_plot
        full_path = abspath(filename)
        savefig(p, filename)
        println("Plot saved as: $full_path")
    end
    
    return p
end

function fit_all_models(system_sizes, combined_slopes)
    """
    Fit the combined periodic slopes to multiple models:
    1. slope = a/L²
    2. slope = b/L  
    3. slope = a1/L + a2/L²
    4. slope = b/(L+a)
    5. slope = b/(L²+a)
    
    Returns:
    - results: Dictionary with fit results for each model
    """
    
    # Filter out any NaN or infinite values
    valid_indices = findall(x -> isfinite(x), combined_slopes)
    L_data = Float64.(system_sizes[valid_indices])
    slope_data = combined_slopes[valid_indices]
    
    if length(L_data) < 2
        println("ERROR: Not enough valid data points for fitting!")
        return nothing
    end
    
    results = Dict()
    
    # Model 1: slope = a/L²
    println("\n" * "="^80)
    println("MODEL 1: slope = a/L²")
    println("="^80)
    
    try
        model1_func(L, p) = p[1] ./ (L.^2)
        initial_a = maximum(slope_data) * maximum(L_data)^2
        fit1 = curve_fit(model1_func, L_data, slope_data, [initial_a])
        
        a_param = fit1.param[1]
        a_error = stderror(fit1)[1]
        y_pred1 = model1_func(L_data, fit1.param)
        r2_1 = 1 - sum((slope_data .- y_pred1).^2) / sum((slope_data .- mean(slope_data)).^2)
        
        results["model1"] = Dict(
            "params" => [a_param],
            "errors" => [a_error],
            "r_squared" => r2_1,
            "function" => L -> a_param / L^2,
            "name" => "a/L²",
            "formula" => "$(round(a_param, sigdigits=5))/L²"
        )
        
        println(@sprintf("Parameter a = %.6f ± %.6f", a_param, a_error))
        println(@sprintf("R² = %.6f", r2_1))
        println("Fitted function: slope(L) = $(round(a_param, sigdigits=5))/L²")
        
    catch e
        println("ERROR: Model 1 fitting failed: $e")
        results["model1"] = nothing
    end
    
    # Model 2: slope = b/L
    println("\n" * "="^80)
    println("MODEL 2: slope = b/L")
    println("="^80)
    
    try
        model2_func(L, p) = p[1] ./ L
        initial_b = maximum(slope_data) * maximum(L_data)
        fit2 = curve_fit(model2_func, L_data, slope_data, [initial_b])
        
        b_param = fit2.param[1]
        b_error = stderror(fit2)[1]
        y_pred2 = model2_func(L_data, fit2.param)
        r2_2 = 1 - sum((slope_data .- y_pred2).^2) / sum((slope_data .- mean(slope_data)).^2)
        
        results["model2"] = Dict(
            "params" => [b_param],
            "errors" => [b_error],
            "r_squared" => r2_2,
            "function" => L -> b_param / L,
            "name" => "b/L",
            "formula" => "$(round(b_param, sigdigits=5))/L"
        )
        
        println(@sprintf("Parameter b = %.6f ± %.6f", b_param, b_error))
        println(@sprintf("R² = %.6f", r2_2))
        println("Fitted function: slope(L) = $(round(b_param, sigdigits=5))/L")
        
    catch e
        println("ERROR: Model 2 fitting failed: $e")
        results["model2"] = nothing
    end
    
    # Model 3: slope = a1/L + a2/L²
    println("\n" * "="^80)
    println("MODEL 3: slope = a1/L + a2/L²")
    println("="^80)
    
    if length(L_data) >= 3  # Need at least 3 points for 2-parameter fit
        try
            model3_func(L, p) = p[1] ./ L .+ p[2] ./ (L.^2)
            initial_a1 = maximum(slope_data) * maximum(L_data)
            initial_a2 = maximum(slope_data) * maximum(L_data)^2 * 0.1
            fit3 = curve_fit(model3_func, L_data, slope_data, [initial_a1, initial_a2])
            
            a1_param, a2_param = fit3.param
            a1_error, a2_error = stderror(fit3)
            y_pred3 = model3_func(L_data, fit3.param)
            r2_3 = 1 - sum((slope_data .- y_pred3).^2) / sum((slope_data .- mean(slope_data)).^2)
            
            results["model3"] = Dict(
                "params" => [a1_param, a2_param],
                "errors" => [a1_error, a2_error],
                "r_squared" => r2_3,
                "function" => L -> a1_param / L + a2_param / L^2,
                "name" => "a1/L + a2/L²",
                "formula" => "$(round(a1_param, sigdigits=4))/L + $(round(a2_param, sigdigits=4))/L²"
            )
            
            println(@sprintf("Parameter a1 = %.6f ± %.6f", a1_param, a1_error))
            println(@sprintf("Parameter a2 = %.6f ± %.6f", a2_param, a2_error))
            println(@sprintf("R² = %.6f", r2_3))
            println("Fitted function: slope(L) = $(round(a1_param, sigdigits=5))/L + $(round(a2_param, sigdigits=5))/L²")
            
        catch e
            println("ERROR: Model 3 fitting failed: $e")
            results["model3"] = nothing
        end
    else
        println("WARNING: Not enough data points for 2-parameter fit")
        results["model3"] = nothing
    end
    
    # Model 4: slope = b/(L+a)
    println("\n" * "="^80)
    println("MODEL 4: slope = b/(L+a)")
    println("="^80)
    
    if length(L_data) >= 3  # Need at least 3 points for 2-parameter fit
        try
            model4_func(L, p) = p[1] ./ (L .+ p[2])
            initial_b = maximum(slope_data) * maximum(L_data)
            initial_a = 1.0  # Small offset
            fit4 = curve_fit(model4_func, L_data, slope_data, [initial_b, initial_a])
            
            b_param, a_param = fit4.param
            b_error, a_error = stderror(fit4)
            y_pred4 = model4_func(L_data, fit4.param)
            r2_4 = 1 - sum((slope_data .- y_pred4).^2) / sum((slope_data .- mean(slope_data)).^2)
            
            results["model4"] = Dict(
                "params" => [b_param, a_param],
                "errors" => [b_error, a_error],
                "r_squared" => r2_4,
                "function" => L -> b_param / (L + a_param),
                "name" => "b/(L+a)",
                "formula" => "$(round(b_param, sigdigits=4))/(L+$(round(a_param, sigdigits=4)))"
            )
            
            println(@sprintf("Parameter b = %.6f ± %.6f", b_param, b_error))
            println(@sprintf("Parameter a = %.6f ± %.6f", a_param, a_error))
            println(@sprintf("R² = %.6f", r2_4))
            println("Fitted function: slope(L) = $(round(b_param, sigdigits=5))/(L+$(round(a_param, sigdigits=5)))")
            
        catch e
        println("ERROR: Model 4 fitting failed: $e")
        results["model4"] = nothing
    end
    else
        println("WARNING: Not enough data points for 2-parameter fit")
        results["model4"] = nothing
    end
    
    # Model 5: slope = b/(L²+a)
    println("\n" * "="^80)
    println("MODEL 5: slope = b/(L²+a)")
    println("="^80)
    
    if length(L_data) >= 3  # Need at least 3 points for 2-parameter fit
        try
            model5_func(L, p) = p[1] ./ (L.^2 .+ p[2])
            initial_b = maximum(slope_data) * maximum(L_data)^2
            initial_a = 1.0  # Small offset
            fit5 = curve_fit(model5_func, L_data, slope_data, [initial_b, initial_a])
            
            b_param, a_param = fit5.param
            b_error, a_error = stderror(fit5)
            y_pred5 = model5_func(L_data, fit5.param)
            r2_5 = 1 - sum((slope_data .- y_pred5).^2) / sum((slope_data .- mean(slope_data)).^2)
            
            results["model5"] = Dict(
                "params" => [b_param, a_param],
                "errors" => [b_error, a_error],
                "r_squared" => r2_5,
                "function" => L -> b_param / (L^2 + a_param),
                "name" => "b/(L²+a)",
                "formula" => "$(round(b_param, sigdigits=4))/(L²+$(round(a_param, sigdigits=4)))"
            )
            
            println(@sprintf("Parameter b = %.6f ± %.6f", b_param, b_error))
            println(@sprintf("Parameter a = %.6f ± %.6f", a_param, a_error))
            println(@sprintf("R² = %.6f", r2_5))
            println("Fitted function: slope(L) = $(round(b_param, sigdigits=5))/(L²+$(round(a_param, sigdigits=5)))")
            
        catch e
            println("ERROR: Model 5 fitting failed: $e")
            results["model5"] = nothing
        end
    else
        println("WARNING: Not enough data points for 2-parameter fit")
        results["model5"] = nothing
    end
    
    # Compare models
    println("\n" * "="^80)
    println("MODEL COMPARISON")
    println("="^80)
    println(@sprintf("%-20s %-15s", "Model", "R²"))
    println("-"^35)
    
    for (model_key, model_result) in results
        if model_result !== nothing
            model_name = model_result["name"]
            r2 = model_result["r_squared"]
            println(@sprintf("%-20s %-15.6f", model_name, r2))
        end
    end
    
    return results
end

function print_summary_table(system_sizes, combined_slopes, left_slopes, right_slopes, gamma_values)
    """
    Print a summary table of the results.
    """
    
    # Sort by system size
    sorted_indices = sortperm(system_sizes)
    
    println("\n" * "="^100)
    println("SUMMARY TABLE: SLOPES VS SYSTEM SIZE")
    println("="^100)
    println(@sprintf("%-8s %-10s %-15s %-15s %-15s %-10s", 
                     "L", "γ", "Combined", "Left", "Right", "Asymmetry"))
    println(@sprintf("%-8s %-10s %-15s %-15s %-15s %-10s", 
                     "", "", "Slope", "Slope", "Slope", "|L-R|"))
    println("-"^100)
    
    for i in sorted_indices
        L = system_sizes[i]
        gamma = gamma_values[i]
        combined = combined_slopes[i]
        left = left_slopes[i]
        right = right_slopes[i]
        asymmetry = abs(left - right)
        
        println(@sprintf("%-8d %-10.2f %-15.6e %-15.6e %-15.6e %-10.6e", 
                        L, gamma, combined, left, right, asymmetry))
    end
    println("="^100)
end

function analyze_log_log_scaling(system_sizes, combined_slopes; save_plot=true, filename="slope_vs_size_log_log.png")
    """
    Create a log-log plot and extract the power law scaling exponent.
    Fits: log(slope) = log(A) + α * log(L)  =>  slope = A * L^α
    
    Returns:
    - power_law_exponent: The scaling exponent α
    - amplitude: The amplitude A
    - r_squared: R² of the log-log fit
    - fit_results: Dictionary with detailed fit results
    """
    
    println("\n" * "="^80)
    println("LOG-LOG POWER LAW ANALYSIS")
    println("="^80)
    
    # Filter out any non-positive values (can't take log of negative numbers)
    valid_indices = findall(x -> x > 0, combined_slopes)
    L_data = Float64.(system_sizes[valid_indices])
    slope_data = combined_slopes[valid_indices]
    
    if length(L_data) < 2
        println("ERROR: Not enough valid positive data points for log-log fitting!")
        return nothing
    end
    
    # Take logarithms
    log_L = log.(L_data)
    log_slope = log.(abs.(slope_data))  # Use abs to handle any small negative slopes
    
    # Perform linear regression on log-log data: log(slope) = log(A) + α * log(L)
    try
        # Using simple linear regression
        n = length(log_L)
        sum_x = sum(log_L)
        sum_y = sum(log_slope)
        sum_xy = sum(log_L .* log_slope)
        sum_x2 = sum(log_L.^2)
        sum_y2 = sum(log_slope.^2)
        
        # Calculate slope (α) and intercept (log(A))
        alpha = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x^2)
        log_A = (sum_y - alpha * sum_x) / n
        A = exp(log_A)
        
        # Calculate R²
        y_pred = log_A .+ alpha .* log_L
        ss_res = sum((log_slope .- y_pred).^2)
        ss_tot = sum((log_slope .- mean(log_slope)).^2)
        r2 = 1 - ss_res / ss_tot
        
        # Calculate errors (standard errors)
        residuals = log_slope .- y_pred
        s2 = sum(residuals.^2) / (n - 2)  # Mean squared error
        alpha_error = sqrt(s2 * n / (n * sum_x2 - sum_x^2))
        log_A_error = sqrt(s2 * sum_x2 / (n * (n * sum_x2 - sum_x^2)))
        A_error = A * log_A_error  # Error propagation for A = exp(log_A)
        
        println(@sprintf("Power law fit: slope(L) = A * L^α"))
        println(@sprintf("Amplitude A = %.6f ± %.6f", A, A_error))
        println(@sprintf("Exponent α = %.6f ± %.6f", alpha, alpha_error))
        println(@sprintf("R² = %.6f", r2))
        println(@sprintf("Fitted function: slope(L) = %.6f * L^(%.6f)", A, alpha))
        
        # Create log-log plot
        p = plot(xlabel="log(System Size L)", 
                 ylabel="log(|Density Slope|)", 
                 title="Log-Log Plot: Power Law Scaling Analysis\nslope(L) = $(round(A, sigdigits=4)) × L^($(round(alpha, sigdigits=4)))",
                 legend=:topright,
                 size=(800, 600),
                 dpi=300)
        
        # Sort data for better visualization
        sorted_indices = sortperm(L_data)
        sorted_L = L_data[sorted_indices]
        sorted_slopes = slope_data[sorted_indices]
        sorted_log_L = log.(sorted_L)
        sorted_log_slopes = log.(abs.(sorted_slopes))
        
        # Plot data points
        scatter!(p, sorted_log_L, sorted_log_slopes,
                 label="Data Points",
                 color=:black,
                 markersize=8,
                 markerstrokewidth=2,
                 markerstrokecolor=:black,
                 markershape=:circle)
        
        # Plot fitted line
        log_L_range = range(minimum(log_L), maximum(log_L), length=100)
        fitted_log_slopes = log_A .+ alpha .* log_L_range
        
        plot!(p, log_L_range, fitted_log_slopes,
              color=:red,
              linewidth=3,
              linestyle=:solid,
              label="Power Law Fit (R²=$(round(r2, digits=4)))",
              alpha=0.8)
        
        # Add grid
        plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
        
        # Add text annotation with the fit equation
        annotate!(p, [(mean(log_L), maximum(sorted_log_slopes) * 0.9, 
                      text("slope ∝ L^$(round(alpha, digits=3))", :black, :left, 12))])
        
        if save_plot
            full_path = abspath(filename)
            savefig(p, filename)
            println("Log-log plot saved as: $full_path")
        end
        
        # Also create a regular scale plot with the power law fit
        p2 = plot(xlabel="System Size (L)", 
                  ylabel="Density Slope", 
                  title="Power Law Fit on Linear Scale\nslope(L) = $(round(A, sigdigits=4)) × L^($(round(alpha, sigdigits=4)))",
                  legend=:topright,
                  size=(800, 600),
                  dpi=300)
        
        # Plot data points
        scatter!(p2, sorted_L, sorted_slopes,
                 label="Data Points",
                 color=:black,
                 markersize=8,
                 markerstrokewidth=2,
                 markerstrokecolor=:black,
                 markershape=:circle)
        
        # Plot fitted curve
        L_range = range(minimum(L_data), maximum(L_data), length=200)
        fitted_slopes = A .* (L_range.^alpha)
        
        plot!(p2, L_range, fitted_slopes,
              color=:red,
              linewidth=3,
              linestyle=:solid,
              label="Power Law Fit (R²=$(round(r2, digits=4)))",
              alpha=0.8)
        
        # Add grid
        plot!(p2, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)
        
        if save_plot
            linear_filename = replace(filename, ".png" => "_linear_scale.png")
            full_path2 = abspath(linear_filename)
            savefig(p2, linear_filename)
            println("Linear scale power law plot saved as: $full_path2")
        end
        
        fit_results = Dict(
            "amplitude" => A,
            "amplitude_error" => A_error,
            "exponent" => alpha,
            "exponent_error" => alpha_error,
            "r_squared" => r2,
            "function" => L -> A * L^alpha,
            "log_log_plot" => p,
            "linear_plot" => p2
        )
        
        return alpha, A, r2, fit_results
        
    catch e
        println("ERROR: Log-log fitting failed: $e")
        return nothing
    end
end

function main()
    println("DENSITY SLOPE SCALING ANALYSIS")
    println("="^80)
    
    # Directory containing the files to analyze
    analysis_dir = "../dummy_states/for_current_analysis"
    
    if !isdir(analysis_dir)
        println("ERROR: Directory $analysis_dir not found!")
        return
    end
    
    # Analyze all files
    system_sizes, combined_slopes, left_slopes, right_slopes, gamma_values, file_names = analyze_all_files_in_directory(analysis_dir)
    
    if isempty(system_sizes)
        println("ERROR: No files were successfully analyzed!")
        return
    end
    
    # Print summary table
    print_summary_table(system_sizes, combined_slopes, left_slopes, right_slopes, gamma_values)
    
    # Perform curve fitting with multiple models
    println("\nPerforming curve fitting with multiple models...")
    fit_results = fit_all_models(system_sizes, combined_slopes)
    
    # Create and display plot with all fits
    println("\nCreating slope vs system size plot with all fits...")
    p = plot_slope_vs_size(system_sizes, combined_slopes, fit_results)
    
    display(p)
    
    # Perform log-log analysis
    println("\nPerforming log-log power law analysis...")
    log_log_results = analyze_log_log_scaling(system_sizes, combined_slopes)
    
    # Additional analysis
    println("\nADDITIONAL ANALYSIS:")
    println("-"^50)
    
    # Check for scaling relationships
    sorted_indices = sortperm(system_sizes)
    sorted_sizes = system_sizes[sorted_indices]
    sorted_combined = combined_slopes[sorted_indices]
    
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
    
    # Check if slopes scale with 1/L²
    println("\nTesting 1/L² scaling:")
    for i in eachindex(sorted_sizes)
        L = sorted_sizes[i]
        slope = sorted_combined[i]
        scaled_slope = slope * L^2
        println("  L=$L: slope × L² = $(round(scaled_slope, sigdigits=6))")
    end
    
    println("\nAnalysis completed successfully!")
    return system_sizes, combined_slopes, left_slopes, right_slopes, gamma_values, file_names, fit_results, log_log_results
end

# Run the analysis
results = main()
