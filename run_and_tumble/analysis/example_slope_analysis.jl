# example_slope_analysis.jl
include("density_slope_analyzer.jl")

println("Example: Density Slope Analysis")
println("="^50)

# Example 1: Analyze a specific file
println("\n1. Analyzing available 1D saved states:")

# Find available 1D files
available_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), readdir("../saved_states"))

if !isempty(available_files)
    test_file = joinpath("../saved_states", available_files[1])
    println("Using file: $(basename(test_file))")
    
    results = analyze_slopes_from_file(test_file)
    print_slope_analysis(results)
    
    # Create plots
    println("\nCreating visualizations...")
    p1 = plot_density_simple(results; save_plot=true, auto_filename=true)
    p2 = plot_density_with_slopes(results; save_plot=true, auto_filename=true)
    
    println("Plots saved with system length in filename:")
    L = results["system_length"]
    println("- density_profile_L_$(L).png")
    println("- density_slope_analysis_L_$(L)_exclude_1.png")
    
    display(p1)
    display(p2)
else
    println("No 1D files found in saved_states directory")
end

# Example 2: Analyze multiple files and compare slopes
function example_multiple_files()
    println("\nExample 2: Comparing slopes across multiple files")
    
    available_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), readdir("../saved_states"))
    
    if length(available_files) < 2
        println("Need at least 2 files for comparison. Found $(length(available_files)).")
        return
    end
    
    # Analyze up to 3 files
    files_to_analyze = available_files[1:min(3, length(available_files))]
    
    println("Analyzing $(length(files_to_analyze)) files:")
    println("File\t\t\tLeft Slope\tRight Slope\tDifference\tAsymmetric?")
    println("-" ^ 80)
    
    all_results = []
    
    for filename in files_to_analyze
        full_path = joinpath("../saved_states", filename)
        try
            results = analyze_slopes_from_file(full_path; exclude_middle_points=1)
            push!(all_results, (filename, results))
            
            asymmetric = abs(results["slope_difference"]) > 1e-6 ? "Yes" : "No"
            short_name = length(filename) > 20 ? filename[1:17] * "..." : filename
            
            left_slope = round(results["left_slope"], sigdigits=4)
            right_slope = round(results["right_slope"], sigdigits=4)
            slope_diff = round(results["slope_difference"], sigdigits=4)
            
            println("$short_name\t$left_slope\t\t$right_slope\t\t$slope_diff\t\t$asymmetric")
            
        catch e
            println("$filename\t\tError: $e")
        end
    end
    
    return all_results
end

# Quick analysis function
function quick_density_slope_analysis(filename::String; show_plots::Bool=true)
    """
    Quick function to analyze density slopes and optionally show plots.
    """
    try
        results = analyze_slopes_from_file(filename)
        
        # Print basic results
        println("\nQuick Analysis Results:")
        println("File: $(basename(filename))")
        println("System length: $(results["system_length"])")
        println("Left slope: $(round(results["left_slope"], sigdigits=5))")
        println("Right slope: $(round(results["right_slope"], sigdigits=5))")
        println("Slope difference: $(round(results["slope_difference"], sigdigits=5))")
        
        if abs(results["slope_difference"]) > 1e-6
            println("→ System shows asymmetry")
        else
            println("→ System appears symmetric")
        end
        
        if show_plots
            p = plot_density_with_slopes(results; save_plot=false)
            display(p)
        end
        
        return results
        
    catch e
        println("Error analyzing $filename: $e")
        return nothing
    end
end

# Run examples if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^50)
    println("2. Comparing multiple files:")
    example_multiple_files()
    
    println("\n" * "="^50)
    println("3. Quick analysis example:")
    available_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), readdir("../saved_states"))
    if !isempty(available_files)
        test_file = joinpath("../saved_states", available_files[1])
        quick_density_slope_analysis(test_file; show_plots=false)
    end
    
    println("\n" * "="^50)
    println("Example completed!")
end 
               "Slope_Difference", "Slope_Ratio"]
    
    data = []
    
    for filename in filenames
        try
            results = analyze_slopes_from_file(filename; exclude_middle_points=1)
            row = [
                basename(filename),
                results["system_length"],
                results["left_slope"],
                results["left_r_squared"],
                results["right_slope"],
                results["right_r_squared"],
                results["slope_difference"],
                results["slope_ratio"]
            ]
            push!(data, row)
        catch e
            println("Error with $filename: $e")
            # Add a row with NaNs for failed files
            row = [basename(filename), NaN, NaN, NaN, NaN, NaN, NaN, NaN]
            push!(data, row)
        end
    end
    
    # Combine headers and data
    all_data = vcat([headers], data)
    
    # Write to CSV
    writedlm(output_file, all_data, ',')
    println("Summary saved to: $output_file")
    
    return all_data
end

# Main function to run examples
function run_examples()
    println("Density Slope Analyzer - Example Usage")
    println("="^50)
    
    # Run example 1
    example_single_file()
    
    # Uncomment these to run other examples:
    # example_multiple_files()
    # example_parameter_sweep()
    
    # Example of creating a CSV summary for multiple files
    # example_files = ["file1.jld2", "file2.jld2", "file3.jld2"]
    # create_slope_summary_csv(example_files)
    
    println("\nExamples completed!")
end

# Run the examples
if abspath(PROGRAM_FILE) == @__FILE__
    run_examples()
end
