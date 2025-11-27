# test_slope_analyzer.jl
include("density_slope_analyzer.jl")

# Test the density slope analyzer with a 1D saved state file
function test_slope_analyzer()
    println("Testing Density Slope Analyzer")
    println("="^50)
    
    # Use one of the 1D files found in the workspace
    #γ=1
    # test_file = "../saved_states/1D_potential-smudge_Vscale-2.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-1600.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    #test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-1600.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gamma-1.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-64_rho-1.0e+02_alpha-0.00_gamma-1.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"

    # γ=0.5
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gammap-1600.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-800.00_D-1.0_f_-0.0_ffr-0.00_t-319016.jld2"
    # test_file = "../dummy_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-800.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-800.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file ="../dummy_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-800.00_D-1.0_f_-0.0_ffr-0.00_t-4000000.jld2"
    test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-64_rho-1.0e+02_alpha-0.00_gamma-0.50_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"

    # γ=0.25
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-16_rho-1.0e+02_alpha-0.00_gammap-400.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gammap-800.00_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"
    # test_file = "../saved_states/1D_potential-smudge_Vscale-4.0_fluctuation-reflection_activity-0.00_L-64_rho-1.0e+02_alpha-0.00_gamma-0.25_D-1.0_f_-0.0_ffr-0.00_t-1000000.jld2"

    if !isfile(test_file)
        println("Test file not found: $test_file")
        println("Available 1D files in saved_states:")
        
        # List available 1D files
        for file in readdir("../saved_states")
            if startswith(file, "1D_") && endswith(file, ".jld2")
                println("  $file")
            end
        end
        
        # Try to use the first available 1D file
        available_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), readdir("../saved_states"))
        if !isempty(available_files)
            test_file = joinpath("../saved_states", available_files[1])
            println("\nUsing first available file: $test_file")
        else
            println("No 1D files found in saved_states directory!")
            return
        end
    end
    
    try
        # Test basic slope analysis
        println("\n1. Basic slope analysis:")
        results = analyze_slopes_from_file(test_file)
        print_slope_analysis(results)
        
        # Create visualizations
        println("\n2. Creating visualizations...")
        
        # First show simple density plot
        println("   a) Simple density profile:")
        p_simple = plot_density_simple(results; save_plot=true)  # Will auto-generate filename with L and gamma
        display(p_simple)
        
        # Then show density with slope analysis
        println("   b) Density profile with fitted slopes:")
        p_slopes = plot_density_with_slopes(results; save_plot=true)  # Will auto-generate filename with L and gamma
        display(p_slopes)  # Display the plot during the test
        
        # Extract key statistics
        println("\n3. Key findings:")
        if abs(results["left_slope"]) > abs(results["right_slope"])
            stronger_side = "left"
            stronger_slope = results["left_slope"]
        else
            stronger_side = "right"
            stronger_slope = results["right_slope"]
        end
        
        println("- System length: $(results["system_length"]) points")
        println("- Stronger slope on $stronger_side side: $(round(stronger_slope, sigdigits=4))")
        println("- Asymmetry indicator (|difference|): $(round(abs(results["slope_difference"]), sigdigits=4))")
        
        if abs(results["slope_difference"]) > 1e-6
            println("- System shows density asymmetry")
        else
            println("- System appears symmetric")
        end
        
        println("\n4. Quality of fits:")
        if results["left_r_squared"] > 0.8 && results["right_r_squared"] > 0.8
            println("- Both regions show good linear fits (R² > 0.8)")
        elseif results["left_r_squared"] > 0.8
            println("- Left region shows good linear fit, right region may be nonlinear")
        elseif results["right_r_squared"] > 0.8
            println("- Right region shows good linear fit, left region may be nonlinear")
        else
            println("- Both regions may show nonlinear behavior (low R² values)")
        end
        
        println("\nTest completed successfully!")
        return results
        
    catch e
        println("Error during analysis: $e")
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
    end
end

# Function to analyze multiple files and create comparison
function test_multiple_files()
    println("\n\nTesting Multiple Files Comparison")
    println("="^50)
    
    # Get some 1D files from saved_states
    saved_files = readdir("../saved_states")
    id_files = filter(f -> startswith(f, "1D_") && endswith(f, ".jld2"), saved_files)
    
    if length(id_files) < 2
        println("Need at least 2 1D files for comparison. Found $(length(id_files)).")
        return
    end
    
    # Analyze up to 5 files
    files_to_analyze = id_files[1:min(5, length(id_files))]
    
    println("Analyzing $(length(files_to_analyze)) files:")
    for (i, file) in enumerate(files_to_analyze)
        println("$i. $file")
    end
    
    println("\nComparison Results:")
    println("File\tL\tLeft Slope\tRight Slope\tDifference\tAsymmetry")
    println("-" * 90)
    
    all_results = []
    for (i, file) in enumerate(files_to_analyze)
        full_path = joinpath("../saved_states", file)
        try
            results = analyze_slopes_from_file(full_path)
            push!(all_results, (file, results))
            
            asymmetry = abs(results["slope_difference"]) > 1e-6 ? "Yes" : "No"
            short_file = length(file) > 15 ? file[1:12] * "..." : file
            
            println("$short_file\t$(results["system_length"])\t$(round(results["left_slope"], sigdigits=3))\t$(round(results["right_slope"], sigdigits=3))\t$(round(results["slope_difference"], sigdigits=3))\t$asymmetry")
            
        catch e
            println("$file\tError: $e")
        end
    end
    
    return all_results
end

# Run the tests
if abspath(PROGRAM_FILE) == @__FILE__
    results = test_slope_analyzer()
    
    # Uncomment to test multiple files
    # multiple_results = test_multiple_files()
end
