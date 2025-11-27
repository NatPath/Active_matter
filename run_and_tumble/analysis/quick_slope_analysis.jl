#!/usr/bin/env julia
# quick_slope_analysis.jl - Quick command-line tool for slope analysis

include("density_slope_analyzer.jl")

function main()
    if length(ARGS) < 1
        println("Usage:")
        println("  julia quick_slope_analysis.jl <saved_state_file.jld2> [--exclude-specific]")
        println("")
        println("Options:")
        println("  --exclude-specific    Exclude points 7,8,9 (for L=16 systems)")
        println("")
        println("Examples:")
        println("  julia quick_slope_analysis.jl saved_states/your_file.jld2")
        println("  julia quick_slope_analysis.jl saved_states/your_file.jld2 --exclude-specific")
        return
    end
    
    filename = ARGS[1]
    exclude_specific = "--exclude-specific" in ARGS
    
    if !isfile(filename)
        println("Error: File not found: $filename")
        return
    end
    
    try
        println("Analyzing: $(basename(filename))")
        println("Exclusion method: $(exclude_specific ? "Specific points 7,8,9" : "Standard middle points")")
        println("-" ^ 60)
        
        results = analyze_slopes_from_file(filename; exclude_specific_points=exclude_specific)
        
        # Print compact results
        println("System length: $(results["system_length"])")
        if exclude_specific
            println("Excluded points: $(results["excluded_points"])")
        else
            println("Excluded middle points: $(results["exclude_middle_points"])")
        end
        println("Left region: $(results["left_region"][1]):$(results["left_region"][2])")
        println("Right region: $(results["right_region"][1]):$(results["right_region"][2])")
        println()
        
        println("SLOPES:")
        println("  Left slope:  $(round(results["left_slope"], sigdigits=6))")
        println("  Right slope: $(round(results["right_slope"], sigdigits=6))")
        println("  Difference:  $(round(results["slope_difference"], sigdigits=6))")
        println()
        
        println("FIT QUALITY:")
        println("  Left R²:  $(round(results["left_r_squared"], sigdigits=4))")
        println("  Right R²: $(round(results["right_r_squared"], sigdigits=4))")
        println()
        
        if abs(results["slope_difference"]) > 1e-6
            println("→ ASYMMETRIC system detected")
        else
            println("→ SYMMETRIC system")
        end
        
        # Create and display plot
        p = plot_density_with_slopes(results; save_plot=true, filename="quick_analysis_plot.png")
        display(p)
        println("\nPlot saved as: quick_analysis_plot.png")
        
    catch e
        println("Error analyzing file: $e")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
