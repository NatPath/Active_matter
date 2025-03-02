############### plot_avg.jl ###############
using JLD2
using ArgParse
using Plots
using Statistics

# Parse a single argument: the path to the aggregated results file.
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "aggregated_results"
            help = "Path to the aggregated results file (.jld2) containing normalized_dists and avg_corr"
            required = true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    agg_file = args["aggregated_results"]
    println("Loading aggregated results from: ", agg_file)
    @load agg_file normalized_dists corr_mats avg_corr avg_dists

    # Create density plot similar to your plot_utils style.
    p1 = plot(avg_dists,
              title = "Average Normalized Density",
              xlabel = "Position",
              ylabel = "Density",
              legend = false)

    # Create correlation heatmap.
    # If avg_corr is already a matrix, we use it directly.
    println(" avg_corr: $avg_corr")
    println(" avg_corr size: $(size(avg_corr))")
    p2 = heatmap(avg_corr,
                 title = "Average Correlation Matrix",
                 xlabel = "x", ylabel = "y",
                 color = :viridis)

    # Save the figures.
    figures_dir = "results_figures"
    mkpath(figures_dir)
    density_file = joinpath(figures_dir, "avg_density.png")
    corr_file    = joinpath(figures_dir, "avg_correlation.png")
    savefig(p1, density_file)
    savefig(p2, corr_file)
    
    println("Plots saved:")
    println("  Density plot: ", density_file)
    println("  Correlation plot: ", corr_file)
end

main()

