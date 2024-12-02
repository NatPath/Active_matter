# load_and_plot.jl
using JLD2
using ArgParse
include("plot_utils.jl")
using Plots
using .PlotUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_state"
            help = "Path to the saved state file (.jld2)"
            required = true
    end
    return parse_args(s)
end

function main()
    # Parse arguments
    args = parse_commandline()
    
    # Load saved state
    println("Loading state from: $(args["saved_state"])")
    @load args["saved_state"] state param potential
    
    # Create array of tuples for plot_data_colapse
    states_params = [(state, param)]
    
    # Generate plot
    p=plot_data_colapse(states_params)
    display(p)
    
    figures_dir = "results_figures"
    # Save plot
    savefig(p,"$(figures_dir)/data_collapse_plot.png")
    println("Plot saved as data_collapse_plot.png")
end

main()