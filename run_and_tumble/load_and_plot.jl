# load_and_plot.jl
using JLD2
using ArgParse
include("plot_utils.jl")
using Plots
using .PlotUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_states"
            help = "Path(s) to the saved state file(s) (.jld2)"
            required = true
            nargs = '+'  # This allows one or more arguments
    end
    return parse_args(s)
end

function main()
    # Parse arguments
    args = parse_commandline()
    figures_dir = "results_figures"
    
    # Create array to store all states, params, and filenames
    states_params_names = []
    
    # Load each saved state
    for saved_state in args["saved_states"]
        println("Loading state from: $(saved_state)")
        @load saved_state state param potential
        
        # Extract filename without extension and path for legend
        filename = basename(saved_state)
        filename = replace(filename, ".jld2" => "")
        
        # Create legend label using relevant parameters
        legend_label = "γ′=$(param.γ*param.N)"
        
        push!(states_params_names, (state, param, legend_label))
    end
    
    # Generate data collapse plot with all states
    p = plot_data_colapse(states_params_names)  # You'll need to update this function to use labels
    display(p)
    savefig(p, "$(figures_dir)/data_collapse_plot.png")
    println("Plot saved as data_collapse_plot.png")
    
    # Generate sweep plots for each state
    for (i, (state, param, label)) in enumerate(states_params_names)
        normalized_dist, corr_mat = plot_sweep(state.t, state, param; label=label)
        savefig("$(figures_dir)/sweep_plot_$(i).png")
        println("Plot saved as sweep_plot_$(i).png")
    end
end

main()