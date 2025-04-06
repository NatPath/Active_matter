# load_and_plot.jl
using JLD2
using ArgParse
include("plot_utils.jl")
using Plots
using .PlotUtils

# Custom function to parse a range from a string
function parse_range(range_str::String)
    parts = split(range_str, ":")
    if length(parts) != 3
        error("Invalid format for --indices. Use 'start:step:end'.")
    end
    start, step, stop = parse.(Int, parts)
    return start:step:stop
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "saved_states"
            help = "Path(s) to the saved state file(s) (.jld2)"
            required = true
            nargs = '+'  # This allows one or more arguments
        "--n"
            help = "the power of y in the scaling from"
            arg_type= Float16
            default = 2.0
        "--indices"
            help = "Range of indices in the format 'start:step:end'"
            default = "1:1:10"  # Default range if not provided

    end
    return parse_args(s)
end

function main()
    # Parse arguments
    args = parse_commandline()
    joined_filename=join(args["saved_states"],"+")
    joined_filename=replace(joined_filename,".jld2"=>"")
    only_joined_filename=split(joined_filename,"/")|>last
    figures_dir = "results_figures/$(only_joined_filename)"
    mkpath(figures_dir)

    n = args["n"]
    indices= parse_range(args["indices"])
    # Create array to store all states, params, and filenames
    states_params_names = []

    
    # Load each saved state
    for saved_state in args["saved_states"]
        println("Loading state from: $(saved_state)")
        @load saved_state state param potential
        
        # # Extract filename without extension and path for legend
        # filename = basename(saved_state)
        # filename = replace(filename, ".jld2" => "")
        
        # Create legend label using relevant parameters
        legend_label = "γ′=$(param.γ*param.N)"
        
        push!(states_params_names, (state, param, legend_label))
    end
    
    # Generate data collapse plot with all states
    p = plot_data_colapse(states_params_names,n,indices)  # You'll need to update this function to use labels
    display(p)
    savefig(p, "$(figures_dir)/data_collapse_plot_y^$(n)_indices-$(args["indices"]).png")
    println("Plot saved as data_collapse_plot.png")
    
    # Generate sweep plots for each state
    for (i, (state, param, label)) in enumerate(states_params_names)
        filename = basename(args["saved_states"][i])
        filename = replace(filename, ".jld2" => "")
        only_filename = split(filename,"/")|>last
        normalized_dist, corr_mat = plot_sweep(state.t, state, param; label=label)
        specific_state_dir= "$(figures_dir)/$(only_filename)"
        mkpath(specific_state_dir)
        savefig("$(specific_state_dir)/sweep_plot_$(i).png")
        println("Plot saved as sweep_plot_$(i).png")
    end
end

main()