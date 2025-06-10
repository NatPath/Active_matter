using JLD2
using ArgParse
using Statistics
include("modules_run_and_tumble.jl")
include("save_utils.jl")
using .FP
using .SaveUtils

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
    println("Starting aggregation of simulation results...")
    
    args = parse_commandline()
    
    println("Processing $(length(args["saved_states"])) state files...")
    
    states = []
    params = []
    normalized_dists = []
    corr_mats = []
    stats_arr = []
    total_t = 0
    
    for (i, saved_state) in enumerate(args["saved_states"])
        println("  Loading file $i/$(length(args["saved_states"])): $(basename(saved_state))")
        try
            @load saved_state state param potential
            states = [states; state]
            params = [params; param]
            total_t += state.t
            stats = calculate_statistics(state)
            stats_arr = [stats_arr; stats]
        catch e
            println("  ERROR: Failed to load $saved_state: $e")
            continue
        end
    end
    
    if isempty(states)
        println("ERROR: No valid states loaded for aggregation")
        exit(1)
    end
    
    println("Successfully loaded $(length(states)) states")
    println("Total simulation time: $total_t")
    
    normalized_dists = [stats[1] for stats in stats_arr]
    corr_mats = [stats[2] for stats in stats_arr]

    # Handle different dimensions for stacking
    println("Aggregating correlation matrices and densities...")
    if ndims(corr_mats[1]) == 2  # 1D case
        println("  Processing 1D correlation matrices")
        stacked_corr = cat(corr_mats..., dims=3)
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
    elseif ndims(corr_mats[1]) == 4  # 2D case
        println("  Processing 2D correlation matrices")
        stacked_corr = cat(corr_mats..., dims=5)
        avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
    else
        error("Unsupported correlation matrix dimensions: $(ndims(corr_mats[1]))")
    end
    
    println("Creating aggregated dummy state...")
    dummy_state = FP.setDummyState(states[1], avg_dists, avg_corr, total_t)

    dummy_state_save_dir = "dummy_states_agg"
    println("Saving aggregated state to: $dummy_state_save_dir")
    
    saved_filename = save_state(dummy_state, params[1], dummy_state_save_dir)
    println("✓ SUCCESS: Aggregated state saved as: $saved_filename")
    println("Aggregation completed successfully!")
end

main()
