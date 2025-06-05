using JLD2
using ArgParse
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
# function aggregate_results(simulation_results...)
#     # Initialize an empty aggregated state
#     aggregated_state = []

#     # Iterate over each simulation result
#     for result in simulation_results
#         # Update the aggregated state
#         if isempty(aggregated_state)
#             aggregated_state = result
#         else
#             aggregated_state .+= result
#         end
#     end

#     # Calculate the averages of the attributes
#     aggregated_state ./= length(simulation_results)

#     # Return the aggregated state
#     return aggregated_state
# end

function main()
    args = parse_commandline()
    states = []
    params = []
    normalized_dists = []
    corr_mats = []
    stats_arr = []
    total_t=0
    for saved_state in args["saved_states"]
        @load saved_state state param potential
        states = [states; state]
        params = [params; param]
        total_t+=state.t
        stats = calculate_statistics(state)
        stats_arr = [stats_arr ; stats]
    end
    normalized_dists = [stats[1] for stats in stats_arr]
    corr_mats = [stats[2] for stats in stats_arr]

    # Handle different dimensions for stacking
    if ndims(corr_mats[1]) == 2  # 1D case
        stacked_corr = cat(corr_mats..., dims=3)  # Stack matrices along a new third dimension
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)  # Average over the third dimension and drop it
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
    elseif ndims(corr_mats[1]) == 4  # 2D case
        stacked_corr = cat(corr_mats..., dims=5)  # Stack 4D tensors along 5th dimension
        avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)  # Average over 5th dimension
        stacked_dists = cat(normalized_dists..., dims=3)  # Stack 2D matrices along 3rd dimension
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)  # Average over 3rd dimension
    else
        error("Unsupported correlation matrix dimensions: $(ndims(corr_mats[1]))")
    end
    
    dummy_state = FP.setDummyState(states[1],avg_dists,avg_corr,total_t)

    dummy_state_save_dir = "dummy_states_agg"
    save_state(dummy_state,params[1],dummy_state_save_dir)

end
main()
