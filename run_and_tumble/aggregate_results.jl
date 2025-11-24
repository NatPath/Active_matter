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

function main()
    println("Starting aggregation of simulation results...")
    
    args = parse_commandline()
    
    println("Processing $(length(args["saved_states"])) state files...")
    
    params = []
    total_t = 0
    total_weight = 0.0
    state_count = 0
    reference_state = nothing
    ρ_avg_sum = nothing
    cut_sums = Dict{Symbol,Array{Float64}}()
    
    for (i, saved_state) in enumerate(args["saved_states"])
        println("  Loading file $i/$(length(args["saved_states"])): $(basename(saved_state))")
        try
            @load saved_state state param potential
            params = [params; param]
            total_t += state.t
            if reference_state === nothing
                reference_state = state
                ρ_avg_sum = zeros(size(state.ρ_avg))
                for (key, arr) in state.ρ_matrix_avg_cuts
                    cut_sums[key] = zeros(size(arr))
                end
            end
            weight = max(state.t, 1)
            total_weight += weight
            ρ_avg_sum .+= state.ρ_avg .* weight
            for (key, arr) in state.ρ_matrix_avg_cuts
                cut_sums[key] .+= arr .* weight
            end
            state_count += 1
        catch e
            println("  ERROR: Failed to load $saved_state: $e")
            continue
        end
    end
    
    if state_count == 0 || reference_state === nothing
        println("ERROR: No valid states loaded for aggregation")
        exit(1)
    end
    
    println("Successfully loaded $state_count states")
    println("Total simulation time: $total_t")
    
    println("Aggregating correlation matrices and densities...")
    avg_ρ = ρ_avg_sum ./ total_weight
    aggregated_cuts = Dict{Symbol,Array{Float64}}()
    for (key, arr_sum) in cut_sums
        aggregated_cuts[key] = arr_sum ./ total_weight
    end
    
    println("Creating aggregated dummy state...")
    dummy_state = FP.setDummyState(reference_state, avg_ρ, aggregated_cuts, total_t)

    dummy_state_save_dir = "dummy_states_agg"
    println("Saving aggregated state to: $dummy_state_save_dir")
    
    saved_filename = save_state(dummy_state, params[1], dummy_state_save_dir)
    println("✓ SUCCESS: Aggregated state saved as: $saved_filename")
    println("Aggregation completed successfully!")
end

main()
