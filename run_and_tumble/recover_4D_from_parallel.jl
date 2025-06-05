using JLD2
using ArgParse
using Statistics
using Random
include("modules_run_and_tumble.jl")
include("save_utils.jl")
include("potentials.jl")
using .FP
using .SaveUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "parallel_results_file"
            help = "Path to the parallel results .jld2 file"
            required = true
        "--output_dir"
            help = "Output directory for the recovered dummy state"
            default = "recovered_states"
    end
    return parse_args(s)
end

function recover_4D_correlation(normalized_dists, corr_mats, param)
    """
    Attempt to recover proper 4D correlation tensor from parallel results
    """
    println("Original correlation matrix dimensions: ", size(corr_mats[1]))
    println("Number of parallel results: ", length(corr_mats))
    
    # Check what we're dealing with
    corr_dims = ndims(corr_mats[1])
    dist_dims = ndims(normalized_dists[1])
    
    println("Correlation tensor has $corr_dims dimensions")
    println("Density arrays have $dist_dims dimensions") 
    
    # Aggregate the results properly based on dimensions
    if corr_dims == 2 && dist_dims == 1
        # 1D case - this should work fine
        println("Detected 1D case")
        stacked_corr = cat(corr_mats..., dims=3)
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
        
    elseif corr_dims == 4 && dist_dims == 2
        # 2D case - this is what we want
        println("Detected proper 2D case with 4D correlation tensor")
        stacked_corr = cat(corr_mats..., dims=5)
        avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
    elseif corr_dims == 3 && dist_dims == 2
        # This is likely the problematic case from cluster
        println("Detected problematic 2D case with 3D correlation tensor")
        println("Correlation tensor size: ", size(corr_mats[1]))
        
        # We can't fully recover the 4D tensor, but we can at least aggregate properly
        # and flag this as problematic
        stacked_corr = cat(corr_mats..., dims=4)
        avg_corr = dropdims(mean(stacked_corr, dims=4), dims=4)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
        println("WARNING: Cannot recover full 4D correlation tensor from 3D data!")
        println("The aggregated result will have reduced correlation information.")
        
    else
        error("Unexpected dimensions: corr_dims=$corr_dims, dist_dims=$dist_dims")
    end
    
    println("Final averaged correlation dimensions: ", size(avg_corr))
    println("Final averaged density dimensions: ", size(avg_dists))
    
    return avg_dists, avg_corr
end

function main()
    args = parse_commandline()
    
    println("Loading parallel results from: $(args["parallel_results_file"])")
    @load args["parallel_results_file"] normalized_dists corr_mats avg_corr avg_dists
    
    println("Analyzing loaded data...")
    println("Number of runs: ", length(normalized_dists))
    println("Density dimensions: ", size(normalized_dists[1]))
    println("Correlation dimensions: ", size(corr_mats[1]))
    
    # Load parameters from your saved dummy state file
    dummy_state_file = "dummy_states/passive_case/2D_potential-xy_slides_Vscale-16.0_fluctuation-profile_switch_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gammap-32.00_D-1.0_t-9000000.jld2"
    
    if isfile(dummy_state_file)
        println("Loading parameters from: $dummy_state_file")
        @load dummy_state_file state param potential
        println("Loaded param: L=$(param.dims), α=$(param.α), γ′=$(param.γ*param.N)")
    else
        println("Warning: Could not find dummy state file, using default parameters")
        # Fallback to creating minimal param structure
        L = size(normalized_dists[1], 1)  # Assume square lattice
        dims = (L, L)
        
        param = FP.setParam(
            0.0,    # α
            1.0,    # γ 
            0.0,    # ϵ
            dims,   # dims
            100.0,  # ρ₀
            1.0,    # D
            "xy_slides",  # potential_type
            "profile_switch",  # fluctuation_type
            16.0    # potential_magnitude
        )
        
        # Create reference state and potential
        rng = MersenneTwister(123)
        v_args = Potentials.potential_args("xy_slides", dims; magnitude=16.0)
        potential = Potentials.choose_potential(v_args, dims; fluctuation_type="profile_switch", rng=rng)
        reference_state = FP.setState(0, rng, param, 1.0, potential)
        state = reference_state
    end
    
    # Recover the correlation tensor
    recovered_dists, recovered_corr = recover_4D_correlation(normalized_dists, corr_mats, param)
    
    # Create dummy state with recovered data
    # Use the number of steps from the state struct
    nsteps = state.t
    dummy_state = FP.setDummyState(state, recovered_dists, recovered_corr, nsteps)
    
    # Save the recovered dummy state
    output_dir = args["output_dir"]
    filename = save_state(dummy_state, param, output_dir)
    println("Recovered dummy state saved to: $filename")
    
    # Print diagnostic information
    println("\n=== RECOVERY DIAGNOSTIC ===")
    println("Original avg_corr dimensions: ", size(avg_corr))
    println("Recovered correlation dimensions: ", size(recovered_corr))
    println("Is 4D tensor recovered: ", ndims(recovered_corr) == 4)
    
    if ndims(recovered_corr) == 4
        println("SUCCESS: 4D correlation tensor recovered!")
    else
        println("WARNING: Could not recover 4D tensor. Data may be incomplete.")
    end
end

main()
