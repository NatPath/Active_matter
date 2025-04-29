############### run_and_tumble.jl ###############

using Distributed
using Printf
using Dates
using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
using LinearAlgebra
using JLD2
using YAML
using ArgParse

# First, on the main process, include all necessary files.
include("potentials.jl")
include("modules_run_and_tumble.jl")
include("plot_utils.jl")
include("save_utils.jl")
using .FP
using .PlotUtils
using .SaveUtils

# Ensure all worker processes load the same files and modules.
@everywhere begin
    using Printf, Dates, Plots, Random, FFTW, ProgressMeter, Statistics, LsqFit, LinearAlgebra, JLD2, YAML, ArgParse
    include("potentials.jl")
    include("modules_run_and_tumble.jl")
    using .FP
    include("plot_utils.jl")
    using .PlotUtils
    # include("save_utils.jl")
    # using .SaveUtils
end

# Command-line parser with extended options for parallel runs.
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--config"
            help = "Configuration file path"
            required = false
        "--continue"
            help = "Path to saved state file to continue from"
            required = false
        "--continue_sweeps"
            help = "Number of sweeps to continue for (overrides config file)"
            arg_type = Int
            required = false
        "--num_runs"
            help = "Number of independent simulation runs to execute in parallel"
            arg_type = Int
            required = false
            default = 1
    end
    return parse_args(s)
end
# Estimate the total run time by running a sample of sweeps.
@everywhere function estimate_run_time(state, param, n_sweeps, rng; sample_size=1000)
    println("Estimating run time using $sample_size sweeps...")
    # Clone the current state so that our sample does not affect the actual simulation.
    state_copy = deepcopy(state)
    t0 = time()
    for i in 1:sample_size
         FP.update!(param, state_copy, rng)
    end
    elapsed = time() - t0
    avg_time = elapsed / sample_size
    estimated_total = avg_time * n_sweeps
    println("Average time per sweep: $(round(avg_time, digits=4)) sec")
    println("Estimated total run time for $n_sweeps sweeps: $(round(estimated_total, digits=2)) sec")
    flush(stdout)
    return estimated_total
end

# Default parameters (used when no config file is provided)
@everywhere function get_default_params()
    L= 64 
    return Dict(
        "dim_num" => 1,
        "D" => 1.0,
        "α" => 0.0,
        "L" => L,
        "N" => L*100,
        "T" => 1.0,
        "γ′" => 1,
        "ϵ" => 0.0,
        "n_sweeps" => 1*10^3,
        "potential_type" => "zero",
        "fluctuation_type" => "independent-points-discrete_with_0",
        "potential_magnitude" => 16,
        "save_dir" => "saved_states",
        "show_times" => [j*10^i for i in 3:12 for j in 1:9],
        "save_times" => [j*10^i for i in 6:12 for j in 1:9]
    )
end


# Function to set up and run one independent simulation.
@everywhere function run_one_simulation_from_state(param, state, seed,n_sweeps)
    println("Continuing simulation with seed $seed for $n_sweeps")
    rng = MersenneTwister(seed)
    # defaults = get_default_params()
    # if haskey(args, "config") && !isnothing(args["config"])
    #     println("Using configuration from file: $(args["config"])")
    #     params = TOML.parsefile(args["config"])
    # else
    #     println("No config file provided. Using default parameters.")
    #     params = get_default_params()
    # end
    # # For parallel runs, disable any visualization or state saving I/O.
    # n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
    show_times = Int[]
    save_times = Int[]
    
    # # Set up simulation parameters.
    # dim_num            = get(params, "dim_num", defaults["dim_num"])
    # potential_type     = get(params, "potential_type", defaults["potential_type"])
    # fluctuation_type   = get(params, "fluctuation_type", defaults["fluctuation_type"])
    # potential_magnitude= get(params, "potential_magnitude", defaults["potential_magnitude"])
    # D                  = get(params, "D", defaults["D"])
    # α                  = get(params, "α", defaults["α"])
    # L                  = get(params, "L", defaults["L"])
    # N                  = get(params, "N", defaults["N"])
    # T                  = get(params, "T", defaults["T"])
    # γ′                 = get(params, "γ′", defaults["γ′"])
    # ϵ                  = get(params, "ϵ", defaults["ϵ"])
    
    # dims = ntuple(i -> L, dim_num)
    # ρ₀ = N / L
    # γ = γ′ / N

    # Initialize simulation parameters and state.
    # param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude)
    # v_smudge_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    # potential = Potentials.choose_potential(v_smudge_args, dims; fluctuation_type=fluctuation_type)
    # state = FP.setState(0, rng, param, T, potential)
    dummy_state = setDummyState(state,state.ρ_avg,state.ρ_matrix_avg,state.t)
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=1000)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")
    # Run the simulation (calculating correlations).
    normalized_dist, corr_mat = run_simulation!(dummy_state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=show_times,
                                                 save_times=save_times)
    return normalized_dist, corr_mat, dummy_state, param
end
# A very shitty function, find a cleaner way to do it! this is just a recepie for spagheti
function n_sweeps_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        println("Using configuration from file: $(args["config"])")
        params = YAML.load_file(args["config"])
    else
        println("No config file provided. Using default parameters.")
        params = get_default_params()
    end
    # For parallel runs, disable any visualization or state saving I/O.
    n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
    return n_sweeps
end
@everywhere function run_one_simulation_from_config(args, seed)
    println("Starting simulation with seed $seed")
    rng = MersenneTwister(seed)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        println("Using configuration from file: $(args["config"])")
        params = YAML.load_file(args["config"])
    else
        println("No config file provided. Using default parameters.")
        params = get_default_params()
    end
    # For parallel runs, disable any visualization or state saving I/O.
    n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
    params["show_times"] = Int[]
    params["save_times"] = Int[]
    
    # Set up simulation parameters.
    dim_num            = get(params, "dim_num", defaults["dim_num"])
    potential_type     = get(params, "potential_type", defaults["potential_type"])
    fluctuation_type   = get(params, "fluctuation_type", defaults["fluctuation_type"])
    potential_magnitude= get(params, "potential_magnitude", defaults["potential_magnitude"])
    D                  = get(params, "D", defaults["D"])
    α                  = get(params, "α", defaults["α"])
    L                  = get(params, "L", defaults["L"])
    N                  = get(params, "N", defaults["N"])
    T                  = get(params, "T", defaults["T"])
    γ′                 = get(params, "γ′", defaults["γ′"])
    ϵ                  = get(params, "ϵ", defaults["ϵ"])
    
    dims = ntuple(i -> L, dim_num)
    ρ₀ = N / L
    γ = γ′ / N

    # Initialize simulation parameters and state.
    param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude)
    v_smudge_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_smudge_args, dims; fluctuation_type=fluctuation_type,rng=rng)
    state = FP.setState(0, rng, param, T, potential)
   
    #estimate run time
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=1000)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")

    # Run the simulation (calculating correlations).
    normalized_dist, corr_mat = run_simulation!(state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=params["show_times"],
                                                 save_times=params["save_times"])
    return normalized_dist, corr_mat, state, param
end

# Main function.
function main()
    args = parse_commandline()
    num_runs = get(args, "num_runs", 1)
    seeds = rand(1:2^30,num_runs)
    println("Running $num_runs independent simulations in parallel.")
    if num_runs > 1
        if haskey(args, "continue") && !isnothing(args["continue"])
            # # When continuing from a saved state, parallel runs are not allowed.
            # error("Parallel runs are not supported when continuing from a saved state.")
            println("Continuing from saved aggregation: $(args["continue"])")
            @load args["continue"] state param potential
            if haskey(args, "config") && !isnothing(args["config"])
                println("Using configuration from file: $(args["config"])")
                params = YAML.load_file(args["config"])
                # params = TOML.parsefile(args["config"])
            else
                println("No config file provided. Using default parameters.")
                params = get_default_params()
            end
            defaults = get_default_params()
            if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
                n_sweeps = args["continue_sweeps"]
                println("Continuing for specified $n_sweeps sweeps")
            else
                n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
                println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
            end
            results = pmap(seed -> run_one_simulation_from_state(param,state,seed,n_sweeps), seeds)
        else
            results = pmap(seed -> run_one_simulation_from_config(args, seed), seeds)
            n_sweeps=n_sweeps_from_args(args)
        end
        # Use different seeds for each independent simulation.
        # Aggregate results (here we average the correlation matrices).
        normalized_dists = [res[1] for res in results]
        corr_mats = [res[2] for res in results]
        states = [res[3] for res in results]
        params = [res[4] for res in results]
        
        #avg_corr = mean(corr_mats, dims=1)
        #avg_dists = mean(normalized_dists,dims=1)
        stacked_corr = cat(corr_mats..., dims=3)  # Stack matrices along a new third dimension
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)  # Average over the third dimension and drop it
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
        total_t = n_sweeps*(num_runs-1)+states[1].t 
        dummy_state = FP.setDummyState(states[1],avg_dists,avg_corr,total_t)

        dummy_state_save_dir = "dummy_states"
        save_state(dummy_state,params[1],dummy_state_save_dir)

        
        # Save aggregated results to a separate file.
        save_dir = "saved_states_parallel"
        mkpath(save_dir)
        now_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        filename = @sprintf("%s/parallel_results_%s.jld2", save_dir, now_str)
        @save filename normalized_dists corr_mats avg_corr avg_dists
        println("Parallel results saved to: $filename")
    else
        # Single-run mode.
        if haskey(args, "continue") && !isnothing(args["continue"])
            println("Continuing from saved state: $(args["continue"])")
            @load args["continue"] state param potential
            if haskey(args, "config") && !isnothing(args["config"])
                println("Using configuration from file: $(args["config"])")
                #params = TOML.parsefile(args["config"])
                params = YAML.load_file(args["config"])
            else
                println("No config file provided. Using default parameters.")
                params = get_default_params()
            end
            defaults = get_default_params()
            if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
                n_sweeps = args["continue_sweeps"]
                println("Continuing for specified $n_sweeps sweeps")
            else
                n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
                println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
            end
            seed = rand(1:2^30)
            rng = MersenneTwister(seed)
        else
            if haskey(args, "config") && !isnothing(args["config"])
                println("Using configuration from file: $(args["config"])")
                # params = TOML.parsefile(args["config"])
                params = YAML.load_file(args["config"])
            else
                println("No config file provided. Using default parameters.")
                params = get_default_params()
            end
            defaults = get_default_params()
            dim_num            = get(params, "dim_num", defaults["dim_num"])
            potential_type     = get(params, "potential_type", defaults["potential_type"])
            fluctuation_type   = get(params, "fluctuation_type", defaults["fluctuation_type"])
            potential_magnitude= get(params, "potential_magnitude", defaults["potential_magnitude"])
            D                  = get(params, "D", defaults["D"])
            α                  = get(params, "α", defaults["α"])
            L                  = get(params, "L", defaults["L"])
            N                  = get(params, "N", defaults["N"])
            T                  = get(params, "T", defaults["T"])
            γ′                 = get(params, "γ′", defaults["γ′"])
            ϵ                  = get(params, "ϵ", defaults["ϵ"])
            n_sweeps           = get(params, "n_sweeps", defaults["n_sweeps"])
            
            dims = ntuple(i -> L, dim_num)
            ρ₀ = N / L
            γ = γ′ / N
            
            param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude)
            v_smudge_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
            seed = rand(1:2^30)
            #rng = MersenneTwister(123)
            rng = MersenneTwister(seed)
            potential = Potentials.choose_potential(v_smudge_args, dims; fluctuation_type=fluctuation_type,rng=rng,plot_flag=true)
            state = FP.setState(0, rng, param, T, potential)
        end
        
        show_times = get(params, "show_times", get_default_params()["show_times"])
        save_times = get(params, "save_times", get_default_params()["save_times"])
        
        # Register an exit hook to save state at exit.
        atexit() do
            println("\nSaving current state...")
            try
                save_dir = get(params, "save_dir", get_default_params()["save_dir"])
                SaveUtils.save_state(state, param, save_dir)
                println("State saved successfully")
            catch e
                println("Error saving state: ", e)
            end
        end

        Base.exit_on_sigint(false)
        Base.sigatomic_begin()
        ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)
        Base.sigatomic_end()
        
        try
            res_dist, corr_mat = run_simulation!(state, param, n_sweeps, rng;
                                                 show_times=show_times,
                                                 save_times=save_times,
                                                 plot_flag=true)
        catch e
            if isa(e, InterruptException)
                println("\nInterrupt detected, initiating cleanup...")
                exit()
            else
                rethrow(e)
            end
        end
        
        save_dir = get(params, "save_dir", get_default_params()["save_dir"])
        filename = save_state(state, param, save_dir)
        println("Final state saved to: ", filename)
   end
end

main()
