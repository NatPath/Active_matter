using Plots
using Random
using FFTW
using ProgressMeter
using Statistics
using LsqFit 
using LinearAlgebra
import Printf.@sprintf
using JLD2
using TOML
using ArgParse

include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "Configuration file path"
            required = false
        "--continue"
            help = "Path to saved state file to continue from"
            required = false
        "--continue-sweeps"
            help = "Number of sweeps to continue for (overrides config file)"
            arg_type = Int
            required = false
    end

    return parse_args(s)
end

function get_default_params()
    # Default parameters when no config file is provided
    return Dict(
        "dim_num" => 1,
        "D" => 1.0,
        "α" => 0.0,
        "L" => 64,
        "N" => 64*100,
        "T" => 1.0,
        "β′" => 0.1,
        "n_sweeps" => 10^6,
        "potential_type" => "smudge",
        "fluctuation_type" => "zero-potential",
        "potential_magnitude" => 2,
        "save_dir" => "saved_states",
        "show_times" => [j*10^i for i in 3:12 for j in 1:9],  # Default visualization times
        "save_times" => Int[10^6]                 # Empty by default
        
    )
end

function save_state(state,param,save_dir)
    mkpath(save_dir)
    β′ = param.β*param.N
    filename = @sprintf("%s/potential-%s_fluctuation-%s_L-%d_rho-%.1e_alpha-%.2f_betaprime-%.2f_D-%.1f_t-%d.jld2",
        save_dir,
        param.potential_type,
        param.fluctuation_type,
        param.dims[1],    # System size
        param.ρ₀,         # Density
        param.α,          # Tumbling rate
        β′,          # Potential fluctuation rate
        param.D,          # Diffusion coefficient
        state.t          # Final time
    )
    potential = state.potential 
    @save filename state param potential
    return filename
end

function main()
    rng = MersenneTwister(123)
    # Parse command line arguments
    args = parse_commandline()
    
    # First check if we're continuing from a saved state
    if haskey(args, "continue") && !isnothing(args["continue"])
        println("Continuing from saved state: $(args["continue"])")
        @load args["continue"] state param potential
        
        # Load configuration for remaining parameters
        if haskey(args, "config") && !isnothing(args["config"])
            println("Using configuration from file: $(args["config"])")
            params = TOML.parsefile(args["config"])
        else
            println("No config file provided. Using default parameters.")
            params = get_default_params()
        end
        
        # Check if continue-sweeps was provided, otherwise use n_sweeps from config/defaults
        defaults = get_default_params()
        if haskey(args, "continue-sweeps") && !isnothing(args["continue-sweeps"])
            n_sweeps = args["continue-sweeps"]
            println("Continuing for specified $n_sweeps sweeps")
        else
            n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
            println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
        end
        
    else
        # Original initialization code
        if haskey(args, "config") && !isnothing(args["config"])
            println("Using configuration from file: $(args["config"])")
            params = TOML.parsefile(args["config"])
        else
            println("No config file provided. Using default parameters.")
            params = get_default_params()
        end
        
        # Extract parameters with fallback to defaults for missing values
        defaults = get_default_params()
        dim_num = get(params, "dim_num", defaults["dim_num"])
        potential_type = get(params, "potential_type", defaults["potential_type"])
        fluctuation_type = get(params, "fluctuation_type", defaults["fluctuation_type"])
        potential_magnitude = get(params, "potential_magnitude", defaults["potential_magnitude"])
        D = get(params, "D", defaults["D"])
        α = get(params, "α", defaults["α"])
        L = get(params, "L", defaults["L"])
        N = get(params, "N", defaults["N"])
        T = get(params, "T", defaults["T"])
        β′ = get(params, "β′", defaults["β′"])
        n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
        
        # Initialize new simulation
        dims = ntuple(i->L, dim_num)
        ρ₀ = N/L
        β = β′/N
        
        param = FP.setParam(α, β, dims, ρ₀, D, potential_type,fluctuation_type, potential_magnitude)
        v_smudge_args = Potentials.potential_args(
            potential_type,
            dims;   
            magnitude= potential_magnitude
        )
        potential = Potentials.choose_potential(v_smudge_args,dims;fluctuation_type=fluctuation_type)
        state = FP.setState(0, rng, param, T, potential)
    end

    show_times = get(params, "show_times", defaults["show_times"])
    save_times = get(params, "save_times", defaults["save_times"])
    β′ = get(params, "β′", defaults["β′"])
    # Make movie
    #make_movie!(state, param, n_frame, rng, file_name, in_fps, show_directions=false, show_times=show_times, save_times=save_times)

    # Register the cleanup function to run at exit
    atexit() do
        println("\nSaving current state...")
        try
            save_dir = "saved_states"
            save_state(state, param, save_dir)
            println("State saved successfully")
        catch e
            println("Error saving state: ", e)
        end
    end

    # Add specific interrupt handler
    Base.exit_on_sigint(false)  # Don't exit immediately on Ctrl-C
    Base.sigatomic_begin()
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)  # Disable Julia's default Ctrl-C handler
    Base.sigatomic_end()

    try
        # Run the simulation
        res_dist, corr_mat = run_simulation!(state, param, n_sweeps, rng;
                                           show_times=show_times,
                                           save_times=save_times)
    catch e
        if isa(e, InterruptException)
            println("\nInterrupt detected, initiating cleanup...")
            exit()  # This will trigger atexit
        else
            rethrow(e)
        end
    end

    # Normal exit save (this will still happen even if interrupted)
    save_dir = "saved_states"
    save_dir = get(params, "save_dir", defaults["save_dir"])
    filename = save_state(state,param,save_dir)
    println("Final state saved to: ", filename)

end

# Run the main function
main()

