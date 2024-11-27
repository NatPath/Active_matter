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

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--config"
            help = "Configuration file path"
            required = false
        "--continue"
            help = "Path to saved state file to continue from"
            required = false
    end

    return parse_args(s)
end

function get_default_params()
    # Default parameters when no config file is provided
    return Dict(
        "dim_num" => 1,
        "D" => 0.0,
        "α" => 0.3,
        "L" => 64,
        "N" => 6400,
        "T" => 1.0,
        "β′" => 0.03,
        "n_sweeps" => 10^2,
        "potential_type" => "smudge",
        "potential_magnitude" => 0.4,
        "save_dir" => "saved_states",
        "show_times" => [10^i for i in 1:5],  # Default visualization times
        "save_times" => Int[]                 # Empty by default
    )
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
        
        # Only use n_sweeps from config/defaults
        defaults = get_default_params()
        n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
        
        println("Continuing simulation for $n_sweeps more sweeps")
        
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
        β = β′/(ρ₀*L)
        
        param = FP.setParam(α, β, dims, ρ₀, D, potential_type, potential_magnitude)
        v_smudge_args = Potentials.potential_args(
            potential_type,
            dims;   
            magnitude= potential_magnitude
        )
        potential = Potentials.choose_potential(v_smudge_args,dims)
        state = FP.setState(0, rng, param, T, potential)
    end

    # Common simulation code
    show_times = get(params, "show_times", defaults["show_times"])
    save_times = get(params, "save_times", defaults["save_times"])
    # Make movie
    #make_movie!(state, param, t_gap, n_frame, rng, file_name, in_fps, show_directions=false, show_times=show_times, save_times=save_times)

    res_dist, corr_mat = run_simulation!(state, param, 1, n_sweeps, rng;
                                       show_times=show_times,
                                       save_times=save_times)

    # Save final state
    save_dir = "saved_states"
    mkpath(save_dir)
    filename = @sprintf("%s/potential-%s_L-%d_rho-%.1e_alpha-%.2f_beta-%.2f_D-%.1f_t-%.1e.jld2",
        save_dir,
        param.potential_type,
        param.dims[1],    # System size
        param.ρ₀,         # Density
        param.α,          # Tumbling rate
        param.β,          # Potential fluctuation rate
        param.D,          # Diffusion coefficient
        state.t          # Final time
    )
    
    @save filename state param potential
    println("Final state saved to: ", filename)

end

# Run the main function
main()

