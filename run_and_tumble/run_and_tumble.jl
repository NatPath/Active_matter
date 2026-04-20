############### run_and_tumble.jl ###############

using Distributed
using Printf
using Dates
if haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
    # Under WSLg, Qt/GR can pick Wayland and then fail EGL initialization.
    # Force the stable XWayland path before Plots/GR initialize.
    ENV["QT_QPA_PLATFORM"] = "xcb"
end
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
# using BenchmarkTools

const SRC_ROOT = joinpath(@__DIR__, "src")
const COMMON_DIR = joinpath(SRC_ROOT, "common")
const RUN_AND_TUMBLE_DIR = joinpath(SRC_ROOT, "run_and_tumble")

# First, on the main process, include all necessary files.
include(joinpath(COMMON_DIR, "potentials.jl"))
include(joinpath(RUN_AND_TUMBLE_DIR, "modules_run_and_tumble.jl"))
include(joinpath(COMMON_DIR, "plot_utils.jl"))
include(joinpath(COMMON_DIR, "save_utils.jl"))
using .FP
using .PlotUtils
using .SaveUtils

# Ensure all worker processes load the same files and modules.
@everywhere const RUN_AND_TUMBLE_COMMON_DIR_WORKER = $COMMON_DIR
@everywhere const RUN_AND_TUMBLE_MODULE_DIR_WORKER = $RUN_AND_TUMBLE_DIR
@everywhere begin
    using Printf, Dates, Random, FFTW, ProgressMeter, Statistics, LsqFit, LinearAlgebra, JLD2, YAML, ArgParse
    if haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
        ENV["QT_QPA_PLATFORM"] = "xcb"
    end
    using Plots
    include(joinpath(RUN_AND_TUMBLE_COMMON_DIR_WORKER, "potentials.jl"))
    include(joinpath(RUN_AND_TUMBLE_MODULE_DIR_WORKER, "modules_run_and_tumble.jl"))
    using .FP
    include(joinpath(RUN_AND_TUMBLE_COMMON_DIR_WORKER, "plot_utils.jl"))
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
        "--initial_state"
            help = "Path to saved state file to start from as t=0 (statistics reset)"
            required = false
        "--int_type"
            help = "Integer type for positions/densities (e.g., Int16, Int32, Int64)"
            arg_type = String
            required = false
        "--estimate_only"
            help = "Estimate runtime and exit without running the simulation (single-run mode)"
            action = :store_true
        "--estimate_sample_size"
            help = "Number of sample sweeps to use for runtime estimation"
            arg_type = Int
            default = 100
            required = false
    end
    return parse_args(s)
end

@everywhere const INT_TYPE_MAP = Dict(
    "Int8" => Int8,
    "Int16" => Int16,
    "Int32" => Int32,
    "Int64" => Int64,
    "UInt8" => UInt8,
    "UInt16" => UInt16,
    "UInt32" => UInt32,
    "UInt64" => UInt64,
)

@everywhere function parse_int_type(value)
    if value isa DataType && value <: Integer
        return value
    elseif value isa Symbol
        return parse_int_type(String(value))
    elseif value isa AbstractString
        if haskey(INT_TYPE_MAP, value)
            return INT_TYPE_MAP[value]
        end
        error("Unsupported int_type: $value. Use one of $(collect(keys(INT_TYPE_MAP))).")
    else
        error("Unsupported int_type: $value. Use a string like \"Int32\".")
    end
end

@everywhere function resolve_int_type(args, params, defaults)
    if haskey(args, "int_type") && !isnothing(args["int_type"])
        return parse_int_type(args["int_type"])
    end
    return parse_int_type(get(params, "int_type", defaults["int_type"]))
end
# Load a saved state and reset its statistics so it can be used as a fresh initial condition.
function load_initial_state(path)
    println("Loading initial state from $path")
    @load path state param potential
    FP.reset_statistics!(state)
    state.potential = potential
    return state, param
end
# Estimate the total run time by running a sample of sweeps.
@everywhere function estimate_run_time(state, param, n_sweeps, rng; sample_size=100)
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
    L= 16  
    d = 2
    return Dict(
        "dim_num" => d,
        "D" => 1.0,
        "α" => 0.0,
        "L" => L,
        "ρ₀" => 100, # particles per site
        "T" => 1.0,
        "γ" => 0.0,
        "ϵ" => 0,
        "n_sweeps" => 10^6,
        "description" => "",
        # "potential_type" => "well",
        # "fluctuation_type" => "reflection",
        "potential_type" => "zero",
        "fluctuation_type" => "no-fluctuation",
        "potential_magnitude" => 0.0,
        "save_dir" => "saved_states",
        "show_times" => [j*10^i for i in 0:12 for j in 1:9],
        # "show_times" => [i for i in 1:1:1000000],
        "save_times" => [j*10^i for i in 6:12 for j in 1:9],
        "forcing_type" => "center_bond_x",
        "forcing_types" => ["center_bond_x"],
        "ffr" => 1.0,
        "ffrs" => [1.0],
        "forcing_fluctuation_type" => "alternating_direction",
        "forcing_magnitude" => 1.0,
        "forcing_magnitudes" => [1.0],
        "forcing_direction_flags" => [true],
        "forcing_rate_scheme" => "legacy_penalty",
        "bond_pass_count_mode" => "nonzero_magnitude",
        "ic" => "random",
        "int_type" => "Int32",
    )
end

@everywhere function to_string_vector(value, key_name::String)
    if value isa AbstractString
        return [String(value)]
    elseif value isa AbstractVector
        return [String(v) for v in value]
    end
    error("$key_name must be a string or a list of strings.")
end

@everywhere function to_float_vector(value, key_name::String)
    if value isa Number
        return [Float64(value)]
    elseif value isa AbstractVector
        return [Float64(v) for v in value]
    end
    error("$key_name must be a number or a list of numbers.")
end

@everywhere function to_bool_vector(value, key_name::String)
    if value isa Bool
        return [value]
    elseif value isa Number
        return [value != 0]
    elseif value isa AbstractVector
        flags = Bool[]
        for v in value
            if v isa Bool
                push!(flags, v)
            elseif v isa Number
                push!(flags, v != 0)
            elseif v isa AbstractString
                lv = lowercase(strip(v))
                if lv in ("true", "t", "1", "yes", "y")
                    push!(flags, true)
                elseif lv in ("false", "f", "0", "no", "n")
                    push!(flags, false)
                else
                    error("$key_name has an invalid boolean value: $v")
                end
            else
                error("$key_name contains unsupported value type: $(typeof(v))")
            end
        end
        return flags
    end
    error("$key_name must be a boolean or a list of booleans.")
end

@everywhere function expand_to_length(values::Vector{T}, n::Int, key_name::String) where {T}
    if length(values) == n
        return values
    elseif length(values) == 1 && n > 1
        return fill(values[1], n)
    end
    error("$key_name must have length 1 or length $n.")
end

@everywhere function param_ffr_input(param)
    if hasfield(typeof(param), :ffr)
        return getfield(param, :ffr)
    elseif hasfield(typeof(param), :ffrs)
        return getfield(param, :ffrs)
    end
    return 0.0
end

@everywhere function maybe_override_forcing_rate_scheme(param, args, params)
    has_explicit_config = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "forcing_rate_scheme")
    if !has_explicit_config
        return param
    end
    return FP.setParam(param.α, param.γ, param.ϵ, param.dims, param.ρ₀, param.D,
                       param.potential_type, param.fluctuation_type, param.potential_magnitude,
                       param_ffr_input(param);
                       forcing_rate_scheme=String(params["forcing_rate_scheme"]))
end

@everywhere function forcing_bond_indices_from_type(forcing_type::AbstractString, dim_num::Int, L::Int)
    if forcing_type == "center_bond_x"
        if dim_num == 1
            return ([L ÷ 2], [L ÷ 2 + 1])
        elseif dim_num == 2
            return ([L ÷ 2, L ÷ 2], [L ÷ 2 + 1, L ÷ 2])
        end
    elseif forcing_type == "center_bond_y"
        if dim_num == 1
            return ([L ÷ 2], [L ÷ 2 + 1])
        elseif dim_num == 2
            return ([L ÷ 2, L ÷ 2], [L ÷ 2, L ÷ 2 + 1])
        end
    end
    error("Unsupported forcing type: $forcing_type")
end

@everywhere function parse_forcing_bond_pair(raw_pair, dim_num::Int, L::Int)
    if !(raw_pair isa AbstractVector) || length(raw_pair) != 2
        error("Each entry in forcing_bond_pairs must contain exactly two bond endpoints.")
    end
    if dim_num == 1
        i = mod1(Int(raw_pair[1]), L)
        j = mod1(Int(raw_pair[2]), L)
        return ([i], [j])
    end

    first_endpoint = raw_pair[1]
    second_endpoint = raw_pair[2]
    if !(first_endpoint isa AbstractVector) || !(second_endpoint isa AbstractVector)
        error("For dim_num=$dim_num, each forcing_bond_pairs entry must look like [[x1,...],[x2,...]].")
    end
    if length(first_endpoint) != dim_num || length(second_endpoint) != dim_num
        error("For dim_num=$dim_num, each endpoint must have exactly $dim_num coordinates.")
    end

    b1 = [mod1(Int(coord), L) for coord in first_endpoint]
    b2 = [mod1(Int(coord), L) for coord in second_endpoint]
    return (b1, b2)
end

@everywhere function build_forcings_and_ffrs(params, defaults, dim_num::Int, L::Int)
    bond_indices_list = Tuple{Vector{Int}, Vector{Int}}[]

    if haskey(params, "forcing_bond_pairs") && haskey(params, "forcing_distance_d")
        error("Use either forcing_bond_pairs or forcing_distance_d, not both.")
    elseif haskey(params, "forcing_bond_pairs")
        raw_pairs = params["forcing_bond_pairs"]
        if !(raw_pairs isa AbstractVector)
            error("forcing_bond_pairs must be a list of bond pairs.")
        end
        for raw_pair in raw_pairs
            push!(bond_indices_list, parse_forcing_bond_pair(raw_pair, dim_num, L))
        end
    elseif haskey(params, "forcing_distance_d")
        if dim_num != 1
            error("forcing_distance_d is currently supported only for dim_num=1.")
        end
        d_raw = params["forcing_distance_d"]
        if !(d_raw isa Number)
            error("forcing_distance_d must be numeric.")
        end
        d = mod(Int(round(Float64(d_raw))), L)

        base_site = if haskey(params, "forcing_base_site")
            base_site_raw = params["forcing_base_site"]
            if !(base_site_raw isa Number)
                error("forcing_base_site must be numeric.")
            end
            mod1(Int(round(Float64(base_site_raw))), L)
        else
            # Default: center the midpoint between the two forces on the origin bond.
            # For odd d this can only be approximate due to lattice discretization.
            mod1((L ÷ 2) - fld(d, 2), L)
        end
        second_site = mod1(base_site + d, L)

        push!(bond_indices_list, ([base_site], [mod1(base_site + 1, L)]))
        push!(bond_indices_list, ([second_site], [mod1(second_site + 1, L)]))
    else
        forcing_types_raw = haskey(params, "forcing_types") ? params["forcing_types"] : get(params, "forcing_type", defaults["forcing_type"])
        forcing_types = to_string_vector(forcing_types_raw, "forcing_types")
        for forcing_type in forcing_types
            push!(bond_indices_list, forcing_bond_indices_from_type(forcing_type, dim_num, L))
        end
    end

    n_forces = length(bond_indices_list)
    if n_forces == 0
        return Potentials.BondForce[], Float64[]
    end
    forcing_magnitudes_raw = haskey(params, "forcing_magnitudes") ? params["forcing_magnitudes"] : get(params, "forcing_magnitude", defaults["forcing_magnitude"])
    ffrs_raw = haskey(params, "ffrs") ? params["ffrs"] : get(params, "ffr", defaults["ffr"])
    direction_flags_raw = haskey(params, "forcing_direction_flags") ? params["forcing_direction_flags"] : [true]

    forcing_magnitudes = expand_to_length(to_float_vector(forcing_magnitudes_raw, "forcing_magnitudes"), n_forces, "forcing_magnitudes")
    ffrs = expand_to_length(to_float_vector(ffrs_raw, "ffrs"), n_forces, "ffrs")
    direction_flags = expand_to_length(to_bool_vector(direction_flags_raw, "forcing_direction_flags"), n_forces, "forcing_direction_flags")

    forcings = [Potentials.setBondForce(bond_indices_list[i], direction_flags[i], forcing_magnitudes[i]) for i in 1:n_forces]
    return forcings, ffrs
end

@everywhere function average_array_list(arrays::Vector{<:AbstractArray})
    if isempty(arrays)
        error("Cannot average an empty array list.")
    end
    stack_dim = ndims(arrays[1]) + 1
    stacked = cat(arrays..., dims=stack_dim)
    return dropdims(mean(stacked, dims=stack_dim), dims=stack_dim)
end


# Function to set up and run one independent simulation.
@everywhere function run_one_simulation_from_state(param, state, seed,n_sweeps; relaxed_ic::Bool=false)
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
    dummy_state = FP.setDummyState(state,state.ρ_avg,state.ρ_matrix_avg_cuts,state.t)
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=100)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")
    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(dummy_state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=show_times,
                                                 save_times=save_times,
                                                 relaxed_ic=relaxed_ic)
    return dist, corr_mat_cuts, dummy_state, param
end
# A very shitty function, find a cleaner way to do it! this is just a recipe for spaghetti
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
    int_type = resolve_int_type(args, params, defaults)
    
    # Set up simulation parameters.
    dim_num            = get(params, "dim_num", defaults["dim_num"])
    potential_type     = get(params, "potential_type", defaults["potential_type"])
    fluctuation_type   = get(params, "fluctuation_type", defaults["fluctuation_type"])
    potential_magnitude= get(params, "potential_magnitude", defaults["potential_magnitude"])
    D                  = get(params, "D", defaults["D"])
    α                  = get(params, "α", defaults["α"])
    L                  = get(params, "L", defaults["L"])
    ρ₀              = get(params, "ρ₀", defaults["ρ₀"])
    T                  = get(params, "T", defaults["T"])
    γ                 = get(params, "γ", defaults["γ"])
    ϵ                  = get(params, "ϵ", defaults["ϵ"])
    forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))

    forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)

    ic = get(params, "ic", defaults["ic"])

    dims = ntuple(i -> L, dim_num)
    # ρ₀ = N / (L^dim_num)

    # Initialize simulation parameters and state.
    param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                        forcing_rate_scheme=forcing_rate_scheme)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=false)
    state = FP.setState(0, rng, param, T, potential, forcings; ic=ic, int_type=int_type, bond_pass_count_mode=bond_pass_count_mode)
   
    #estimate run time
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=100)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")

    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=params["show_times"],
                                                 save_times=params["save_times"])
    return dist, corr_mat_cuts, state, param
end

# Main function.
function main()
    args = parse_commandline()
    num_runs = get(args, "num_runs", 1)
    estimate_only = get(args, "estimate_only", false)
    estimate_sample_size = max(get(args, "estimate_sample_size", 100), 1)
    seeds = rand(1:2^30,num_runs)
    println("Running $num_runs independent simulations in parallel.")
    if estimate_only && num_runs != 1
        error("--estimate_only is supported only with --num_runs 1.")
    end
    using_initial_state = haskey(args, "initial_state") && !isnothing(args["initial_state"])
    initial_state = nothing
    initial_param = nothing
    if using_initial_state
        initial_state, initial_param = load_initial_state(args["initial_state"])
    end
    if num_runs > 1
        if haskey(args, "continue") && !isnothing(args["continue"])
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
            param = maybe_override_forcing_rate_scheme(param, args, params)
            defaults = get_default_params()
            if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
                n_sweeps = args["continue_sweeps"]
                println("Continuing for specified $n_sweeps sweeps")
            else
                n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
                println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
            end
            results = pmap(seed -> run_one_simulation_from_state(param,state,seed,n_sweeps), seeds)
        elseif using_initial_state
            defaults = get_default_params()
            if haskey(args, "config") && !isnothing(args["config"])
                params = YAML.load_file(args["config"])
            else
                params = get_default_params()
            end
            initial_param = maybe_override_forcing_rate_scheme(initial_param, args, params)
            n_sweeps = n_sweeps_from_args(args)
            results = pmap(seed -> begin
                    run_state = deepcopy(initial_state)
                    run_param = deepcopy(initial_param)
                    run_one_simulation_from_state(run_param, run_state, seed, n_sweeps; relaxed_ic=true)
                end, seeds)
        else
            results = pmap(seed -> run_one_simulation_from_config(args, seed), seeds)
            n_sweeps=n_sweeps_from_args(args)
        end
        # Use different seeds for each independent simulation.
        # Aggregate results (here we average the correlation matrices).
        dists = [res[1] for res in results]
        mat_cuts = [res[2] for res in results]
        states = [res[3] for res in results]
        params = [res[4] for res in results]
        
        dim_num = length(params[1].dims)
        if dim_num == 1
            full_mats = [mat_cut[:full] for mat_cut in mat_cuts]
            stacked_corr = cat(full_mats..., dims=3)  # Stack matrices along a new third dimension
            avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)  # Average over the third dimension and drop it
            stacked_dists = cat(dists..., dims=2)
            avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
        elseif dim_num == 2
            if haskey(mat_cuts[1],:full)
                full_mats = [mat_cut[:full] for mat_cut in mat_cuts]
                stacked_corr = cat(full_mats..., dims=5)  # Stack 4D tensors along 5th dimension
                avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)  # Average over 5th dimension
            else
                x_cut_mats = [mat_cut[:x_cut] for mat_cut in mat_cuts]
                y_cut_mats = [mat_cut[:y_cut] for mat_cut in mat_cuts]
                diagonal_cut_mats = [mat_cut[:diag_cut] for mat_cut in mat_cuts]
                stacked_corr_x_cut = cat(x_cut_mats..., dims=3)
                stacked_corr_y_cut = cat(y_cut_mats..., dims=3)
                stacked_corr_diagonal_cut = cat(diagonal_cut_mats..., dims=3)
                avg_corr_x_cut = dropdims(mean(stacked_corr_x_cut, dims=3), dims=3)
                avg_corr_y_cut = dropdims(mean(stacked_corr_y_cut, dims=3), dims=3)
                avg_corr_diagonal_cut = dropdims(mean(stacked_corr_diagonal_cut, dims=3), dims=3)
            end
            stacked_dists = cat(dists..., dims=3)  # Stack 2D matrices along 3rd dimension
            avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)  # Average over 3rd dimension
        else
            error("Unsupported correlation matrix dimensions: $(ndims(corr_mats[1]))")
        end
        if haskey(mat_cuts[1],:full)
            mat_cuts_averaged = Dict(:full => avg_corr)
        else
            mat_cuts_averaged = Dict(:x_cut => avg_corr_x_cut, :y_cut => avg_corr_y_cut, :diag_cut => avg_corr_diagonal_cut)
        end

        shared_extra_keys = Set(keys(mat_cuts[1]))
        for mat_cut in mat_cuts[2:end]
            intersect!(shared_extra_keys, Set(keys(mat_cut)))
        end
        for base_key in (:full, :x_cut, :y_cut, :diag_cut)
            delete!(shared_extra_keys, base_key)
        end
        for key in shared_extra_keys
            arrays_to_average = [mat_cut[key] for mat_cut in mat_cuts]
            mat_cuts_averaged[key] = average_array_list(arrays_to_average)
        end
        
        total_t = n_sweeps*(num_runs-1)+states[1].t 
        dummy_state = FP.setDummyState(states[1],avg_dists,mat_cuts_averaged,total_t)

        dummy_state_save_dir = "dummy_states"
        save_state(dummy_state,params[1],dummy_state_save_dir; relaxed_ic=using_initial_state)

        
        # Save aggregated results to a separate file.
        # save_dir = "saved_states_parallel"
        # mkpath(save_dir)
        # now_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        # filename = @sprintf("%s/parallel_results_%s.jld2", save_dir, now_str)
        # @save filename states params
        # println("Parallel results saved to: $filename")
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
            param = maybe_override_forcing_rate_scheme(param, args, params)
            ic = get(params, "ic", get_default_params()["ic"])
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
        elseif using_initial_state
            state = deepcopy(initial_state)
            param = deepcopy(initial_param)
            defaults = get_default_params()
            if haskey(args, "config") && !isnothing(args["config"])
                println("Using configuration from file: $(args["config"])")
                params = YAML.load_file(args["config"])
            else
                println("No config file provided. Using default parameters.")
                params = get_default_params()
            end
            param = maybe_override_forcing_rate_scheme(param, args, params)
            n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
            seed = rand(1:2^30)
            rng = MersenneTwister(seed)
            ic = get(params, "ic", defaults["ic"])
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
            ρ₀                = get(params, "ρ₀", defaults["ρ₀"])
            T                  = get(params, "T", defaults["T"])
            γ                 = get(params, "γ", defaults["γ"])
            ϵ                  = get(params, "ϵ", defaults["ϵ"])
            forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
            bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))
            n_sweeps           = get(params, "n_sweeps", defaults["n_sweeps"])
            forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
            ic = get(params, "ic", defaults["ic"])
            int_type = resolve_int_type(args, params, defaults)

            ic = get(params, "ic", defaults["ic"])
            
            dims = ntuple(i -> L, dim_num)
            # ρ₀ = N / prod(dims)
            
            param = FP.setParam(α, γ, ϵ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                                forcing_rate_scheme=forcing_rate_scheme)
            v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
            seed = rand(1:2^30)
            #rng = MersenneTwister(123)
            rng = MersenneTwister(seed)
            potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type,rng=rng,plot_flag=true)
            
            state = FP.setState(0, rng, param, T, potential, forcings; ic=ic, int_type=int_type, bond_pass_count_mode=bond_pass_count_mode)
        end
        
        defaults = get_default_params()
        description = String(get(params, "description", defaults["description"]))
        show_times = get(params, "show_times", defaults["show_times"])
        save_times = get(params, "save_times", defaults["save_times"])
        save_dir = get(params, "save_dir", defaults["save_dir"])
        progress_file = String(get(params, "progress_file", ""))
        progress_interval_raw = get(params, "progress_interval", 25)
        progress_interval = max(Int(round(Float64(progress_interval_raw))), 1)
        snapshot_request_file = String(get(params, "snapshot_request_file", ""))
        snapshot_tag_prefix = String(get(params, "snapshot_tag_prefix", "snapshot"))

        if estimate_only
            estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
            estimated_time_hours = estimated_time / 3600
            println("Estimated run time for this simulation: $estimated_time_hours hours")
            return
        end
        
        # Register an exit hook to save state at exit.
        final_state_saved = Ref(false)
        atexit() do
            if final_state_saved[]
                return
            end
            println("\nSaving current state...")
            try
                SaveUtils.save_state(state, param, save_dir; ic=ic, relaxed_ic=using_initial_state)
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
            res_dist, corr_mat_cuts = run_simulation!(state, param, n_sweeps, rng;
                                                 show_times=show_times,
                                                 save_times=save_times,
                                                 save_dir=save_dir,
                                                 plot_flag=true,
                                                 save_description=description,
                                                 progress_file=progress_file,
                                                 progress_interval=progress_interval,
                                                 snapshot_request_file=snapshot_request_file,
                                                 snapshot_tag_prefix=snapshot_tag_prefix,
                                                 relaxed_ic=using_initial_state)
        catch e
            if isa(e, InterruptException)
                println("\nInterrupt detected, initiating cleanup...")
                exit()
            else
                rethrow(e)
            end
        end
        
        filename = save_state(state, param, save_dir; ic=ic, relaxed_ic=using_initial_state)
        final_state_saved[] = true
        println("Final state saved to: ", filename)
   end
end

main()
