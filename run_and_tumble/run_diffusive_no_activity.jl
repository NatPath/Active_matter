############### run_diffusive_no_activity.jl ###############
############### Diffusive Fork (No α/ϵ) ###############

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
# using BenchmarkTools

# First, on the main process, include all necessary files.
include("potentials.jl")
include("modules_diffusive_no_activity.jl")
include("plot_utils.jl")
include("save_utils.jl")
using .FPDiffusive
using .PlotUtils
using .SaveUtils

# Ensure all worker processes load the same files and modules.
@everywhere begin
    using Printf, Dates, Plots, Random, FFTW, ProgressMeter, Statistics, LsqFit, LinearAlgebra, JLD2, YAML, ArgParse
    include("potentials.jl")
    include("modules_diffusive_no_activity.jl")
    using .FPDiffusive
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
        "--initial_state"
            help = "Path to saved state file to start from as t=0 (statistics reset)"
            required = false
        "--int_type"
            help = "Integer type for positions/densities (e.g., Int16, Int32, Int64)"
            arg_type = String
            required = false
        "--save_tag"
            help = "Optional save tag override (mainly used for aggregated multi-run outputs)"
            arg_type = String
            required = false
        "--aggregate_state_list"
            help = "Path to newline-delimited state files to aggregate (one .jld2 path per line)"
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
    if !hasfield(typeof(state), :bond_pass_stats)
        legacy_stats = Dict{Symbol,Vector{Float64}}()
        if hasfield(typeof(state), :ρ_matrix_avg_cuts)
            legacy_cuts = getfield(state, :ρ_matrix_avg_cuts)
            for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                        :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                        :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
                if haskey(legacy_cuts, key)
                    legacy_stats[key] = Float64.(legacy_cuts[key])
                end
            end
        end
        state = FPDiffusive.setDummyState(state, state.ρ_avg, state.ρ_matrix_avg_cuts, state.t, legacy_stats)
    end
    FPDiffusive.reset_statistics!(state)
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
         FPDiffusive.update!(param, state_copy, rng)
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
        "L" => L,
        "ρ₀" => 100, # particles per site
        "T" => 1.0,
        "γ" => 0.0,
        "n_sweeps" => 10^6,
        "warmup_sweeps" => 0,
        "cluster_mode" => false,
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

@everywhere function to_bool(value, key_name::String)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lv = lowercase(strip(value))
        if lv in ("true", "t", "1", "yes", "y")
            return true
        elseif lv in ("false", "f", "0", "no", "n")
            return false
        end
        error("$key_name has an invalid boolean value: $value")
    end
    error("$key_name must be boolean-like (Bool/0-1/string).")
end

@everywhere function get_warmup_sweeps(params, defaults)
    warmup_raw = get(params, "warmup_sweeps", defaults["warmup_sweeps"])
    if !(warmup_raw isa Number)
        error("warmup_sweeps must be numeric.")
    end
    warmup_sweeps = Int(round(Float64(warmup_raw)))
    return max(warmup_sweeps, 0)
end

@everywhere function get_cluster_mode(params, defaults)
    return to_bool(get(params, "cluster_mode", defaults["cluster_mode"]), "cluster_mode")
end

@everywhere function get_description(params, defaults)
    raw_description = get(params, "description", defaults["description"])
    if isnothing(raw_description)
        return ""
    end
    return String(strip(String(raw_description)))
end

@everywhere function expand_to_length(values::Vector{T}, n::Int, key_name::String) where {T}
    if length(values) == n
        return values
    elseif length(values) == 1 && n > 1
        return fill(values[1], n)
    end
    error("$key_name must have length 1 or length $n.")
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

@everywhere function bond_pass_stats_with_weights(state)
    stats_dict = Dict{Symbol,Vector{Float64}}()
    if hasfield(typeof(state), :bond_pass_stats)
        raw_stats = getfield(state, :bond_pass_stats)
        for (k, v) in raw_stats
            stats_dict[k] = Float64.(v)
        end
    elseif hasfield(typeof(state), :ρ_matrix_avg_cuts)
        raw_cuts = getfield(state, :ρ_matrix_avg_cuts)
        for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                    :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                    :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
            if haskey(raw_cuts, key)
                stats_dict[key] = Float64.(raw_cuts[key])
            end
        end
    end

    bond_w = 1.0
    spatial_w = 1.0
    if haskey(stats_dict, :bond_pass_sample_count) && !isempty(stats_dict[:bond_pass_sample_count])
        bond_w = max(stats_dict[:bond_pass_sample_count][1], 1.0)
    elseif hasfield(typeof(state), :t)
        bond_w = max(Float64(state.t), 1.0)
    end
    if haskey(stats_dict, :bond_pass_spatial_sample_count) && !isempty(stats_dict[:bond_pass_spatial_sample_count])
        spatial_w = max(stats_dict[:bond_pass_spatial_sample_count][1], 1.0)
    else
        spatial_w = bond_w
    end
    return stats_dict, bond_w, spatial_w
end

@everywhere function average_bond_pass_stats_from_prepared(
    stats_list::Vector{Dict{Symbol,Vector{Float64}}},
    bond_weights::Vector{Float64},
    spatial_weights::Vector{Float64},
)
    if isempty(stats_list)
        return Dict{Symbol,Vector{Float64}}()
    end
    non_empty_indices = [i for i in eachindex(stats_list) if !isempty(stats_list[i])]
    non_empty_stats = [stats_list[i] for i in non_empty_indices]
    if isempty(non_empty_stats)
        return Dict{Symbol,Vector{Float64}}()
    end
    bond_weights = [bond_weights[i] for i in non_empty_indices]
    spatial_weights = [spatial_weights[i] for i in non_empty_indices]

    shared_keys = Set(keys(non_empty_stats[1]))
    for stats in non_empty_stats[2:end]
        intersect!(shared_keys, Set(keys(stats)))
    end

    averaged = Dict{Symbol,Vector{Float64}}()
    for key in shared_keys
        vectors = [stats[key] for stats in non_empty_stats]
        if isempty(vectors)
            continue
        end
        if key == :bond_pass_sample_count
            averaged[key] = [sum(bond_weights)]
        elseif key == :bond_pass_spatial_sample_count
            averaged[key] = [sum(spatial_weights)]
        elseif key == :bond_pass_track_mask
            averaged[key] = Float64.(vectors[1])
        else
            weights = if key in (:bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg)
                spatial_weights
            else
                bond_weights
            end
            total_weight = sum(weights)
            if total_weight <= 0
                averaged[key] = average_array_list(vectors)
            else
                weighted_sum = zeros(Float64, size(vectors[1]))
                for (vec, w) in zip(vectors, weights)
                    weighted_sum .+= vec .* w
                end
                averaged[key] = weighted_sum ./ total_weight
            end
        end
    end
    return averaged
end

@everywhere function average_bond_pass_stats(states::Vector)
    if isempty(states)
        return Dict{Symbol,Vector{Float64}}()
    end
    stats_list = Vector{Dict{Symbol,Vector{Float64}}}(undef, length(states))
    bond_weights = Float64[]
    spatial_weights = Float64[]
    for (i, state) in enumerate(states)
        stats_dict, bond_w, spatial_w = bond_pass_stats_with_weights(state)
        stats_list[i] = stats_dict
        push!(bond_weights, bond_w)
        push!(spatial_weights, spatial_w)
    end
    return average_bond_pass_stats_from_prepared(stats_list, bond_weights, spatial_weights)
end


# Function to set up and run one independent simulation.
@everywhere function run_one_simulation_from_state(param, state, seed, n_sweeps; relaxed_ic::Bool=false, warmup_sweeps::Int=0, cluster_mode::Bool=false, description::String="")
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
    # L                  = get(params, "L", defaults["L"])
    # N                  = get(params, "N", defaults["N"])
    # T                  = get(params, "T", defaults["T"])
    # γ′                 = get(params, "γ′", defaults["γ′"])
    
    # dims = ntuple(i -> L, dim_num)
    # ρ₀ = N / L
    # γ = γ′ / N

    # Initialize simulation parameters and state.
    # param = FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude)
    # v_smudge_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    # potential = Potentials.choose_potential(v_smudge_args, dims; fluctuation_type=fluctuation_type)
    # state = FPDiffusive.setState(0, rng, param, T, potential)
    state_bond_pass_stats = hasfield(typeof(state), :bond_pass_stats) ? deepcopy(getfield(state, :bond_pass_stats)) : Dict{Symbol,Vector{Float64}}()
    dummy_state = FPDiffusive.setDummyState(state, state.ρ_avg, state.ρ_matrix_avg_cuts, state.t, state_bond_pass_stats)
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=100)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")
    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(dummy_state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=show_times,
                                                 save_times=save_times,
                                                 warmup_sweeps=warmup_sweeps,
                                                 plot_label=description,
                                                 save_description=description,
                                                 show_progress=!cluster_mode,
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

function warmup_sweeps_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        println("Using configuration from file: $(args["config"])")
        params = YAML.load_file(args["config"])
    else
        println("No config file provided. Using default parameters.")
        params = get_default_params()
    end
    return get_warmup_sweeps(params, defaults)
end

function cluster_mode_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        println("Using configuration from file: $(args["config"])")
        params = YAML.load_file(args["config"])
    else
        println("No config file provided. Using default parameters.")
        params = get_default_params()
    end
    return get_cluster_mode(params, defaults)
end

function description_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        params = YAML.load_file(args["config"])
    else
        params = defaults
    end
    return get_description(params, defaults)
end

function save_dir_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        params = YAML.load_file(args["config"])
    else
        params = defaults
    end
    return get(params, "save_dir", defaults["save_dir"])
end

function legacy_bond_pass_stats(state)
    legacy_stats = Dict{Symbol,Vector{Float64}}()
    if hasfield(typeof(state), :ρ_matrix_avg_cuts)
        legacy_cuts = getfield(state, :ρ_matrix_avg_cuts)
        for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                    :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                    :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
            if haskey(legacy_cuts, key)
                legacy_stats[key] = Float64.(legacy_cuts[key])
            end
        end
    end
    return legacy_stats
end

function normalize_state_for_aggregation(state)
    if hasfield(typeof(state), :bond_pass_stats)
        return state
    end
    return FPDiffusive.setDummyState(state, state.ρ_avg, state.ρ_matrix_avg_cuts, state.t, legacy_bond_pass_stats(state))
end

function read_state_list(path::String)
    files = String[]
    for raw_line in eachline(path)
        line = strip(raw_line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        push!(files, line)
    end
    return files
end

function aggregate_and_save_states(states::Vector, result_params::Vector, args; using_initial_state::Bool=false, run_description::String="")
    if isempty(states) || isempty(result_params)
        error("No states were provided for aggregation.")
    end
    first_state = states[1]
    state_weights = [max(Float64(averaged_sample_count(state)), 1.0) for state in states]
    total_weight = sum(state_weights)
    if total_weight <= 0
        error("Invalid aggregation weights: total effective sample weight is non-positive.")
    end

    avg_dists = zeros(Float64, size(first_state.ρ_avg))
    for (state, weight) in zip(states, state_weights)
        avg_dists .+= state.ρ_avg .* weight
    end
    avg_dists ./= total_weight

    shared_corr_keys = Set(keys(first_state.ρ_matrix_avg_cuts))
    for state in states[2:end]
        intersect!(shared_corr_keys, Set(keys(state.ρ_matrix_avg_cuts)))
    end
    mat_cuts_averaged = Dict{Symbol,AbstractArray{Float64}}()
    for key in shared_corr_keys
        weighted_sum = zeros(Float64, size(first_state.ρ_matrix_avg_cuts[key]))
        for (state, weight) in zip(states, state_weights)
            weighted_sum .+= state.ρ_matrix_avg_cuts[key] .* weight
        end
        mat_cuts_averaged[key] = weighted_sum ./ total_weight
    end

    total_t = Int(round(total_weight))
    avg_bond_pass_stats = average_bond_pass_stats(states)
    dummy_state = FPDiffusive.setDummyState(first_state, avg_dists, mat_cuts_averaged, total_t, avg_bond_pass_stats)

    dummy_state_save_dir = save_dir_from_args(args)
    save_tag = if haskey(args, "save_tag") && !isnothing(args["save_tag"])
        String(args["save_tag"])
    else
        Dates.format(now(), "yyyymmdd-HHMMSS")
    end
    if !startswith(save_tag, "aggregated_")
        save_tag = "aggregated_" * save_tag
    end
    filename = save_state(dummy_state, result_params[1], dummy_state_save_dir;
                          tag=save_tag, ic="aggregated", relaxed_ic=using_initial_state, description=run_description)
    println("Final aggregated state saved to: ", filename)
    return filename
end

function aggregate_state_list_and_save(args; run_description::String="")
    if !haskey(args, "aggregate_state_list") || isnothing(args["aggregate_state_list"])
        error("Internal error: aggregate_state_list_and_save called without --aggregate_state_list.")
    end
    state_list_path = String(args["aggregate_state_list"])
    state_files = read_state_list(state_list_path)
    if isempty(state_files)
        error("No state files found in aggregate list: $state_list_path")
    end
    println("Aggregating $(length(state_files)) precomputed states from list: $state_list_path")

    first_state = nothing
    first_param = nothing
    avg_dists = nothing
    mat_cuts_weighted = Dict{Symbol,Any}()
    shared_corr_keys = Set{Symbol}()
    total_weight = 0.0

    stats_list = Dict{Symbol,Vector{Float64}}[]
    bond_weights = Float64[]
    spatial_weights = Float64[]

    for (idx, state_path) in enumerate(state_files)
        println("Loading state for aggregation ($idx/$(length(state_files))): $state_path")
        @load state_path state param potential
        norm_state = normalize_state_for_aggregation(state)
        state_weight = max(Float64(averaged_sample_count(norm_state)), 1.0)

        if idx == 1
            first_state = norm_state
            first_param = param
            avg_dists = zeros(Float64, size(norm_state.ρ_avg))
            avg_dists .+= norm_state.ρ_avg .* state_weight
            shared_corr_keys = Set{Symbol}(keys(norm_state.ρ_matrix_avg_cuts))
            for key in shared_corr_keys
                weighted_sum = zeros(Float64, size(norm_state.ρ_matrix_avg_cuts[key]))
                weighted_sum .+= norm_state.ρ_matrix_avg_cuts[key] .* state_weight
                mat_cuts_weighted[key] = weighted_sum
            end
        else
            if size(avg_dists) != size(norm_state.ρ_avg)
                error("Cannot aggregate states with different ρ_avg sizes: got $(size(norm_state.ρ_avg)), expected $(size(avg_dists)).")
            end
            avg_dists .+= norm_state.ρ_avg .* state_weight

            current_keys = Set{Symbol}(keys(norm_state.ρ_matrix_avg_cuts))
            new_shared = intersect(shared_corr_keys, current_keys)
            for key in collect(keys(mat_cuts_weighted))
                if !(key in new_shared)
                    delete!(mat_cuts_weighted, key)
                end
            end
            for key in new_shared
                if size(mat_cuts_weighted[key]) != size(norm_state.ρ_matrix_avg_cuts[key])
                    error("Cannot aggregate key $(key): shape mismatch $(size(norm_state.ρ_matrix_avg_cuts[key])) vs $(size(mat_cuts_weighted[key])).")
                end
                mat_cuts_weighted[key] .+= norm_state.ρ_matrix_avg_cuts[key] .* state_weight
            end
            shared_corr_keys = new_shared
        end

        stats_dict, bond_w, spatial_w = bond_pass_stats_with_weights(norm_state)
        push!(stats_list, stats_dict)
        push!(bond_weights, bond_w)
        push!(spatial_weights, spatial_w)
        total_weight += state_weight

        if idx % 5 == 0
            GC.gc(false)
        end
    end

    if isnothing(first_state) || isnothing(first_param)
        error("No valid states were loaded for aggregation.")
    end
    if total_weight <= 0
        error("Invalid aggregation weights: total effective sample weight is non-positive.")
    end

    avg_dists ./= total_weight
    mat_cuts_averaged = Dict{Symbol,AbstractArray{Float64}}()
    for (key, weighted_sum) in mat_cuts_weighted
        mat_cuts_averaged[key] = weighted_sum ./ total_weight
    end

    total_t = Int(round(total_weight))
    avg_bond_pass_stats = average_bond_pass_stats_from_prepared(stats_list, bond_weights, spatial_weights)
    dummy_state = FPDiffusive.setDummyState(first_state, avg_dists, mat_cuts_averaged, total_t, avg_bond_pass_stats)

    dummy_state_save_dir = save_dir_from_args(args)
    save_tag = if haskey(args, "save_tag") && !isnothing(args["save_tag"])
        String(args["save_tag"])
    else
        Dates.format(now(), "yyyymmdd-HHMMSS")
    end
    if !startswith(save_tag, "aggregated_")
        save_tag = "aggregated_" * save_tag
    end
    filename = save_state(dummy_state, first_param, dummy_state_save_dir;
                          tag=save_tag, ic="aggregated", relaxed_ic=false, description=run_description)
    println("Final aggregated state saved to: ", filename)
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
    warmup_sweeps = get_warmup_sweeps(params, defaults)
    cluster_mode = get_cluster_mode(params, defaults)
    description = get_description(params, defaults)
    params["show_times"] = Int[]
    params["save_times"] = Int[]
    int_type = resolve_int_type(args, params, defaults)
    
    # Set up simulation parameters.
    dim_num            = get(params, "dim_num", defaults["dim_num"])
    potential_type     = get(params, "potential_type", defaults["potential_type"])
    fluctuation_type   = get(params, "fluctuation_type", defaults["fluctuation_type"])
    potential_magnitude= get(params, "potential_magnitude", defaults["potential_magnitude"])
    D                  = get(params, "D", defaults["D"])
    L                  = get(params, "L", defaults["L"])
    ρ₀              = get(params, "ρ₀", defaults["ρ₀"])
    T                  = get(params, "T", defaults["T"])
    γ                 = get(params, "γ", defaults["γ"])
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))

    forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)

    ic = get(params, "ic", defaults["ic"])

    dims = ntuple(i -> L, dim_num)
    # ρ₀ = N / (L^dim_num)

    # Initialize simulation parameters and state.
    param = FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type,rng=rng)
    state = FPDiffusive.setState(0, rng, param, T, potential, forcings; ic=ic, int_type=int_type, bond_pass_count_mode=bond_pass_count_mode)
   
    #estimate run time
    estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=100)
    estimated_time_hours = estimated_time / 3600
    println("Estimated run time for this simulation: $estimated_time_hours hours")

    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=params["show_times"],
                                                 save_times=params["save_times"],
                                                 plot_label=description,
                                                 save_description=description,
                                                 warmup_sweeps=warmup_sweeps,
                                                 show_progress=!cluster_mode)
    return dist, corr_mat_cuts, state, param
end

# Main function.
function main()
    args = parse_commandline()
    run_description = description_from_args(args)
    if haskey(args, "aggregate_state_list") && !isnothing(args["aggregate_state_list"])
        aggregate_state_list_and_save(args; run_description=run_description)
        return
    end
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
            if !hasfield(typeof(state), :bond_pass_stats)
                legacy_stats = Dict{Symbol,Vector{Float64}}()
                if hasfield(typeof(state), :ρ_matrix_avg_cuts)
                    legacy_cuts = getfield(state, :ρ_matrix_avg_cuts)
                    for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                                :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                                :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
                        if haskey(legacy_cuts, key)
                            legacy_stats[key] = Float64.(legacy_cuts[key])
                        end
                    end
                end
                state = FPDiffusive.setDummyState(state, state.ρ_avg, state.ρ_matrix_avg_cuts, state.t, legacy_stats)
            end
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
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            cluster_mode = get_cluster_mode(params, defaults)
            run_description = get_description(params, defaults)
            println("Warmup sweeps per restarted run: $warmup_sweeps")
            results = pmap(seed -> run_one_simulation_from_state(param, state, seed, n_sweeps; warmup_sweeps=warmup_sweeps, cluster_mode=cluster_mode, description=run_description), seeds)
        elseif using_initial_state
            defaults = get_default_params()
            n_sweeps = n_sweeps_from_args(args)
            warmup_sweeps = warmup_sweeps_from_args(args)
            cluster_mode = cluster_mode_from_args(args)
            results = pmap(seed -> begin
                    run_state = deepcopy(initial_state)
                    run_param = deepcopy(initial_param)
                    run_one_simulation_from_state(run_param, run_state, seed, n_sweeps; relaxed_ic=true, warmup_sweeps=warmup_sweeps, cluster_mode=cluster_mode, description=run_description)
                end, seeds)
        else
            results = pmap(seed -> run_one_simulation_from_config(args, seed), seeds)
            n_sweeps=n_sweeps_from_args(args)
        end
        states = [res[3] for res in results]
        result_params = [res[4] for res in results]
        aggregate_and_save_states(states, result_params, args; using_initial_state=using_initial_state, run_description=run_description)

        
        # Save aggregated results to a separate file.
        # save_dir = "saved_states_parallel"
        # mkpath(save_dir)
        # now_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        # filename = @sprintf("%s/parallel_results_%s.jld2", save_dir, now_str)
        # @save filename states params
        # println("Parallel results saved to: $filename")
    else
        # Single-run mode.
        description = run_description
        if haskey(args, "continue") && !isnothing(args["continue"])
            println("Continuing from saved state: $(args["continue"])")
            @load args["continue"] state param potential
            if !hasfield(typeof(state), :bond_pass_stats)
                legacy_stats = Dict{Symbol,Vector{Float64}}()
                if hasfield(typeof(state), :ρ_matrix_avg_cuts)
                    legacy_cuts = getfield(state, :ρ_matrix_avg_cuts)
                    for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                                :bond_pass_total_sq_avg, :bond_pass_sample_count, :bond_pass_track_mask,
                                :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
                        if haskey(legacy_cuts, key)
                            legacy_stats[key] = Float64.(legacy_cuts[key])
                        end
                    end
                end
                state = FPDiffusive.setDummyState(state, state.ρ_avg, state.ρ_matrix_avg_cuts, state.t, legacy_stats)
            end
            if haskey(args, "config") && !isnothing(args["config"])
                println("Using configuration from file: $(args["config"])")
                #params = TOML.parsefile(args["config"])
                params = YAML.load_file(args["config"])
            else
                println("No config file provided. Using default parameters.")
                params = get_default_params()
            end
            ic = get(params, "ic", get_default_params()["ic"])
            defaults = get_default_params()
            if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
                n_sweeps = args["continue_sweeps"]
                println("Continuing for specified $n_sweeps sweeps")
            else
                n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
                println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
            end
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            cluster_mode = get_cluster_mode(params, defaults)
            description = get_description(params, defaults)
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
            n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            cluster_mode = get_cluster_mode(params, defaults)
            description = get_description(params, defaults)
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
            L                  = get(params, "L", defaults["L"])
            ρ₀                = get(params, "ρ₀", defaults["ρ₀"])
            T                  = get(params, "T", defaults["T"])
            γ                 = get(params, "γ", defaults["γ"])
            bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))
            n_sweeps           = get(params, "n_sweeps", defaults["n_sweeps"])
            warmup_sweeps      = get_warmup_sweeps(params, defaults)
            cluster_mode       = get_cluster_mode(params, defaults)
            description        = get_description(params, defaults)
            forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
            ic = get(params, "ic", defaults["ic"])
            int_type = resolve_int_type(args, params, defaults)

            ic = get(params, "ic", defaults["ic"])
            
            dims = ntuple(i -> L, dim_num)
            # ρ₀ = N / prod(dims)
            
            param = FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs)
            v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
            seed = rand(1:2^30)
            #rng = MersenneTwister(123)
            rng = MersenneTwister(seed)
            potential_plot_flag = !cluster_mode
            potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type,rng=rng,plot_flag=potential_plot_flag)
            
            state = FPDiffusive.setState(0, rng, param, T, potential, forcings; ic=ic, int_type=int_type, bond_pass_count_mode=bond_pass_count_mode)
        end
        
        defaults = get_default_params()
        show_times = get(params, "show_times", defaults["show_times"])
        save_times = get(params, "save_times", defaults["save_times"])
        save_dir = get(params, "save_dir", defaults["save_dir"])
        progress_file = String(get(params, "progress_file", ""))
        progress_interval_raw = get(params, "progress_interval", 25)
        progress_interval = max(Int(round(Float64(progress_interval_raw))), 1)
        snapshot_request_file = String(get(params, "snapshot_request_file", ""))
        snapshot_tag_prefix = String(get(params, "snapshot_tag_prefix", "snapshot"))
        if !(@isdefined warmup_sweeps)
            warmup_sweeps = get_warmup_sweeps(params, defaults)
        end
        if !(@isdefined cluster_mode)
            cluster_mode = get_cluster_mode(params, defaults)
        end
        if cluster_mode
            show_times = Int[]
        end

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
                SaveUtils.save_state(state, param, save_dir; ic=ic, relaxed_ic=using_initial_state, description=description)
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
                                                 plot_flag=!cluster_mode,
                                                 plot_label=description,
                                                 save_description=description,
                                                 warmup_sweeps=warmup_sweeps,
                                                 show_progress=!cluster_mode,
                                                 save_dir=save_dir,
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
        
        filename = save_state(state, param, save_dir; ic=ic, relaxed_ic=using_initial_state, description=description)
        final_state_saved[] = true
        println("Final state saved to: ", filename)
   end
end

main()
