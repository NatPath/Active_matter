############### run_diffusive_no_activity.jl ###############
############### Diffusive Fork (No α/ϵ) ###############

using Distributed
using Printf
using Dates

function _parse_bool_literal(raw)
    raw === nothing && return nothing
    value = lowercase(strip(String(raw)))
    if value in ("1", "true", "yes", "on", "y", "t")
        return true
    elseif value in ("0", "false", "no", "off", "n", "f", "")
        return false
    end
    return nothing
end

function _argv_value(flag::AbstractString)
    idx = findfirst(==(flag), ARGS)
    if idx === nothing || idx >= length(ARGS)
        return nothing
    end
    return ARGS[idx + 1]
end

function _read_config_bool(path::AbstractString, key::AbstractString)
    !isfile(path) && return nothing
    prefix = string(key, ":")
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, prefix) || continue
        value = strip(line[length(prefix) + 1:end])
        comment_idx = findfirst(==('#'), value)
        if comment_idx !== nothing
            value = strip(value[1:prevind(value, comment_idx)])
        end
        value = strip(replace(value, "\"" => "", "'" => ""))
        parsed = _parse_bool_literal(value)
        parsed === nothing || return parsed
    end
    return nothing
end

function _config_requests_performance_mode()
    config_path = _argv_value("--config")
    config_path === nothing && return false
    performance_mode = _read_config_bool(String(config_path), "performance_mode")
    performance_mode !== nothing && return performance_mode
    cluster_mode = _read_config_bool(String(config_path), "cluster_mode")
    cluster_mode !== nothing && return cluster_mode
    return false
end

const RUN_AND_TUMBLE_HEADLESS = begin
    env_headless = _parse_bool_literal(get(ENV, "RUN_AND_TUMBLE_HEADLESS", "0")) === true
    aggregate_mode = "--aggregate_state_list" in ARGS
    cli_performance_mode = "--performance_mode" in ARGS
    env_headless || aggregate_mode || cli_performance_mode || _config_requests_performance_mode()
end

if !RUN_AND_TUMBLE_HEADLESS && haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
    # Under WSLg, Qt/GR can pick Wayland and then fail EGL initialization.
    # Force the stable XWayland path before Plots/GR initialize.
    ENV["QT_QPA_PLATFORM"] = "xcb"
end
if !RUN_AND_TUMBLE_HEADLESS
    using Plots
end
using Random
using ProgressMeter
using Statistics
using LinearAlgebra
using JLD2
using YAML
using ArgParse
# using BenchmarkTools

const SRC_ROOT = joinpath(@__DIR__, "src")
const COMMON_DIR = joinpath(SRC_ROOT, "common")
const DIFFUSIVE_DIR = joinpath(SRC_ROOT, "diffusive")

# First, on the main process, include all necessary files.
include(joinpath(DIFFUSIVE_DIR, "modules_diffusive_no_activity.jl"))
include(joinpath(COMMON_DIR, "save_utils.jl"))
using .FPDiffusive
using .SaveUtils
if !RUN_AND_TUMBLE_HEADLESS
    include(joinpath(COMMON_DIR, "plot_utils.jl"))
    using .PlotUtils
    plot_sweep_runner = PlotUtils.plot_sweep
else
    plot_sweep_runner = nothing
end

const AGG_TWO_FORCE_REPLICA_COUNT_KEY = :agg_two_force_replica_count
const AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY = :agg_two_force_var_slot_mean
const AGG_TWO_FORCE_VAR_SLOT_SEM_KEY = :agg_two_force_var_slot_sem
const AGG_TWO_FORCE_VAR_SLOT_CI95_KEY = :agg_two_force_var_slot_ci95
const AGG_TWO_FORCE_VAR_SLOT_N_KEY = :agg_two_force_var_slot_n
const AGG_TWO_FORCE_VAR_SLOT_WEIGHT_KEY = :agg_two_force_var_slot_weight
const AGG_TWO_FORCE_VAR_SLOT_SUM_KEY = :agg_two_force_var_slot_sum
const AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY = :agg_two_force_var_slot_sumsq
const AGG_TWO_FORCE_VAR_SLOT_WEIGHTSQ_KEY = :agg_two_force_var_slot_weightsq
const AGG_TWO_FORCE_VAR_MEAN_KEY = :agg_two_force_var_mean
const AGG_TWO_FORCE_VAR_MEAN_SEM_KEY = :agg_two_force_var_mean_sem
const AGG_TWO_FORCE_VAR_MEAN_CI95_KEY = :agg_two_force_var_mean_ci95
const AGG_TWO_FORCE_VAR_MEAN_N_KEY = :agg_two_force_var_mean_n
const AGG_TWO_FORCE_VAR_MEAN_WEIGHT_KEY = :agg_two_force_var_mean_weight
const AGG_TWO_FORCE_VAR_MEAN_SUM_KEY = :agg_two_force_var_mean_sum
const AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY = :agg_two_force_var_mean_sumsq
const AGG_TWO_FORCE_VAR_MEAN_WEIGHTSQ_KEY = :agg_two_force_var_mean_weightsq
const AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY = :agg_two_force_var_raw_slot_mean
const AGG_TWO_FORCE_VAR_RAW_SLOT_SEM_KEY = :agg_two_force_var_raw_slot_sem
const AGG_TWO_FORCE_VAR_RAW_SLOT_CI95_KEY = :agg_two_force_var_raw_slot_ci95
const AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY = :agg_two_force_var_raw_slot_n
const AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHT_KEY = :agg_two_force_var_raw_slot_weight
const AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY = :agg_two_force_var_raw_slot_sum
const AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY = :agg_two_force_var_raw_slot_sumsq
const AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHTSQ_KEY = :agg_two_force_var_raw_slot_weightsq
const AGG_TWO_FORCE_VAR_RAW_MEAN_KEY = :agg_two_force_var_raw_mean
const AGG_TWO_FORCE_VAR_RAW_MEAN_SEM_KEY = :agg_two_force_var_raw_mean_sem
const AGG_TWO_FORCE_VAR_RAW_MEAN_CI95_KEY = :agg_two_force_var_raw_mean_ci95
const AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY = :agg_two_force_var_raw_mean_n
const AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHT_KEY = :agg_two_force_var_raw_mean_weight
const AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY = :agg_two_force_var_raw_mean_sum
const AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY = :agg_two_force_var_raw_mean_sumsq
const AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHTSQ_KEY = :agg_two_force_var_raw_mean_weightsq
const AGG_TWO_FORCE_J2_SLOT_MEAN_KEY = :agg_two_force_j2_slot_mean
const AGG_TWO_FORCE_J2_SLOT_SEM_KEY = :agg_two_force_j2_slot_sem
const AGG_TWO_FORCE_J2_SLOT_CI95_KEY = :agg_two_force_j2_slot_ci95
const AGG_TWO_FORCE_J2_SLOT_N_KEY = :agg_two_force_j2_slot_n
const AGG_TWO_FORCE_J2_SLOT_WEIGHT_KEY = :agg_two_force_j2_slot_weight
const AGG_TWO_FORCE_J2_SLOT_SUM_KEY = :agg_two_force_j2_slot_sum
const AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY = :agg_two_force_j2_slot_sumsq
const AGG_TWO_FORCE_J2_SLOT_WEIGHTSQ_KEY = :agg_two_force_j2_slot_weightsq
const AGG_TWO_FORCE_J2_MEAN_KEY = :agg_two_force_j2_mean
const AGG_TWO_FORCE_J2_MEAN_SEM_KEY = :agg_two_force_j2_mean_sem
const AGG_TWO_FORCE_J2_MEAN_CI95_KEY = :agg_two_force_j2_mean_ci95
const AGG_TWO_FORCE_J2_MEAN_N_KEY = :agg_two_force_j2_mean_n
const AGG_TWO_FORCE_J2_MEAN_WEIGHT_KEY = :agg_two_force_j2_mean_weight
const AGG_TWO_FORCE_J2_MEAN_SUM_KEY = :agg_two_force_j2_mean_sum
const AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY = :agg_two_force_j2_mean_sumsq
const AGG_TWO_FORCE_J2_MEAN_WEIGHTSQ_KEY = :agg_two_force_j2_mean_weightsq
const AGG_CONNECTED_FULL_EXACT_KEY = :agg_connected_corr_full_exact

# Ensure all worker processes load the same files and modules.
@everywhere const RUN_AND_TUMBLE_HEADLESS_WORKER = $RUN_AND_TUMBLE_HEADLESS
@everywhere const RUN_AND_TUMBLE_COMMON_DIR_WORKER = $COMMON_DIR
@everywhere const RUN_AND_TUMBLE_DIFFUSIVE_DIR_WORKER = $DIFFUSIVE_DIR
@everywhere begin
    using Printf, Dates, Random, ProgressMeter, Statistics, LinearAlgebra, JLD2, YAML, ArgParse
    local run_and_tumble_headless = RUN_AND_TUMBLE_HEADLESS_WORKER
    if !run_and_tumble_headless && haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
        ENV["QT_QPA_PLATFORM"] = "xcb"
    end
    if !run_and_tumble_headless
        using Plots
    end
    include(joinpath(RUN_AND_TUMBLE_DIFFUSIVE_DIR_WORKER, "modules_diffusive_no_activity.jl"))
    using .FPDiffusive
    if !run_and_tumble_headless
        include(joinpath(RUN_AND_TUMBLE_COMMON_DIR_WORKER, "plot_utils.jl"))
        using .PlotUtils
        global plot_sweep_runner = PlotUtils.plot_sweep
    else
        global plot_sweep_runner = nothing
    end
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
            help = "Density integer type (signed; e.g., Int16, Int32, Int64, or auto)"
            arg_type = String
            required = false
        "--position_int_type"
            help = "Particle-position integer type (signed; e.g., Int16, Int32, Int64, or auto)"
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
        "--performance_mode"
            help = "Disable progress bars and plotting for lean runtime"
            action = :store_true
        "--estimate_only"
            help = "Estimate runtime and exit without running the simulation (single-run mode)"
            action = :store_true
        "--estimate_runtime"
            help = "Estimate runtime before running the simulation"
            action = :store_true
        "--estimate_sample_size"
            help = "Number of sample sweeps to use for runtime estimation"
            arg_type = Int
            required = false
    end
    return parse_args(s)
end

function explicit_save_tag_from_args(args)
    if haskey(args, "save_tag") && !isnothing(args["save_tag"])
        return String(args["save_tag"])
    end
    return nothing
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

@everywhere function parse_int_type_or_auto(value)
    value === nothing && return nothing
    if value isa Symbol
        return value == :auto ? nothing : parse_int_type(String(value))
    elseif value isa AbstractString
        return lowercase(strip(String(value))) == "auto" ? nothing : parse_int_type(value)
    end
    return parse_int_type(value)
end

@everywhere function smallest_signed_int_type(max_value::Integer)
    max_value < 0 && error("smallest_signed_int_type expects a non-negative bound. Got $(max_value).")
    if max_value <= typemax(Int8)
        return Int8
    elseif max_value <= typemax(Int16)
        return Int16
    elseif max_value <= typemax(Int32)
        return Int32
    elseif max_value <= typemax(Int64)
        return Int64
    end
    error("Requested integer bound $(max_value) exceeds Int64 capacity.")
end

@everywhere function resolve_density_int_type(args, params, defaults, max_occupancy::Integer)
    requested = if haskey(args, "int_type") && !isnothing(args["int_type"])
        parse_int_type_or_auto(args["int_type"])
    else
        parse_int_type_or_auto(get(params, "int_type", defaults["int_type"]))
    end
    resolved = isnothing(requested) ? smallest_signed_int_type(max_occupancy) : requested
    resolved <: Signed || error("Density int_type must be signed to avoid underflow/overflow surprises. Got $(resolved).")
    max_occupancy <= typemax(resolved) || error("Density int_type $(resolved) cannot represent max site occupancy $(max_occupancy).")
    return resolved
end

@everywhere function resolve_position_int_type(args, params, defaults, dims)
    requested = if haskey(args, "position_int_type") && !isnothing(args["position_int_type"])
        parse_int_type_or_auto(args["position_int_type"])
    else
        parse_int_type_or_auto(get(params, "position_int_type", defaults["position_int_type"]))
    end
    max_coord = maximum(Int.(dims))
    resolved = isnothing(requested) ? smallest_signed_int_type(max_coord) : requested
    resolved <: Signed || error("position_int_type must be signed. Got $(resolved).")
    max_coord <= typemax(resolved) || error("position_int_type $(resolved) cannot represent coordinates up to $(max_coord).")
    return resolved
end

@everywhere function get_keep_directional_densities(params, defaults)
    if haskey(params, "keep_directional_densities")
        return to_bool(params["keep_directional_densities"], "keep_directional_densities")
    end
    return to_bool(get(defaults, "keep_directional_densities", false), "keep_directional_densities")
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
        "performance_mode" => false,
        "cluster_mode" => false,
        "estimate_runtime" => false,
        "estimate_sample_size" => 100,
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
        "int_type" => "auto",
        "position_int_type" => "auto",
        "keep_directional_densities" => false,
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

@everywhere function param_ffr_input(param)
    if loaded_param_has_field(param, :ffr)
        return loaded_param_field(param, :ffr)
    elseif loaded_param_has_field(param, :ffrs)
        return loaded_param_field(param, :ffrs)
    end
    return 0.0
end

@everywhere function maybe_override_forcing_rate_scheme(param, args, params)
    param = canonicalize_loaded_param(param)
    has_explicit_config = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "forcing_rate_scheme")
    if !has_explicit_config
        return param
    end
    return FPDiffusive.setParam(param.γ, param.dims, param.ρ₀, param.D,
                                param.potential_type, param.fluctuation_type, param.potential_magnitude,
                                param_ffr_input(param);
                                forcing_rate_scheme=String(params["forcing_rate_scheme"]))
end

@everywhere function get_warmup_sweeps(params, defaults)
    warmup_raw = get(params, "warmup_sweeps", defaults["warmup_sweeps"])
    if !(warmup_raw isa Number)
        error("warmup_sweeps must be numeric.")
    end
    warmup_sweeps = Int(round(Float64(warmup_raw)))
    return max(warmup_sweeps, 0)
end

@everywhere function to_int_vector(value, key_name::String)
    if value isa Number
        return [Int(round(Float64(value)))]
    elseif value isa AbstractVector
        return [Int(round(Float64(v))) for v in value]
    end
    error("$key_name must be numeric or a list of numerics.")
end

@everywhere function get_show_times(params, defaults, warmup_sweeps::Int; has_explicit_show_times::Bool=false)
    raw_show_times = if has_explicit_show_times
        params["show_times"]
    else
        to_int_vector(defaults["show_times"], "show_times") .+ warmup_sweeps
    end
    return sort(unique(to_int_vector(raw_show_times, "show_times")))
end

@everywhere function get_performance_mode(params, defaults)
    if haskey(params, "performance_mode")
        return to_bool(params["performance_mode"], "performance_mode")
    elseif haskey(params, "cluster_mode")
        return to_bool(params["cluster_mode"], "cluster_mode")
    end
    return to_bool(get(defaults, "performance_mode", false), "performance_mode")
end

@everywhere function get_cluster_mode(params, defaults)
    return get_performance_mode(params, defaults)
end

@everywhere function get_estimate_runtime(params, defaults)
    return to_bool(get(params, "estimate_runtime", defaults["estimate_runtime"]), "estimate_runtime")
end

@everywhere function get_estimate_sample_size(params, defaults, cli_value=nothing)
    if !isnothing(cli_value)
        return max(Int(cli_value), 1)
    end
    raw_value = get(params, "estimate_sample_size", defaults["estimate_sample_size"])
    if !(raw_value isa Number)
        error("estimate_sample_size must be numeric.")
    end
    return max(Int(round(Float64(raw_value))), 1)
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

@everywhere function infer_requested_force_count(params, defaults)
    candidate_lengths = Int[]

    if haskey(params, "forcing_magnitudes")
        push!(candidate_lengths, length(to_float_vector(params["forcing_magnitudes"], "forcing_magnitudes")))
    elseif haskey(params, "forcing_magnitude")
        push!(candidate_lengths, 1)
    end

    if haskey(params, "ffrs")
        push!(candidate_lengths, length(to_float_vector(params["ffrs"], "ffrs")))
    elseif haskey(params, "ffr")
        push!(candidate_lengths, 1)
    end

    if haskey(params, "forcing_direction_flags")
        push!(candidate_lengths, length(to_bool_vector(params["forcing_direction_flags"], "forcing_direction_flags")))
    end

    return isempty(candidate_lengths) ? 1 : maximum(candidate_lengths)
end

@everywhere function inferred_bond_indices_list(params, defaults, dim_num::Int, L::Int)
    requested_force_count = infer_requested_force_count(params, defaults)
    forcing_type = String(get(params, "forcing_type", defaults["forcing_type"]))

    if requested_force_count == 1
        return [forcing_bond_indices_from_type(forcing_type, dim_num, L)]
    end

    has_distance = haskey(params, "distance_between_forces") || haskey(params, "forcing_distance_d")
    if requested_force_count == 2 && has_distance
        dim_num == 1 || error("distance_between_forces is currently supported only for dim_num=1.")
        if haskey(params, "distance_between_forces") && haskey(params, "forcing_distance_d")
            error("Use either distance_between_forces or forcing_distance_d, not both.")
        end

        d_raw = haskey(params, "distance_between_forces") ? params["distance_between_forces"] : params["forcing_distance_d"]
        d_raw isa Number || error("distance_between_forces must be numeric.")
        d = mod(Int(round(Float64(d_raw))), L)

        left_site = if haskey(params, "forcing_base_site")
            base_site_raw = params["forcing_base_site"]
            base_site_raw isa Number || error("forcing_base_site must be numeric.")
            mod1(Int(round(Float64(base_site_raw))), L)
        else
            # Center the midpoint of the two bond midpoints on the middle bond of the system.
            # Here d is the gap between the right site of the left bond and the left site of the right bond.
            mod1((L ÷ 2) - fld(d + 1, 2), L)
        end
        right_site = mod1(left_site + d + 1, L)

        return [
            ([left_site], [mod1(left_site + 1, L)]),
            ([right_site], [mod1(right_site + 1, L)]),
        ]
    end

    error("Unable to infer forcing bond locations. Provide forcing_bond_pairs, explicit forcing_types, or for two 1D forces set distance_between_forces.")
end

@everywhere function build_forcings_and_ffrs(params, defaults, dim_num::Int, L::Int)
    bond_indices_list = Tuple{Vector{Int}, Vector{Int}}[]

    has_distance = haskey(params, "distance_between_forces") || haskey(params, "forcing_distance_d")

    if haskey(params, "forcing_bond_pairs") && has_distance
        error("Use either forcing_bond_pairs or distance_between_forces, not both.")
    elseif haskey(params, "forcing_bond_pairs")
        raw_pairs = params["forcing_bond_pairs"]
        if !(raw_pairs isa AbstractVector)
            error("forcing_bond_pairs must be a list of bond pairs.")
        end
        for raw_pair in raw_pairs
            push!(bond_indices_list, parse_forcing_bond_pair(raw_pair, dim_num, L))
        end
    elseif has_distance
        append!(bond_indices_list, inferred_bond_indices_list(params, defaults, dim_num, L))
    elseif haskey(params, "forcing_types") || haskey(params, "forcing_type")
        forcing_types_raw = haskey(params, "forcing_types") ? params["forcing_types"] : get(params, "forcing_type", defaults["forcing_type"])
        forcing_types = to_string_vector(forcing_types_raw, "forcing_types")
        for forcing_type in forcing_types
            push!(bond_indices_list, forcing_bond_indices_from_type(forcing_type, dim_num, L))
        end
    else
        append!(bond_indices_list, inferred_bond_indices_list(params, defaults, dim_num, L))
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

function force_signature(force)
    return (
        Tuple(Int.(force.bond_indices[1])),
        Tuple(Int.(force.bond_indices[2])),
        Float64(force.magnitude),
    )
end

function loaded_param_has_field(param, field_name::Symbol)
    if hasfield(typeof(param), field_name)
        return true
    end
    if hasfield(typeof(param), :fields)
        field_names = Tuple(typeof(param).parameters[2])
        return field_name in field_names
    end
    return false
end

function loaded_param_field(param, field_name::Symbol)
    if hasfield(typeof(param), field_name)
        return getfield(param, field_name)
    end
    if hasfield(typeof(param), :fields)
        field_names = Tuple(typeof(param).parameters[2])
        idx = findfirst(==(field_name), field_names)
        idx === nothing && error("Loaded param does not contain field $(field_name).")
        raw_fields = getfield(param, :fields)
        values = if raw_fields isa AbstractVector && length(raw_fields) == 1 && raw_fields[1] isa AbstractVector
            raw_fields[1]
        else
            raw_fields
        end
        return values[idx]
    end
    error("Unsupported loaded param type $(typeof(param)); cannot read field $(field_name).")
end

function reconcile_loaded_state_with_config!(state, loaded_param, args, params, defaults)
    dim_num = Int(get(params, "dim_num", length(normalized_dims_tuple(loaded_param_field(loaded_param, :dims)))))
    loaded_dims = normalized_dims_tuple(loaded_param_field(loaded_param, :dims))
    L = Int(get(params, "L", loaded_dims[1]))
    dims = ntuple(_ -> L, dim_num)
    if dims != loaded_dims
        error("Loaded initial_state dims=$(loaded_dims) do not match config dims=$(dims).")
    end

    potential_type = String(get(params, "potential_type", loaded_param_field(loaded_param, :potential_type)))
    fluctuation_type = String(get(params, "fluctuation_type", loaded_param_field(loaded_param, :fluctuation_type)))
    potential_magnitude = Float64(get(params, "potential_magnitude", loaded_param_field(loaded_param, :potential_magnitude)))
    D = Float64(get(params, "D", loaded_param_field(loaded_param, :D)))
    ρ₀ = Float64(get(params, "ρ₀", loaded_param_field(loaded_param, :ρ₀)))
    γ = Float64(get(params, "γ", loaded_param_field(loaded_param, :γ)))
    forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))

    config_forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
    saved_forcings = deepcopy(FPDiffusive.get_state_forcings!(state))

    if length(saved_forcings) == length(config_forcings)
        saved_sigs = force_signature.(saved_forcings)
        config_sigs = force_signature.(config_forcings)
        if saved_sigs == config_sigs
            for i in eachindex(config_forcings)
                config_forcings[i].direction_flag = saved_forcings[i].direction_flag
            end
        else
            println("WARNING: initial_state forcing definitions differ from config; using config forcings and config default direction flags.")
        end
    else
        println("WARNING: initial_state force count ($(length(saved_forcings))) differs from config force count ($(length(config_forcings))); using config forcings.")
    end

    state.forcing = config_forcings
    FPDiffusive.configure_bond_passage_tracking!(state, bond_pass_count_mode)

    reconciled_param = FPDiffusive.setParam(
        γ,
        dims,
        ρ₀,
        D,
        potential_type,
        fluctuation_type,
        potential_magnitude,
        ffrs;
        forcing_rate_scheme=forcing_rate_scheme,
    )
    density_int_type = resolve_density_int_type(args, params, defaults, reconciled_param.N)
    position_int_type = resolve_position_int_type(args, params, defaults, dims)
    keep_directional_densities = get_keep_directional_densities(params, defaults)
    state = FPDiffusive.setDummyState(
        state,
        state.ρ_avg,
        state.ρ_matrix_avg_cuts,
        state.t,
        state.bond_pass_stats;
        density_int_type=density_int_type,
        position_int_type=position_int_type,
        keep_directional_densities=keep_directional_densities,
    )
    FPDiffusive.configure_bond_passage_tracking!(state, bond_pass_count_mode)

    println("Reconciled initial_state with config:")
    println("  dims=$(dims)")
    println("  forcing_rate_scheme=$(forcing_rate_scheme)")
    println("  bond_pass_count_mode=$(bond_pass_count_mode)")
    println("  density_int_type=$(density_int_type)")
    println("  position_int_type=$(position_int_type)")
    println("  keep_directional_densities=$(keep_directional_densities)")
    println("  num_forces=$(length(config_forcings))")
    for (i, force) in enumerate(config_forcings)
        println("    force[$i] bond=$(force.bond_indices) magnitude=$(force.magnitude) direction=$(force.direction_flag) ffr=$(i <= length(ffrs) ? ffrs[i] : NaN)")
    end

    return state, reconciled_param
end

@everywhere function average_array_list(arrays::Vector{<:AbstractArray})
    if isempty(arrays)
        error("Cannot average an empty array list.")
    end
    stack_dim = ndims(arrays[1]) + 1
    stacked = cat(arrays..., dims=stack_dim)
    return dropdims(mean(stacked, dims=stack_dim), dims=stack_dim)
end

@everywhere function raw_sweep_weight(state)
    if hasfield(typeof(state), :bond_pass_stats)
        stats = getfield(state, :bond_pass_stats)
        if haskey(stats, :bond_pass_sample_count) && !isempty(stats[:bond_pass_sample_count])
            return max(Float64(stats[:bond_pass_sample_count][1]), 1.0)
        end
    elseif hasfield(typeof(state), :ρ_matrix_avg_cuts)
        raw_cuts = getfield(state, :ρ_matrix_avg_cuts)
        if haskey(raw_cuts, :bond_pass_sample_count) && !isempty(raw_cuts[:bond_pass_sample_count])
            return max(Float64(raw_cuts[:bond_pass_sample_count][1]), 1.0)
        end
    end
    if hasfield(typeof(state), :t)
        return max(Float64(getfield(state, :t)), 1.0)
    end
    return 1.0
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

    bond_w = raw_sweep_weight(state)
    spatial_w = raw_sweep_weight(state)
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
            sample_total = sum(!isempty(vec) ? Float64(vec[1]) : 0.0 for vec in vectors)
            averaged[key] = [sample_total > 0 ? sample_total : sum(bond_weights)]
        elseif key == :bond_pass_spatial_sample_count
            sample_total = sum(!isempty(vec) ? Float64(vec[1]) : 0.0 for vec in vectors)
            averaged[key] = [sample_total > 0 ? sample_total : sum(spatial_weights)]
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

function finite_mean_nonan(values::AbstractVector{<:Real})
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    return isempty(vals) ? NaN : mean(vals)
end

function finite_sem_nonan(values::AbstractVector{<:Real})
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    n = length(vals)
    if n < 2
        return NaN
    end
    return std(vals; corrected=true) / sqrt(n)
end

function mean_sem_ci95(values::AbstractVector{<:Real})
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    n = length(vals)
    if n == 0
        return NaN, NaN, NaN, 0
    end
    μ = mean(vals)
    sem = n >= 2 ? std(vals; corrected=true) / sqrt(n) : NaN
    ci95 = isfinite(sem) ? 1.96 * sem : NaN
    return μ, sem, ci95, n
end

function metric_accumulator_from_values(values::AbstractVector{<:Real}; weight::Real=1.0)
    vals = Float64.(values)
    n = zeros(Float64, length(vals))
    w = zeros(Float64, length(vals))
    wsq = zeros(Float64, length(vals))
    sums = zeros(Float64, length(vals))
    sumsq = zeros(Float64, length(vals))
    finite_weight = isfinite(Float64(weight)) && Float64(weight) > 0 ? Float64(weight) : 1.0
    for i in eachindex(vals)
        v = vals[i]
        if isfinite(v)
            n[i] = 1.0
            w[i] = finite_weight
            wsq[i] = finite_weight^2
            sums[i] = finite_weight * v
            sumsq[i] = finite_weight * v^2
        end
    end
    return (n=n, w=w, wsq=wsq, sum=sums, sumsq=sumsq)
end

function combine_metric_accumulators(accumulators::Vector{<:NamedTuple})
    if isempty(accumulators)
        return (n=Float64[], w=Float64[], wsq=Float64[], sum=Float64[], sumsq=Float64[])
    end
    max_len = maximum(length(acc.n) for acc in accumulators)
    n = zeros(Float64, max_len)
    w = zeros(Float64, max_len)
    wsq = zeros(Float64, max_len)
    sums = zeros(Float64, max_len)
    sumsq = zeros(Float64, max_len)
    for acc in accumulators
        len = length(acc.n)
        n[1:len] .+= Float64.(acc.n)
        if hasproperty(acc, :w)
            w[1:len] .+= Float64.(acc.w)
        else
            w[1:len] .+= Float64.(acc.n)
        end
        if hasproperty(acc, :wsq)
            wsq[1:len] .+= Float64.(acc.wsq)
        else
            wsq[1:len] .+= Float64.(acc.n)
        end
        sums[1:len] .+= Float64.(acc.sum)
        sumsq[1:len] .+= Float64.(acc.sumsq)
    end
    return (n=n, w=w, wsq=wsq, sum=sums, sumsq=sumsq)
end

function stats_from_metric_accumulator(accum::NamedTuple)
    len = length(accum.n)
    means = fill(NaN, len)
    sems = fill(NaN, len)
    ci95s = fill(NaN, len)
    for i in 1:len
        count = Float64(accum.n[i])
        weight = hasproperty(accum, :w) ? Float64(accum.w[i]) : count
        weight_sq = hasproperty(accum, :wsq) ? Float64(accum.wsq[i]) : count
        if !(isfinite(weight) && weight > 0)
            continue
        end
        sum_val = Float64(accum.sum[i])
        sumsq_val = Float64(accum.sumsq[i])
        means[i] = sum_val / weight
        if isfinite(count) && count >= 2 && isfinite(weight_sq) && weight_sq > 0
            centered_ss = max(sumsq_val - (sum_val^2 / weight), 0.0)
            dof = weight - (weight_sq / weight)
            n_eff = weight^2 / weight_sq
            if dof > 0 && n_eff > 0
                sample_var = centered_ss / dof
                sems[i] = sqrt(sample_var / n_eff)
            end
        elseif isfinite(count) && count >= 2
            centered_ss = max(sumsq_val - (sum_val^2 / weight), 0.0)
            sample_var = centered_ss / (count - 1.0)
            sems[i] = sqrt(sample_var / count)
        end
        if isfinite(sems[i])
            ci95s[i] = 1.96 * sems[i]
        end
    end
    return means, sems, ci95s
end

function emit_metric_accumulator_summary!(
    metadata::Dict{Symbol,Vector{Float64}},
    accum::NamedTuple;
    mean_key::Symbol,
    sem_key::Symbol,
    ci95_key::Symbol,
    n_key::Symbol,
    weight_key::Union{Nothing,Symbol}=nothing,
    sum_key::Symbol,
    sumsq_key::Symbol,
    weightsq_key::Union{Nothing,Symbol}=nothing,
)
    means, sems, ci95s = stats_from_metric_accumulator(accum)
    metadata[mean_key] = means
    metadata[sem_key] = sems
    metadata[ci95_key] = ci95s
    metadata[n_key] = Float64.(accum.n)
    if !isnothing(weight_key)
        metadata[weight_key] = hasproperty(accum, :w) ? Float64.(accum.w) : Float64.(accum.n)
    end
    metadata[sum_key] = Float64.(accum.sum)
    metadata[sumsq_key] = Float64.(accum.sumsq)
    if !isnothing(weightsq_key)
        metadata[weightsq_key] = hasproperty(accum, :wsq) ? Float64.(accum.wsq) : Float64.(accum.n)
    end
end

function forcing_bonds_1d_from_state(state, L::Int)
    if !hasfield(typeof(state), :forcing)
        return Tuple{Int,Int}[], Float64[]
    end
    forcing_raw = getfield(state, :forcing)
    forcings = forcing_raw isa AbstractVector ? forcing_raw : [forcing_raw]
    bonds = Tuple{Int,Int}[]
    magnitudes = Float64[]
    for force in forcings
        if !hasproperty(force, :bond_indices) || !hasproperty(force, :magnitude)
            continue
        end
        bond_indices = getproperty(force, :bond_indices)
        if length(bond_indices) < 2
            continue
        end
        if length(bond_indices[1]) == 1 && length(bond_indices[2]) == 1
            b1 = mod1(Int(round(Float64(bond_indices[1][1]))), L)
            b2 = mod1(Int(round(Float64(bond_indices[2][1]))), L)
            push!(bonds, (b1, b2))
            push!(magnitudes, Float64(getproperty(force, :magnitude)))
        end
    end
    return bonds, magnitudes
end

function tracked_force_indices_from_stats(stats_dict::Dict{Symbol,Vector{Float64}}, magnitudes::Vector{Float64})
    n_forces = length(magnitudes)
    if haskey(stats_dict, :bond_pass_track_mask)
        mask = stats_dict[:bond_pass_track_mask]
        if length(mask) == n_forces
            tracked = [i for i in 1:n_forces if mask[i] > 0.5]
            if !isempty(tracked)
                return tracked
            end
        end
    end
    tracked = [i for i in 1:n_forces if abs(magnitudes[i]) > 0]
    return isempty(tracked) ? collect(1:n_forces) : tracked
end

function bond_centered_cut_1d_from_corr(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    L = size(corr_mat, 2)
    cut = 0.5 .* (Float64.(corr_mat[b1, :]) .+ Float64.(corr_mat[b2, :]))
    if smooth_diagonal
        b1_left = mod1(b1 - 1, L)
        b1_right = mod1(b1 + 1, L)
        b2_left = mod1(b2 - 1, L)
        b2_right = mod1(b2 + 1, L)
        cut[b1] = 0.5 * (cut[b1_left] + cut[b1_right])
        cut[b2] = 0.5 * (cut[b2_left] + cut[b2_right])
    end
    return cut
end

function bond_center_value_from_corr(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    cut = bond_centered_cut_1d_from_corr(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
    return 0.5 * (cut[b1] + cut[b2])
end

function rho_outer_from_avg(rho_avg::AbstractArray{<:Real})
    nd = ndims(rho_avg)
    left_shape = (size(rho_avg)..., ntuple(_ -> 1, nd)...)
    right_shape = (ntuple(_ -> 1, nd)..., size(rho_avg)...)
    rho_float = Float64.(rho_avg)
    return reshape(rho_float, left_shape...) .* reshape(rho_float, right_shape...)
end

function exact_connected_corr_full(state)
    if !hasfield(typeof(state), :ρ_matrix_avg_cuts) || !hasfield(typeof(state), :ρ_avg)
        return nothing
    end
    corr_cuts = getfield(state, :ρ_matrix_avg_cuts)
    if haskey(corr_cuts, AGG_CONNECTED_FULL_EXACT_KEY)
        return Float64.(corr_cuts[AGG_CONNECTED_FULL_EXACT_KEY])
    end
    if !haskey(corr_cuts, :full)
        return nothing
    end
    rho_avg = Float64.(getfield(state, :ρ_avg))
    full_corr = Float64.(corr_cuts[:full])
    return full_corr .- rho_outer_from_avg(rho_avg)
end

function extract_two_force_replica_metrics(state)
    if !hasfield(typeof(state), :ρ_avg) || !hasfield(typeof(state), :ρ_matrix_avg_cuts)
        return nothing
    end
    rho_avg = Float64.(getfield(state, :ρ_avg))
    ndims(rho_avg) == 1 || return nothing
    L = length(rho_avg)

    corr_mat = exact_connected_corr_full(state)
    if isnothing(corr_mat)
        return nothing
    end
    if ndims(corr_mat) != 2 || size(corr_mat, 1) != L || size(corr_mat, 2) != L
        return nothing
    end

    bonds, magnitudes = forcing_bonds_1d_from_state(state, L)
    if length(bonds) < 2
        return nothing
    end

    stats_dict, _, _ = bond_pass_stats_with_weights(state)
    tracked = tracked_force_indices_from_stats(stats_dict, magnitudes)
    if isempty(tracked)
        tracked = collect(1:length(bonds))
    end

    var_vals_smoothed = Float64[]
    var_vals_raw = Float64[]
    for idx in tracked
        if idx <= length(bonds)
            b1, b2 = bonds[idx]
            push!(var_vals_smoothed, bond_center_value_from_corr(corr_mat, b1, b2; smooth_diagonal=true))
            push!(var_vals_raw, bond_center_value_from_corr(corr_mat, b1, b2; smooth_diagonal=false))
        else
            push!(var_vals_smoothed, NaN)
            push!(var_vals_raw, NaN)
        end
    end

    j2_all = haskey(stats_dict, :bond_pass_total_sq_avg) ? Float64.(stats_dict[:bond_pass_total_sq_avg]) : Float64[]
    j2_vals = [idx <= length(j2_all) ? j2_all[idx] : NaN for idx in tracked]
    return (
        var_vals_smoothed=var_vals_smoothed,
        var_vals_raw=var_vals_raw,
        j2_vals=j2_vals,
    )
end

function has_metric_accumulator_stats(stats::AbstractDict, sum_key::Symbol, sumsq_key::Symbol, n_key::Symbol)
    return haskey(stats, sum_key) && haskey(stats, sumsq_key) && haskey(stats, n_key)
end

function metric_accumulator_from_stats_dict(stats::AbstractDict, sum_key::Symbol, sumsq_key::Symbol, n_key::Symbol;
                                            weight_key::Union{Nothing,Symbol}=nothing,
                                            weightsq_key::Union{Nothing,Symbol}=nothing)
    n_vals = Float64.(stats[n_key])
    w_vals = if !isnothing(weight_key) && haskey(stats, weight_key)
        Float64.(stats[weight_key])
    else
        copy(n_vals)
    end
    wsq_vals = if !isnothing(weightsq_key) && haskey(stats, weightsq_key)
        Float64.(stats[weightsq_key])
    else
        copy(n_vals)
    end
    return (
        n=n_vals,
        w=w_vals,
        wsq=wsq_vals,
        sum=Float64.(stats[sum_key]),
        sumsq=Float64.(stats[sumsq_key]),
    )
end

function two_force_replica_ci_metadata_from_accumulators(accum_bundles::Vector{<:NamedTuple})
    if isempty(accum_bundles)
        return Dict{Symbol,Vector{Float64}}()
    end

    metadata = Dict{Symbol,Vector{Float64}}()
    metadata[AGG_TWO_FORCE_REPLICA_COUNT_KEY] = [sum(Float64(bundle.replica_count) for bundle in accum_bundles)]

    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.var_slot_accum_smoothed for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_VAR_SLOT_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_VAR_SLOT_CI95_KEY,
        n_key=AGG_TWO_FORCE_VAR_SLOT_N_KEY,
        weight_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_VAR_SLOT_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHTSQ_KEY,
    )
    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.var_mean_accum_smoothed for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_VAR_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_VAR_MEAN_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_VAR_MEAN_CI95_KEY,
        n_key=AGG_TWO_FORCE_VAR_MEAN_N_KEY,
        weight_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_VAR_MEAN_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHTSQ_KEY,
    )
    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.var_slot_accum_raw for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_VAR_RAW_SLOT_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_VAR_RAW_SLOT_CI95_KEY,
        n_key=AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY,
        weight_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHTSQ_KEY,
    )
    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.var_mean_accum_raw for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_VAR_RAW_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_VAR_RAW_MEAN_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_VAR_RAW_MEAN_CI95_KEY,
        n_key=AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY,
        weight_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHTSQ_KEY,
    )
    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.j2_slot_accum for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_J2_SLOT_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_J2_SLOT_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_J2_SLOT_CI95_KEY,
        n_key=AGG_TWO_FORCE_J2_SLOT_N_KEY,
        weight_key=AGG_TWO_FORCE_J2_SLOT_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_J2_SLOT_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_J2_SLOT_WEIGHTSQ_KEY,
    )
    emit_metric_accumulator_summary!(
        metadata,
        combine_metric_accumulators([bundle.j2_mean_accum for bundle in accum_bundles]);
        mean_key=AGG_TWO_FORCE_J2_MEAN_KEY,
        sem_key=AGG_TWO_FORCE_J2_MEAN_SEM_KEY,
        ci95_key=AGG_TWO_FORCE_J2_MEAN_CI95_KEY,
        n_key=AGG_TWO_FORCE_J2_MEAN_N_KEY,
        weight_key=AGG_TWO_FORCE_J2_MEAN_WEIGHT_KEY,
        sum_key=AGG_TWO_FORCE_J2_MEAN_SUM_KEY,
        sumsq_key=AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY,
        weightsq_key=AGG_TWO_FORCE_J2_MEAN_WEIGHTSQ_KEY,
    )
    return metadata
end

function two_force_metric_accumulator_bundle(state)
    metrics = extract_two_force_replica_metrics(state)
    isnothing(metrics) && return nothing

    stats_dict, _, _ = bond_pass_stats_with_weights(state)
    state_weight = raw_sweep_weight(state)
    replica_count = if haskey(stats_dict, AGG_TWO_FORCE_REPLICA_COUNT_KEY) && !isempty(stats_dict[AGG_TWO_FORCE_REPLICA_COUNT_KEY])
        max(Float64(stats_dict[AGG_TWO_FORCE_REPLICA_COUNT_KEY][1]), 1.0)
    else
        1.0
    end

    var_slot_accum_smoothed = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_VAR_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_SLOT_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_VAR_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_SLOT_N_KEY;
                                           weight_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHTSQ_KEY) :
        metric_accumulator_from_values(metrics.var_vals_smoothed; weight=state_weight)
    var_mean_accum_smoothed = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_VAR_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_MEAN_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_VAR_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_MEAN_N_KEY;
                                           weight_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHTSQ_KEY) :
        metric_accumulator_from_values([finite_mean_nonan(metrics.var_vals_smoothed)]; weight=state_weight)
    var_slot_accum_raw = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY;
                                           weight_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHTSQ_KEY) :
        metric_accumulator_from_values(metrics.var_vals_raw; weight=state_weight)
    var_mean_accum_raw = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY;
                                           weight_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHTSQ_KEY) :
        metric_accumulator_from_values([finite_mean_nonan(metrics.var_vals_raw)]; weight=state_weight)
    j2_slot_accum = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_J2_SLOT_SUM_KEY, AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_J2_SLOT_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_J2_SLOT_SUM_KEY, AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_J2_SLOT_N_KEY;
                                           weight_key=AGG_TWO_FORCE_J2_SLOT_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_J2_SLOT_WEIGHTSQ_KEY) :
        metric_accumulator_from_values(metrics.j2_vals; weight=state_weight)
    j2_mean_accum = has_metric_accumulator_stats(stats_dict, AGG_TWO_FORCE_J2_MEAN_SUM_KEY, AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_J2_MEAN_N_KEY) ?
        metric_accumulator_from_stats_dict(stats_dict, AGG_TWO_FORCE_J2_MEAN_SUM_KEY, AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_J2_MEAN_N_KEY;
                                           weight_key=AGG_TWO_FORCE_J2_MEAN_WEIGHT_KEY,
                                           weightsq_key=AGG_TWO_FORCE_J2_MEAN_WEIGHTSQ_KEY) :
        metric_accumulator_from_values([finite_mean_nonan(metrics.j2_vals)]; weight=state_weight)

    return (
        replica_count=replica_count,
        var_slot_accum_smoothed=var_slot_accum_smoothed,
        var_mean_accum_smoothed=var_mean_accum_smoothed,
        var_slot_accum_raw=var_slot_accum_raw,
        var_mean_accum_raw=var_mean_accum_raw,
        j2_slot_accum=j2_slot_accum,
        j2_mean_accum=j2_mean_accum,
    )
end

function two_force_replica_ci_metadata(metric_rows::Vector{<:NamedTuple})
    if isempty(metric_rows)
        return Dict{Symbol,Vector{Float64}}()
    end

    row_weight(row) = hasproperty(row, :weight) ? max(Float64(getproperty(row, :weight)), 1.0) : 1.0
    accum_bundles = [
        (
            replica_count=1.0,
            var_slot_accum_smoothed=metric_accumulator_from_values(row.var_vals_smoothed; weight=row_weight(row)),
            var_mean_accum_smoothed=metric_accumulator_from_values([finite_mean_nonan(row.var_vals_smoothed)]; weight=row_weight(row)),
            var_slot_accum_raw=metric_accumulator_from_values(row.var_vals_raw; weight=row_weight(row)),
            var_mean_accum_raw=metric_accumulator_from_values([finite_mean_nonan(row.var_vals_raw)]; weight=row_weight(row)),
            j2_slot_accum=metric_accumulator_from_values(row.j2_vals; weight=row_weight(row)),
            j2_mean_accum=metric_accumulator_from_values([finite_mean_nonan(row.j2_vals)]; weight=row_weight(row)),
        )
        for row in metric_rows
    ]
    return two_force_replica_ci_metadata_from_accumulators(accum_bundles)
end


# Function to set up and run one independent simulation.
@everywhere function run_one_simulation_from_state(
    param,
    state,
    seed,
    n_sweeps;
    relaxed_ic::Bool=false,
    warmup_sweeps::Int=0,
    performance_mode::Bool=false,
    estimate_runtime::Bool=false,
    estimate_sample_size::Int=100,
    description::String="",
)
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
    if estimate_runtime
        estimated_time = estimate_run_time(dummy_state, param, n_sweeps, rng; sample_size=max(estimate_sample_size, 1))
        estimated_time_hours = estimated_time / 3600
        println("Estimated run time for this simulation: $estimated_time_hours hours")
    end
    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(dummy_state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=show_times,
                                                 save_times=save_times,
                                                 plot_flag=!performance_mode,
                                                 plotter=plot_sweep_runner,
                                                 warmup_sweeps=warmup_sweeps,
                                                 plot_label=description,
                                                 save_description=description,
                                                 show_progress=!performance_mode,
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

function performance_mode_from_args(args)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        println("Using configuration from file: $(args["config"])")
        params = YAML.load_file(args["config"])
    else
        println("No config file provided. Using default parameters.")
        params = get_default_params()
    end
    if get(args, "performance_mode", false)
        return true
    end
    return get_performance_mode(params, defaults)
end

function cluster_mode_from_args(args)
    return performance_mode_from_args(args)
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

function normalized_dims_tuple(raw_dims)
    if raw_dims isa Tuple
        return Tuple(Int.(collect(raw_dims)))
    elseif raw_dims isa AbstractVector
        return Tuple(Int.(collect(raw_dims)))
    end
    return (Int(raw_dims),)
end

function forcing_rate_scheme_from_param(param)
    if loaded_param_has_field(param, :forcing_rate_scheme)
        return String(loaded_param_field(param, :forcing_rate_scheme))
    end
    return FPDiffusive.LEGACY_FORCING_RATE_SCHEME
end

function canonicalize_loaded_param(param)
    return FPDiffusive.setParam(
        Float64(loaded_param_field(param, :γ)),
        normalized_dims_tuple(loaded_param_field(param, :dims)),
        Float64(loaded_param_field(param, :ρ₀)),
        Float64(loaded_param_field(param, :D)),
        String(loaded_param_field(param, :potential_type)),
        String(loaded_param_field(param, :fluctuation_type)),
        Float64(loaded_param_field(param, :potential_magnitude)),
        param_ffr_input(param);
        forcing_rate_scheme=forcing_rate_scheme_from_param(param),
    )
end

function aggregation_save_param_from_args(args; fallback_param=nothing)
    defaults = get_default_params()
    if haskey(args, "config") && !isnothing(args["config"])
        params = YAML.load_file(args["config"])
        dim_num = get(params, "dim_num", defaults["dim_num"])
        potential_type = get(params, "potential_type", defaults["potential_type"])
        fluctuation_type = get(params, "fluctuation_type", defaults["fluctuation_type"])
        potential_magnitude = get(params, "potential_magnitude", defaults["potential_magnitude"])
        D = get(params, "D", defaults["D"])
        L = get(params, "L", defaults["L"])
        ρ₀ = get(params, "ρ₀", defaults["ρ₀"])
        γ = get(params, "γ", defaults["γ"])
        forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
        _, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
        dims = ntuple(i -> L, dim_num)
        return FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                                    forcing_rate_scheme=forcing_rate_scheme)
    end
    if !isnothing(fallback_param)
        return canonicalize_loaded_param(fallback_param)
    end
    dim_num = defaults["dim_num"]
    L = defaults["L"]
    dims = ntuple(i -> L, dim_num)
    return FPDiffusive.setParam(defaults["γ"], dims, defaults["ρ₀"], defaults["D"],
                                defaults["potential_type"], defaults["fluctuation_type"],
                                defaults["potential_magnitude"], defaults["ffrs"];
                                forcing_rate_scheme=String(defaults["forcing_rate_scheme"]))
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
    state_weights = [raw_sweep_weight(state) for state in states]
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
    exact_connected_sum = nothing
    exact_connected_shape = nothing
    for (state, weight) in zip(states, state_weights)
        connected_full = exact_connected_corr_full(state)
        if isnothing(connected_full)
            exact_connected_sum = nothing
            exact_connected_shape = nothing
            break
        end
        if isnothing(exact_connected_sum)
            exact_connected_sum = zeros(Float64, size(connected_full))
            exact_connected_shape = size(connected_full)
        elseif size(connected_full) != exact_connected_shape
            exact_connected_sum = nothing
            exact_connected_shape = nothing
            break
        end
        exact_connected_sum .+= connected_full .* weight
    end
    if !isnothing(exact_connected_sum)
        mat_cuts_averaged[AGG_CONNECTED_FULL_EXACT_KEY] = exact_connected_sum ./ total_weight
    end

    total_t = Int(round(total_weight))
    avg_bond_pass_stats = average_bond_pass_stats(states)
    replica_accum_bundles = NamedTuple[]
    for state in states
        norm_state = normalize_state_for_aggregation(state)
        accum_bundle = two_force_metric_accumulator_bundle(norm_state)
        if !isnothing(accum_bundle)
            push!(replica_accum_bundles, accum_bundle)
        end
    end
    if !isempty(replica_accum_bundles)
        metadata = two_force_replica_ci_metadata_from_accumulators(replica_accum_bundles)
        for (key, values) in metadata
            avg_bond_pass_stats[key] = values
        end
    end
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
    save_param = aggregation_save_param_from_args(args; fallback_param=result_params[1])
    filename = save_state(dummy_state, save_param, dummy_state_save_dir;
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
    exact_connected_weighted = nothing
    exact_connected_shape = nothing
    total_weight = 0.0

    stats_list = Dict{Symbol,Vector{Float64}}[]
    bond_weights = Float64[]
    spatial_weights = Float64[]
    replica_accum_bundles = NamedTuple[]

    for (idx, state_path) in enumerate(state_files)
        println("Loading state for aggregation ($idx/$(length(state_files))): $state_path")
        @load state_path state param potential
        norm_state = normalize_state_for_aggregation(state)
        state_weight = raw_sweep_weight(norm_state)

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
            connected_full = exact_connected_corr_full(norm_state)
            if !isnothing(connected_full)
                exact_connected_weighted = zeros(Float64, size(connected_full))
                exact_connected_weighted .+= connected_full .* state_weight
                exact_connected_shape = size(connected_full)
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
            if !isnothing(exact_connected_weighted)
                connected_full = exact_connected_corr_full(norm_state)
                if isnothing(connected_full) || size(connected_full) != exact_connected_shape
                    exact_connected_weighted = nothing
                    exact_connected_shape = nothing
                else
                    exact_connected_weighted .+= connected_full .* state_weight
                end
            end
        end

        stats_dict, bond_w, spatial_w = bond_pass_stats_with_weights(norm_state)
        push!(stats_list, stats_dict)
        push!(bond_weights, bond_w)
        push!(spatial_weights, spatial_w)
        accum_bundle = two_force_metric_accumulator_bundle(norm_state)
        if !isnothing(accum_bundle)
            push!(replica_accum_bundles, accum_bundle)
        end
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
    if !isnothing(exact_connected_weighted)
        mat_cuts_averaged[AGG_CONNECTED_FULL_EXACT_KEY] = exact_connected_weighted ./ total_weight
    end

    total_t = Int(round(total_weight))
    avg_bond_pass_stats = average_bond_pass_stats_from_prepared(stats_list, bond_weights, spatial_weights)
    if !isempty(replica_accum_bundles)
        metadata = two_force_replica_ci_metadata_from_accumulators(replica_accum_bundles)
        for (key, values) in metadata
            avg_bond_pass_stats[key] = values
        end
    end
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
    save_param = aggregation_save_param_from_args(args; fallback_param=first_param)
    filename = save_state(dummy_state, save_param, dummy_state_save_dir;
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
    performance_mode = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
    estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
    estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
    description = get_description(params, defaults)
    params["show_times"] = Int[]
    params["save_times"] = Int[]
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
    forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))

    forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)

    ic = get(params, "ic", defaults["ic"])

    dims = ntuple(i -> L, dim_num)
    # ρ₀ = N / (L^dim_num)

    # Initialize simulation parameters and state.
    param = FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                                 forcing_rate_scheme=forcing_rate_scheme)
    density_int_type = resolve_density_int_type(args, params, defaults, param.N)
    position_int_type = resolve_position_int_type(args, params, defaults, dims)
    keep_directional_densities = get_keep_directional_densities(params, defaults)
    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=!performance_mode)
    state = FPDiffusive.setState(
        0,
        rng,
        param,
        T,
        potential,
        forcings;
        ic=ic,
        int_type=density_int_type,
        position_int_type=position_int_type,
        keep_directional_densities=keep_directional_densities,
        bond_pass_count_mode=bond_pass_count_mode,
    )
   
    if estimate_runtime
        estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
        estimated_time_hours = estimated_time / 3600
        println("Estimated run time for this simulation: $estimated_time_hours hours")
    end

    # Run the simulation (calculating correlations).
    dist, corr_mat_cuts = run_simulation!(state, param, n_sweeps, rng;
                                                 calc_correlations=false,
                                                 show_times=params["show_times"],
                                                 save_times=params["save_times"],
                                                 plot_flag=!performance_mode,
                                                 plotter=plot_sweep_runner,
                                                 plot_label=description,
                                                 save_description=description,
                                                 warmup_sweeps=warmup_sweeps,
                                                 show_progress=!performance_mode)
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
    explicit_save_tag = explicit_save_tag_from_args(args)
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
            param = maybe_override_forcing_rate_scheme(param, args, params)
            defaults = get_default_params()
            if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
                n_sweeps = args["continue_sweeps"]
                println("Continuing for specified $n_sweeps sweeps")
            else
                n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
                println("Continuing simulation for $n_sweeps more sweeps (from config/defaults)")
            end
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            performance_mode = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
            estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
            run_description = get_description(params, defaults)
            println("Warmup sweeps per restarted run: $warmup_sweeps")
            results = pmap(seed -> run_one_simulation_from_state(
                    param,
                    state,
                    seed,
                    n_sweeps;
                    warmup_sweeps=warmup_sweeps,
                    performance_mode=performance_mode,
                    estimate_runtime=estimate_runtime,
                    estimate_sample_size=estimate_sample_size,
                    description=run_description,
                ), seeds)
        elseif using_initial_state
            defaults = get_default_params()
            if haskey(args, "config") && !isnothing(args["config"])
                params = YAML.load_file(args["config"])
            else
                params = get_default_params()
            end
            initial_state, initial_param = reconcile_loaded_state_with_config!(initial_state, initial_param, args, params, defaults)
            n_sweeps = n_sweeps_from_args(args)
            warmup_sweeps = warmup_sweeps_from_args(args)
            performance_mode = performance_mode_from_args(args)
            estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
            results = pmap(seed -> begin
                    run_state = deepcopy(initial_state)
                    run_param = deepcopy(initial_param)
                    run_one_simulation_from_state(
                        run_param,
                        run_state,
                        seed,
                        n_sweeps;
                        relaxed_ic=true,
                        warmup_sweeps=warmup_sweeps,
                        performance_mode=performance_mode,
                        estimate_runtime=estimate_runtime,
                        estimate_sample_size=estimate_sample_size,
                        description=run_description,
                    )
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
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            performance_mode = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
            estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
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
            state, param = reconcile_loaded_state_with_config!(state, param, args, params, defaults)
            n_sweeps = get(params, "n_sweeps", defaults["n_sweeps"])
            warmup_sweeps = get_warmup_sweeps(params, defaults)
            performance_mode = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
            estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
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
            forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
            bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))
            n_sweeps           = get(params, "n_sweeps", defaults["n_sweeps"])
            warmup_sweeps      = get_warmup_sweeps(params, defaults)
            performance_mode   = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
            estimate_runtime   = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
            description        = get_description(params, defaults)
            forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
            ic = get(params, "ic", defaults["ic"])
            ic = get(params, "ic", defaults["ic"])
            
            dims = ntuple(i -> L, dim_num)
            # ρ₀ = N / prod(dims)
            
            param = FPDiffusive.setParam(γ, dims, ρ₀, D, potential_type, fluctuation_type, potential_magnitude, ffrs;
                                         forcing_rate_scheme=forcing_rate_scheme)
            density_int_type = resolve_density_int_type(args, params, defaults, param.N)
            position_int_type = resolve_position_int_type(args, params, defaults, dims)
            keep_directional_densities = get_keep_directional_densities(params, defaults)
            v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
            seed = rand(1:2^30)
            #rng = MersenneTwister(123)
            rng = MersenneTwister(seed)
            potential_plot_flag = !performance_mode
            potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type,rng=rng,plot_flag=potential_plot_flag)
            
            state = FPDiffusive.setState(
                0,
                rng,
                param,
                T,
                potential,
                forcings;
                ic=ic,
                int_type=density_int_type,
                position_int_type=position_int_type,
                keep_directional_densities=keep_directional_densities,
                bond_pass_count_mode=bond_pass_count_mode,
            )
        end
        
        defaults = get_default_params()
        has_explicit_show_times = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "show_times")
        show_times = get_show_times(params, defaults, warmup_sweeps; has_explicit_show_times=has_explicit_show_times)
        save_times = to_int_vector(get(params, "save_times", defaults["save_times"]), "save_times")
        save_dir = get(params, "save_dir", defaults["save_dir"])
        progress_file = String(get(params, "progress_file", ""))
        progress_interval_raw = get(params, "progress_interval", 25)
        progress_interval = max(Int(round(Float64(progress_interval_raw))), 1)
        snapshot_request_file = String(get(params, "snapshot_request_file", ""))
        snapshot_tag_prefix = String(get(params, "snapshot_tag_prefix", "snapshot"))
        if !(@isdefined warmup_sweeps)
            warmup_sweeps = get_warmup_sweeps(params, defaults)
        end
        if !(@isdefined performance_mode)
            performance_mode = get(args, "performance_mode", false) ? true : get_performance_mode(params, defaults)
        end
        if !(@isdefined estimate_runtime)
            estimate_runtime = get(args, "estimate_runtime", false) || get_estimate_runtime(params, defaults)
        end
        if !(@isdefined estimate_sample_size)
            estimate_sample_size = get_estimate_sample_size(params, defaults, get(args, "estimate_sample_size", nothing))
        end
        if performance_mode
            show_times = Int[]
        end

        if estimate_only
            estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
            estimated_time_hours = estimated_time / 3600
            println("Estimated run time for this simulation: $estimated_time_hours hours")
            return
        end

        if estimate_runtime
            estimated_time = estimate_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
            estimated_time_hours = estimated_time / 3600
            println("Estimated run time for this simulation: $estimated_time_hours hours")
        end
        
        # Register an exit hook to save state at exit.
        final_state_saved = Ref(false)
        atexit() do
            if final_state_saved[]
                return
            end
            println("\nSaving current state...")
            try
                SaveUtils.save_state(state, param, save_dir;
                                     tag=explicit_save_tag,
                                     ic=ic,
                                     relaxed_ic=using_initial_state,
                                     description=description)
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
                                                 plot_flag=!performance_mode,
                                                 plotter=plot_sweep_runner,
                                                 plot_label=description,
                                                 save_description=description,
                                                 warmup_sweeps=warmup_sweeps,
                                                 show_progress=!performance_mode,
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
        
        filename = save_state(state, param, save_dir;
                              tag=explicit_save_tag,
                              ic=ic,
                              relaxed_ic=using_initial_state,
                              description=description)
        final_state_saved[] = true
        println("Final state saved to: ", filename)
   end
end

main()
