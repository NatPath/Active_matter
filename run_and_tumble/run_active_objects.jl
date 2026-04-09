############### run_active_objects.jl ###############
############### Diffusive Active Objects (1D) ###############

using Dates
using Printf

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

const RUN_ACTIVE_OBJECTS_HEADLESS = begin
    env_headless = _parse_bool_literal(get(ENV, "RUN_ACTIVE_OBJECTS_HEADLESS", "0")) === true
    cli_performance_mode = "--performance_mode" in ARGS
    env_headless || cli_performance_mode || _config_requests_performance_mode()
end

if !RUN_ACTIVE_OBJECTS_HEADLESS && haskey(ENV, "WSL_DISTRO_NAME") && get(ENV, "QT_QPA_PLATFORM", "") in ("", "wayland", "wayland-egl")
    ENV["QT_QPA_PLATFORM"] = "xcb"
end
if !RUN_ACTIVE_OBJECTS_HEADLESS
    using Plots
end

using Random
using ProgressMeter
using JLD2
using YAML
using ArgParse

const SRC_ROOT = joinpath(@__DIR__, "src")
const COMMON_DIR = joinpath(SRC_ROOT, "common")
const DIFFUSIVE_DIR = joinpath(SRC_ROOT, "diffusive")
const ACTIVE_OBJECTS_DIR = joinpath(SRC_ROOT, "active_objects")

include(joinpath(DIFFUSIVE_DIR, "modules_diffusive_no_activity.jl"))
include(joinpath(ACTIVE_OBJECTS_DIR, "modules_active_objects.jl"))
include(joinpath(COMMON_DIR, "save_utils.jl"))

using .Potentials
using .FPDiffusive
using .FPActiveObjects
using .SaveUtils

if !RUN_ACTIVE_OBJECTS_HEADLESS
    include(joinpath(COMMON_DIR, "plot_utils.jl"))
    using .PlotUtils
end

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--config"
            help = "Configuration file path"
            required = false
        "--continue"
            help = "Path to a saved active-object state to continue from"
            required = false
        "--continue_sweeps"
            help = "Number of sweeps to continue for (overrides config)"
            arg_type = Int
            required = false
        "--save_tag"
            help = "Optional save tag override"
            arg_type = String
            required = false
        "--performance_mode"
            help = "Disable plotting and progress bars for lean runtime"
            action = :store_true
        "--estimate_only"
            help = "Estimate runtime and exit without running the simulation"
            action = :store_true
        "--estimate_runtime"
            help = "Estimate runtime before running the simulation"
            action = :store_true
        "--estimate_sample_size"
            help = "Number of sample sweeps to use for runtime estimation"
            arg_type = Int
            required = false
    end
    return parse_args(settings)
end

function get_default_params()
    return Dict(
        "dim_num" => 1,
        "L" => 128,
        "ρ₀" => 0.5,
        "D" => 1.0,
        "T" => 1.0,
        "γ" => 0.0,
        "n_sweeps" => 10_000,
        "warmup_sweeps" => 0,
        "show_times" => Int[],
        "show_on_object_move" => false,
        "performance_mode" => false,
        "cluster_mode" => false,
        "estimate_runtime" => false,
        "estimate_sample_size" => 100,
        "description" => "",
        "potential_type" => "zero",
        "fluctuation_type" => "no-fluctuation",
        "potential_magnitude" => 0.0,
        "save_dir" => "saved_states_active_objects",
        "forcing_type" => "center_bond_x",
        "forcing_types" => ["center_bond_x"],
        "forcing_magnitude" => 1.0,
        "forcing_magnitudes" => [1.0],
        "ffr" => 1.0,
        "ffrs" => [1.0],
        "forcing_direction_flags" => [true],
        "forcing_rate_scheme" => "legacy_penalty",
        "bond_pass_count_mode" => "all_forcing_bonds",
        "ic" => "random",
        "int_type" => "Int32",
        "plot_final" => true,
        "save_final_plot" => true,
        "live_plot" => false,
        "live_plot_interval" => 100,
        "save_live_plot" => true,
        "object_motion_scheme" => "hard_refresh",
        "object_refresh_sweeps" => 50,
        "object_memory_sweeps" => 50.0,
        "object_kappa" => 0.2,
        "object_D0" => 0.0,
        "object_history_interval" => 1,
        "object_history_on_move_only" => false,
    )
end

to_int(value, key_name::String) = value isa Number ? Int(round(Float64(value))) : error("$key_name must be numeric.")
to_float(value, key_name::String) = value isa Number ? Float64(value) : error("$key_name must be numeric.")

function to_int_vector(value, key_name::String)
    if value isa Number
        return [Int(round(Float64(value)))]
    elseif value isa AbstractVector
        return [Int(round(Float64(v))) for v in value]
    end
    error("$key_name must be numeric or a list of numerics.")
end

function to_bool(value, key_name::String)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lowered = lowercase(strip(value))
        if lowered in ("true", "t", "1", "yes", "y")
            return true
        elseif lowered in ("false", "f", "0", "no", "n")
            return false
        end
    end
    error("$key_name must be boolean-like.")
end

function to_string_vector(value, key_name::String)
    if value isa AbstractString
        return [String(value)]
    elseif value isa AbstractVector
        return [String(v) for v in value]
    end
    error("$key_name must be a string or a list of strings.")
end

function to_float_vector(value, key_name::String)
    if value isa Number
        return [Float64(value)]
    elseif value isa AbstractVector
        return [Float64(v) for v in value]
    end
    error("$key_name must be numeric or a list of numerics.")
end

function to_bool_vector(value, key_name::String)
    if value isa Bool || value isa Number || value isa AbstractString
        return [to_bool(value, key_name)]
    elseif value isa AbstractVector
        return [to_bool(v, key_name) for v in value]
    end
    error("$key_name must be boolean-like or a list of boolean-like values.")
end

function expand_to_length(values::Vector{T}, n::Int, key_name::String) where {T}
    if length(values) == n
        return values
    elseif length(values) == 1 && n > 1
        return fill(values[1], n)
    end
    error("$key_name must have length 1 or length $n.")
end

function resolve_int_type(params, defaults)
    raw = String(get(params, "int_type", defaults["int_type"]))
    lowered = lowercase(strip(raw))
    if lowered == "int32"
        return Int32
    elseif lowered == "int64"
        return Int64
    end
    error("Unsupported int_type: $raw. Use Int32 or Int64.")
end

function forcing_bond_indices_from_type(forcing_type::AbstractString, dim_num::Int, L::Int)
    dim_num == 1 || error("run_active_objects currently supports only dim_num=1.")
    if forcing_type == "center_bond_x"
        return ([L ÷ 2], [L ÷ 2 + 1])
    end
    error("Unsupported forcing_type: $forcing_type")
end

function parse_forcing_bond_pair(raw_pair, dim_num::Int, L::Int)
    dim_num == 1 || error("run_active_objects currently supports only dim_num=1.")
    if !(raw_pair isa AbstractVector) || length(raw_pair) != 2
        error("Each forcing_bond_pairs entry must contain exactly two endpoints.")
    end
    return ([mod1(Int(raw_pair[1]), L)], [mod1(Int(raw_pair[2]), L)])
end

function infer_requested_force_count(params, defaults)
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

function inferred_bond_indices_list(params, defaults, dim_num::Int, L::Int)
    requested_force_count = infer_requested_force_count(params, defaults)
    forcing_type = String(get(params, "forcing_type", defaults["forcing_type"]))
    if requested_force_count == 1
        return [forcing_bond_indices_from_type(forcing_type, dim_num, L)]
    end

    has_distance = haskey(params, "distance_between_forces") || haskey(params, "forcing_distance_d")
    if requested_force_count == 2 && has_distance
        d_raw = haskey(params, "distance_between_forces") ? params["distance_between_forces"] : params["forcing_distance_d"]
        d = mod(Int(round(Float64(d_raw))), L)
        d <= L - 2 || error("For two inferred active-object bonds, forcing_distance_d must satisfy 0 <= d <= L - 2. Got d=$(d) for L=$(L).")
        left_site = if haskey(params, "forcing_base_site")
            mod1(Int(round(Float64(params["forcing_base_site"]))), L)
        else
            mod1((L ÷ 2) - fld(d + 1, 2), L)
        end
        right_site = mod1(left_site + d + 1, L)
        return [
            ([left_site], [mod1(left_site + 1, L)]),
            ([right_site], [mod1(right_site + 1, L)]),
        ]
    end

    error("Unable to infer active-object bond locations. Provide forcing_bond_pairs, explicit forcing_types, or for two objects set distance_between_forces.")
end

function build_forcings_and_ffrs(params, defaults, dim_num::Int, L::Int)
    bond_indices_list = Tuple{Vector{Int},Vector{Int}}[]
    has_distance = haskey(params, "distance_between_forces") || haskey(params, "forcing_distance_d")

    if haskey(params, "forcing_bond_pairs") && has_distance
        error("Use either forcing_bond_pairs or distance_between_forces, not both.")
    elseif haskey(params, "forcing_bond_pairs")
        raw_pairs = params["forcing_bond_pairs"]
        raw_pairs isa AbstractVector || error("forcing_bond_pairs must be a list.")
        for raw_pair in raw_pairs
            push!(bond_indices_list, parse_forcing_bond_pair(raw_pair, dim_num, L))
        end
    elseif has_distance
        append!(bond_indices_list, inferred_bond_indices_list(params, defaults, dim_num, L))
    elseif haskey(params, "forcing_types") || haskey(params, "forcing_type")
        forcing_types_raw = haskey(params, "forcing_types") ? params["forcing_types"] : get(params, "forcing_type", defaults["forcing_type"])
        for forcing_type in to_string_vector(forcing_types_raw, "forcing_types")
            push!(bond_indices_list, forcing_bond_indices_from_type(forcing_type, dim_num, L))
        end
    else
        append!(bond_indices_list, inferred_bond_indices_list(params, defaults, dim_num, L))
    end

    n_forces = length(bond_indices_list)
    n_forces == 0 && return Potentials.BondForce[], Float64[]

    forcing_magnitudes_raw = haskey(params, "forcing_magnitudes") ? params["forcing_magnitudes"] : get(params, "forcing_magnitude", defaults["forcing_magnitude"])
    ffrs_raw = haskey(params, "ffrs") ? params["ffrs"] : get(params, "ffr", defaults["ffr"])
    direction_flags_raw = haskey(params, "forcing_direction_flags") ? params["forcing_direction_flags"] : get(params, "forcing_direction_flags", defaults["forcing_direction_flags"])

    forcing_magnitudes = expand_to_length(to_float_vector(forcing_magnitudes_raw, "forcing_magnitudes"), n_forces, "forcing_magnitudes")
    ffrs = expand_to_length(to_float_vector(ffrs_raw, "ffrs"), n_forces, "ffrs")
    direction_flags = expand_to_length(to_bool_vector(direction_flags_raw, "forcing_direction_flags"), n_forces, "forcing_direction_flags")

    forcings = [Potentials.setBondForce(bond_indices_list[i], direction_flags[i], forcing_magnitudes[i]) for i in 1:n_forces]
    return forcings, ffrs
end

function description_from_params(params, defaults)
    raw_description = get(params, "description", defaults["description"])
    isnothing(raw_description) && return ""
    return String(strip(String(raw_description)))
end

function performance_mode_from_params(params, defaults, cli_override::Bool)
    if cli_override
        return true
    elseif haskey(params, "performance_mode")
        return to_bool(params["performance_mode"], "performance_mode")
    elseif haskey(params, "cluster_mode")
        return to_bool(params["cluster_mode"], "cluster_mode")
    end
    return to_bool(get(defaults, "performance_mode", false), "performance_mode")
end

function load_params(args)
    defaults = get_default_params()
    params = if haskey(args, "config") && !isnothing(args["config"])
        YAML.load_file(args["config"])
    else
        deepcopy(defaults)
    end
    return params, defaults
end

function estimate_runtime_from_params(args, params, defaults)
    if get(args, "estimate_runtime", false)
        return true
    elseif haskey(params, "estimate_runtime")
        return to_bool(params["estimate_runtime"], "estimate_runtime")
    end
    return to_bool(get(defaults, "estimate_runtime", false), "estimate_runtime")
end

function estimate_sample_size_from_params(args, params, defaults)
    if haskey(args, "estimate_sample_size") && !isnothing(args["estimate_sample_size"])
        return max(Int(args["estimate_sample_size"]), 1)
    end
    return max(to_int(get(params, "estimate_sample_size", defaults["estimate_sample_size"]), "estimate_sample_size"), 1)
end

function estimate_active_object_run_time(state, param, n_sweeps, rng; sample_size::Int=100)
    sample_sweeps = max(min(Int(sample_size), Int(n_sweeps)), 1)
    println("Estimating active-object run time using $(sample_sweeps) sweeps...")
    state_copy = deepcopy(state)
    warmup_sweeps = min(5, sample_sweeps)
    for _ in 1:warmup_sweeps
        update_and_compute_correlations!(state_copy, param, nothing, state_copy.t + 1, rng; collect_statistics=true)
        rightward_counts, leftward_counts = FPDiffusive.latest_bond_passage_counts(state_copy)
        FPActiveObjects.apply_object_dynamics!(state_copy, param, rng, rightward_counts, leftward_counts)
        FPActiveObjects.record_object_history!(state_copy, param)
    end
    t0 = time()
    for _ in 1:sample_sweeps
        update_and_compute_correlations!(state_copy, param, nothing, state_copy.t + 1, rng; collect_statistics=true)
        rightward_counts, leftward_counts = FPDiffusive.latest_bond_passage_counts(state_copy)
        FPActiveObjects.apply_object_dynamics!(state_copy, param, rng, rightward_counts, leftward_counts)
        FPActiveObjects.record_object_history!(state_copy, param)
    end
    elapsed = time() - t0
    avg_time = elapsed / sample_sweeps
    estimated_total = avg_time * n_sweeps
    println("Average time per active-object sweep: $(round(avg_time, digits=6)) sec")
    println("Estimated total run time for $(n_sweeps) sweeps: $(round(estimated_total, digits=2)) sec ($(round(estimated_total / 3600, digits=3)) hours)")
    flush(stdout)
    return estimated_total
end

function get_show_times(params, defaults, warmup_sweeps::Int; has_explicit_show_times::Bool=false)
    raw_show_times = if has_explicit_show_times
        params["show_times"]
    else
        to_int_vector(get(defaults, "show_times", Int[]), "show_times") .+ warmup_sweeps
    end
    return sort(unique(to_int_vector(raw_show_times, "show_times")))
end

function create_state_from_params(params, defaults, rng)
    dim_num = to_int(get(params, "dim_num", defaults["dim_num"]), "dim_num")
    dim_num == 1 || error("run_active_objects currently supports only dim_num=1.")

    L = to_int(get(params, "L", defaults["L"]), "L")
    dims = (L,)
    ρ₀ = to_float(get(params, "ρ₀", defaults["ρ₀"]), "ρ₀")
    D = to_float(get(params, "D", defaults["D"]), "D")
    T = to_float(get(params, "T", defaults["T"]), "T")
    γ = to_float(get(params, "γ", defaults["γ"]), "γ")
    potential_type = String(get(params, "potential_type", defaults["potential_type"]))
    fluctuation_type = String(get(params, "fluctuation_type", defaults["fluctuation_type"]))
    potential_magnitude = to_float(get(params, "potential_magnitude", defaults["potential_magnitude"]), "potential_magnitude")
    forcing_rate_scheme = String(get(params, "forcing_rate_scheme", defaults["forcing_rate_scheme"]))
    bond_pass_count_mode = String(get(params, "bond_pass_count_mode", defaults["bond_pass_count_mode"]))
    ic = String(get(params, "ic", defaults["ic"]))
    int_type = resolve_int_type(params, defaults)

    forcings, ffrs = build_forcings_and_ffrs(params, defaults, dim_num, L)
    param = FPActiveObjects.setParam(
        γ,
        dims,
        ρ₀,
        D,
        potential_type,
        fluctuation_type,
        potential_magnitude,
        ffrs;
        forcing_rate_scheme=forcing_rate_scheme,
        object_motion_scheme=String(get(params, "object_motion_scheme", defaults["object_motion_scheme"])),
        object_refresh_sweeps=to_int(get(params, "object_refresh_sweeps", defaults["object_refresh_sweeps"]), "object_refresh_sweeps"),
        object_memory_sweeps=to_float(get(params, "object_memory_sweeps", get(params, "object_refresh_sweeps", defaults["object_memory_sweeps"])), "object_memory_sweeps"),
        object_kappa=to_float(get(params, "object_kappa", defaults["object_kappa"]), "object_kappa"),
        object_D0=to_float(get(params, "object_D0", defaults["object_D0"]), "object_D0"),
        object_history_interval=to_int(get(params, "object_history_interval", defaults["object_history_interval"]), "object_history_interval"),
        object_history_on_move_only=to_bool(get(params, "object_history_on_move_only", defaults["object_history_on_move_only"]), "object_history_on_move_only"),
    )

    v_args = Potentials.potential_args(potential_type, dims; magnitude=potential_magnitude)
    potential = Potentials.choose_potential(v_args, dims; fluctuation_type=fluctuation_type, rng=rng, plot_flag=!RUN_ACTIVE_OBJECTS_HEADLESS)
    state = FPActiveObjects.setState(
        0,
        rng,
        param,
        T,
        potential,
        forcings;
        ic=ic,
        int_type=int_type,
        bond_pass_count_mode=bond_pass_count_mode,
    )
    return state, param
end

function load_active_object_state(path::AbstractString)
    println("Loading active-object state from $path")
    @load path state param potential
    state.potential = potential
    return state, param
end

function n_sweeps_from_continue(args, params, defaults)
    if haskey(args, "continue_sweeps") && !isnothing(args["continue_sweeps"])
        return Int(args["continue_sweeps"])
    elseif haskey(params, "n_sweeps")
        return to_int(params["n_sweeps"], "n_sweeps")
    elseif haskey(defaults, "n_sweeps")
        return to_int(defaults["n_sweeps"], "n_sweeps")
    end
    error("Provide n_sweeps in the config or via --continue_sweeps when continuing a saved state.")
end

function run_active_object_simulation!(
    state,
    param,
    n_sweeps,
    rng;
    warmup_sweeps::Int=0,
    show_progress::Bool=true,
    show_times=Int[],
    show_on_object_move::Bool=false,
    plot_flag::Bool=false,
    plotter=nothing,
    plot_label::AbstractString="",
    live_plot::Bool=false,
    live_plot_interval::Int=100,
    live_plot_path::Union{Nothing,AbstractString}=nothing,
)
    println("Starting active-object simulation")
    progress = show_progress ? Progress(n_sweeps) : nothing
    t_init = state.t + 1
    t_end = state.t + n_sweeps
    warmup_sweeps = max(warmup_sweeps, 0)
    live_plot_interval = max(live_plot_interval, 1)
    show_time_lookup = normalize_time_lookup(show_times)
    plot_label = String(plot_label)
    if live_plot && isnothing(show_time_lookup)
        show_time_lookup = Set{Int}(collect(t_init:live_plot_interval:t_end))
        push!(show_time_lookup, t_end)
    end

    for sweep in t_init:t_end
        sweep_since_start = sweep - t_init + 1
        if warmup_sweeps > 0 && sweep_since_start == warmup_sweeps + 1
            FPActiveObjects.reset_measurement_statistics_preserve_time!(state)
            println("Warmup complete at sweep $sweep. Reset measurement statistics; object memory state preserved.")
        end

        update_and_compute_correlations!(state, param, nothing, sweep, rng; collect_statistics=true)
        rightward_counts, leftward_counts = FPDiffusive.latest_bond_passage_counts(state)
        move_deltas = FPActiveObjects.apply_object_dynamics!(state, param, rng, rightward_counts, leftward_counts)
        FPActiveObjects.record_object_history!(state, param)

        moved_this_sweep = any(!iszero, move_deltas)
        should_plot_this_sweep = plot_flag && !isnothing(plotter) && (
            (!isnothing(show_time_lookup) && in(sweep, show_time_lookup)) ||
            (show_on_object_move && moved_this_sweep)
        )
        if should_plot_this_sweep
            plot_obj = plotter(sweep, state, param; label=plot_label)
            if !isnothing(live_plot_path)
                mkpath(dirname(String(live_plot_path)))
                savefig(plot_obj, String(live_plot_path))
            end
        end

        if show_progress && !isnothing(progress)
            next!(progress)
        end
    end

    println("Active-object simulation complete")
    return state.ρ_avg, state.ρ_matrix_avg_cuts
end

function sanitize_filename_token(value::AbstractString)
    token = replace(strip(value), r"[^A-Za-z0-9._-]+" => "-")
    token = replace(token, r"-{2,}" => "-")
    token = replace(token, r"^-+" => "")
    token = replace(token, r"-+$" => "")
    return isempty(token) ? "active_objects" : token
end

function active_object_filename_suffix(param)
    scheme_raw = lowercase(strip(String(param.object_motion_scheme)))
    scheme_token = if scheme_raw == FPActiveObjects.HARD_REFRESH_SCHEME
        "obj-hr"
    elseif scheme_raw == FPActiveObjects.EXPONENTIAL_MEMORY_SCHEME
        "obj-em"
    elseif scheme_raw == FPActiveObjects.PER_HOP_PROBABILITY_SCHEME
        "obj-hop"
    else
        "obj-" * sanitize_filename_token(scheme_raw)
    end
    return @sprintf(
        "%s_or-%d_om-%.1f_ok-%.2e_oD0-%.2e",
        scheme_token,
        Int(param.object_refresh_sweeps),
        Float64(param.object_memory_sweeps),
        Float64(param.object_kappa),
        Float64(param.object_D0),
    )
end

function active_object_file_description(description::AbstractString, param)
    obj_suffix = active_object_filename_suffix(param)
    desc = strip(String(description))
    isempty(desc) && return obj_suffix
    return string(desc, "_", obj_suffix)
end

function build_live_plot_path(save_dir, param; description="", save_tag=nothing)
    prefix = sanitize_filename_token(active_object_file_description(description, param))
    tag = isnothing(save_tag) ? "live" : sanitize_filename_token(String(save_tag))
    return joinpath(save_dir, "$(prefix)_live_$(tag).png")
end

@inline direction_arrow(direction_flag::Bool) = direction_flag ? "→" : "←"

function object_direction_summary(direction_flags::AbstractVector{Bool})
    isempty(direction_flags) && return ""
    return join(["obj $(idx) $(direction_arrow(direction_flags[idx]))" for idx in eachindex(direction_flags)], ", ")
end

function plot_active_object_live_snapshot(state, param; label="", sweep=state.t)
    L = param.dims[1]
    x_range = 1:L
    stats = state.object_stats
    left_sites = FPActiveObjects.object_left_sites(state, param)
    bonds, direction_flags, _ = PlotUtils.force_bond_sites_1d(state, L)
    n_objects = length(left_sites)
    colors = [:red, :blue, :darkgreen, :orange, :magenta, :cyan]

    inst_density = Float64.(vec(state.ρ))
    y_max = isempty(inst_density) ? 1.0 : maximum(inst_density)
    if !isfinite(y_max) || y_max <= 0
        y_max = 1.0
    end
    y_upper = max(1.15 * y_max, 1.0)
    y_marker = 0.94 * y_upper
    y_label = 0.985 * y_upper

    p_inst = bar(
        x_range,
        inst_density;
        title=isempty(label) ? "Instantaneous density (sweep $(sweep))" : "Instantaneous density (sweep $(sweep)) | $(label)",
        xlabel="Site",
        ylabel="ρ(x,t)",
        color=:steelblue,
        alpha=0.75,
        bar_width=0.9,
        legend=false,
        framestyle=:box,
        grid=:y,
        xlim=(1, L),
        ylim=(0, y_upper),
    )

    PlotUtils.annotate_force_markers_minimal_1d!(p_inst, bonds; direction_flags=direction_flags, colors=colors)
    for (obj_idx, left_site) in enumerate(left_sites)
        color = colors[mod1(obj_idx, length(colors))]
        right_site = mod1(left_site + 1, L)
        if right_site == left_site + 1
            plot!(p_inst, [left_site, right_site], [y_marker, y_marker], color=color, lw=4, alpha=0.95, label=false)
        else
            scatter!(p_inst, [left_site, right_site], [y_marker, y_marker], color=color, markersize=4, label=false)
        end
        dir_text = obj_idx <= length(direction_flags) ? direction_arrow(direction_flags[obj_idx]) : "?"
        annotate!(p_inst, (left_site, y_label, text("obj $(obj_idx) $(dir_text)", color, 8)))
    end

    sweeps = stats[:history_sweeps]
    left_site_history = stats[:history_left_sites]
    isempty(left_site_history) && return p_inst

    dir_summary = object_direction_summary(direction_flags)
    has_edge_distance_history = haskey(stats, :history_min_edge_distance) &&
                                length(stats[:history_min_edge_distance]) == length(sweeps)
    has_gap_history = haskey(stats, :history_forward_gap) &&
                      haskey(stats, :history_min_gap) &&
                      length(stats[:history_forward_gap]) == length(sweeps) &&
                      length(stats[:history_min_gap]) == length(sweeps)
    p_history = if n_objects == 2 &&
                   haskey(stats, :history_forward_distance) &&
                   haskey(stats, :history_min_distance) &&
                   length(stats[:history_forward_distance]) == length(sweeps) &&
                   length(stats[:history_min_distance]) == length(sweeps)
        distance_series = has_edge_distance_history ? stats[:history_min_edge_distance] :
                          (has_gap_history ? stats[:history_min_gap] : stats[:history_min_distance])
        ylabel_text = has_edge_distance_history ? "Edge Distance" :
                      (has_gap_history ? "Bond Gap" : "Distance")
        distance_label = has_edge_distance_history ? "minimum edge distance" :
                         (has_gap_history ? "minimum bond gap" : "minimum arc distance")
        title_text = has_edge_distance_history ? "Two-object edge distance so far" :
                     (has_gap_history ? "Two-object bond gap so far" : "Two-object separation so far")
        plot(
            sweeps,
            distance_series;
            lw=2.4,
            color=:purple,
            label=distance_label,
            title=isempty(dir_summary) ? title_text : "$(title_text) | $(dir_summary)",
            xlabel="Sweep",
            ylabel=ylabel_text,
            framestyle=:box,
            grid=:y,
        )
    else
        p = plot(
            title="Active-object positions so far",
            xlabel="Sweep",
            ylabel="Left bond site",
            framestyle=:box,
            grid=:y,
            legend=:outertopright,
        )
        for obj_idx in 1:n_objects
            series = [sites[obj_idx] for sites in left_site_history]
            dir_text = obj_idx <= length(direction_flags) ? " $(direction_arrow(direction_flags[obj_idx]))" : ""
            plot!(p, sweeps, series, lw=2.0, color=colors[mod1(obj_idx, length(colors))], label="object $(obj_idx)$(dir_text)")
        end
        p
    end

    return plot(p_inst, p_history, layout=(2, 1), size=(1200, 800))
end

function plot_active_object_sweep(sweep, state, param; label="")
    p = plot_active_object_live_snapshot(state, param; label=label, sweep=sweep)
    PlotUtils.present_plot!(p)
    return p
end

function plot_active_object_trajectories(state, param; label="")
    stats = state.object_stats
    sweeps = stats[:history_sweeps]
    left_site_history = stats[:history_left_sites]
    isempty(left_site_history) && return plot(title="Active-object trajectory (no recorded history)", axis=false, legend=false)

    bonds, direction_flags, _ = PlotUtils.force_bond_sites_1d(state, param.dims[1])
    n_objects = length(left_site_history[1])
    colors = [:red, :blue, :darkgreen, :orange, :magenta, :cyan]
    p_positions = plot(
        title=isempty(label) ? "Active-object positions" : "Active-object positions | $(label)",
        xlabel="Sweep",
        ylabel="Left bond site",
        framestyle=:box,
        grid=:y,
        legend=:outertopright,
    )
    for obj_idx in 1:n_objects
        series = [sites[obj_idx] for sites in left_site_history]
        dir_text = obj_idx <= length(direction_flags) ? " $(direction_arrow(direction_flags[obj_idx]))" : ""
        plot!(p_positions, sweeps, series, lw=2.0, color=colors[mod1(obj_idx, length(colors))], label="object $(obj_idx)$(dir_text)")
    end

    if n_objects == 1
        return p_positions
    end

    has_edge_distance_history = haskey(stats, :history_min_edge_distance) &&
                                length(stats[:history_min_edge_distance]) == length(sweeps)
    has_gap_history = haskey(stats, :history_forward_gap) &&
                      haskey(stats, :history_min_gap) &&
                      length(stats[:history_forward_gap]) == length(sweeps) &&
                      length(stats[:history_min_gap]) == length(sweeps)
    distance_series = has_edge_distance_history ? stats[:history_min_edge_distance] :
                      (has_gap_history ? stats[:history_min_gap] : stats[:history_min_distance])
    ylabel_text = has_edge_distance_history ? "Edge Distance" :
                  (has_gap_history ? "Bond Gap" : "Distance")
    distance_label = has_edge_distance_history ? "minimum edge distance" :
                     (has_gap_history ? "minimum bond gap" : "minimum arc distance")
    title_text = has_edge_distance_history ? "Two-object edge distance" :
                 (has_gap_history ? "Two-object bond gap" : "Two-object separation")
    dir_summary = object_direction_summary(direction_flags)
    p_distance = plot(
        sweeps,
        distance_series;
        lw=2.0,
        color=:purple,
        label=distance_label,
        title=isempty(dir_summary) ? title_text : "$(title_text) | $(dir_summary)",
        xlabel="Sweep",
        ylabel=ylabel_text,
        framestyle=:box,
        grid=:y,
    )
    return plot(p_positions, p_distance, layout=(2, 1), size=(1200, 850))
end

function maybe_save_final_plots(state, param, save_dir; description="", save_tag=nothing, present_plots::Bool=true)
    RUN_ACTIVE_OBJECTS_HEADLESS && return nothing

    tag = isnothing(save_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : String(save_tag)
    prefix = sanitize_filename_token(active_object_file_description(description, param))

    traj_plot = plot_active_object_trajectories(state, param; label=description)
    if present_plots
        PlotUtils.present_plot!(traj_plot)
    end
    traj_path = joinpath(save_dir, "$(prefix)_object_trajectory_$(tag).png")
    savefig(traj_plot, traj_path)
    println("Saved active-object trajectory plot to $traj_path")

    avg_plot = PlotUtils.plot_average_density_and_correlation(state, param; label=description)
    if present_plots
        PlotUtils.present_plot!(avg_plot)
    end
    avg_path = joinpath(save_dir, "$(prefix)_average_observables_$(tag).png")
    savefig(avg_plot, avg_path)
    println("Saved average-observables plot to $avg_path")
    return nothing
end

function save_active_object_outputs(state, param, save_dir;
    save_tag=nothing,
    ic::AbstractString="random",
    description::AbstractString="",
    save_plots::Bool=false,
    present_plots::Bool=false,
    continued::Bool=false,
)
    filename_description = active_object_file_description(description, param)
    filename = SaveUtils.save_state(state, param, save_dir; tag=save_tag, ic=ic, description=filename_description)
    if continued
        println("Saved continued active-object state to $filename")
    else
        println("Saved active-object state to $filename")
    end
    if save_plots
        maybe_save_final_plots(state, param, save_dir; description=description, save_tag=save_tag, present_plots=present_plots)
    end
    return filename
end

function register_active_object_exit_hook(state, param, save_dir, saved_ref::Base.RefValue{Bool};
    save_tag=nothing,
    ic::AbstractString="random",
    description::AbstractString="",
    save_plots::Bool=false,
    continued::Bool=false,
)
    atexit() do
        if saved_ref[]
            return
        end
        println("\nSaving current active-object state...")
        try
            save_active_object_outputs(
                state,
                param,
                save_dir;
                save_tag=save_tag,
                ic=ic,
                description=description,
                save_plots=save_plots,
                present_plots=false,
                continued=continued,
            )
            println("Active-object outputs saved successfully")
        catch e
            println("Error saving active-object outputs: ", e)
        end
    end
    return nothing
end

function main()
    args = parse_commandline()
    defaults = get_default_params()
    performance_mode_cli = get(args, "performance_mode", false)

    if haskey(args, "continue") && !isnothing(args["continue"])
        continue_path = String(args["continue"])
        state, param = load_active_object_state(continue_path)
        params, defaults = load_params(args)
        has_config = haskey(args, "config") && !isnothing(args["config"])
        performance_mode = performance_mode_from_params(params, defaults, performance_mode_cli)
        estimate_only = get(args, "estimate_only", false)
        estimate_runtime = estimate_runtime_from_params(args, params, defaults)
        estimate_sample_size = estimate_sample_size_from_params(args, params, defaults)
        n_sweeps = n_sweeps_from_continue(args, params, defaults)
        warmup_sweeps = haskey(params, "warmup_sweeps") ? max(to_int(params["warmup_sweeps"], "warmup_sweeps"), 0) : 0
        description = haskey(params, "description") ? description_from_params(params, defaults) : ""
        save_dir = if has_config && haskey(params, "save_dir")
            String(params["save_dir"])
        else
            continue_dir = dirname(continue_path)
            isempty(continue_dir) ? get(defaults, "save_dir", "saved_states_active_objects") : continue_dir
        end
        plot_final = haskey(params, "plot_final") ? to_bool(params["plot_final"], "plot_final") : get(defaults, "plot_final", true)
        save_final_plot = haskey(params, "save_final_plot") ? to_bool(params["save_final_plot"], "save_final_plot") : get(defaults, "save_final_plot", true)
        has_explicit_show_times = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "show_times")
        show_times = get_show_times(params, defaults, warmup_sweeps; has_explicit_show_times=has_explicit_show_times)
        show_on_object_move = to_bool(get(params, "show_on_object_move", defaults["show_on_object_move"]), "show_on_object_move")
        live_plot = to_bool(get(params, "live_plot", defaults["live_plot"]), "live_plot")
        live_plot_interval = max(to_int(get(params, "live_plot_interval", defaults["live_plot_interval"]), "live_plot_interval"), 1)
        save_live_plot = to_bool(get(params, "save_live_plot", defaults["save_live_plot"]), "save_live_plot")
        live_plot_path = live_plot && save_live_plot ? build_live_plot_path(save_dir, param; description=description, save_tag=get(args, "save_tag", nothing)) : nothing
        if live_plot && RUN_ACTIVE_OBJECTS_HEADLESS
            println("live_plot requested, but plotting is headless in this run. Skipping live plot refresh.")
        end
        plotter = (!RUN_ACTIVE_OBJECTS_HEADLESS && !performance_mode) ? plot_active_object_sweep : nothing
        rng = MersenneTwister(rand(1:2^30))

        if estimate_only
            estimate_active_object_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
            return
        end
        if estimate_runtime
            estimate_active_object_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
        end

        final_outputs_saved = Ref(false)
        register_active_object_exit_hook(
            state,
            param,
            save_dir,
            final_outputs_saved;
            save_tag=get(args, "save_tag", nothing),
            ic="continued",
            description=description,
            save_plots=save_final_plot,
            continued=true,
        )
        Base.exit_on_sigint(false)
        Base.sigatomic_begin()
        ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)
        Base.sigatomic_end()

        try
            run_active_object_simulation!(
                state,
                param,
                n_sweeps,
                rng;
                warmup_sweeps=warmup_sweeps,
                show_progress=!performance_mode,
                show_times=show_times,
                show_on_object_move=show_on_object_move,
                plot_flag=!performance_mode,
                plotter=plotter,
                plot_label=description,
                live_plot=live_plot,
                live_plot_interval=live_plot_interval,
                live_plot_path=live_plot_path,
            )
        catch e
            if isa(e, InterruptException)
                println("\nInterrupt detected, initiating cleanup...")
                exit()
            else
                rethrow(e)
            end
        end

        save_active_object_outputs(
            state,
            param,
            save_dir;
            save_tag=get(args, "save_tag", nothing),
            ic="continued",
            description=description,
            save_plots=save_final_plot,
            present_plots=plot_final,
            continued=true,
        )
        final_outputs_saved[] = true
        return
    end

    params, defaults = load_params(args)
    performance_mode = performance_mode_from_params(params, defaults, performance_mode_cli)
    estimate_only = get(args, "estimate_only", false)
    estimate_runtime = estimate_runtime_from_params(args, params, defaults)
    estimate_sample_size = estimate_sample_size_from_params(args, params, defaults)
    description = description_from_params(params, defaults)
    warmup_sweeps = max(to_int(get(params, "warmup_sweeps", defaults["warmup_sweeps"]), "warmup_sweeps"), 0)
    n_sweeps = to_int(get(params, "n_sweeps", defaults["n_sweeps"]), "n_sweeps")
    plot_final = to_bool(get(params, "plot_final", defaults["plot_final"]), "plot_final")
    save_final_plot = to_bool(get(params, "save_final_plot", defaults["save_final_plot"]), "save_final_plot")
    has_explicit_show_times = haskey(args, "config") && !isnothing(args["config"]) && haskey(params, "show_times")
    show_times = get_show_times(params, defaults, warmup_sweeps; has_explicit_show_times=has_explicit_show_times)
    show_on_object_move = to_bool(get(params, "show_on_object_move", defaults["show_on_object_move"]), "show_on_object_move")
    live_plot = to_bool(get(params, "live_plot", defaults["live_plot"]), "live_plot")
    live_plot_interval = max(to_int(get(params, "live_plot_interval", defaults["live_plot_interval"]), "live_plot_interval"), 1)
    save_live_plot = to_bool(get(params, "save_live_plot", defaults["save_live_plot"]), "save_live_plot")
    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    live_plot_path = nothing
    if live_plot && RUN_ACTIVE_OBJECTS_HEADLESS
        println("live_plot requested, but plotting is headless in this run. Skipping live plot refresh.")
    end
    plotter = (!RUN_ACTIVE_OBJECTS_HEADLESS && !performance_mode) ? plot_active_object_sweep : nothing

    rng = MersenneTwister(rand(1:2^30))
    state, param = create_state_from_params(params, defaults, rng)

    if estimate_only
        estimate_active_object_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
        return
    end
    if estimate_runtime
        estimate_active_object_run_time(state, param, n_sweeps, rng; sample_size=estimate_sample_size)
    end

    if live_plot && save_live_plot
        live_plot_path = build_live_plot_path(save_dir, param; description=description, save_tag=get(args, "save_tag", nothing))
    end
    ic = String(get(params, "ic", defaults["ic"]))
    final_outputs_saved = Ref(false)
    register_active_object_exit_hook(
        state,
        param,
        save_dir,
        final_outputs_saved;
        save_tag=get(args, "save_tag", nothing),
        ic=ic,
        description=description,
        save_plots=save_final_plot,
        continued=false,
    )
    Base.exit_on_sigint(false)
    Base.sigatomic_begin()
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)
    Base.sigatomic_end()

    try
        run_active_object_simulation!(
            state,
            param,
            n_sweeps,
            rng;
            warmup_sweeps=warmup_sweeps,
            show_progress=!performance_mode,
            show_times=show_times,
            show_on_object_move=show_on_object_move,
            plot_flag=!performance_mode,
            plotter=plotter,
            plot_label=description,
            live_plot=live_plot,
            live_plot_interval=live_plot_interval,
            live_plot_path=live_plot_path,
        )
    catch e
        if isa(e, InterruptException)
            println("\nInterrupt detected, initiating cleanup...")
            exit()
        else
            rethrow(e)
        end
    end

    save_active_object_outputs(
        state,
        param,
        save_dir;
        save_tag=get(args, "save_tag", nothing),
        ic=ic,
        description=description,
        save_plots=save_final_plot,
        present_plots=plot_final,
        continued=false,
    )
    final_outputs_saved[] = true
end

main()
