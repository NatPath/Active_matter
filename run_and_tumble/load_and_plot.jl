using ArgParse
using Dates
using JLD2
using LinearAlgebra
using Plots
using Printf
using Statistics

isdefined(@__MODULE__, :Potentials) || include(joinpath(@__DIR__, "src", "common", "potentials.jl"))
include(joinpath(@__DIR__, "src", "diffusive", "modules_diffusive_no_activity.jl"))
include(joinpath(@__DIR__, "src", "ssep", "modules_ssep.jl"))
include(joinpath(@__DIR__, "src", "common", "plot_utils.jl"))

using .PlotUtils

const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask
const TWO_FORCE_J2_BASELINE_RHO_FACTOR = 0.0894
const TWO_FORCE_LOGLOG_TARGET_SLOPE = -2.0
const ENABLE_BASELINE_SEARCH_FIGURES = get(ENV, "LOAD_AND_PLOT_ENABLE_BASELINE_SEARCH_FIGURES", "1") == "1"
const TWO_FORCE_BOND_SERIES_COLORS = [:royalblue, :crimson, :seagreen, :darkorange, :mediumpurple, :sienna, :deeppink, :teal]
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
const RUN_FAMILY_TWO_FORCE_D = "two_force_d"
const RUN_FAMILY_SINGLE_ORIGIN_BOND = "single_origin_bond"
const RUN_FAMILY_SSEP = "ssep"
const RUN_FAMILY_DIFFUSIVE_1D_PMLR = "diffusive_1d_pmlr"
const RUN_FAMILY_DIFFUSIVE_2D_ORIGIN_BOND = "diffusive_2d_origin_bond"
const RUN_FAMILY_BASE_RELPATHS = Dict(
    RUN_FAMILY_TWO_FORCE_D => joinpath("runs", "two_force_d"),
    RUN_FAMILY_SINGLE_ORIGIN_BOND => joinpath("runs", "single_origin_bond"),
    RUN_FAMILY_SSEP => joinpath("runs", "ssep", "single_center_bond"),
    RUN_FAMILY_DIFFUSIVE_1D_PMLR => joinpath("runs", "diffusive_1d_pmlr"),
    RUN_FAMILY_DIFFUSIVE_2D_ORIGIN_BOND => joinpath("runs", "diffusive_2d_origin_bond"),
)
const RUN_ID_FAMILIES_IN_SEARCH_ORDER = [
    RUN_FAMILY_SSEP,
    RUN_FAMILY_TWO_FORCE_D,
    RUN_FAMILY_SINGLE_ORIGIN_BOND,
    RUN_FAMILY_DIFFUSIVE_1D_PMLR,
    RUN_FAMILY_DIFFUSIVE_2D_ORIGIN_BOND,
]

Base.@kwdef mutable struct LegacyPlotState
    legacy_normalized::Bool = true
    t::Int64 = 0
    particles::Vector{Any} = Any[]
    ρ::Any = nothing
    ρ₊::Any = nothing
    ρ₋::Any = nothing
    ρ_avg::Any = nothing
    ρ_matrix_avg_cuts::Any = nothing
    bond_pass_stats::Dict{Symbol,Vector{Float64}} = Dict{Symbol,Vector{Float64}}()
    T::Float64 = 0.0
    potential::Any = nothing
    forcing::Any = Any[]
end

function wildcard_to_regex(pattern::String)
    special = Set(['.', '^', '$', '+', '(', ')', '[', ']', '{', '}', '|', '\\'])
    io = IOBuffer()
    for c in pattern
        if c == '*'
            print(io, ".*")
        elseif c == '?'
            print(io, ".")
        elseif c in special
            print(io, '\\', c)
        else
            print(io, c)
        end
    end
    return Regex("^" * String(take!(io)) * "\$")
end

function collect_matching_files(dir::String, pattern::String; recursive::Bool=false)
    matcher = wildcard_to_regex(pattern)
    files = String[]

    if recursive
        for (root, _, names) in walkdir(dir)
            for name in names
                if occursin(matcher, name)
                    push!(files, joinpath(root, name))
                end
            end
        end
    else
        for name in readdir(dir)
            path = joinpath(dir, name)
            if isfile(path) && occursin(matcher, name)
                push!(files, path)
            end
        end
    end

    sort!(files)
    return files
end

function add_unique!(paths::Vector{String}, seen::Set{String}, candidate::String)
    full_path = abspath(candidate)
    if !(full_path in seen)
        push!(paths, full_path)
        push!(seen, full_path)
    end
end

function collect_input_files(inputs::Vector{String}, default_glob::String; recursive::Bool=false)
    files = String[]
    seen = Set{String}()

    for input in inputs
        if isfile(input)
            if endswith(lowercase(input), ".jld2")
                add_unique!(files, seen, input)
            else
                println("Skipping non-JLD2 file: ", input)
            end
            continue
        end

        if isdir(input)
            for match in collect_matching_files(input, default_glob; recursive=recursive)
                add_unique!(files, seen, match)
            end
            continue
        end

        if occursin('*', input) || occursin('?', input)
            dir = dirname(input)
            dir = isempty(dir) ? "." : dir
            pattern = basename(input)
            if isdir(dir)
                for match in collect_matching_files(dir, pattern; recursive=recursive)
                    add_unique!(files, seen, match)
                end
            else
                println("Skipping glob with missing directory: ", input)
            end
            continue
        end

        println("Skipping missing path: ", input)
    end

    sort!(files)
    return files
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "inputs"
            help = "Input path(s): JLD2 files, directories, or globs (optional when --run_id is used)"
            nargs = '*'
        "--mode"
            help = "Mode: single, two_force_d, or run_id (run_id auto-detects the supported run family)"
            default = "single"
        "--run_id"
            help = "Run folder name under supported runs/<family>/.../<run_id> folders"
            default = ""
        "--run_result_mode"
            help = "When using --run_id: managed, warmup, production, local_warmup, local_production, or auto"
            default = "auto"
        "--cluster_results_root"
            help = "Root directory containing fetched cluster run folders and fitting outputs"
            default = "cluster_results"
        "--glob"
            help = "Filename wildcard for directory inputs"
            default = "*.jld2"
        "--recursive"
            help = "Recursively scan directory inputs"
            action = :store_true
        "--out_dir"
            help = "Output directory"
            default = "results_figures/fitting"
        "--include_abs_mean_in_spatial_f_plot"
            help = "Include |<J>| in spatial J statistics panel"
            action = :store_true
        "--keep_diagonal_in_multiforce_cut"
            help = "Do not smooth diagonal points in bond-centered cuts"
            action = :store_true
        "--skip_per_state_sweep"
            help = "Skip per-state sweep/component plots"
            action = :store_true
        "--with_per_state_sweep"
            help = "When using --run_id, also export per-state sweep/component plots (default with run_id is analysis-only)"
            action = :store_true
        "--baseline_j2"
            help = @sprintf("Baseline override for (<J^2>-baseline) vs d analysis (default: auto %.6g*rho0^2)", TWO_FORCE_J2_BASELINE_RHO_FACTOR)
            arg_type = Float64
            default = NaN
        "--corr_model_c"
            help = "Fixed constant offset C in bond-correlation model fit (default 0)"
            arg_type = Float64
            default = 0.0
        "--collapse_power"
            help = "Power n used for cut-based data-collapse plots C(x,y)*y^n"
            arg_type = Float64
            default = 2.0
        "--collapse_indices"
            help = "Optional subset of positive cut offsets to use for data collapse, e.g. 8,16,32 or 20:10:80. Required for 2D x/diag cut collapse and for legacy/full-matrix 1D states without selected-cut metadata."
            default = ""
        "--bond_centered_collapse"
            help = "Use bond-centered coordinates/reflection for 1D data collapse; this is already the default when a single forcing bond and full correlation matrix are available"
            action = :store_true
    end
    return parse_args(s)
end

function candidate_run_bases(cluster_results_root::AbstractString, family::AbstractString)
    requested_root = abspath(String(cluster_results_root))
    repo_root = abspath(@__DIR__)
    base_rel = get(RUN_FAMILY_BASE_RELPATHS, String(family), nothing)
    isnothing(base_rel) && error("Unsupported run family '$family'.")
    bases = String[]
    seen = Set{String}()
    for root in (repo_root, requested_root)
        base = joinpath(root, base_rel)
        if !(base in seen)
            push!(bases, base)
            push!(seen, base)
        end
    end
    return bases
end

function run_result_mode_rel_paths(run_result_mode::AbstractString)
    run_result_mode_str = String(run_result_mode)
    if run_result_mode_str == "managed"
        return [("managed", "managed")]
    elseif run_result_mode_str == "warmup"
        return [("warmup", "warmup"), (joinpath("local", "warmup"), "local_warmup")]
    elseif run_result_mode_str == "production"
        return [("production", "production"), (joinpath("local", "production"), "local_production")]
    elseif run_result_mode_str == "local_warmup"
        return [(joinpath("local", "warmup"), "local_warmup")]
    elseif run_result_mode_str == "local_production"
        return [(joinpath("local", "production"), "local_production")]
    elseif run_result_mode_str == "auto"
        return [
            ("managed", "managed"),
            ("warmup", "warmup"),
            ("production", "production"),
            (joinpath("local", "warmup"), "local_warmup"),
            (joinpath("local", "production"), "local_production"),
        ]
    end
    error("--run_result_mode must be managed, warmup, production, local_warmup, local_production, or auto. Got '$run_result_mode_str'.")
end

function resolve_run_id_dir(run_id::AbstractString, cluster_results_root::AbstractString, run_result_mode::AbstractString;
                            preferred_families::AbstractVector{<:AbstractString}=String[])
    run_id_str = String(run_id)
    run_result_mode_str = String(run_result_mode)
    family_order = unique(vcat(String.(preferred_families), RUN_ID_FAMILIES_IN_SEARCH_ORDER))
    mode_rel_paths = run_result_mode_rel_paths(run_result_mode_str)
    tried = String[]

    if run_result_mode_str != "auto"
        for family in family_order
            bases = candidate_run_bases(cluster_results_root, family)
            for base in bases
                for (rel_path, resolved_mode) in mode_rel_paths
                    dir = joinpath(base, rel_path, run_id_str)
                    push!(tried, dir)
                    if isdir(dir)
                        return dir, resolved_mode, family
                    end
                end
            end
        end
        error("run_id not found in mode=$run_result_mode_str. Tried: $(join(tried, ", "))")
    end

    matches = Vector{Tuple{String,String,String,Int,Int}}()
    seen_dirs = Set{String}()
    for (family_i, family) in enumerate(family_order)
        bases = candidate_run_bases(cluster_results_root, family)
        for (base_i, base) in enumerate(bases)
            for (rel_path, resolved_mode) in mode_rel_paths
                dir = joinpath(base, rel_path, run_id_str)
                push!(tried, dir)
                if isdir(dir) && !(dir in seen_dirs)
                    push!(matches, (dir, resolved_mode, family, base_i, family_i))
                    push!(seen_dirs, dir)
                end
            end
        end
    end

    if length(matches) == 1
        match = matches[1]
        return match[1], match[2], match[3]
    elseif isempty(matches)
        error("run_id not found for mode=auto. Tried: $(join(tried, ", "))")
    else
        min_family_i = minimum(m[5] for m in matches)
        preferred_family_matches = [m for m in matches if m[5] == min_family_i]
        min_base_i = minimum(m[4] for m in preferred_family_matches)
        preferred = [m for m in preferred_family_matches if m[4] == min_base_i]
        if length(preferred) == 1
            match = preferred[1]
            return match[1], match[2], match[3]
        end

        choices = join(["$(m[1]) [$(m[2]), family=$(m[3])]" for m in preferred], ", ")
        error("run_id is ambiguous (found multiple matches in the preferred search root). Use --run_result_mode/--cluster_results_root to disambiguate. Preferred matches: $choices")
    end
end

function latest_jld2_file_in_dir(dir::AbstractString)
    isdir(dir) || return nothing
    best_path = nothing
    best_mtime = -1.0
    for name in readdir(dir)
        path = joinpath(dir, name)
        isfile(path) || continue
        endswith(lowercase(path), ".jld2") || continue
        mtime = try
            stat(path).mtime
        catch
            0.0
        end
        if best_path === nothing || mtime >= best_mtime
            best_path = path
            best_mtime = mtime
        end
    end
    return best_path
end

function collect_ssep_files_from_run_dir(run_dir::AbstractString)
    candidate_dirs = [
        joinpath(run_dir, "aggregated"),
        joinpath(run_dir, "states", "aggregated"),
        joinpath(run_dir, "states"),
    ]
    for dir in candidate_dirs
        latest = latest_jld2_file_in_dir(dir)
        latest === nothing || return [latest]
    end
    error("No JLD2 files found for SSEP run in aggregated/ or states/ under: $(run_dir)")
end

function collect_diffusive_files_from_run_dir(run_dir::AbstractString)
    candidate_dirs = [
        joinpath(run_dir, "aggregates", "current"),
        joinpath(run_dir, "aggregates"),
        joinpath(run_dir, "aggregated", "current"),
        joinpath(run_dir, "aggregated"),
        joinpath(run_dir, "states", "aggregated"),
        joinpath(run_dir, "states", "raw"),
        joinpath(run_dir, "states"),
    ]
    for dir in candidate_dirs
        latest = latest_jld2_file_in_dir(dir)
        latest === nothing || return [latest]
    end
    error("No JLD2 files found for diffusive run in aggregated/ or states/ under: $(run_dir)")
end

function collect_files_from_run_id(run_id::AbstractString, cluster_results_root::AbstractString, run_result_mode::AbstractString;
                                   preferred_families::AbstractVector{<:AbstractString}=String[])
    run_dir, resolved_mode, resolved_family = resolve_run_id_dir(
        run_id,
        cluster_results_root,
        run_result_mode;
        preferred_families=preferred_families,
    )

    files = if resolved_family == RUN_FAMILY_SSEP
        collect_ssep_files_from_run_dir(run_dir)
    elseif resolved_family in (RUN_FAMILY_DIFFUSIVE_1D_PMLR, RUN_FAMILY_DIFFUSIVE_2D_ORIGIN_BOND) ||
           (resolved_family == RUN_FAMILY_SINGLE_ORIGIN_BOND && resolved_mode == "managed")
        collect_diffusive_files_from_run_dir(run_dir)
    else
        states_dir = joinpath(run_dir, "states")
        isdir(states_dir) || error("States directory not found for run_id: $(states_dir)")

        collected = String[]
        for (root, _, names) in walkdir(states_dir)
            for name in names
                if endswith(lowercase(name), ".jld2")
                    push!(collected, joinpath(root, name))
                end
            end
        end
        sort!(collected)
        isempty(collected) && error("No JLD2 files found in run states directory: $(states_dir)")
        collected
    end

    ts = Dates.format(now(), "yyyymmdd-HHMMSS")
    default_out_dir = joinpath(run_dir, "reports", "load_and_plot_" * ts)
    return files, default_out_dir, resolved_mode, run_dir, resolved_family
end

function ensure_state_potential!(state, potential)
    if isnothing(potential)
        return
    end
    try
        state.potential = potential
    catch
    end
end

function reconstructed_fields_payload(obj)
    if hasfield(typeof(obj), :fields)
        fields = getfield(obj, :fields)
        if fields isa AbstractVector
            if length(fields) == 1 && fields[1] isa AbstractVector
                return fields[1]
            end
            return fields
        end
    end
    return nothing
end

function reconstructed_prop_lookup(obj, name::Symbol, default=nothing)
    names = try
        propertynames(obj)
    catch
        return default
    end
    idx = findfirst(==(name), names)
    idx === nothing && return default

    payload = reconstructed_fields_payload(obj)
    if payload !== nothing && idx <= length(payload)
        return payload[idx]
    end
    return default
end

function has_prop(obj, name::Symbol)
    try
        hasproperty(obj, name) && return true
    catch
    end
    if hasfield(typeof(obj), name)
        return true
    end
    return reconstructed_prop_lookup(obj, name, nothing) !== nothing
end

function get_prop(obj, name::Symbol, default=nothing)
    if hasfield(typeof(obj), name)
        return getfield(obj, name)
    end
    try
        if hasproperty(obj, name)
            return getproperty(obj, name)
        end
    catch
    end
    return reconstructed_prop_lookup(obj, name, default)
end

function param_ffr_input_for_plot(param)
    if has_prop(param, :ffr)
        return get_prop(param, :ffr)
    elseif has_prop(param, :ffrs)
        return get_prop(param, :ffrs)
    elseif has_prop(param, :forcing_fluctuation_rate)
        ffr = Float64(get_prop(param, :forcing_fluctuation_rate))
        if has_prop(param, :N)
            return ffr * Float64(get_prop(param, :N))
        end
        return ffr
    end
    return 0.0
end

function loaded_module_name(obj)
    try
        return String(nameof(parentmodule(typeof(obj))))
    catch
        return ""
    end
end

function canonicalize_loaded_param(param; state=nothing)
    param_module = loaded_module_name(param)
    state_module = isnothing(state) ? "" : loaded_module_name(state)

    if param isa FPDiffusive.Param || param_module in ("FPSSEP", "FPDiffusive", "FP")
        return param
    end
    if state_module == "FPSSEP"
        return param
    end

    required = (:γ, :dims, :ρ₀, :D, :potential_type, :fluctuation_type, :potential_magnitude)
    all(name -> has_prop(param, name), required) || return param

    dims_raw = get_prop(param, :dims)
    dims_tuple = try
        Tuple(Int.(collect(dims_raw)))
    catch
        return param
    end

    forcing_rate_scheme = if has_prop(param, :forcing_rate_scheme)
        String(get_prop(param, :forcing_rate_scheme))
    else
        "legacy_penalty"
    end

    return FPDiffusive.setParam(
        Float64(get_prop(param, :γ)),
        dims_tuple,
        Float64(get_prop(param, :ρ₀)),
        Float64(get_prop(param, :D)),
        String(get_prop(param, :potential_type)),
        String(get_prop(param, :fluctuation_type)),
        Float64(get_prop(param, :potential_magnitude)),
        param_ffr_input_for_plot(param);
        forcing_rate_scheme=forcing_rate_scheme,
    )
end

function legacy_correlation_cuts_for_plot(state)
    if has_prop(state, :ρ_matrix_avg_cuts)
        cuts = get_prop(state, :ρ_matrix_avg_cuts)
        if cuts isa AbstractDict
            return cuts
        end
    end

    if has_prop(state, :ρ_matrix_avg)
        rho_matrix_avg = get_prop(state, :ρ_matrix_avg)
        rho_matrix_avg === nothing || return Dict{Symbol,Any}(:full => Float64.(rho_matrix_avg))
    end

    return nothing
end

function canonicalize_loaded_state(state; potential=nothing)
    cuts = legacy_correlation_cuts_for_plot(state)
    if has_prop(state, :ρ_matrix_avg_cuts) || isnothing(cuts)
        return state
    end

    rho_avg = get_prop(state, :ρ_avg, nothing)
    rho_avg === nothing && return state

    rho_inst = get_prop(state, :ρ, nothing)
    if rho_inst === nothing
        rho_inst = zeros(Int, size(rho_avg))
    end

    rho_plus = get_prop(state, :ρ₊, nothing)
    rho_plus === nothing && (rho_plus = zeros(Int, size(rho_inst)))

    rho_minus = get_prop(state, :ρ₋, nothing)
    rho_minus === nothing && (rho_minus = zeros(Int, size(rho_inst)))

    stats = get_prop(state, :bond_pass_stats, nothing)
    if !(stats isa AbstractDict)
        stats = Dict{Symbol,Vector{Float64}}()
    end

    potential_value = isnothing(potential) ? get_prop(state, :potential, nothing) : potential
    forcing_value = has_prop(state, :forcing) ? get_prop(state, :forcing) : Any[]

    return LegacyPlotState(
        t=Int(get_prop(state, :t, 0)),
        particles=get_prop(state, :particles, Any[]),
        ρ=rho_inst,
        ρ₊=rho_plus,
        ρ₋=rho_minus,
        ρ_avg=Float64.(rho_avg),
        ρ_matrix_avg_cuts=cuts,
        bond_pass_stats=stats,
        T=Float64(get_prop(state, :T, 0.0)),
        potential=potential_value,
        forcing=forcing_value,
    )
end

function has_loaded_key(data, key::AbstractString)
    return haskey(data, key) || haskey(data, Symbol(key))
end

function get_loaded_value(data, key::AbstractString, default=nothing)
    if haskey(data, key)
        return data[key]
    end
    sym = Symbol(key)
    if haskey(data, sym)
        return data[sym]
    end
    return default
end

function load_state_bundle(saved_state::String)
    data = JLD2.load(saved_state)

    state = get_loaded_value(data, "state", nothing)
    if state === nothing
        state = get_loaded_value(data, "dummy_state", nothing)
    end
    if state === nothing && has_loaded_key(data, "states")
        states = get_loaded_value(data, "states", nothing)
        if states isa AbstractVector && !isempty(states)
            state = states[1]
        end
    end

    param = get_loaded_value(data, "param", nothing)
    if param === nothing && has_loaded_key(data, "params")
        params = get_loaded_value(data, "params", nothing)
        if params isa AbstractVector && !isempty(params)
            param = params[1]
        end
    end

    potential = get_loaded_value(data, "potential", nothing)
    if state !== nothing && potential === nothing && has_prop(state, :potential)
        potential = get_prop(state, :potential)
    end

    if state === nothing || param === nothing
        available_keys = join(sort!(String.(collect(keys(data)))), ", ")
        error("Missing state/param payload (keys: $available_keys)")
    end

    param = canonicalize_loaded_param(param; state=state)
    state = canonicalize_loaded_state(state; potential=potential)
    return state, param, potential
end

function param_module_name_for_plot(param)
    return loaded_module_name(param)
end

function is_legacy_normalized_state(state)
    return Bool(get_prop(state, :legacy_normalized, false))
end

function is_ssep_state(state, param)
    return loaded_module_name(state) == "FPSSEP" || param_module_name_for_plot(param) == "FPSSEP"
end

function is_common_diffusive_state(state, param)
    is_ssep_state(state, param) && return false
    rho_avg = get_prop(state, :ρ_avg, nothing)
    rho_matrix_avg_cuts = get_prop(state, :ρ_matrix_avg_cuts, nothing)
    dims = get_prop(param, :dims, nothing)

    rho_avg === nothing && return false
    rho_matrix_avg_cuts === nothing && return false
    dims === nothing && return false
    return dims isa Tuple || dims isa AbstractVector
end

function selected_cut_indices_from_state(state, param)
    length(param.dims) == 1 || return Int[]
    PlotUtils.has_selected_site_cuts_for_plot(state) || return Int[]
    L = Int(param.dims[1])
    selected_sites, origin_site, _ = PlotUtils.selected_site_cut_metadata_for_plot(state, L)
    offsets = Int[]
    for site in selected_sites
        offset = Int(round(abs(periodic_displacement_1d(Float64(site - origin_site), L))))
        offset > 0 && push!(offsets, offset)
    end
    return sort(unique(offsets))
end

function parse_requested_collapse_indices(raw_value)::Union{Nothing,Vector{Int}}
    value = strip(String(raw_value))
    isempty(value) && return nothing
    lowercase(value) == "all" && return nothing

    requested = Int[]
    seen = Set{Int}()
    for chunk in split(value, ',')
        token = strip(chunk)
        isempty(token) && continue
        if occursin(':', token)
            parts = split(token, ':')
            if length(parts) == 2
                start_val = try
                    parse(Int, strip(parts[1]))
                catch
                    error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
                end
                step_val = 1
                stop_val = try
                    parse(Int, strip(parts[2]))
                catch
                    error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
                end
            elseif length(parts) == 3
                start_val = try
                    parse(Int, strip(parts[1]))
                catch
                    error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
                end
                step_val = try
                    parse(Int, strip(parts[2]))
                catch
                    error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
                end
                stop_val = try
                    parse(Int, strip(parts[3]))
                catch
                    error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
                end
            else
                error("Invalid --collapse_indices range '$token'. Use start:stop or start:step:stop with positive integers.")
            end

            start_val > 0 || error("--collapse_indices entries must be positive. Got $start_val in '$token'.")
            step_val > 0 || error("--collapse_indices step must be positive. Got $step_val in '$token'.")
            stop_val >= start_val || error("--collapse_indices range stop must be >= start. Got '$token'.")
            for parsed in start_val:step_val:stop_val
                if !(parsed in seen)
                    push!(requested, parsed)
                    push!(seen, parsed)
                end
            end
        else
            parsed = try
                parse(Int, token)
            catch
                error("Invalid --collapse_indices entry '$token'. Use positive integers or ranges like 20:10:80.")
            end
            parsed > 0 || error("--collapse_indices entries must be positive. Got $parsed.")
            if !(parsed in seen)
                push!(requested, parsed)
                push!(seen, parsed)
            end
        end
    end

    isempty(requested) && error("--collapse_indices did not contain any valid positive integers.")
    return requested
end

function collapse_power_dir_token(collapse_power::Real)
    rounded = round(Float64(collapse_power); digits=6)
    if isinteger(rounded)
        return string(Int(rounded))
    end
    token = @sprintf("%.6f", rounded)
    token = replace(token, r"0+$" => "")
    token = replace(token, r"\.$" => "")
    return replace(token, "." => "p")
end

function collapse_indices_dir_token(indices::AbstractVector{<:Integer})
    isempty(indices) && return "idx-none"
    return "idx-" * join(Int.(indices), "-")
end

function collapse_dir_name(collapse_power::Real, indices::AbstractVector{<:Integer}; bond_centered::Bool=false)
    prefix = bond_centered ? "data_collapse_bond_centered_y" : "data_collapse_y"
    return prefix * collapse_power_dir_token(collapse_power) * "_" * collapse_indices_dir_token(indices)
end

function resolve_collapse_indices(available_indices::Vector{Int}, requested_indices::Union{Nothing,Vector{Int}}, saved_state::AbstractString)
    isnothing(requested_indices) && return available_indices

    missing = [idx for idx in requested_indices if !(idx in available_indices)]
    if !isempty(missing)
        error(
            "Requested collapse indices $(missing) are not available in $(saved_state). " *
            "Available positive cut offsets: $(available_indices)."
        )
    end
    return requested_indices
end

function full_matrix_collapse_indices(param)
    dims = get_prop(param, :dims, nothing)
    if !(dims isa Tuple || dims isa AbstractVector) || length(dims) != 1
        return Int[]
    end
    L = Int(dims[1])
    L <= 1 && return Int[]
    return collect(1:div(L, 2))
end

function resolve_1d_collapse_indices(state, param, requested_indices::Union{Nothing,Vector{Int}}, saved_state::AbstractString)
    selected_indices = selected_cut_indices_from_state(state, param)
    if !isempty(selected_indices)
        return resolve_collapse_indices(selected_indices, requested_indices, saved_state), :selected_site_cuts
    end

    available_indices = full_matrix_collapse_indices(param)
    if isempty(available_indices)
        return Int[], :unsupported
    end

    if isnothing(requested_indices)
        return Int[], :needs_explicit_indices
    end

    missing = [idx for idx in requested_indices if !(idx in available_indices)]
    if !isempty(missing)
        error(
            "Requested collapse indices $(missing) are not valid for $(saved_state). " *
            "Valid positive cut offsets for this 1D state are $(available_indices)."
        )
    end
    return requested_indices, :full_matrix
end

function supports_1d_data_collapse(state, param)
    dims = get_prop(param, :dims, nothing)
    if !(dims isa Tuple || dims isa AbstractVector) || length(dims) != 1
        return false
    end

    rho_avg = get_prop(state, :ρ_avg, nothing)
    rho_matrix_avg_cuts = get_prop(state, :ρ_matrix_avg_cuts, nothing)
    rho_avg === nothing && return false
    rho_matrix_avg_cuts isa AbstractDict || return false

    return PlotUtils.has_selected_site_cuts_for_plot(state) || haskey(rho_matrix_avg_cuts, :full)
end

function default_bond_centered_collapse_1d(state, param)
    dims = get_prop(param, :dims, nothing)
    if !(dims isa Tuple || dims isa AbstractVector) || length(dims) != 1
        return false
    end

    rho_matrix_avg_cuts = get_prop(state, :ρ_matrix_avg_cuts, nothing)
    rho_matrix_avg_cuts isa AbstractDict || return false
    has_full_corr = haskey(rho_matrix_avg_cuts, :full) || haskey(rho_matrix_avg_cuts, AGG_CONNECTED_FULL_EXACT_KEY)
    has_full_corr || return false

    try
        bonds, _, _ = PlotUtils.force_bond_sites_1d(state, Int(dims[1]))
        return length(bonds) == 1
    catch
        return false
    end
end

function available_2d_cut_collapse_indices(param)
    dims = get_prop(param, :dims, nothing)
    if !(dims isa Tuple || dims isa AbstractVector) || length(dims) != 2
        return Int[]
    end

    Lx = Int(dims[1])
    Lx <= 1 && return Int[]
    return collect(1:div(Lx, 2))
end

function resolve_2d_cut_collapse_indices(param, requested_indices::Union{Nothing,Vector{Int}}, saved_state::AbstractString)
    available_indices = available_2d_cut_collapse_indices(param)
    if isempty(available_indices)
        return Int[], :unsupported
    end

    if isnothing(requested_indices)
        return Int[], :needs_explicit_indices
    end

    missing = [idx for idx in requested_indices if !(idx in available_indices)]
    if !isempty(missing)
        error(
            "Requested collapse indices $(missing) are not valid for $(saved_state). " *
            "Valid positive cut offsets for this 2D state are $(available_indices)."
        )
    end
    return requested_indices, :requested
end

function supports_2d_cut_data_collapse(state, param)
    dims = get_prop(param, :dims, nothing)
    if !(dims isa Tuple || dims isa AbstractVector) || length(dims) != 2
        return false
    end

    rho_avg = get_prop(state, :ρ_avg, nothing)
    rho_matrix_avg_cuts = get_prop(state, :ρ_matrix_avg_cuts, nothing)
    rho_avg === nothing && return false
    rho_matrix_avg_cuts isa AbstractDict || return false

    return haskey(rho_matrix_avg_cuts, :full) ||
           (haskey(rho_matrix_avg_cuts, :x_cut) && haskey(rho_matrix_avg_cuts, :diag_cut))
end

function bond_pass_stats_dict(state)
    if has_prop(state, :bond_pass_stats)
        stats = get_prop(state, :bond_pass_stats)
        if stats isa AbstractDict
            return stats
        end
    end
    if has_prop(state, :ρ_matrix_avg_cuts)
        cuts = get_prop(state, :ρ_matrix_avg_cuts)
        if cuts isa AbstractDict
            legacy = Dict{Symbol,Vector{Float64}}()
            for key in (:bond_pass_forward_avg, :bond_pass_reverse_avg, :bond_pass_total_avg,
                        :bond_pass_total_sq_avg, :bond_pass_sample_count, :statistics_sample_count, :bond_pass_track_mask,
                        :bond_pass_spatial_f_avg, :bond_pass_spatial_f2_avg, :bond_pass_spatial_sample_count)
                if haskey(cuts, key)
                    legacy[key] = Float64.(cuts[key])
                end
            end
            return legacy
        end
    end
    return Dict{Symbol,Vector{Float64}}()
end

function forcing_bonds_1d(state, L::Int)
    if !has_prop(state, :forcing)
        return Tuple{Int,Int}[], Float64[]
    end
    forcing_raw = get_prop(state, :forcing)
    forcings = forcing_raw isa AbstractVector ? forcing_raw : [forcing_raw]
    bonds = Tuple{Int,Int}[]
    magnitudes = Float64[]
    for force in forcings
        if length(force.bond_indices[1]) == 1 && length(force.bond_indices[2]) == 1
            b1 = mod1(force.bond_indices[1][1], L)
            b2 = mod1(force.bond_indices[2][1], L)
            push!(bonds, (b1, b2))
            push!(magnitudes, Float64(force.magnitude))
        end
    end
    return bonds, magnitudes
end

function tracked_force_indices(state, magnitudes::AbstractVector{<:Real})
    n_forces = length(magnitudes)
    stats = bond_pass_stats_dict(state)
    if haskey(stats, BOND_PASS_TRACK_MASK_KEY)
        mask = stats[BOND_PASS_TRACK_MASK_KEY]
        if length(mask) == n_forces
            return [i for i in 1:n_forces if mask[i] > 0.5]
        end
    end
    return [i for i in 1:n_forces if abs(Float64(magnitudes[i])) > 0]
end

function connected_corr_mat_1d(state, param)
    fix_term = PlotUtils.connected_correlation_fix_term(param)
    if haskey(state.ρ_matrix_avg_cuts, :full)
        rho_avg = Float64.(state.ρ_avg)
        derived = Float64.(state.ρ_matrix_avg_cuts[:full]) .- (rho_avg * transpose(rho_avg)) .+ fix_term
        if haskey(state.ρ_matrix_avg_cuts, AGG_CONNECTED_FULL_EXACT_KEY)
            stored_exact = Float64.(state.ρ_matrix_avg_cuts[AGG_CONNECTED_FULL_EXACT_KEY]) .+ fix_term
            if maximum(abs, stored_exact) <= 1e-12 && maximum(abs, derived) > 1e-12
                return derived
            end
            return stored_exact
        end
        return derived
    end
    return Float64.(state.ρ_matrix_avg_cuts[AGG_CONNECTED_FULL_EXACT_KEY]) .+ fix_term
end

function bond_centered_cut_1d(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
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

function bond_center_value(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
    return 0.5 * (cut[b1] + cut[b2])
end

function pair_distance_d(bonds::Vector{Tuple{Int,Int}}, L::Int)
    if length(bonds) < 2
        return nothing
    end
    (a1, a2) = bonds[1]
    (b1, b2) = bonds[2]
    d1 = mod(b1 - a2, L)
    d2 = mod(a1 - b2, L)
    d = min(d1, d2)
    if d == 0
        d = L
    end
    return d
end

function finite_mean(values::AbstractVector{<:Real})
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    return isempty(vals) ? NaN : mean(vals)
end

function finite_sem(values::AbstractVector{<:Real})
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

function preferred_ci95(values::AbstractVector{<:Real}, ci_candidates::AbstractVector{<:Real})
    ci_vals = [Float64(v) for v in ci_candidates if isfinite(Float64(v)) && Float64(v) >= 0]
    if !isempty(ci_vals)
        return mean(ci_vals)
    end
    _, _, ci95, _ = mean_sem_ci95(values)
    return ci95
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

function has_metric_accumulator(stats::AbstractDict, sum_key::Symbol, sumsq_key::Symbol, n_key::Symbol)
    return haskey(stats, sum_key) && haskey(stats, sumsq_key) && haskey(stats, n_key)
end

function metric_accumulator_from_stats(stats::AbstractDict, sum_key::Symbol, sumsq_key::Symbol, n_key::Symbol;
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

function sanitize_yerror(values::AbstractVector{<:Real})
    return [isfinite(Float64(v)) && Float64(v) > 0 ? Float64(v) : 0.0 for v in values]
end

function sanitize_yerror_log(yerr::AbstractVector{<:Real}, yvals::AbstractVector{<:Real}; lower_fraction::Float64=0.95)
    n = min(length(yerr), length(yvals))
    out = Vector{Float64}(undef, n)
    for i in 1:n
        e = Float64(yerr[i])
        y = Float64(yvals[i])
        if !(isfinite(e) && e > 0 && isfinite(y) && y > 0)
            out[i] = 0.0
            continue
        end
        out[i] = min(e, lower_fraction * y)
    end
    return out
end

bond_series_color(slot::Integer) = TWO_FORCE_BOND_SERIES_COLORS[mod1(Int(slot), length(TWO_FORCE_BOND_SERIES_COLORS))]

function savefig_or_placeholder(plot_obj, out_path::String; placeholder_title::AbstractString="Plot unavailable")
    try
        savefig(plot_obj, out_path)
        println("Saved ", out_path)
        return true
    catch e
        msg = sprint(showerror, e)
        if e isa AssertionError && occursin("total_plotarea_", msg)
            @warn "savefig layout assertion; attempting placeholder." path=out_path error=msg
            try
                Plots.closeall()
            catch
            end
            try
                Plots.gr()
            catch
            end
            try
                p_placeholder = plot(title=String(placeholder_title), axis=false, legend=false, size=(1000, 550))
                annotate!(p_placeholder, 0.5, 0.5, text("Rendering skipped due to GR layout assertion.", 10))
                savefig(p_placeholder, out_path)
                println("Saved ", out_path)
                return true
            catch e_placeholder
                msg_placeholder = sprint(showerror, e_placeholder)
                @warn "Placeholder save failed; skipping file." path=out_path error=msg_placeholder
                return false
            end
        end
        rethrow(e)
    end
end

function remap_vector_by_tracked(meta_all::AbstractVector{<:Real}, tracked::AbstractVector{<:Integer})
    meta = Float64.(meta_all)
    n = length(tracked)
    if n == 0
        return Float64[]
    end
    if length(meta) == n
        return meta
    end
    if !isempty(meta) && maximum(tracked) <= length(meta)
        return [meta[idx] for idx in tracked]
    end
    return [i <= length(meta) ? meta[i] : NaN for i in 1:n]
end

function remap_accumulator_by_tracked(accum::NamedTuple, tracked::AbstractVector{<:Integer})
    return (
        n=remap_vector_by_tracked(accum.n, tracked),
        w=hasproperty(accum, :w) ? remap_vector_by_tracked(accum.w, tracked) : remap_vector_by_tracked(accum.n, tracked),
        wsq=hasproperty(accum, :wsq) ? remap_vector_by_tracked(accum.wsq, tracked) : remap_vector_by_tracked(accum.n, tracked),
        sum=remap_vector_by_tracked(accum.sum, tracked),
        sumsq=remap_vector_by_tracked(accum.sumsq, tracked),
    )
end

function bucket_scalar_summary(bucket, value_field::Symbol, ci95_field::Symbol, accum_field::Symbol, exact_field::Symbol)
    if all(getproperty(row, exact_field) for row in bucket)
        combined = combine_metric_accumulators([getproperty(row, accum_field) for row in bucket])
        means, _, ci95s = stats_from_metric_accumulator(combined)
        mean_val = isempty(means) ? NaN : means[1]
        ci95_val = isempty(ci95s) ? NaN : ci95s[1]
        return mean_val, ci95_val
    end
    vals = [Float64(getproperty(row, value_field)) for row in bucket]
    ci95_candidates = [Float64(getproperty(row, ci95_field)) for row in bucket]
    return finite_mean(vals), preferred_ci95(vals, ci95_candidates)
end

function bucket_vector_summary(bucket, value_field::Symbol, ci95_field::Symbol, accum_field::Symbol, exact_field::Symbol, max_slots::Int)
    if all(getproperty(row, exact_field) for row in bucket)
        combined = combine_metric_accumulators([getproperty(row, accum_field) for row in bucket])
        means, _, ci95s = stats_from_metric_accumulator(combined)
        means_out = fill(NaN, max_slots)
        ci95_out = fill(NaN, max_slots)
        copy_len = min(length(means), max_slots)
        if copy_len > 0
            means_out[1:copy_len] .= means[1:copy_len]
            ci95_out[1:copy_len] .= ci95s[1:copy_len]
        end
        return means_out, ci95_out
    end

    means_out = fill(NaN, max_slots)
    ci95_out = fill(NaN, max_slots)
    for slot in 1:max_slots
        vals = Float64[]
        ci95_candidates = Float64[]
        for row in bucket
            row_vals = getproperty(row, value_field)
            if slot <= length(row_vals)
                push!(vals, Float64(row_vals[slot]))
            end
            row_ci95 = getproperty(row, ci95_field)
            if slot <= length(row_ci95)
                push!(ci95_candidates, Float64(row_ci95[slot]))
            end
        end
        means_out[slot] = finite_mean(vals)
        ci95_out[slot] = preferred_ci95(vals, ci95_candidates)
    end
    return means_out, ci95_out
end

function rho0_from_param(param)
    if has_prop(param, :ρ₀)
        value = get_prop(param, :ρ₀)
        return value isa Number ? Float64(value) : NaN
    elseif has_prop(param, :rho0)
        value = get_prop(param, :rho0)
        return value isa Number ? Float64(value) : NaN
    elseif param isa AbstractDict
        for key in ("ρ₀", "rho0")
            value = get_loaded_value(param, key, nothing)
            if value isa Number
                return Float64(value)
            end
        end
    end
    return NaN
end

function periodic_displacement_1d(dx::Float64, L::Int)
    return mod(dx + 0.5 * L, L) - 0.5 * L
end

function bond_centered_axis_1d(L::Int, b1::Int, b2::Int)
    center = bond_center_coordinate_1d(L, b1, b2)
    x_rel = [periodic_displacement_1d(Float64(i) - center, L) for i in 1:L]
    perm = sortperm(x_rel)
    return x_rel, perm
end

function bond_center_coordinate_1d(L::Int, b1::Int, b2::Int)
    if b2 == mod1(b1 + 1, L)
        return Float64(b1) + 0.5
    elseif b1 == mod1(b2 + 1, L)
        return Float64(b2) + 0.5
    else
        return 0.5 * (Float64(b1) + Float64(b2))
    end
end

function bond_cut_model(x::AbstractVector{<:Real}, K1::Float64, K2::Float64; x0::Float64=0.0, orientation_sign::Float64=1.0, C::Float64=0.0)
    x_vals = Float64.(x)
    x_shifted = x_vals .- x0
    return C .+ orientation_sign .* (K1 * K2) .* x_shifted ./ (x_shifted .^ 2 .+ K2^2) .^ 2
end

function fit_bond_cut_profile(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; x0::Float64=0.0, orientation_sign::Float64=1.0, C_fixed::Float64=0.0)
    x_data = Float64.(x)
    y_data = Float64.(y)
    mask = isfinite.(x_data) .& isfinite.(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]

    if length(x_data) < 6
        return nothing
    end

    abs_x = abs.(x_data)
    nonzero = abs_x[abs_x .> 0]
    if isempty(nonzero)
        return nothing
    end

    k2_min = max(0.5 * minimum(nonzero), 1e-3)
    k2_max = max(maximum(abs_x), 1.2 * k2_min)
    k2_grid = exp.(range(log(k2_min), log(k2_max), length=220))

    best_sse = Inf
    best_k2 = NaN
    best_A = NaN
    for k2 in k2_grid
        basis_x = x_data .- x0
        basis = orientation_sign .* basis_x ./ (basis_x .^ 2 .+ k2^2) .^ 2
        denom = dot(basis, basis)
        if !(isfinite(denom) && denom > eps(Float64))
            continue
        end
        centered_y = y_data .- C_fixed
        A = dot(basis, centered_y) / denom
        residual = y_data .- (A .* basis .+ C_fixed)
        sse = dot(residual, residual)
        if isfinite(sse) && sse < best_sse
            best_sse = sse
            best_k2 = k2
            best_A = A
        end
    end

    if !isfinite(best_sse)
        return nothing
    end

    y_mean = mean(y_data)
    ss_tot = sum((y_data .- y_mean) .^ 2)
    r2 = ss_tot > 0 ? 1 - best_sse / ss_tot : NaN
    K1 = best_A / best_k2
    return (K1=K1, K2=best_k2, x0=x0, orientation_sign=orientation_sign, A=best_A, C=C_fixed, r2=r2)
end

function fit_loglog_powerlaw(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; fixed_slope=nothing)
    x_data = Float64.(x)
    y_data = Float64.(y)
    mask = isfinite.(x_data) .& isfinite.(y_data) .& (x_data .> 0) .& (y_data .> 0)
    if count(mask) < 2
        return nothing
    end

    x_fit = x_data[mask]
    y_fit = y_data[mask]
    lx = log10.(x_fit)
    ly = log10.(y_fit)

    y_mean = mean(ly)
    if isnothing(fixed_slope)
        x_mean = mean(lx)
        denom = sum((lx .- x_mean) .^ 2)
        if denom <= eps(Float64)
            return nothing
        end

        slope = sum((lx .- x_mean) .* (ly .- y_mean)) / denom
        intercept = y_mean - slope * x_mean
    else
        slope = Float64(fixed_slope)
        intercept = mean(ly .- slope .* lx)
    end

    pred = intercept .+ slope .* lx
    ss_tot = sum((ly .- y_mean) .^ 2)
    ss_res = sum((ly .- pred) .^ 2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    return (slope=slope, intercept=intercept, r2=r2, x=x_fit)
end

function loglog_fit_param_count(fixed_slope)
    return isnothing(fixed_slope) ? 2 : 1
end

function decimal_tick_label(x::Float64)
    if !isfinite(x)
        return ""
    end
    if x == 0.0
        return "0"
    end

    ax = abs(x)
    if ax >= 1.0
        if ax < 1e7
            return @sprintf("%.0f", x)
        else
            return @sprintf("%.3g", x)
        end
    end

    n_decimals = min(8, max(1, ceil(Int, -log10(ax))))
    s = @sprintf("%.*f", n_decimals, x)
    s = replace(s, r"0+$" => "")
    s = replace(s, r"\.$" => "")
    return s
end

function log10_power_with_decimal_tick_label(x)
    xv = Float64(x)
    if !(isfinite(xv) && xv > 0)
        return ""
    end
    k = round(Int, log10(xv))
    power_value = 10.0^k
    if isapprox(xv, power_value; rtol=1e-8, atol=1e-12)
        return "10^$(k)\n" * decimal_tick_label(power_value)
    end
    return decimal_tick_label(xv)
end

function apply_log10_decimal_x_ticks!(p)
    plot!(p; xformatter=log10_power_with_decimal_tick_label, bottom_margin=9Plots.mm)
    return p
end

function adjusted_r2(r2::Float64, n_points::Int; n_params::Int=2)
    if !isfinite(r2)
        return NaN
    end
    dof = n_points - n_params
    if dof <= 0
        return NaN
    end
    return 1 - (1 - r2) * (n_points - 1) / dof
end

function scan_loglog_baseline(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                              n_grid::Int=360,
                              min_points::Int=4,
                              target_slope=nothing)
    x_data = Float64.(x)
    y_data = Float64.(y)
    base_mask = isfinite.(x_data) .& isfinite.(y_data) .& (x_data .> 0)
    if count(base_mask) < min_points
        return nothing
    end

    y_finite = y_data[base_mask]
    y_min = minimum(y_finite)
    y_max = maximum(y_finite)
    y_span = max(y_max - y_min, abs(y_max), abs(y_min), 1e-12)

    baseline_lo_raw = y_min - 0.9 * y_span
    baseline_hi_raw = y_min - max(1e-12, 1e-6 * max(abs(y_min), 1.0))
    if !(isfinite(baseline_lo_raw) && isfinite(baseline_hi_raw))
        return nothing
    end

    # Physical constraint: baseline variance must be non-negative (b >= 0).
    baseline_lo = max(0.0, baseline_lo_raw)
    baseline_hi = max(0.0, baseline_hi_raw)
    n_scan = max(n_grid, 12)
    baselines = if baseline_hi > baseline_lo + eps(Float64)
        collect(range(baseline_lo, baseline_hi, length=n_scan))
    else
        [baseline_lo]
    end
    n_candidates = length(baselines)
    r2_vals = fill(NaN, n_candidates)
    adj_r2_vals = fill(NaN, n_candidates)
    slope_vals = fill(NaN, n_candidates)
    point_counts = fill(0, n_candidates)
    n_fit_params = loglog_fit_param_count(target_slope)

    best_idx = 0
    best_score = -Inf
    best_points = 0
    for (i, baseline) in enumerate(baselines)
        fit = fit_loglog_powerlaw(x_data, y_data .- baseline; fixed_slope=target_slope)
        if isnothing(fit)
            continue
        end
        n_pts = length(fit.x)
        point_counts[i] = n_pts
        if n_pts < min_points
            continue
        end
        r2 = fit.r2
        adj = adjusted_r2(r2, n_pts; n_params=n_fit_params)
        if !isfinite(adj)
            adj = r2
        end
        if !(isfinite(adj) && isfinite(r2))
            continue
        end

        r2_vals[i] = r2
        adj_r2_vals[i] = adj
        slope_vals[i] = fit.slope

        better = adj > best_score + 1e-12
        tie = abs(adj - best_score) <= 1e-12 && n_pts > best_points
        if better || tie
            best_idx = i
            best_score = adj
            best_points = n_pts
        end
    end

    if best_idx == 0
        return nothing
    end

    best_baseline = baselines[best_idx]
    best_fit = fit_loglog_powerlaw(x_data, y_data .- best_baseline; fixed_slope=target_slope)
    isnothing(best_fit) && return nothing

    return (
        baselines=baselines,
        r2_vals=r2_vals,
        adj_r2_vals=adj_r2_vals,
        slope_vals=slope_vals,
        point_counts=point_counts,
        best_idx=best_idx,
        best_baseline=best_baseline,
        best_fit=best_fit,
        y_shifted_best=(y_data .- best_baseline),
    )
end

function save_loglog_baseline_search_plot(file_name::String,
                                          analysis_dir::String,
                                          x::AbstractVector{<:Real},
                                          y::AbstractVector{<:Real};
                                          quantity_label::AbstractString,
                                          min_points::Int=4,
                                          target_slope=nothing)
    scan = scan_loglog_baseline(x, y; min_points=min_points, target_slope=target_slope)
    out_path = joinpath(analysis_dir, file_name)
    scan_title = isnothing(target_slope) ?
        "$quantity_label baseline scan" :
        @sprintf("%s baseline scan (target slope=%.3f)", quantity_label, target_slope)
    best_fit_title = isnothing(target_slope) ?
        @sprintf("%s - baseline(best) fit", quantity_label) :
        @sprintf("%s - baseline(best) fit (slope fixed at %.3f)", quantity_label, target_slope)

    if isnothing(scan)
        p_empty = plot(title=scan_title,
                       axis=false,
                       legend=false)
        annotate!(p_empty, 0.5, 0.5, text("No valid baseline scan (need ≥ $(min_points) positive points)", 10))
        savefig_or_placeholder(
            p_empty,
            out_path;
            placeholder_title="Baseline scan unavailable",
        )
        return nothing
    end

    x_data = Float64.(x)
    best_idx = scan.best_idx
    best_baseline = scan.best_baseline
    best_fit = scan.best_fit
    y_best = scan.y_shifted_best
    best_adj = scan.adj_r2_vals[best_idx]
    best_n = scan.point_counts[best_idx]

    if !ENABLE_BASELINE_SEARCH_FIGURES
        println("Skipping baseline-search figures (set LOAD_AND_PLOT_ENABLE_BASELINE_SEARCH_FIGURES=1 to enable): ", out_path)
        return (
            baseline=best_baseline,
            slope=best_fit.slope,
            r2=best_fit.r2,
            adj_r2=best_adj,
            n_points=best_n,
        )
    end

    p_scan = plot(title=scan_title,
                  xlabel="baseline to subtract (b ≥ 0)",
                  ylabel="adjusted R² (higher is better)",
                  framestyle=:box,
                  legend=:topright,
                  size=(1200, 700),
                  titlefontsize=10,
                  guidefontsize=11,
                  tickfontsize=9,
                  legendfontsize=8)
    mask_adj = isfinite.(scan.adj_r2_vals)
    if any(mask_adj)
        plot!(p_scan, scan.baselines[mask_adj], scan.adj_r2_vals[mask_adj],
              lw=1.6, color=:blue, marker=:circle, markersize=3.8,
              markerstrokecolor=:blue, label="adjusted R² samples")
    end
    vline!(p_scan, [best_baseline], color=:red, alpha=0.35, linestyle=:dash, label=false)
    scatter!(p_scan, [best_baseline], [best_adj],
             marker=:diamond, markersize=8, color=:red,
             label=@sprintf("best b=%.6g (adjR²=%.4f)", best_baseline, best_adj))

    p_best = plot(title=best_fit_title,
                  xlabel="d",
                  ylabel="value - baseline(best)",
                  xscale=:log10,
                  yscale=:log10,
                  framestyle=:box,
                  legend=:topright,
                  size=(1200, 700),
                  titlefontsize=10,
                  guidefontsize=11,
                  tickfontsize=9,
                  legendfontsize=8)
    apply_log10_decimal_x_ticks!(p_best)
    mask_best = isfinite.(x_data) .& isfinite.(y_best) .& (x_data .> 0) .& (y_best .> 0)
    if any(mask_best)
        plot!(p_best, x_data[mask_best], y_best[mask_best],
              marker=:diamond, lw=2.5, color=:black, label="mean - b(best)")
    end

    x_fit = best_fit.x
    y_fit = 10 .^ (best_fit.intercept .+ best_fit.slope .* log10.(x_fit))
    plot!(p_best, x_fit, y_fit,
          lw=2.0,
          color=:gray20,
          linestyle=:dashdot,
          label=isnothing(target_slope) ?
              @sprintf("fit slope=%.3f (R²=%.3f)", best_fit.slope, best_fit.r2) :
              @sprintf("fit slope=%.3f fixed (R²=%.3f)", best_fit.slope, best_fit.r2))
    anchor_x = exp(mean(log.(x_fit)))
    anchor_y = 10^(best_fit.intercept + best_fit.slope * log10(anchor_x))
    add_reference_slopes!(p_best, x_fit, anchor_x, anchor_y)
    base, ext = splitext(out_path)
    scan_out = base * "_scan" * ext
    savefig_or_placeholder(
        p_scan,
        scan_out;
        placeholder_title="Baseline scan unavailable",
    )
    savefig_or_placeholder(
        p_best,
        out_path;
        placeholder_title="Baseline fit unavailable",
    )

    return (
        baseline=best_baseline,
        slope=best_fit.slope,
        r2=best_fit.r2,
        adj_r2=best_adj,
        n_points=best_n,
    )
end

function ci95_to_log10_sigma(ci95::Real, y::Real)
    ci95_val = Float64(ci95)
    y_val = Float64(y)
    if !(isfinite(ci95_val) && ci95_val > 0 && isfinite(y_val) && y_val > 0)
        return NaN
    end
    sigma_y = ci95_val / 1.96
    return sigma_y / (log(10.0) * y_val)
end

function fit_loglog_powerlaw_weighted(x::AbstractVector{<:Real},
                                      y::AbstractVector{<:Real},
                                      ci95::AbstractVector{<:Real};
                                      fixed_slope=nothing)
    x_data = Float64.(x)
    y_data = Float64.(y)
    ci95_data = Float64.(ci95)
    sigma_log10 = [ci95_to_log10_sigma(ci95_data[i], y_data[i]) for i in eachindex(y_data)]
    mask = isfinite.(x_data) .& isfinite.(y_data) .& isfinite.(ci95_data) .&
           isfinite.(sigma_log10) .& (x_data .> 0) .& (y_data .> 0) .&
           (ci95_data .> 0) .& (sigma_log10 .> 0)
    if count(mask) < 2
        return nothing
    end

    x_fit = x_data[mask]
    y_fit = y_data[mask]
    ci95_fit = ci95_data[mask]
    sigma_log10_fit = sigma_log10[mask]
    lx = log10.(x_fit)
    ly = log10.(y_fit)
    w = 1.0 ./ (sigma_log10_fit .^ 2)
    if !(all(isfinite.(w)) && all(w .> 0))
        return nothing
    end

    slope = NaN
    intercept = NaN
    s = sum(w)
    sy = sum(w .* ly)
    if !(isfinite(s) && s > 0)
        return nothing
    end
    if isnothing(fixed_slope)
        sx = sum(w .* lx)
        sxx = sum(w .* lx .* lx)
        sxy = sum(w .* lx .* ly)
        denom = s * sxx - sx^2
        if !(isfinite(denom) && denom > eps(Float64))
            return nothing
        end
        slope = (s * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / s
    else
        slope = Float64(fixed_slope)
        intercept = sum(w .* (ly .- slope .* lx)) / s
    end

    pred = intercept .+ slope .* lx
    residual = ly .- pred
    chi2 = sum(w .* residual .^ 2)
    y_mean_w = sy / s
    ss_tot = sum(w .* (ly .- y_mean_w) .^ 2)
    r2 = ss_tot > 0 ? 1 - chi2 / ss_tot : NaN
    n_params = loglog_fit_param_count(fixed_slope)
    dof = length(lx) - n_params
    reduced_chi2 = dof > 0 ? chi2 / dof : NaN
    return (
        slope=slope,
        intercept=intercept,
        r2=r2,
        chi2=chi2,
        reduced_chi2=reduced_chi2,
        dof=dof,
        x=x_fit,
        y=y_fit,
        ci95=ci95_fit,
    )
end

function scan_loglog_baseline_weighted(x::AbstractVector{<:Real},
                                       y::AbstractVector{<:Real},
                                       ci95::AbstractVector{<:Real};
                                       n_grid::Int=360,
                                       min_points::Int=4,
                                       target_slope=nothing)
    x_data = Float64.(x)
    y_data = Float64.(y)
    ci95_data = Float64.(ci95)
    base_mask = isfinite.(x_data) .& isfinite.(y_data) .& isfinite.(ci95_data) .&
                (x_data .> 0) .& (ci95_data .> 0)
    if count(base_mask) < min_points
        return nothing
    end

    y_finite = y_data[base_mask]
    y_min = minimum(y_finite)
    y_max = maximum(y_finite)
    y_span = max(y_max - y_min, abs(y_max), abs(y_min), 1e-12)

    baseline_lo_raw = y_min - 0.9 * y_span
    baseline_hi_raw = y_min - max(1e-12, 1e-6 * max(abs(y_min), 1.0))
    if !(isfinite(baseline_lo_raw) && isfinite(baseline_hi_raw))
        return nothing
    end

    baseline_lo = max(0.0, baseline_lo_raw)
    baseline_hi = max(0.0, baseline_hi_raw)
    n_scan = max(n_grid, 12)
    baselines = if baseline_hi > baseline_lo + eps(Float64)
        collect(range(baseline_lo, baseline_hi, length=n_scan))
    else
        [baseline_lo]
    end

    n_candidates = length(baselines)
    r2_vals = fill(NaN, n_candidates)
    chi2_vals = fill(NaN, n_candidates)
    reduced_chi2_vals = fill(NaN, n_candidates)
    slope_vals = fill(NaN, n_candidates)
    point_counts = fill(0, n_candidates)

    best_idx = 0
    best_score = Inf
    best_points = 0
    for (i, baseline) in enumerate(baselines)
        fit = fit_loglog_powerlaw_weighted(x_data, y_data .- baseline, ci95_data; fixed_slope=target_slope)
        if isnothing(fit)
            continue
        end
        n_pts = length(fit.x)
        point_counts[i] = n_pts
        if n_pts < min_points
            continue
        end
        if !(isfinite(fit.reduced_chi2) && isfinite(fit.r2))
            continue
        end

        r2_vals[i] = fit.r2
        chi2_vals[i] = fit.chi2
        reduced_chi2_vals[i] = fit.reduced_chi2
        slope_vals[i] = fit.slope

        better = fit.reduced_chi2 < best_score - 1e-12
        tie = abs(fit.reduced_chi2 - best_score) <= 1e-12 && n_pts > best_points
        if better || tie
            best_idx = i
            best_score = fit.reduced_chi2
            best_points = n_pts
        end
    end

    if best_idx == 0
        return nothing
    end

    best_baseline = baselines[best_idx]
    best_fit = fit_loglog_powerlaw_weighted(x_data, y_data .- best_baseline, ci95_data; fixed_slope=target_slope)
    isnothing(best_fit) && return nothing

    return (
        baselines=baselines,
        r2_vals=r2_vals,
        chi2_vals=chi2_vals,
        reduced_chi2_vals=reduced_chi2_vals,
        slope_vals=slope_vals,
        point_counts=point_counts,
        best_idx=best_idx,
        best_baseline=best_baseline,
        best_fit=best_fit,
        y_shifted_best=(y_data .- best_baseline),
        ci95_data=ci95_data,
    )
end

function save_loglog_baseline_search_plot_weighted(file_name::String,
                                                   analysis_dir::String,
                                                   x::AbstractVector{<:Real},
                                                   y::AbstractVector{<:Real},
                                                   y_ci95::AbstractVector{<:Real};
                                                   quantity_label::AbstractString,
                                                   min_points::Int=4,
                                                   target_slope=nothing)
    scan = scan_loglog_baseline_weighted(x, y, y_ci95; min_points=min_points, target_slope=target_slope)
    out_path = joinpath(analysis_dir, file_name)
    scan_title = isnothing(target_slope) ?
        "$quantity_label baseline scan [CI95-weighted]" :
        @sprintf("%s baseline scan [CI95-weighted, target slope=%.3f]", quantity_label, target_slope)
    best_fit_title = isnothing(target_slope) ?
        @sprintf("%s - baseline(best) fit [CI95-weighted]", quantity_label) :
        @sprintf("%s - baseline(best) fit [CI95-weighted, slope fixed at %.3f]", quantity_label, target_slope)

    if isnothing(scan)
        p_empty = plot(title=scan_title,
                       axis=false,
                       legend=false)
        annotate!(p_empty, 0.5, 0.5, text("No valid CI95-weighted baseline scan (need ≥ $(min_points) positive points with finite CI95)", 10))
        savefig_or_placeholder(
            p_empty,
            out_path;
            placeholder_title="CI95-weighted baseline scan unavailable",
        )
        return nothing
    end

    x_data = Float64.(x)
    best_idx = scan.best_idx
    best_baseline = scan.best_baseline
    best_fit = scan.best_fit
    y_best = scan.y_shifted_best
    best_redchi2 = scan.reduced_chi2_vals[best_idx]
    best_n = scan.point_counts[best_idx]
    ci95_data = scan.ci95_data

    if !ENABLE_BASELINE_SEARCH_FIGURES
        println("Skipping CI95-weighted baseline-search figures (set LOAD_AND_PLOT_ENABLE_BASELINE_SEARCH_FIGURES=1 to enable): ", out_path)
        return (
            baseline=best_baseline,
            slope=best_fit.slope,
            r2=best_fit.r2,
            reduced_chi2=best_redchi2,
            n_points=best_n,
        )
    end

    p_scan = plot(title=scan_title,
                  xlabel="baseline to subtract (b ≥ 0)",
                  ylabel="reduced χ² (lower is better)",
                  framestyle=:box,
                  legend=:topright,
                  size=(1200, 700),
                  titlefontsize=10,
                  guidefontsize=11,
                  tickfontsize=9,
                  legendfontsize=8)
    mask_score = isfinite.(scan.reduced_chi2_vals)
    if any(mask_score)
        plot!(p_scan, scan.baselines[mask_score], scan.reduced_chi2_vals[mask_score],
              lw=1.6, color=:darkgreen, marker=:circle, markersize=3.8,
              markerstrokecolor=:darkgreen, label="reduced χ² samples")
    end
    vline!(p_scan, [best_baseline], color=:red, alpha=0.35, linestyle=:dash, label=false)
    scatter!(p_scan, [best_baseline], [best_redchi2],
             marker=:diamond, markersize=8, color=:red,
             label=@sprintf("best b=%.6g (χ²ν=%.4f)", best_baseline, best_redchi2))

    p_best = plot(title=best_fit_title,
                  xlabel="d",
                  ylabel="value - baseline(best)",
                  xscale=:log10,
                  yscale=:log10,
                  framestyle=:box,
                  legend=:topright,
                  size=(1200, 700),
                  titlefontsize=10,
                  guidefontsize=11,
                  tickfontsize=9,
                  legendfontsize=8)
    apply_log10_decimal_x_ticks!(p_best)
    mask_best = isfinite.(x_data) .& isfinite.(y_best) .& isfinite.(ci95_data) .&
                (x_data .> 0) .& (y_best .> 0) .& (ci95_data .> 0)
    if any(mask_best)
        y_best_sel = y_best[mask_best]
        plot!(p_best, x_data[mask_best], y_best_sel,
              yerror=sanitize_yerror_log(ci95_data[mask_best], y_best_sel),
              marker=:diamond, lw=2.5, color=:black, label="mean - b(best) [CI95]")
    end

    x_fit = best_fit.x
    y_fit = 10 .^ (best_fit.intercept .+ best_fit.slope .* log10.(x_fit))
    plot!(p_best, x_fit, y_fit,
          lw=2.0,
          color=:gray20,
          linestyle=:dashdot,
          label=isnothing(target_slope) ?
              @sprintf("weighted fit slope=%.3f (R²=%.3f)", best_fit.slope, best_fit.r2) :
              @sprintf("weighted fit slope=%.3f fixed (R²=%.3f)", best_fit.slope, best_fit.r2))
    anchor_x = exp(mean(log.(x_fit)))
    anchor_y = 10^(best_fit.intercept + best_fit.slope * log10(anchor_x))
    add_reference_slopes!(p_best, x_fit, anchor_x, anchor_y)

    base, ext = splitext(out_path)
    scan_out = base * "_scan" * ext
    savefig_or_placeholder(
        p_scan,
        scan_out;
        placeholder_title="CI95-weighted baseline scan unavailable",
    )
    savefig_or_placeholder(
        p_best,
        out_path;
        placeholder_title="CI95-weighted baseline fit unavailable",
    )

    return (
        baseline=best_baseline,
        slope=best_fit.slope,
        r2=best_fit.r2,
        reduced_chi2=best_redchi2,
        n_points=best_n,
    )
end

function add_reference_slopes!(p, x::Vector{Float64}, anchor_x::Float64, anchor_y::Float64)
    if !(isfinite(anchor_x) && isfinite(anchor_y) && anchor_x > 0 && anchor_y > 0)
        return
    end
    x_ref = sort([xi for xi in x if isfinite(xi) && xi > 0])
    isempty(x_ref) && return
    x_max = maximum(x_ref)
    for n in 1:4
        y_ref = anchor_y .* (x_ref ./ anchor_x) .^ (-n)
        mask = isfinite.(y_ref) .& (y_ref .> 0)
        if !any(mask)
            continue
        end
        plot!(p, x_ref[mask], y_ref[mask], color=:gray40, lw=1.0, linestyle=:dash, alpha=0.2, label=false)
        y_lab = anchor_y * (x_max / anchor_x)^(-n)
        if isfinite(y_lab) && y_lab > 0
            annotate!(p, x_max, y_lab, text("d^(-$(n))", 7, :gray40))
        end
    end
end

function center_origin_bond_1d(L::Int)
    b1 = max(1, div(L, 2))
    b2 = mod1(b1 + 1, L)
    return b1, b2
end

function is_single_origin_fluctuating_state_1d(state, param)
    if length(param.dims) != 1
        return false, 0, 0
    end
    L = param.dims[1]
    bonds, magnitudes = forcing_bonds_1d(state, L)
    tracked = tracked_force_indices(state, magnitudes)
    if isempty(tracked)
        tracked = collect(1:length(bonds))
    end
    if length(tracked) != 1
        return false, 0, 0
    end
    idx = tracked[1]
    if idx > length(bonds)
        return false, 0, 0
    end
    b1, b2 = bonds[idx]
    origin_b1, origin_b2 = center_origin_bond_1d(L)
    is_origin = (b1 == origin_b1 && b2 == origin_b2) || (b1 == origin_b2 && b2 == origin_b1)
    return is_origin, b1, b2
end

function symmetrized_varj_shifted_profile_1d(state, param; ref_site::Int)
    L = param.dims[1]
    f_avg, f2_avg, samples = PlotUtils.spatial_force_moments_1d(state, L)
    var_f = max.(0.0, f2_avg .- f_avg .^ 2)
    baseline = PlotUtils.single_origin_varj_baseline(param)
    dist_var, var_sym = PlotUtils.average_by_abs_distance(var_f, ref_site)
    var_sym_shifted = var_sym .- baseline
    mask = (dist_var .> 0) .& isfinite.(var_sym_shifted) .& (var_sym_shifted .> 0)
    x = Float64.(dist_var[mask])
    y = Float64.(var_sym_shifted[mask])
    return x, y, samples, baseline
end

function plot_single_origin_varj_reference_slopes(state, param; title_suffix::AbstractString="")
    is_origin, b1, _ = is_single_origin_fluctuating_state_1d(state, param)
    if !is_origin
        return nothing
    end
    x, y, samples, baseline = symmetrized_varj_shifted_profile_1d(state, param; ref_site=b1)
    if isempty(x)
        return nothing
    end

    p = plot(title=@sprintf("Symmetrized Var(J)-%.4g vs |Δx|%s (%d sweeps)", baseline, title_suffix, samples),
             xlabel="|Δx| from origin bond",
             ylabel="Symmetrized Var(J)-baseline (log-log)",
             xscale=:log10,
             yscale=:log10,
             framestyle=:box,
             legend=:outerright,
             grid=:both,
             size=(980, 620),
             top_margin=8Plots.mm,
             left_margin=4Plots.mm,
             right_margin=4Plots.mm)
    apply_log10_decimal_x_ticks!(p)

    fit = fit_loglog_powerlaw(x, y)
    if !isnothing(fit)
        anchor_x = exp(mean(log.(fit.x)))
        anchor_y = 10^(fit.intercept + fit.slope * log10(anchor_x))
        add_reference_slopes!(p, x, anchor_x, anchor_y)
    else
        anchor_x = exp(mean(log.(x)))
        anchor_y = exp(mean(log.(y)))
        add_reference_slopes!(p, x, anchor_x, anchor_y)
    end

    plot!(p, x, y,
          lw=2.3,
          color=:purple,
          marker=:circle,
          markersize=3.8,
          alpha=0.95,
          label="Var(J)-baseline")
    return p
end

function plot_single_origin_varj_loglog_fit(state, param; title_suffix::AbstractString="")
    is_origin, b1, _ = is_single_origin_fluctuating_state_1d(state, param)
    if !is_origin
        return nothing
    end
    x, y, samples, baseline = symmetrized_varj_shifted_profile_1d(state, param; ref_site=b1)
    if isempty(x)
        return nothing
    end

    p = plot(x, y,
             title=@sprintf("Symmetrized Var(J)-%.4g log-log fit%s (%d sweeps)", baseline, title_suffix, samples),
             xlabel="|Δx| from origin bond",
             ylabel="Symmetrized Var(J)-baseline (log-log)",
             lw=2.1,
             color=:purple,
             marker=:circle,
             markersize=3.5,
             label="Var(J)-baseline",
             xscale=:log10,
             yscale=:log10,
             framestyle=:box,
             legend=:outerright,
             grid=:both,
             size=(980, 620),
             top_margin=8Plots.mm,
             left_margin=4Plots.mm,
             right_margin=4Plots.mm)
    apply_log10_decimal_x_ticks!(p)

    fit = fit_loglog_powerlaw(x, y)
    if !isnothing(fit)
        x_fit = fit.x
        y_fit = 10 .^ (fit.intercept .+ fit.slope .* log10.(x_fit))
        plot!(p, x_fit, y_fit,
              lw=2.3,
              color=:black,
              linestyle=:dashdot,
              alpha=0.9,
              label=@sprintf("log-log fit slope=%.3f (R²=%.3f)", fit.slope, fit.r2))

        anchor_x = exp(mean(log.(x_fit)))
        anchor_y = 10^(fit.intercept + fit.slope * log10(anchor_x))
        add_reference_slopes!(p, x_fit, anchor_x, anchor_y)
    end

    return p
end

function plot_bond_cut_model_fits_1d(state, param; smooth_diagonal::Bool=true, corr_model_c::Float64=0.0)
    L = param.dims[1]
    bonds, magnitudes = forcing_bonds_1d(state, L)
    tracked = tracked_force_indices(state, magnitudes)
    if isempty(tracked)
        tracked = collect(1:length(bonds))
    end
    if isempty(tracked)
        return plot(title="Bond cut fit unavailable", axis=false, legend=false)
    end

    corr_mat = connected_corr_mat_1d(state, param)
    colors = [:red, :orange, :green, :blue, :magenta, :cyan, :black]
    centers = Dict{Int,Float64}()
    for idx in tracked
        if idx <= length(bonds)
            b1, b2 = bonds[idx]
            centers[idx] = bond_center_coordinate_1d(L, b1, b2)
        end
    end

    p = plot(title="Bond Correlation Model Fit",
             xlabel="Δx = x' - x_bond",
             ylabel="C(x_bond, x')",
             framestyle=:box,
             legend=:outerright,
             titlefontsize=9,
             legendfontsize=7,
             guidefontsize=9,
             tickfontsize=8)
    model_label = iszero(corr_model_c) ?
        "Model: s*K1*K2*(Δx-x0)/((Δx-x0)^2+K2^2)^2" :
        @sprintf("Model: C + s*K1*K2*(Δx-x0)/((Δx-x0)^2+K2^2)^2, C=%.3g", corr_model_c)
    plot!(p, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=model_label)
    hline!(p, [0.0], color=:gray, linestyle=:dot, label=false)
    vline!(p, [0.0], color=:gray, linestyle=:dash, label=false)

    for (draw_i, idx) in enumerate(tracked)
        if idx > length(bonds)
            continue
        end
        b1, b2 = bonds[idx]
        color = colors[mod1(draw_i, length(colors))]
        cut = bond_centered_cut_1d(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
        x_rel, perm = bond_centered_axis_1d(L, b1, b2)
        x_plot = x_rel[perm]
        y_plot = cut[perm]

        bond_short = "b$(idx)"
        bond_label = "$(bond_short) ($(b1),$(b2))"
        plot!(p, x_plot, y_plot, lw=1.8, color=color, alpha=0.35, label="$(bond_label) data")

        x0_fixed = 0.0
        orientation_sign = 1.0
        if haskey(centers, idx)
            center_i = centers[idx]
            best_disp = NaN
            best_abs = Inf
            for jdx in tracked
                if jdx == idx || !haskey(centers, jdx)
                    continue
                end
                disp = periodic_displacement_1d(centers[jdx] - center_i, L)
                abs_disp = abs(disp)
                if abs_disp > 1e-9 && abs_disp < best_abs
                    best_abs = abs_disp
                    best_disp = disp
                end
            end
            if isfinite(best_disp)
                x0_fixed = best_disp
                orientation_sign = best_disp >= 0 ? 1.0 : -1.0
            end
        end

        fit = fit_bond_cut_profile(x_plot, y_plot; x0=x0_fixed, orientation_sign=orientation_sign, C_fixed=corr_model_c)
        if isnothing(fit)
            plot!(p, [NaN], [NaN], color=color, linestyle=:dash, label="$(bond_short) fit: failed")
            continue
        end

        y_fit = bond_cut_model(x_plot, fit.K1, fit.K2; x0=fit.x0, orientation_sign=fit.orientation_sign, C=fit.C)
        fit_label = @sprintf("%s fit: K1=%.3g, K2=%.3g, x0=%.2f, s=%.0f, R²=%.2f",
                             bond_short, fit.K1, fit.K2, fit.x0, fit.orientation_sign, fit.r2)
        if !iszero(fit.C)
            fit_label = string(fit_label, @sprintf(", C=%.3g", fit.C))
        end
        plot!(p, x_plot, y_fit, lw=2.2, color=color, linestyle=:dash, alpha=0.95, label=fit_label)
    end

    return p
end

function save_diffusive_sweep_components(saved_state::String, state, param, out_dir::String;
                                         include_abs_mean_in_spatial_f_plot::Bool=false,
                                         keep_diagonal_in_multiforce_cut::Bool=false,
                                         corr_model_c::Float64=0.0)
    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name, "current_sweep_statistics")
    mkpath(state_dir)

    if length(param.dims) == 1
        components = PlotUtils.plot_sweep_1d_multiforce(
            state.t,
            state,
            param;
            remove_diagonal_for_cuts=!keep_diagonal_in_multiforce_cut,
            include_abs_mean_in_spatial_f_plot=include_abs_mean_in_spatial_f_plot,
            return_components=true,
        )

        # Keep 1D output lean: the detailed correlation/J panels remain visible in
        # the composite, while separate files are limited to the density views.
        panel_map = [
            ("00_composite", components.final_plot),
            ("01_avg_density", components.avg_density),
            ("02_inst_density", components.inst_density),
        ]

        for (name, plt) in panel_map
            output_file = joinpath(state_dir, string(name, ".png"))
            savefig(plt, output_file)
            println("Saved ", output_file)
        end
    else
        PlotUtils.plot_sweep(state.t, state, param)
        output_file = joinpath(state_dir, "00_composite.png")
        savefig(output_file)
        println("Saved ", output_file)
    end
end

function save_ssep_sweep_and_collapse(saved_state::String, state, param, out_dir::String;
                                      collapse_power::Float64=2.0,
                                      requested_collapse_indices::Union{Nothing,Vector{Int}}=nothing,
                                      bond_centered_collapse::Bool=false)
    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name)
    sweep_dir = joinpath(state_dir, "current_sweep_statistics")
    mkpath(sweep_dir)

    p_sweep = PlotUtils.plot_sweep(state.t, state, param)
    savefig_or_placeholder(
        p_sweep,
        joinpath(sweep_dir, "00_composite.png");
        placeholder_title="SSEP sweep plot unavailable",
    )

    save_1d_data_collapse_if_possible(
        saved_state,
        state,
        param,
        out_dir;
        collapse_power=collapse_power,
        requested_collapse_indices=requested_collapse_indices,
        bond_centered_collapse=bond_centered_collapse,
    )
end

function save_1d_data_collapse_if_possible(saved_state::String, state, param, out_dir::String;
                                           collapse_power::Float64=2.0,
                                           requested_collapse_indices::Union{Nothing,Vector{Int}}=nothing,
                                           bond_centered_collapse::Bool=false)
    supports_1d_data_collapse(state, param) || return false

    base_name = replace(basename(saved_state), ".jld2" => "")
    indices, source = resolve_1d_collapse_indices(state, param, requested_collapse_indices, saved_state)
    if isempty(indices)
        if source == :selected_site_cuts
            println("Skipping 1D data collapse for ", saved_state, ": no positive selected cut indices were found.")
        elseif source == :needs_explicit_indices
            println("Skipping 1D data collapse for ", saved_state, ": pass --collapse_indices for legacy/full-matrix 1D states.")
        end
        return false
    end
    effective_bond_centered_collapse = bond_centered_collapse || default_bond_centered_collapse_1d(state, param)
    collapse_dir = joinpath(out_dir, base_name, collapse_dir_name(collapse_power, indices; bond_centered=effective_bond_centered_collapse))
    mkpath(collapse_dir)

    PlotUtils.plot_data_colapse(
        [(state, param, base_name)],
        collapse_power,
        indices,
        collapse_dir,
        true;
        bond_centered_1d=effective_bond_centered_collapse,
    )
    center_mode = effective_bond_centered_collapse ? ", bond_centered=true" : ""
    println("Saved 1D data collapse using indices ", indices, " under ", collapse_dir, " (source=", source, center_mode, ")")
    return true
end

function save_2d_cut_data_collapse_if_possible(saved_state::String, state, param, out_dir::String;
                                               collapse_power::Float64=2.0,
                                               requested_collapse_indices::Union{Nothing,Vector{Int}}=nothing)
    supports_2d_cut_data_collapse(state, param) || return false

    base_name = replace(basename(saved_state), ".jld2" => "")
    indices, source = resolve_2d_cut_collapse_indices(param, requested_collapse_indices, saved_state)
    if isempty(indices)
        if source == :needs_explicit_indices
            println("Skipping 2D cut data collapse for ", saved_state, ": pass --collapse_indices to choose x/diag cut offsets.")
        end
        return false
    end
    collapse_dir = joinpath(out_dir, base_name, collapse_dir_name(collapse_power, indices))
    mkpath(collapse_dir)

    PlotUtils.plot_data_colapse(
        [(state, param, base_name)],
        collapse_power,
        indices,
        collapse_dir,
        true,
    )
    println("Saved 2D cut data collapse using indices ", indices, " under ", collapse_dir, " (source=", source, ")")
    return true
end

function save_legacy_sweep_plot(saved_state::String, state, param, out_dir::String)
    base_name = replace(basename(saved_state), ".jld2" => "")
    state_dir = joinpath(out_dir, base_name, "current_sweep_statistics")
    mkpath(state_dir)
    output_file = joinpath(state_dir, "00_composite.png")
    try
        PlotUtils.plot_sweep(state.t, state, param)
        savefig(output_file)
        println("Saved ", output_file)
    catch e
        p = if has_prop(state, :ρ_avg)
            ρ_avg = Float64.(get_prop(state, :ρ_avg))
            if ndims(ρ_avg) == 1
                plot(1:length(ρ_avg), ρ_avg,
                     lw=2.2,
                     color=:black,
                     title="Legacy fallback: time-averaged density",
                     xlabel="Site",
                     ylabel="⟨ρ⟩",
                     framestyle=:box)
            else
                heatmap(ρ_avg',
                        title="Legacy fallback: time-averaged density",
                        xlabel="x",
                        ylabel="y",
                        aspect_ratio=1,
                        framestyle=:box)
            end
        else
            plot(title="Legacy fallback: unsupported state format", axis=false, legend=false)
        end
        savefig(p, output_file)
        println("Saved ", output_file, " (legacy fallback)")
        println("Legacy sweep fallback used for ", saved_state, " due to: ", e)
    end
end

function write_two_force_d_analysis_report(rows::AbstractVector,
                                           analysis_dir::AbstractString;
                                           baseline_j2::Float64=NaN,
                                           smooth_diagonal::Bool=true)
    isempty(rows) && return

    d_values = sort(unique(row.d for row in rows))
    max_var_slots = maximum(max(length(row.var_vals_smoothed), length(row.var_vals_raw)) for row in rows)
    max_j2_slots = maximum(length(row.j2_vals) for row in rows)
    grouped = Dict(d => [row for row in rows if row.d == d] for d in d_values)

    x = Float64.(d_values)

    function build_var_aggregates(use_smoothed::Bool)
        y_var_mean = fill(NaN, length(d_values))
        y_var_mean_ci95 = fill(NaN, length(d_values))
        y_var_slots = [fill(NaN, length(d_values)) for _ in 1:max_var_slots]
        y_var_slots_ci95 = [fill(NaN, length(d_values)) for _ in 1:max_var_slots]

        for (di, d) in enumerate(d_values)
            bucket = grouped[d]
            if use_smoothed
                y_var_mean[di], y_var_mean_ci95[di] = bucket_scalar_summary(bucket, :var_mean_smoothed, :var_mean_ci95_smoothed, :var_mean_accum_smoothed, :var_smoothed_exact)
                slot_means, slot_ci95 = bucket_vector_summary(bucket, :var_vals_smoothed, :var_slot_ci95_smoothed, :var_slot_accum_smoothed, :var_smoothed_exact, max_var_slots)
            else
                y_var_mean[di], y_var_mean_ci95[di] = bucket_scalar_summary(bucket, :var_mean_raw, :var_mean_ci95_raw, :var_mean_accum_raw, :var_raw_exact)
                slot_means, slot_ci95 = bucket_vector_summary(bucket, :var_vals_raw, :var_slot_ci95_raw, :var_slot_accum_raw, :var_raw_exact, max_var_slots)
            end
            for slot in 1:max_var_slots
                y_var_slots[slot][di] = slot_means[slot]
                y_var_slots_ci95[slot][di] = slot_ci95[slot]
            end
        end

        var_baseline_candidates = Float64[]
        append!(var_baseline_candidates, [v for v in y_var_mean if isfinite(v)])
        for slot in 1:max_var_slots
            append!(var_baseline_candidates, [v for v in y_var_slots[slot] if isfinite(v)])
        end
        var_baseline = isempty(var_baseline_candidates) ? NaN : minimum(var_baseline_candidates)
        y_var_mean_shifted = [
            (isfinite(v) && isfinite(var_baseline)) ? (v - var_baseline) : NaN
            for v in y_var_mean
        ]
        y_var_mean_shifted_ci95 = copy(y_var_mean_ci95)
        y_var_slots_shifted = [
            [
                (isfinite(v) && isfinite(var_baseline)) ? (v - var_baseline) : NaN
                for v in y_var_slots[slot]
            ]
            for slot in 1:max_var_slots
        ]
        y_var_slots_shifted_ci95 = [copy(y_var_slots_ci95[slot]) for slot in 1:max_var_slots]

        return (
            y_var_mean=y_var_mean,
            y_var_mean_ci95=y_var_mean_ci95,
            y_var_slots=y_var_slots,
            y_var_slots_ci95=y_var_slots_ci95,
            var_baseline=var_baseline,
            y_var_mean_shifted=y_var_mean_shifted,
            y_var_mean_shifted_ci95=y_var_mean_shifted_ci95,
            y_var_slots_shifted=y_var_slots_shifted,
            y_var_slots_shifted_ci95=y_var_slots_shifted_ci95,
        )
    end

    var_smoothed = build_var_aggregates(true)
    var_raw = build_var_aggregates(false)
    var_primary = smooth_diagonal ? var_smoothed : var_raw
    var_secondary = smooth_diagonal ? var_raw : var_smoothed
    primary_mode_label = smooth_diagonal ? "smoothed" : "raw"
    secondary_mode_label = smooth_diagonal ? "raw" : "smoothed"

    y_var_mean = var_primary.y_var_mean

    y_j2_mean = fill(NaN, length(d_values))
    y_j2_mean_ci95 = fill(NaN, length(d_values))
    y_baseline_mean = [finite_mean([row.baseline_j2 for row in grouped[d]]) for d in d_values]
    y_j2_mean_shifted = fill(NaN, length(d_values))
    y_j2_mean_shifted_ci95 = fill(NaN, length(d_values))
    y_j2_slots = [fill(NaN, length(d_values)) for _ in 1:max_j2_slots]
    y_j2_slots_ci95 = [fill(NaN, length(d_values)) for _ in 1:max_j2_slots]
    y_j2_slots_shifted = [fill(NaN, length(d_values)) for _ in 1:max_j2_slots]
    y_j2_slots_shifted_ci95 = [fill(NaN, length(d_values)) for _ in 1:max_j2_slots]

    for (di, d) in enumerate(d_values)
        bucket = grouped[d]
        y_j2_mean[di], y_j2_mean_ci95[di] = bucket_scalar_summary(bucket, :j2_mean, :j2_mean_ci95, :j2_mean_accum, :j2_exact)
        y_j2_mean_shifted[di] = isfinite(y_j2_mean[di]) ? (y_j2_mean[di] - y_baseline_mean[di]) : NaN
        y_j2_mean_shifted_ci95[di] = y_j2_mean_ci95[di]
        slot_means, slot_ci95 = bucket_vector_summary(bucket, :j2_vals, :j2_slot_ci95, :j2_slot_accum, :j2_exact, max_j2_slots)
        for slot in 1:max_j2_slots
            y_j2_slots[slot][di] = slot_means[slot]
            y_j2_slots_ci95[slot][di] = slot_ci95[slot]
            y_j2_slots_shifted[slot][di] = isfinite(slot_means[slot]) ? (slot_means[slot] - y_baseline_mean[di]) : NaN
            y_j2_slots_shifted_ci95[slot][di] = slot_ci95[slot]
        end
    end

    j2_min_baseline_candidates = Float64[]
    append!(j2_min_baseline_candidates, [v for v in y_j2_mean if isfinite(v)])
    for slot in 1:max_j2_slots
        append!(j2_min_baseline_candidates, [v for v in y_j2_slots[slot] if isfinite(v)])
    end
    j2_min_baseline = isempty(j2_min_baseline_candidates) ? NaN : minimum(j2_min_baseline_candidates)
    y_j2_mean_min_shifted = [
        (isfinite(v) && isfinite(j2_min_baseline)) ? (v - j2_min_baseline) : NaN
        for v in y_j2_mean
    ]
    y_j2_mean_min_shifted_ci95 = copy(y_j2_mean_ci95)
    y_j2_slots_min_shifted = [
        [
            (isfinite(v) && isfinite(j2_min_baseline)) ? (v - j2_min_baseline) : NaN
            for v in y_j2_slots[slot]
        ]
        for slot in 1:max_j2_slots
    ]
    y_j2_slots_min_shifted_ci95 = [copy(y_j2_slots_ci95[slot]) for slot in 1:max_j2_slots]

    baseline_label = isfinite(baseline_j2) ? @sprintf("baseline=%.6g", baseline_j2) : @sprintf("baseline=%.6g*rho0^2 (per-state)", TWO_FORCE_J2_BASELINE_RHO_FACTOR)
    j2_min_baseline_label = @sprintf("baseline(min ⟨J²⟩)=%.6g", j2_min_baseline)

    mkpath(analysis_dir)

    function save_var_plots(variant;
                            file_suffix::String="",
                            mode_label::AbstractString="")
        suffix_token = isempty(strip(file_suffix)) ? "" : "_" * strip(file_suffix)
        mode_key_raw = lowercase(strip(String(mode_label)))
        mode_key = if occursin("smooth", mode_key_raw)
            "smoothed"
        elseif occursin("raw", mode_key_raw)
            "raw"
        else
            isempty(mode_key_raw) ? "primary" : mode_key_raw
        end
        mode_title = " [" * mode_key * "]"
        baseline_note = isfinite(variant.var_baseline) ? @sprintf("baseline b = %.6g", variant.var_baseline) : "baseline b = NaN"

        function annotate_baseline_linear!(p, x_vals::Vector{Float64}, y_vals::Vector{Float64})
            mask = isfinite.(x_vals) .& isfinite.(y_vals)
            any(mask) || return
            xs = x_vals[mask]
            ys = y_vals[mask]
            x0 = minimum(xs)
            y_min = minimum(ys)
            y_max = maximum(ys)
            span = max(abs(y_max - y_min), abs(y_max), 1e-12)
            y0 = y_max - 0.08 * span
            annotate!(p, x0, y0, text(baseline_note, 9, :black, :left))
        end

        function annotate_baseline_loglog!(p, x_vals::Vector{Float64}, y_vals::Vector{Float64})
            mask = isfinite.(x_vals) .& isfinite.(y_vals) .& (x_vals .> 0) .& (y_vals .> 0)
            any(mask) || return
            xs = x_vals[mask]
            ys = y_vals[mask]
            x0 = minimum(xs)
            y0 = maximum(ys) / 1.6
            if !(isfinite(y0) && y0 > 0)
                y0 = maximum(ys)
            end
            annotate!(p, x0, y0, text(baseline_note, 9, :black, :left))
        end

        p_var = plot(title="C_bond(0) vs d (log-log)" * mode_title,
                     xlabel="d",
                     ylabel="C_bond(0)",
                     xscale=:log10,
                     yscale=:log10,
                     framestyle=:box,
                     legend=:topright,
                     size=(1180, 760),
                     titlefontsize=11,
                     guidefontsize=13,
                     tickfontsize=10)
        apply_log10_decimal_x_ticks!(p_var)
        for slot in 1:max_var_slots
            y = variant.y_var_slots[slot]
            mask = isfinite.(y) .& (y .> 0)
            if any(mask)
                y_sel = y[mask]
                slot_color = bond_series_color(slot)
                plot!(p_var, x[mask], y[mask],
                      yerror=sanitize_yerror_log(variant.y_var_slots_ci95[slot][mask], y_sel),
                      marker=:circle, lw=2, color=slot_color,
                      markerstrokecolor=slot_color, label="bond $(slot)")
            end
        end
        mask_var_mean = isfinite.(variant.y_var_mean) .& (variant.y_var_mean .> 0)
        if any(mask_var_mean)
            y_mean_sel = variant.y_var_mean[mask_var_mean]
            plot!(p_var, x[mask_var_mean], variant.y_var_mean[mask_var_mean],
                  yerror=sanitize_yerror_log(variant.y_var_mean_ci95[mask_var_mean], y_mean_sel),
                  marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        fit_var = fit_loglog_powerlaw(x, variant.y_var_mean)
        if !isnothing(fit_var)
            x_fit = fit_var.x
            y_fit = 10 .^ (fit_var.intercept .+ fit_var.slope .* log10.(x_fit))
            plot!(p_var, x_fit, y_fit, lw=2.2, color=:black, linestyle=:dashdot, alpha=0.9,
                  label=@sprintf("mean fit slope=%.3f (R²=%.3f)", fit_var.slope, fit_var.r2))
            anchor_x = exp(mean(log.(x_fit)))
            anchor_y = 10^(fit_var.intercept + fit_var.slope * log10(anchor_x))
            add_reference_slopes!(p_var, x_fit, anchor_x, anchor_y)
        elseif any(mask_var_mean)
            x_ref = x[mask_var_mean]
            y_ref = variant.y_var_mean[mask_var_mean]
            anchor_x = exp(mean(log.(x_ref)))
            anchor_y = exp(mean(log.(y_ref)))
            add_reference_slopes!(p_var, x_ref, anchor_x, anchor_y)
        end
        out_var_loglog = joinpath(analysis_dir, "00_bond_center_variance_vs_d_loglog" * suffix_token * ".png")
        savefig_or_placeholder(
            p_var,
            out_var_loglog;
            placeholder_title="C_bond(0) vs d (log-log) unavailable",
        )

        p_var_linear = plot(title="C_bond(0) vs d" * mode_title,
                            xlabel="d",
                            ylabel="C_bond(0)",
                            framestyle=:box,
                            legend=:topright,
                            size=(1180, 760),
                            titlefontsize=11,
                            guidefontsize=13,
                            tickfontsize=10)
        for slot in 1:max_var_slots
            y = variant.y_var_slots[slot]
            mask = isfinite.(y)
            if any(mask)
                slot_color = bond_series_color(slot)
                plot!(p_var_linear, x[mask], y[mask],
                      yerror=sanitize_yerror(variant.y_var_slots_ci95[slot][mask]),
                      marker=:circle, lw=2, color=slot_color,
                      markerstrokecolor=slot_color, label="bond $(slot)")
            end
        end
        mask_var_mean_linear = isfinite.(variant.y_var_mean)
        if any(mask_var_mean_linear)
            plot!(p_var_linear, x[mask_var_mean_linear], variant.y_var_mean[mask_var_mean_linear],
                  yerror=sanitize_yerror(variant.y_var_mean_ci95[mask_var_mean_linear]),
                  marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        out_var_linear = joinpath(analysis_dir, "01_bond_center_variance_vs_d_linear" * suffix_token * ".png")
        savefig_or_placeholder(
            p_var_linear,
            out_var_linear;
            placeholder_title="C_bond(0) vs d unavailable",
        )

        p_var_shifted_linear = plot(title="C_bond(0)-b vs d" * mode_title,
                                    xlabel="d",
                                    ylabel="C_bond(0) - b",
                                    framestyle=:box,
                                    legend=:topright,
                                    size=(1180, 760),
                                    titlefontsize=11,
                                    guidefontsize=13,
                                    tickfontsize=10)
        hline!(p_var_shifted_linear, [0.0], color=:gray55, linestyle=:dash, label=false)
        plot!(p_var_shifted_linear, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=baseline_note)
        for slot in 1:max_var_slots
            y = variant.y_var_slots_shifted[slot]
            mask = isfinite.(y)
            if any(mask)
                slot_color = bond_series_color(slot)
                plot!(p_var_shifted_linear, x[mask], y[mask],
                      yerror=sanitize_yerror(variant.y_var_slots_shifted_ci95[slot][mask]),
                      marker=:circle, lw=2, color=slot_color,
                      markerstrokecolor=slot_color, label="bond $(slot)")
            end
        end
        mask_var_mean_shifted_linear = isfinite.(variant.y_var_mean_shifted)
        if any(mask_var_mean_shifted_linear)
            plot!(p_var_shifted_linear, x[mask_var_mean_shifted_linear], variant.y_var_mean_shifted[mask_var_mean_shifted_linear],
                  yerror=sanitize_yerror(variant.y_var_mean_shifted_ci95[mask_var_mean_shifted_linear]),
                  marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        annotate_baseline_linear!(p_var_shifted_linear, x, variant.y_var_mean_shifted)
        out_var_shifted_linear = joinpath(analysis_dir, "06_bond_center_variance_minus_min_baseline_vs_d_linear" * suffix_token * ".png")
        savefig_or_placeholder(
            p_var_shifted_linear,
            out_var_shifted_linear;
            placeholder_title="C_bond(0)-b vs d unavailable",
        )

        p_var_shifted_loglog = plot(title="C_bond(0)-b vs d (log-log)" * mode_title,
                                    xlabel="d",
                                    ylabel="C_bond(0) - b",
                                    xscale=:log10,
                                    yscale=:log10,
                                    framestyle=:box,
                                    legend=:topright,
                                    size=(1180, 760),
                                    titlefontsize=11,
                                    guidefontsize=13,
                                    tickfontsize=10)
        apply_log10_decimal_x_ticks!(p_var_shifted_loglog)
        plot!(p_var_shifted_loglog, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=baseline_note)
        for slot in 1:max_var_slots
            y = variant.y_var_slots_shifted[slot]
            mask = isfinite.(y) .& (y .> 0)
            if any(mask)
                y_sel = y[mask]
                slot_color = bond_series_color(slot)
                plot!(p_var_shifted_loglog, x[mask], y[mask],
                      yerror=sanitize_yerror_log(variant.y_var_slots_shifted_ci95[slot][mask], y_sel),
                      marker=:circle, lw=2, color=slot_color,
                      markerstrokecolor=slot_color, label="bond $(slot)")
            end
        end
        mask_var_mean_shifted_loglog = isfinite.(variant.y_var_mean_shifted) .& (variant.y_var_mean_shifted .> 0)
        if any(mask_var_mean_shifted_loglog)
            y_mean_sel = variant.y_var_mean_shifted[mask_var_mean_shifted_loglog]
            plot!(p_var_shifted_loglog, x[mask_var_mean_shifted_loglog], variant.y_var_mean_shifted[mask_var_mean_shifted_loglog],
                  yerror=sanitize_yerror_log(variant.y_var_mean_shifted_ci95[mask_var_mean_shifted_loglog], y_mean_sel),
                  marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        fit_var_shifted = fit_loglog_powerlaw(x, variant.y_var_mean_shifted)
        if !isnothing(fit_var_shifted)
            x_fit = fit_var_shifted.x
            y_fit = 10 .^ (fit_var_shifted.intercept .+ fit_var_shifted.slope .* log10.(x_fit))
            plot!(p_var_shifted_loglog, x_fit, y_fit, lw=2.2, color=:black, linestyle=:dashdot, alpha=0.9,
                  label=@sprintf("mean fit slope=%.3f (R²=%.3f)", fit_var_shifted.slope, fit_var_shifted.r2))
            anchor_x = exp(mean(log.(x_fit)))
            anchor_y = 10^(fit_var_shifted.intercept + fit_var_shifted.slope * log10(anchor_x))
            add_reference_slopes!(p_var_shifted_loglog, x_fit, anchor_x, anchor_y)
        elseif any(mask_var_mean_shifted_loglog)
            x_ref = x[mask_var_mean_shifted_loglog]
            y_ref = variant.y_var_mean_shifted[mask_var_mean_shifted_loglog]
            anchor_x = exp(mean(log.(x_ref)))
            anchor_y = exp(mean(log.(y_ref)))
            add_reference_slopes!(p_var_shifted_loglog, x_ref, anchor_x, anchor_y)
        end
        annotate_baseline_loglog!(p_var_shifted_loglog, x, variant.y_var_mean_shifted)
        out_var_shifted_loglog = joinpath(analysis_dir, "07_bond_center_variance_minus_min_baseline_vs_d_loglog" * suffix_token * ".png")
        savefig_or_placeholder(
            p_var_shifted_loglog,
            out_var_shifted_loglog;
            placeholder_title="C_bond(0)-b vs d (log-log) unavailable",
        )

        classic_scan = save_loglog_baseline_search_plot(
            "10_bond_center_variance_baseline_search_loglog_fit" * suffix_token * ".png",
            analysis_dir,
            x,
            variant.y_var_mean;
            quantity_label="C_bond(0)" * mode_title,
            target_slope=TWO_FORCE_LOGLOG_TARGET_SLOPE,
        )
        weighted_scan = save_loglog_baseline_search_plot_weighted(
            "12_bond_center_variance_baseline_search_loglog_fit_weighted_ci95" * suffix_token * ".png",
            analysis_dir,
            x,
            variant.y_var_mean,
            variant.y_var_mean_ci95;
            quantity_label="C_bond(0)" * mode_title,
            target_slope=TWO_FORCE_LOGLOG_TARGET_SLOPE,
        )
        return (classic=classic_scan, weighted=weighted_scan)
    end

    best_var_scans_primary = save_var_plots(var_primary; file_suffix="", mode_label=primary_mode_label)
    secondary_suffix = smooth_diagonal ? "raw_center" : "smoothed_center"
    best_var_scans_secondary = save_var_plots(var_secondary; file_suffix=secondary_suffix, mode_label=secondary_mode_label)
    best_var_scan_smoothed = smooth_diagonal ? best_var_scans_primary.classic : best_var_scans_secondary.classic
    best_var_scan_raw = smooth_diagonal ? best_var_scans_secondary.classic : best_var_scans_primary.classic
    best_var_scan_smoothed_weighted = smooth_diagonal ? best_var_scans_primary.weighted : best_var_scans_secondary.weighted
    best_var_scan_raw_weighted = smooth_diagonal ? best_var_scans_secondary.weighted : best_var_scans_primary.weighted

    function save_j2_plot(file_name::String, title::String, ylabel::String;
                          shifted::Bool=false,
                          loglog::Bool=false,
                          fit_mean_loglog::Bool=false,
                          y_slots_override=nothing,
                          y_slots_ci95_override=nothing,
                          y_mean_override=nothing,
                          y_mean_ci95_override=nothing,
                          baseline_annot::AbstractString="",
                          zero_line::Bool=false,
                          add_reference_slope_guides::Bool=false)
        title_text = shifted ? replace(title, ", " => ",\n"; count=1) : title
        p = plot(title=title_text,
                 xlabel="d",
                 ylabel=ylabel,
                 framestyle=:box,
                 legend=:outerright,
                 size=(1080, 620),
                 top_margin=10Plots.mm,
                 left_margin=4Plots.mm,
                 right_margin=4Plots.mm)
        if loglog
            plot!(p; xscale=:log10, yscale=:log10)
            apply_log10_decimal_x_ticks!(p)
        end
        if zero_line && !loglog
            hline!(p, [0.0], color=:gray55, linestyle=:dash, label=false)
        end
        if !isempty(strip(baseline_annot))
            plot!(p, [NaN], [NaN], lw=0, marker=:none, color=:transparent, label=baseline_annot)
        end
        y_slots = isnothing(y_slots_override) ? (shifted ? y_j2_slots_shifted : y_j2_slots) : y_slots_override
        y_slots_ci95 = isnothing(y_slots_ci95_override) ? (shifted ? y_j2_slots_shifted_ci95 : y_j2_slots_ci95) : y_slots_ci95_override
        y_mean = isnothing(y_mean_override) ? (shifted ? y_j2_mean_shifted : y_j2_mean) : y_mean_override
        y_mean_ci95 = isnothing(y_mean_ci95_override) ? (shifted ? y_j2_mean_shifted_ci95 : y_j2_mean_ci95) : y_mean_ci95_override
        for slot in 1:max_j2_slots
            y = y_slots[slot]
            mask = isfinite.(y)
            if loglog
                mask .&= (y .> 0)
            end
            if any(mask)
                y_sel = y[mask]
                yerr_sel = loglog ?
                    sanitize_yerror_log(y_slots_ci95[slot][mask], y_sel) :
                    sanitize_yerror(y_slots_ci95[slot][mask])
                slot_color = bond_series_color(slot)
                plot!(p, x[mask], y[mask],
                      yerror=yerr_sel,
                      marker=:circle, lw=2, color=slot_color,
                      markerstrokecolor=slot_color, label="bond $(slot)")
            end
        end
        mask_mean = isfinite.(y_mean)
        if loglog
            mask_mean .&= (y_mean .> 0)
        end
        if any(mask_mean)
            y_mean_sel = y_mean[mask_mean]
            yerr_mean_sel = loglog ?
                sanitize_yerror_log(y_mean_ci95[mask_mean], y_mean_sel) :
                sanitize_yerror(y_mean_ci95[mask_mean])
            plot!(p, x[mask_mean], y_mean[mask_mean],
                  yerror=yerr_mean_sel,
                  marker=:diamond, lw=2.8, color=:black, label="mean")
        end
        if fit_mean_loglog && loglog
            fit = fit_loglog_powerlaw(x, y_mean)
            if !isnothing(fit)
                x_fit = fit.x
                y_fit = 10 .^ (fit.intercept .+ fit.slope .* log10.(x_fit))
                plot!(p, x_fit, y_fit, lw=2.2, color=:gray20, linestyle=:dashdot,
                      label=@sprintf("mean fit slope=%.3f (R²=%.3f)", fit.slope, fit.r2))
                if add_reference_slope_guides
                    anchor_x = exp(mean(log.(x_fit)))
                    anchor_y = 10^(fit.intercept + fit.slope * log10(anchor_x))
                    add_reference_slopes!(p, x_fit, anchor_x, anchor_y)
                end
            elseif add_reference_slope_guides && any(mask_mean)
                x_ref = x[mask_mean]
                y_ref = y_mean[mask_mean]
                anchor_x = exp(mean(log.(x_ref)))
                anchor_y = exp(mean(log.(y_ref)))
                add_reference_slopes!(p, x_ref, anchor_x, anchor_y)
            end
        end
        out_path = joinpath(analysis_dir, file_name)
        savefig_or_placeholder(
            p,
            out_path;
            placeholder_title=title * " unavailable",
        )
    end

    save_j2_plot("02_j2_vs_d_linear.png",
                 "⟨J²⟩ vs d",
                 "⟨J²⟩";
                 shifted=false,
                 loglog=false)
    save_j2_plot("03_j2_vs_d_loglog.png",
                 "⟨J²⟩ vs d (log-log)",
                 "⟨J²⟩";
                 shifted=false,
                 loglog=true)
    save_j2_plot("04_j2_minus_baseline_vs_d_linear.png",
                 "⟨J²⟩-baseline vs d, " * baseline_label,
                 "⟨J²⟩ - baseline";
                 shifted=true,
                 loglog=false)
    save_j2_plot("05_j2_minus_baseline_vs_d_loglog.png",
                 "⟨J²⟩-baseline vs d (log-log), " * baseline_label,
                 "⟨J²⟩ - baseline";
                 shifted=true,
                 loglog=true,
                 fit_mean_loglog=true)
    save_j2_plot("08_j2_minus_min_baseline_vs_d_linear.png",
                 "⟨J²⟩-baseline(min) vs d, " * j2_min_baseline_label,
                 "⟨J²⟩ - baseline(min)";
                 shifted=true,
                 y_slots_override=y_j2_slots_min_shifted,
                 y_slots_ci95_override=y_j2_slots_min_shifted_ci95,
                 y_mean_override=y_j2_mean_min_shifted,
                 y_mean_ci95_override=y_j2_mean_min_shifted_ci95,
                 baseline_annot=j2_min_baseline_label,
                 zero_line=true)
    save_j2_plot("09_j2_minus_min_baseline_vs_d_loglog.png",
                 "⟨J²⟩-baseline(min) vs d (log-log), " * j2_min_baseline_label,
                 "⟨J²⟩ - baseline(min)";
                 shifted=true,
                 loglog=true,
                 fit_mean_loglog=true,
                 y_slots_override=y_j2_slots_min_shifted,
                 y_slots_ci95_override=y_j2_slots_min_shifted_ci95,
                 y_mean_override=y_j2_mean_min_shifted,
                 y_mean_ci95_override=y_j2_mean_min_shifted_ci95,
                 baseline_annot=j2_min_baseline_label,
                 add_reference_slope_guides=true)

    best_j2_scan = save_loglog_baseline_search_plot(
        "11_j2_baseline_search_loglog_fit.png",
        analysis_dir,
        x,
        y_j2_mean;
        quantity_label="⟨J²⟩",
        target_slope=TWO_FORCE_LOGLOG_TARGET_SLOPE,
    )
    best_j2_scan_weighted = save_loglog_baseline_search_plot_weighted(
        "13_j2_baseline_search_loglog_fit_weighted_ci95.png",
        analysis_dir,
        x,
        y_j2_mean,
        y_j2_mean_ci95;
        quantity_label="⟨J²⟩",
        target_slope=TWO_FORCE_LOGLOG_TARGET_SLOPE,
    )

    baseline_scan_summary_file = joinpath(analysis_dir, "baseline_search_loglog_summary.csv")
    open(baseline_scan_summary_file, "w") do io
        println(io, "quantity,best_baseline,slope,r2,adjusted_r2,n_points")
        if isnothing(best_var_scan_smoothed)
            println(io, "bond_center_variance_smoothed,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("bond_center_variance_smoothed,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_var_scan_smoothed.baseline,
                                 best_var_scan_smoothed.slope,
                                 best_var_scan_smoothed.r2,
                                 best_var_scan_smoothed.adj_r2,
                                 best_var_scan_smoothed.n_points))
        end
        if isnothing(best_var_scan_raw)
            println(io, "bond_center_variance_raw,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("bond_center_variance_raw,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_var_scan_raw.baseline,
                                 best_var_scan_raw.slope,
                                 best_var_scan_raw.r2,
                                 best_var_scan_raw.adj_r2,
                                 best_var_scan_raw.n_points))
        end
        if isnothing(best_j2_scan)
            println(io, "j2,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("j2,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_j2_scan.baseline,
                                 best_j2_scan.slope,
                                 best_j2_scan.r2,
                                 best_j2_scan.adj_r2,
                                 best_j2_scan.n_points))
        end
    end
    println("Saved ", baseline_scan_summary_file)

    baseline_scan_weighted_summary_file = joinpath(analysis_dir, "baseline_search_loglog_summary_weighted_ci95.csv")
    open(baseline_scan_weighted_summary_file, "w") do io
        println(io, "quantity,best_baseline,slope,r2,reduced_chi2,n_points")
        if isnothing(best_var_scan_smoothed_weighted)
            println(io, "bond_center_variance_smoothed,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("bond_center_variance_smoothed,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_var_scan_smoothed_weighted.baseline,
                                 best_var_scan_smoothed_weighted.slope,
                                 best_var_scan_smoothed_weighted.r2,
                                 best_var_scan_smoothed_weighted.reduced_chi2,
                                 best_var_scan_smoothed_weighted.n_points))
        end
        if isnothing(best_var_scan_raw_weighted)
            println(io, "bond_center_variance_raw,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("bond_center_variance_raw,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_var_scan_raw_weighted.baseline,
                                 best_var_scan_raw_weighted.slope,
                                 best_var_scan_raw_weighted.r2,
                                 best_var_scan_raw_weighted.reduced_chi2,
                                 best_var_scan_raw_weighted.n_points))
        end
        if isnothing(best_j2_scan_weighted)
            println(io, "j2,NaN,NaN,NaN,NaN,0")
        else
            println(io, @sprintf("j2,%.10g,%.10g,%.10g,%.10g,%d",
                                 best_j2_scan_weighted.baseline,
                                 best_j2_scan_weighted.slope,
                                 best_j2_scan_weighted.r2,
                                 best_j2_scan_weighted.reduced_chi2,
                                 best_j2_scan_weighted.n_points))
        end
    end
    println("Saved ", baseline_scan_weighted_summary_file)

    summary_file = joinpath(analysis_dir, "summary_vs_d.csv")
    open(summary_file, "w") do io
        println(io, "d,var_mean,j2_mean,baseline_mean,j2_minus_baseline")
        for (i, d) in enumerate(d_values)
            println(io, @sprintf("%d,%.10g,%.10g,%.10g,%.10g", d, y_var_mean[i], y_j2_mean[i], y_baseline_mean[i], y_j2_mean_shifted[i]))
        end
    end
    println("Saved ", summary_file)

    summary_var_compare_file = joinpath(analysis_dir, "summary_vs_d_variance_smoothed_vs_raw.csv")
    open(summary_var_compare_file, "w") do io
        println(io, "d,var_mean_smoothed,var_mean_raw,var_mean_primary")
        for (i, d) in enumerate(d_values)
            println(io, @sprintf("%d,%.10g,%.10g,%.10g", d, var_smoothed.y_var_mean[i], var_raw.y_var_mean[i], y_var_mean[i]))
        end
    end
    println("Saved ", summary_var_compare_file)
end

function analyze_two_force_d(files::Vector{String}, out_dir::String;
                             baseline_j2::Float64=NaN,
                             smooth_diagonal::Bool=true)
    rows = NamedTuple[]

    for saved_state in files
        try
            state, param, potential = load_state_bundle(saved_state)
            ensure_state_potential!(state, potential)

            if !is_common_diffusive_state(state, param)
                println("Skipping non-diffusive state for d-analysis: ", saved_state)
                continue
            end

            if length(param.dims) != 1
                println("Skipping non-1D state for d-analysis: ", saved_state)
                continue
            end

            L = param.dims[1]
            bonds, magnitudes = forcing_bonds_1d(state, L)
            if length(bonds) < 2
                println("Skipping state without two 1D force bonds: ", saved_state)
                continue
            end

            d = pair_distance_d(bonds, L)
            if isnothing(d)
                println("Skipping state with undefined d: ", saved_state)
                continue
            end

            tracked = tracked_force_indices(state, magnitudes)
            if isempty(tracked)
                tracked = collect(1:length(bonds))
            end

            corr_mat = connected_corr_mat_1d(state, param)
            var_vals_smoothed = [bond_center_value(corr_mat, bonds[i][1], bonds[i][2]; smooth_diagonal=true) for i in tracked]
            var_vals_raw = [bond_center_value(corr_mat, bonds[i][1], bonds[i][2]; smooth_diagonal=false) for i in tracked]

            stats = bond_pass_stats_dict(state)
            j2_all = haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) ? Float64.(stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) : Float64[]
            j2_vals = Float64[]
            for idx in tracked
                if idx <= length(j2_all)
                    push!(j2_vals, j2_all[idx])
                else
                    push!(j2_vals, NaN)
                end
            end

            legacy_aggregate = haskey(stats, AGG_TWO_FORCE_REPLICA_COUNT_KEY)

            var_slot_accum_smoothed = has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_SLOT_N_KEY) ?
                remap_accumulator_by_tracked(metric_accumulator_from_stats(stats, AGG_TWO_FORCE_VAR_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_SLOT_N_KEY;
                                                                           weight_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHT_KEY,
                                                                           weightsq_key=AGG_TWO_FORCE_VAR_SLOT_WEIGHTSQ_KEY), tracked) :
                metric_accumulator_from_values(var_vals_smoothed)
            var_mean_accum_smoothed = has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_MEAN_N_KEY) ?
                metric_accumulator_from_stats(stats, AGG_TWO_FORCE_VAR_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_MEAN_N_KEY;
                                              weight_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHT_KEY,
                                              weightsq_key=AGG_TWO_FORCE_VAR_MEAN_WEIGHTSQ_KEY) :
                metric_accumulator_from_values([finite_mean(var_vals_smoothed)])
            var_smoothed_exact = (
                has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_SLOT_N_KEY) &&
                has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_MEAN_N_KEY)
            ) || !legacy_aggregate

            var_slot_accum_raw = has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY) ?
                remap_accumulator_by_tracked(metric_accumulator_from_stats(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY;
                                                                           weight_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHT_KEY,
                                                                           weightsq_key=AGG_TWO_FORCE_VAR_RAW_SLOT_WEIGHTSQ_KEY), tracked) :
                metric_accumulator_from_values(var_vals_raw)
            var_mean_accum_raw = has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY) ?
                metric_accumulator_from_stats(stats, AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY;
                                              weight_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHT_KEY,
                                              weightsq_key=AGG_TWO_FORCE_VAR_RAW_MEAN_WEIGHTSQ_KEY) :
                metric_accumulator_from_values([finite_mean(var_vals_raw)])
            var_raw_exact = (
                has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_SLOT_N_KEY) &&
                has_metric_accumulator(stats, AGG_TWO_FORCE_VAR_RAW_MEAN_SUM_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_VAR_RAW_MEAN_N_KEY)
            ) || !legacy_aggregate

            j2_slot_accum = has_metric_accumulator(stats, AGG_TWO_FORCE_J2_SLOT_SUM_KEY, AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_J2_SLOT_N_KEY) ?
                remap_accumulator_by_tracked(metric_accumulator_from_stats(stats, AGG_TWO_FORCE_J2_SLOT_SUM_KEY, AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_J2_SLOT_N_KEY;
                                                                           weight_key=AGG_TWO_FORCE_J2_SLOT_WEIGHT_KEY,
                                                                           weightsq_key=AGG_TWO_FORCE_J2_SLOT_WEIGHTSQ_KEY), tracked) :
                metric_accumulator_from_values(j2_vals)
            j2_mean_accum = has_metric_accumulator(stats, AGG_TWO_FORCE_J2_MEAN_SUM_KEY, AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_J2_MEAN_N_KEY) ?
                metric_accumulator_from_stats(stats, AGG_TWO_FORCE_J2_MEAN_SUM_KEY, AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_J2_MEAN_N_KEY;
                                              weight_key=AGG_TWO_FORCE_J2_MEAN_WEIGHT_KEY,
                                              weightsq_key=AGG_TWO_FORCE_J2_MEAN_WEIGHTSQ_KEY) :
                metric_accumulator_from_values([finite_mean(j2_vals)])
            j2_exact = (
                has_metric_accumulator(stats, AGG_TWO_FORCE_J2_SLOT_SUM_KEY, AGG_TWO_FORCE_J2_SLOT_SUMSQ_KEY, AGG_TWO_FORCE_J2_SLOT_N_KEY) &&
                has_metric_accumulator(stats, AGG_TWO_FORCE_J2_MEAN_SUM_KEY, AGG_TWO_FORCE_J2_MEAN_SUMSQ_KEY, AGG_TWO_FORCE_J2_MEAN_N_KEY)
            ) || !legacy_aggregate

            var_vals_smoothed_exact, _, var_slot_ci95_smoothed_exact = stats_from_metric_accumulator(var_slot_accum_smoothed)
            var_mean_smoothed_exact_vec, _, var_mean_ci95_smoothed_exact_vec = stats_from_metric_accumulator(var_mean_accum_smoothed)
            var_vals_raw_exact, _, var_slot_ci95_raw_exact = stats_from_metric_accumulator(var_slot_accum_raw)
            var_mean_raw_exact_vec, _, var_mean_ci95_raw_exact_vec = stats_from_metric_accumulator(var_mean_accum_raw)
            j2_vals_exact, _, j2_slot_ci95_exact = stats_from_metric_accumulator(j2_slot_accum)
            j2_mean_exact_vec, _, j2_mean_ci95_exact_vec = stats_from_metric_accumulator(j2_mean_accum)

            var_slot_mean_meta_smoothed = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY, Float64[]), tracked)
            var_slot_ci95_meta_smoothed = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_VAR_SLOT_CI95_KEY, Float64[]), tracked)
            var_mean_meta_smoothed = (haskey(stats, AGG_TWO_FORCE_VAR_MEAN_KEY) && !isempty(stats[AGG_TWO_FORCE_VAR_MEAN_KEY])) ? Float64(stats[AGG_TWO_FORCE_VAR_MEAN_KEY][1]) : NaN
            var_mean_ci95_meta_smoothed = (haskey(stats, AGG_TWO_FORCE_VAR_MEAN_CI95_KEY) && !isempty(stats[AGG_TWO_FORCE_VAR_MEAN_CI95_KEY])) ? Float64(stats[AGG_TWO_FORCE_VAR_MEAN_CI95_KEY][1]) : NaN

            var_slot_mean_meta_raw = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY, Float64[]), tracked)
            var_slot_ci95_meta_raw = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_CI95_KEY, Float64[]), tracked)
            var_mean_meta_raw = (haskey(stats, AGG_TWO_FORCE_VAR_RAW_MEAN_KEY) && !isempty(stats[AGG_TWO_FORCE_VAR_RAW_MEAN_KEY])) ? Float64(stats[AGG_TWO_FORCE_VAR_RAW_MEAN_KEY][1]) : NaN
            var_mean_ci95_meta_raw = (haskey(stats, AGG_TWO_FORCE_VAR_RAW_MEAN_CI95_KEY) && !isempty(stats[AGG_TWO_FORCE_VAR_RAW_MEAN_CI95_KEY])) ? Float64(stats[AGG_TWO_FORCE_VAR_RAW_MEAN_CI95_KEY][1]) : NaN

            j2_slot_mean_meta = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_J2_SLOT_MEAN_KEY, Float64[]), tracked)
            j2_slot_ci95_meta = remap_vector_by_tracked(get(stats, AGG_TWO_FORCE_J2_SLOT_CI95_KEY, Float64[]), tracked)
            j2_mean_meta = (haskey(stats, AGG_TWO_FORCE_J2_MEAN_KEY) && !isempty(stats[AGG_TWO_FORCE_J2_MEAN_KEY])) ? Float64(stats[AGG_TWO_FORCE_J2_MEAN_KEY][1]) : NaN
            j2_mean_ci95_meta = (haskey(stats, AGG_TWO_FORCE_J2_MEAN_CI95_KEY) && !isempty(stats[AGG_TWO_FORCE_J2_MEAN_CI95_KEY])) ? Float64(stats[AGG_TWO_FORCE_J2_MEAN_CI95_KEY][1]) : NaN

            row_var_vals_smoothed = var_smoothed_exact ? var_vals_smoothed_exact : (!isempty(var_slot_mean_meta_smoothed) ? var_slot_mean_meta_smoothed : var_vals_smoothed)
            row_var_mean_smoothed = var_smoothed_exact ? (isempty(var_mean_smoothed_exact_vec) ? NaN : var_mean_smoothed_exact_vec[1]) : (isfinite(var_mean_meta_smoothed) ? var_mean_meta_smoothed : finite_mean(var_vals_smoothed))
            row_var_slot_ci95_smoothed = var_smoothed_exact ? var_slot_ci95_smoothed_exact : var_slot_ci95_meta_smoothed
            row_var_mean_ci95_smoothed = var_smoothed_exact ? (isempty(var_mean_ci95_smoothed_exact_vec) ? NaN : var_mean_ci95_smoothed_exact_vec[1]) : var_mean_ci95_meta_smoothed

            row_var_vals_raw = var_raw_exact ? var_vals_raw_exact : (!isempty(var_slot_mean_meta_raw) ? var_slot_mean_meta_raw : var_vals_raw)
            row_var_mean_raw = var_raw_exact ? (isempty(var_mean_raw_exact_vec) ? NaN : var_mean_raw_exact_vec[1]) : (isfinite(var_mean_meta_raw) ? var_mean_meta_raw : finite_mean(var_vals_raw))
            row_var_slot_ci95_raw = var_raw_exact ? var_slot_ci95_raw_exact : var_slot_ci95_meta_raw
            row_var_mean_ci95_raw = var_raw_exact ? (isempty(var_mean_ci95_raw_exact_vec) ? NaN : var_mean_ci95_raw_exact_vec[1]) : var_mean_ci95_meta_raw

            row_j2_vals = j2_exact ? j2_vals_exact : (!isempty(j2_slot_mean_meta) ? j2_slot_mean_meta : j2_vals)
            row_j2_mean = j2_exact ? (isempty(j2_mean_exact_vec) ? NaN : j2_mean_exact_vec[1]) : (isfinite(j2_mean_meta) ? j2_mean_meta : finite_mean(j2_vals))
            row_j2_slot_ci95 = j2_exact ? j2_slot_ci95_exact : j2_slot_ci95_meta
            row_j2_mean_ci95 = j2_exact ? (isempty(j2_mean_ci95_exact_vec) ? NaN : j2_mean_ci95_exact_vec[1]) : j2_mean_ci95_meta

            rho0 = Float64(param.ρ₀)
            baseline_row = isfinite(baseline_j2) ? baseline_j2 : (TWO_FORCE_J2_BASELINE_RHO_FACTOR * rho0^2)
            j2_vals_shifted = [isfinite(v) ? (v - baseline_row) : NaN for v in row_j2_vals]
            row_var_vals_primary = smooth_diagonal ? row_var_vals_smoothed : row_var_vals_raw
            row_var_mean_primary = smooth_diagonal ? row_var_mean_smoothed : row_var_mean_raw
            row_var_slot_ci95_primary = smooth_diagonal ? row_var_slot_ci95_smoothed : row_var_slot_ci95_raw
            row_var_mean_ci95_primary = smooth_diagonal ? row_var_mean_ci95_smoothed : row_var_mean_ci95_raw

            push!(rows, (
                file=saved_state,
                d=Int(d),
                rho0=rho0,
                baseline_j2=baseline_row,
                var_vals=row_var_vals_primary,
                var_mean=row_var_mean_primary,
                var_slot_ci95=row_var_slot_ci95_primary,
                var_mean_ci95=row_var_mean_ci95_primary,
                var_vals_smoothed=row_var_vals_smoothed,
                var_mean_smoothed=row_var_mean_smoothed,
                var_slot_ci95_smoothed=row_var_slot_ci95_smoothed,
                var_mean_ci95_smoothed=row_var_mean_ci95_smoothed,
                var_slot_accum_smoothed=var_slot_accum_smoothed,
                var_mean_accum_smoothed=var_mean_accum_smoothed,
                var_smoothed_exact=var_smoothed_exact,
                var_vals_raw=row_var_vals_raw,
                var_mean_raw=row_var_mean_raw,
                var_slot_ci95_raw=row_var_slot_ci95_raw,
                var_mean_ci95_raw=row_var_mean_ci95_raw,
                var_slot_accum_raw=var_slot_accum_raw,
                var_mean_accum_raw=var_mean_accum_raw,
                var_raw_exact=var_raw_exact,
                j2_vals=row_j2_vals,
                j2_mean=row_j2_mean,
                j2_slot_ci95=row_j2_slot_ci95,
                j2_mean_ci95=row_j2_mean_ci95,
                j2_slot_accum=j2_slot_accum,
                j2_mean_accum=j2_mean_accum,
                j2_exact=j2_exact,
                j2_vals_shifted=j2_vals_shifted,
                j2_mean_shifted=isfinite(row_j2_mean) ? (row_j2_mean - baseline_row) : NaN,
            ))
        catch e
            println("Failed in d-analysis for ", saved_state, ": ", e)
        end
    end

    if isempty(rows)
        println("No valid two-force 1D states found for d-analysis.")
        return
    end

    analysis_dir = joinpath(out_dir, "two_force_d_analysis")
    write_two_force_d_analysis_report(
        rows,
        analysis_dir;
        baseline_j2=baseline_j2,
        smooth_diagonal=smooth_diagonal,
    )

    rows_d_ge_8 = [row for row in rows if row.d >= 8]
    if isempty(rows_d_ge_8)
        println("Skipping d>=8-only two-force d-analysis: no rows with d >= 8.")
        return
    end

    analysis_dir_d_ge_8 = joinpath(analysis_dir, "d_ge_8")
    write_two_force_d_analysis_report(
        rows_d_ge_8,
        analysis_dir_d_ge_8;
        baseline_j2=baseline_j2,
        smooth_diagonal=smooth_diagonal,
    )
end

function main()
    args = parse_commandline()
    recursive = get(args, "recursive", false)
    include_abs_mean = get(args, "include_abs_mean_in_spatial_f_plot", false)
    keep_diag = get(args, "keep_diagonal_in_multiforce_cut", false)
    skip_per_state_flag = get(args, "skip_per_state_sweep", false)
    with_per_state_flag = get(args, "with_per_state_sweep", false)
    corr_model_c = Float64(get(args, "corr_model_c", 0.0))
    collapse_power = Float64(get(args, "collapse_power", 2.0))
    requested_collapse_indices = parse_requested_collapse_indices(get(args, "collapse_indices", ""))
    bond_centered_collapse = get(args, "bond_centered_collapse", false)
    mode_raw = String(args["mode"])
    run_id = strip(String(get(args, "run_id", "")))
    if !isempty(run_id) && mode_raw == "single"
        println("run_id was provided with mode=single; auto-detecting run family.")
        mode_raw = "run_id"
    end
    mode = mode_raw == "run_id" ? "run_id" : mode_raw
    run_result_mode = String(get(args, "run_result_mode", "auto"))
    cluster_results_root = String(get(args, "cluster_results_root", "cluster_results"))
    out_dir_arg = String(args["out_dir"])
    out_dir_default = "results_figures/fitting"

    inputs_raw = get(args, "inputs", Any[])
    inputs = String.(inputs_raw isa AbstractVector ? inputs_raw : Any[])
    default_glob = String(args["glob"])
    files = String[]
    resolved_run_mode = ""
    resolved_run_dir = ""
    resolved_run_family = ""
    out_dir = out_dir_arg

    if mode_raw == "run_id" && isempty(run_id)
        error("--mode run_id requires --run_id.")
    end
    if !isempty(run_id) && !(mode in ("run_id", "two_force_d", "single"))
        error("--run_id is supported only with --mode single, --mode two_force_d, or --mode run_id.")
    end
    if skip_per_state_flag && with_per_state_flag
        error("Use only one of --skip_per_state_sweep or --with_per_state_sweep.")
    end

    skip_per_state = skip_per_state_flag

    if !isempty(run_id)
        preferred_families = if mode_raw == "two_force_d"
            [RUN_FAMILY_TWO_FORCE_D]
        else
            String[]
        end
        files, run_out_dir, resolved_run_mode, resolved_run_dir, resolved_run_family = collect_files_from_run_id(
            run_id,
            cluster_results_root,
            run_result_mode;
            preferred_families=preferred_families,
        )
        if out_dir_arg == out_dir_default
            out_dir = run_out_dir
        end
        if mode_raw == "two_force_d" && resolved_run_family != RUN_FAMILY_TWO_FORCE_D
            error("run_id '$run_id' resolved to family '$resolved_run_family', but --mode two_force_d was requested.")
        end
        if mode_raw == "run_id"
            mode = resolved_run_family == RUN_FAMILY_TWO_FORCE_D ? "two_force_d" : "single"
        end
        println("Using run_id='", run_id, "' (mode=", resolved_run_mode, ", family=", resolved_run_family, ") from ", resolved_run_dir)
    end

    if !isempty(run_id) && !skip_per_state_flag && !with_per_state_flag
        if resolved_run_family == RUN_FAMILY_TWO_FORCE_D
            skip_per_state = true
            println("run_id mode: defaulting to analysis-only for two_force_d runs. Pass --with_per_state_sweep to include per-state plots.")
        else
            skip_per_state = false
        end
    elseif with_per_state_flag
        skip_per_state = false
    end

    if !isempty(inputs)
        file_inputs = collect_input_files(inputs, default_glob; recursive=recursive)
        if isempty(files)
            files = file_inputs
        else
            files = sort(unique(vcat(files, file_inputs)))
        end
    end

    println("Found ", length(files), " matching file(s)")
    println("Output directory: ", out_dir)

    if isempty(files)
        if isempty(run_id) && isempty(inputs)
            error("No inputs were provided. Pass input file/dir/glob paths or use --run_id.")
        end
        return
    end

    if !skip_per_state
        for saved_state in files
            try
                println("Processing sweep plots for ", saved_state)
                state, param, potential = load_state_bundle(saved_state)
                ensure_state_potential!(state, potential)
                if is_ssep_state(state, param)
                    save_ssep_sweep_and_collapse(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        collapse_power=collapse_power,
                        requested_collapse_indices=requested_collapse_indices,
                        bond_centered_collapse=bond_centered_collapse,
                    )
                elseif is_legacy_normalized_state(state)
                    save_legacy_sweep_plot(saved_state, state, param, out_dir)
                    save_1d_data_collapse_if_possible(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        collapse_power=collapse_power,
                        requested_collapse_indices=requested_collapse_indices,
                        bond_centered_collapse=bond_centered_collapse,
                    )
                elseif is_common_diffusive_state(state, param)
                    save_diffusive_sweep_components(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        include_abs_mean_in_spatial_f_plot=include_abs_mean,
                        keep_diagonal_in_multiforce_cut=keep_diag,
                        corr_model_c=corr_model_c,
                    )
                    save_1d_data_collapse_if_possible(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        collapse_power=collapse_power,
                        requested_collapse_indices=requested_collapse_indices,
                        bond_centered_collapse=bond_centered_collapse,
                    )
                    save_2d_cut_data_collapse_if_possible(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        collapse_power=collapse_power,
                        requested_collapse_indices=requested_collapse_indices,
                    )
                else
                    save_legacy_sweep_plot(saved_state, state, param, out_dir)
                    save_1d_data_collapse_if_possible(
                        saved_state,
                        state,
                        param,
                        out_dir;
                        collapse_power=collapse_power,
                        requested_collapse_indices=requested_collapse_indices,
                        bond_centered_collapse=bond_centered_collapse,
                    )
                end
            catch e
                println("Failed to export sweep plots for ", saved_state, ": ", e)
            end
        end
    end

    if mode == "two_force_d"
        analyze_two_force_d(
            files,
            out_dir;
            baseline_j2=Float64(args["baseline_j2"]),
            smooth_diagonal=!keep_diag,
        )
    elseif mode != "single"
        error("Unsupported mode: $mode_raw. Use --mode single, --mode two_force_d, or --mode run_id.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
