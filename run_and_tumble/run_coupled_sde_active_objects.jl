#!/usr/bin/env julia

using ArgParse
using Dates
using JLD2
using Printf
using Random
using SHA
using YAML

const SRC_ROOT = joinpath(@__DIR__, "src")
const SDE_DIR = joinpath(SRC_ROOT, "active_objects_sde")

include(joinpath(SDE_DIR, "modules_coupled_sde_active_objects.jl"))

using .FPCoupledSDEActiveObjects

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--config"
            help = "Configuration file path"
            required = true
        "--save_tag"
            help = "Optional save tag used in the output filename"
            arg_type = String
            required = false
        "--performance_mode"
            help = "Lean cluster mode; currently disables only nonessential console output"
            action = :store_true
        "--estimate_only"
            help = "Print a simple step-count/runtime-scale summary and exit"
            action = :store_true
        "--initial_state"
            help = "Optional completed JLD2 result whose saved state initializes this run; stats are not reused"
            arg_type = String
            required = false
        "--resume_checkpoint"
            help = "Optional runner checkpoint JLD2 to resume warmup/production and accumulated stats"
            arg_type = String
            required = false
        "--checkpoint_path"
            help = "Optional path where the latest checkpoint for this replica is atomically updated"
            arg_type = String
            required = false
        "--checkpoint_interval_steps"
            help = "Save a checkpoint after this many warmup/production steps; 0 disables checkpointing"
            arg_type = Int
            default = 0
    end
    return parse_args(settings)
end

function get_default_params()
    return Dict(
        "mode" => FPCoupledSDEActiveObjects.MOBILE_OBJECTS_MODE,
        "L" => 128.0,
        "rho0" => 10.0,
        "D0" => 1.0,
        "dt" => 1.0e-3,
        "mu_bath" => 1.0,
        "mu_obj" => 1.0e-3,
        "f0" => 1.0,
        "sigma_f" => 1.0,
        "profile_type" => "gaussian",
        "separation" => 16.0,
        "random_initial_objects" => false,
        "initial_min_separation" => 0.0,
        "initial_max_separation" => nothing,
        "n_steps" => 10_000,
        "warmup_steps" => 1_000,
        "sample_interval" => 1,
        "history_interval" => 100,
        "n_bins" => 64,
        "max_history_records" => 100_000,
        "save_raw_history" => true,
        "seed" => 0,
        "description" => "",
        "save_dir" => "saved_states/coupled_sde_active_objects",
        "performance_mode" => false,
        "cluster_mode" => false,
    )
end

function to_float(value, key_name::String)
    value isa Number && return Float64(value)
    value isa AbstractString && return parse(Float64, strip(value))
    error("$key_name must be numeric.")
end

function to_int(value, key_name::String)
    value isa Number && return Int(round(Float64(value)))
    value isa AbstractString && return parse(Int, strip(value))
    error("$key_name must be numeric.")
end

function to_bool(value, key_name::String)
    if value isa Bool
        return value
    elseif value isa Number
        return value != 0
    elseif value isa AbstractString
        lowered = lowercase(strip(value))
        if lowered in ("true", "t", "1", "yes", "y", "on")
            return true
        elseif lowered in ("false", "f", "0", "no", "n", "off", "")
            return false
        end
    end
    error("$key_name must be boolean-like.")
end

function optional_float(params, key::String)
    if !haskey(params, key) || isnothing(params[key])
        return nothing
    end
    return to_float(params[key], key)
end

function get_value(params, defaults, key::String)
    return get(params, key, defaults[key])
end

function get_rho0(params, defaults)
    if haskey(params, "rho0")
        return to_float(params["rho0"], "rho0")
    end
    rho0_unicode_key = "\u03c1\u2080"
    if haskey(params, rho0_unicode_key)
        return to_float(params[rho0_unicode_key], "rho0")
    end
    return to_float(defaults["rho0"], "rho0")
end

function get_warmup_steps(params, defaults)
    if haskey(params, "warmup_steps")
        return to_int(params["warmup_steps"], "warmup_steps")
    elseif haskey(params, "warmup_sweeps")
        return to_int(params["warmup_sweeps"], "warmup_sweeps")
    end
    return to_int(defaults["warmup_steps"], "warmup_steps")
end

function get_n_steps(params, defaults)
    if haskey(params, "n_steps")
        return to_int(params["n_steps"], "n_steps")
    elseif haskey(params, "n_sweeps")
        return to_int(params["n_sweeps"], "n_sweeps")
    end
    return to_int(defaults["n_steps"], "n_steps")
end

function build_param(params, defaults)
    L = to_float(get_value(params, defaults, "L"), "L")
    rho0 = get_rho0(params, defaults)
    N = if haskey(params, "N") && !isnothing(params["N"])
        to_int(params["N"], "N")
    else
        Int(round(rho0 * L))
    end
    N >= 0 || error("N must be nonnegative. Got $N.")

    sample_interval = max(to_int(get_value(params, defaults, "sample_interval"), "sample_interval"), 1)
    history_interval = max(to_int(get_value(params, defaults, "history_interval"), "history_interval"), 1)
    n_bins = max(to_int(get_value(params, defaults, "n_bins"), "n_bins"), 1)
    max_history_records = max(to_int(get_value(params, defaults, "max_history_records"), "max_history_records"), 0)

    mode = FPCoupledSDEActiveObjects.normalize_mode(get(params, "mode", get(params, "simulation_mode", defaults["mode"])))
    return CoupledSDEParam(
        mode=mode,
        L=L,
        rho0=rho0,
        N=N,
        D0=to_float(get_value(params, defaults, "D0"), "D0"),
        dt=to_float(get_value(params, defaults, "dt"), "dt"),
        mu_bath=to_float(get_value(params, defaults, "mu_bath"), "mu_bath"),
        mu_obj=to_float(get_value(params, defaults, "mu_obj"), "mu_obj"),
        f0=to_float(get_value(params, defaults, "f0"), "f0"),
        sigma_f=to_float(get_value(params, defaults, "sigma_f"), "sigma_f"),
        profile_type=String(get_value(params, defaults, "profile_type")),
        separation=to_float(get_value(params, defaults, "separation"), "separation"),
        initial_XA=optional_float(params, "initial_XA"),
        initial_XB=optional_float(params, "initial_XB"),
        random_initial_objects=to_bool(get_value(params, defaults, "random_initial_objects"), "random_initial_objects"),
        initial_min_separation=to_float(get_value(params, defaults, "initial_min_separation"), "initial_min_separation"),
        initial_max_separation=optional_float(params, "initial_max_separation"),
        n_steps=max(get_n_steps(params, defaults), 0),
        warmup_steps=max(get_warmup_steps(params, defaults), 0),
        sample_interval=sample_interval,
        history_interval=history_interval,
        n_bins=n_bins,
        max_history_records=max_history_records,
        save_raw_history=to_bool(get_value(params, defaults, "save_raw_history"), "save_raw_history"),
        seed=to_int(get_value(params, defaults, "seed"), "seed"),
        description=String(get_value(params, defaults, "description")),
    )
end

function performance_mode_from_params(args, params, defaults)
    if get(args, "performance_mode", false)
        return true
    elseif haskey(params, "performance_mode")
        return to_bool(params["performance_mode"], "performance_mode")
    elseif haskey(params, "cluster_mode")
        return to_bool(params["cluster_mode"], "cluster_mode")
    end
    return to_bool(defaults["performance_mode"], "performance_mode")
end

function rng_from_param(param::CoupledSDEParam)
    if param.seed > 0
        return MersenneTwister(param.seed)
    end
    return MersenneTwister(rand(1:2^30))
end

function compatible_physical_params!(saved::CoupledSDEParam, current::CoupledSDEParam, path::AbstractString)
    for key in (:mode, :N, :profile_type, :random_initial_objects)
        getfield(saved, key) == getfield(current, key) || error("Incompatible $key in initial/checkpoint state $path.")
    end
    for key in (:L, :rho0, :D0, :dt, :mu_bath, :mu_obj, :f0, :sigma_f, :separation, :initial_min_separation)
        a = Float64(getfield(saved, key))
        b = Float64(getfield(current, key))
        isapprox(a, b; rtol=1.0e-10, atol=1.0e-12) || error("Incompatible $key in initial/checkpoint state $path: saved=$a current=$b")
    end
    saved.n_bins == current.n_bins || error("Incompatible n_bins in initial/checkpoint state $path.")
    return true
end

function validate_state_shape!(state::CoupledSDEState, param::CoupledSDEParam, path::AbstractString)
    length(state.x) == param.N || error("Initial/checkpoint state $path has N=$(length(state.x)); expected $(param.N).")
    return true
end

function load_initial_state(path::AbstractString, param::CoupledSDEParam)
    data = JLD2.load(path)
    haskey(data, "state") || error("Initial-state file does not contain a saved state: $path")
    if haskey(data, "param")
        compatible_physical_params!(data["param"], param, path)
    end
    state = data["state"]
    validate_state_shape!(state, param, path)
    return state
end

function load_resume_checkpoint(path::AbstractString, param::CoupledSDEParam)
    data = JLD2.load(path)
    haskey(data, "checkpoint") || error("Checkpoint file does not contain checkpoint metadata: $path")
    haskey(data, "state") || error("Checkpoint file does not contain state: $path")
    haskey(data, "stats") || error("Checkpoint file does not contain stats: $path")
    haskey(data, "rng") || error("Checkpoint file does not contain rng: $path")
    if haskey(data, "param")
        compatible_physical_params!(data["param"], param, path)
    end
    state = data["state"]
    validate_state_shape!(state, param, path)
    checkpoint = data["checkpoint"]
    return Dict(
        "state" => state,
        "stats" => data["stats"],
        "rng" => data["rng"],
        "completed_warmup_steps" => Int(get(checkpoint, "completed_warmup_steps", 0)),
        "completed_production_steps" => Int(get(checkpoint, "completed_production_steps", 0)),
    )
end

function sanitize_filename_token(value::AbstractString)
    token = replace(strip(value), r"[^A-Za-z0-9._-]+" => "-")
    token = replace(token, r"-{2,}" => "-")
    token = replace(token, r"^-+" => "")
    token = replace(token, r"-+$" => "")
    return isempty(token) ? "coupled_sde" : token
end

function compact_token(token::AbstractString; max_bytes::Int=120)
    safe = sanitize_filename_token(token)
    ncodeunits(safe) <= max_bytes && return safe
    hash = bytes2hex(sha1(codeunits(safe)))[1:10]
    keep = max(max_bytes - ncodeunits(hash) - 3, 12)
    return first(safe, min(keep, length(safe))) * "_h" * hash
end

function output_filename(save_dir::AbstractString, param::CoupledSDEParam; save_tag=nothing)
    mkpath(save_dir)
    mode_token = param.mode == FPCoupledSDEActiveObjects.FIXED_SEPARATION_MODE ? "fixed" : "mobile"
    sep_token = @sprintf("sep%.6g", param.separation)
    rho_token = @sprintf("rho%.6g", param.rho0)
    mu_token = @sprintf("muo%.3e", param.mu_obj)
    tag = isnothing(save_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : String(save_tag)
    stem = compact_token("coupled_sde_$(mode_token)_L$(round(Int, param.L))_$(rho_token)_$(sep_token)_$(mu_token)_id-$(tag)")
    return joinpath(save_dir, stem * ".jld2")
end

function write_summary(path::AbstractString, result::Dict, output_path::AbstractString)
    open(path, "w") do io
        println(io, "output_path=$(output_path)")
        println(io, "result_type=$(result["result_type"])")
        println(io, "mode=$(result["mode"])")
        println(io, "sample_count=$(result["sample_count"])")
        params = result["parameters"]
        for key in ["L", "rho0", "N", "D0", "dt", "mu_bath", "mu_obj", "f0", "sigma_f", "separation", "n_steps", "warmup_steps", "sample_interval"]
            println(io, "$(key)=$(params[key])")
        end
        if haskey(result, "D_rel_proxy")
            println(io, "D_rel_proxy=$(result["D_rel_proxy"])")
        end
        stability = result["stability"]
        for key in sort(collect(keys(stability)))
            println(io, "stability_$(key)=$(stability[key])")
        end
    end
    return path
end

function print_estimate(param::CoupledSDEParam)
    total_steps = param.warmup_steps + param.n_steps
    println("Coupled-SDE active-object run estimate")
    println("  mode=$(param.mode)")
    println("  N=$(param.N)")
    println("  warmup_steps=$(param.warmup_steps)")
    println("  production_steps=$(param.n_steps)")
    println("  total particle updates=$(total_steps * param.N)")
    println("  thermal_rms=$(sqrt(2.0 * param.D0 * param.dt))")
    println("  thermal_rms/sigma_f=$(sqrt(2.0 * param.D0 * param.dt) / param.sigma_f)")
    println("  mu_bath*f0*sqrt(dt)/sigma_f=$(abs(param.mu_bath * param.f0) * sqrt(param.dt) / param.sigma_f)")
end

function save_checkpoint(
    checkpoint_path::AbstractString,
    param::CoupledSDEParam,
    state::CoupledSDEState,
    stats,
    rng::AbstractRNG,
    completed_warmup_steps::Integer,
    completed_production_steps::Integer,
    phase::AbstractString,
)
    isempty(checkpoint_path) && return nothing
    mkpath(dirname(checkpoint_path))
    checkpoint = Dict(
        "checkpoint_type" => "coupled_sde_mobile_objects",
        "phase" => String(phase),
        "completed_warmup_steps" => Int(completed_warmup_steps),
        "completed_production_steps" => Int(completed_production_steps),
        "target_warmup_steps" => param.warmup_steps,
        "target_production_steps" => param.n_steps,
        "saved_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
    )
    tmp = "$(checkpoint_path).tmp.$(getpid())"
    jldsave(tmp; checkpoint=checkpoint, param=param, state=state, stats=stats, rng=rng)
    mv(tmp, checkpoint_path; force=true)
    return checkpoint_path
end

function maybe_save_checkpoint!(
    checkpoint_path::AbstractString,
    checkpoint_interval_steps::Integer,
    last_checkpoint_total::Base.RefValue{Int},
    param::CoupledSDEParam,
    state::CoupledSDEState,
    stats,
    rng::AbstractRNG,
    completed_warmup_steps::Integer,
    completed_production_steps::Integer,
    phase::AbstractString,
)
    checkpoint_interval_steps > 0 || return nothing
    total_completed = Int(completed_warmup_steps + completed_production_steps)
    if total_completed - last_checkpoint_total[] >= checkpoint_interval_steps
        save_checkpoint(checkpoint_path, param, state, stats, rng, completed_warmup_steps, completed_production_steps, phase)
        last_checkpoint_total[] = total_completed
    end
    return nothing
end

function run_mobile_objects_checkpointed!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    rng::AbstractRNG;
    checkpoint_path::AbstractString="",
    checkpoint_interval_steps::Integer=0,
    resume_data=nothing,
)
    work = SDEWork(param.N)
    completed_warmup_steps = 0
    completed_production_steps = 0
    stats = nothing
    if !isnothing(resume_data)
        completed_warmup_steps = Int(resume_data["completed_warmup_steps"])
        completed_production_steps = Int(resume_data["completed_production_steps"])
        stats = resume_data["stats"]
    end
    last_checkpoint_total = Ref(completed_warmup_steps + completed_production_steps)

    while completed_warmup_steps < max(param.warmup_steps, 0)
        step_mobile_objects!(state, param, rng, work)
        completed_warmup_steps += 1
        maybe_save_checkpoint!(
            checkpoint_path,
            checkpoint_interval_steps,
            last_checkpoint_total,
            param,
            state,
            stats,
            rng,
            completed_warmup_steps,
            completed_production_steps,
            "warmup",
        )
    end

    if isnothing(stats)
        stats = FPCoupledSDEActiveObjects.MobileStats(param)
    end
    while completed_production_steps < max(param.n_steps, 0)
        completed_production_steps += 1
        obs = step_mobile_objects!(state, param, rng, work)
        FPCoupledSDEActiveObjects.accumulate!(stats, obs, state, param, completed_production_steps)
        maybe_save_checkpoint!(
            checkpoint_path,
            checkpoint_interval_steps,
            last_checkpoint_total,
            param,
            state,
            stats,
            rng,
            completed_warmup_steps,
            completed_production_steps,
            "production",
        )
    end

    if checkpoint_interval_steps > 0
        save_checkpoint(checkpoint_path, param, state, stats, rng, completed_warmup_steps, completed_production_steps, "complete")
    end
    return stats
end

function main()
    args = parse_commandline()
    defaults = get_default_params()
    params = YAML.load_file(String(args["config"]))
    param = build_param(params, defaults)
    performance_mode = performance_mode_from_params(args, params, defaults)
    initial_state_path = get(args, "initial_state", nothing)
    resume_checkpoint_path = get(args, "resume_checkpoint", nothing)
    raw_checkpoint_path = get(args, "checkpoint_path", nothing)
    checkpoint_path = isnothing(raw_checkpoint_path) ? "" : String(raw_checkpoint_path)
    checkpoint_interval_steps = max(Int(get(args, "checkpoint_interval_steps", 0)), 0)

    if get(args, "estimate_only", false)
        print_estimate(param)
        return
    end
    if !isnothing(initial_state_path) && !isnothing(resume_checkpoint_path)
        error("Use only one of --initial_state or --resume_checkpoint.")
    end

    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    resume_data = nothing
    rng = rng_from_param(param)
    state = if !isnothing(resume_checkpoint_path)
        resume_data = load_resume_checkpoint(String(resume_checkpoint_path), param)
        rng = resume_data["rng"]
        resume_data["state"]
    elseif !isnothing(initial_state_path)
        load_initial_state(String(initial_state_path), param)
    else
        initialize_state(param, rng)
    end

    !performance_mode && print_estimate(param)
    start_time = time()
    stats = if param.mode == FPCoupledSDEActiveObjects.FIXED_SEPARATION_MODE
        run_fixed_separation!(state, param, rng)
    elseif param.mode == FPCoupledSDEActiveObjects.MOBILE_OBJECTS_MODE
        if checkpoint_interval_steps > 0 || !isnothing(resume_data)
            run_mobile_objects_checkpointed!(
                state,
                param,
                rng;
                checkpoint_path=checkpoint_path,
                checkpoint_interval_steps=checkpoint_interval_steps,
                resume_data=resume_data,
            )
        else
            run_mobile_objects!(state, param, rng)
        end
    else
        error("Unsupported mode stored in param: $(param.mode)")
    end
    elapsed = time() - start_time

    result = if param.mode == FPCoupledSDEActiveObjects.FIXED_SEPARATION_MODE
        fixed_result_dict(param, state, stats)
    else
        mobile_result_dict(param, state, stats)
    end
    metadata = Dict(
        "config_path" => abspath(String(args["config"])),
        "save_tag" => get(args, "save_tag", nothing),
        "wall_time_seconds" => elapsed,
        "saved_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "runner" => "run_coupled_sde_active_objects.jl",
        "initial_state_path" => isnothing(initial_state_path) ? nothing : abspath(String(initial_state_path)),
        "resume_checkpoint_path" => isnothing(resume_checkpoint_path) ? nothing : abspath(String(resume_checkpoint_path)),
        "checkpoint_path" => isempty(checkpoint_path) ? nothing : abspath(checkpoint_path),
        "checkpoint_interval_steps" => checkpoint_interval_steps,
    )

    output_path = output_filename(save_dir, param; save_tag=get(args, "save_tag", nothing))
    jldsave(output_path; result=result, param=param, state=state, stats=stats, metadata=metadata)
    summary_path = replace(output_path, r"\.jld2$" => "_summary.txt")
    write_summary(summary_path, result, output_path)

    println("Saved coupled-SDE active-object result:")
    println("  $(output_path)")
    println("  $(summary_path)")
    println("  wall_time_seconds=$(round(elapsed, digits=3))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
