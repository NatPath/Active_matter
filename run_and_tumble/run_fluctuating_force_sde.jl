#!/usr/bin/env julia

using ArgParse
using Dates
using JLD2
using Printf
using Random
using SHA
using YAML

const SRC_ROOT = joinpath(@__DIR__, "src")
const SDE_DIR = joinpath(SRC_ROOT, "fluctuating_force_sde")

include(joinpath(SDE_DIR, "modules_fluctuating_force_sde.jl"))

using .FPFluctuatingForceSDE

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
    end
    return parse_args(settings)
end

function get_default_params()
    return Dict(
        "dims" => 1,
        "L" => 60.0,
        "N" => 8_000,
        "D_bath" => 1.0,
        "dt" => 0.05,
        "mu_bath" => 5.0,
        "f0" => 1.0,
        "sigma_f" => 0.5,
        "profile_type" => "gaussian",
        "force_centers" => nothing,
        "mobile_forces" => false,
        "force_mobility" => 0.0,
        "force_diffusivity" => 0.0,
        "n_steps" => 10_000,
        "warmup_steps" => 1_000,
        "sample_interval" => 1,
        "n_bins" => 80,
        "n_radial_bins" => 20,
        "radial_min" => 0.8,
        "radial_max" => nothing,
        "edge_bins_for_offset" => 5,
        "variance_floor" => 1.0e-6,
        "history_interval" => 1_000,
        "max_history_records" => 20_000,
        "save_force_history" => true,
        "seed" => 0,
        "description" => "",
        "save_dir" => "saved_states/fluctuating_force_sde",
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

function is_scalar_number(value)
    return value isa Number || value isa AbstractString
end

function vector_to_float(values, key_name::String)
    values isa AbstractVector || error("$key_name must be a vector.")
    return [to_float(v, key_name) for v in values]
end

function force_centers_from_params(params, defaults, dims::Int)
    raw = if haskey(params, "force_centers")
        params["force_centers"]
    elseif haskey(params, "force_center")
        params["force_center"]
    else
        defaults["force_centers"]
    end
    if isnothing(raw)
        return zeros(Float64, dims, 1)
    end
    raw isa AbstractVector || error("force_centers must be a vector or vector of vectors.")
    if dims == 1 && all(is_scalar_number, raw)
        centers = zeros(Float64, 1, length(raw))
        for k in eachindex(raw)
            centers[1, k] = to_float(raw[k], "force_centers")
        end
        return centers
    elseif dims > 1 && length(raw) == dims && all(is_scalar_number, raw)
        centers = zeros(Float64, dims, 1)
        vals = vector_to_float(raw, "force_centers")
        for d in 1:dims
            centers[d, 1] = vals[d]
        end
        return centers
    end

    n_forces = length(raw)
    centers = zeros(Float64, dims, n_forces)
    for k in 1:n_forces
        item = raw[k]
        item isa AbstractVector || error("force_centers[$k] must be a vector of length dims=$dims.")
        length(item) == dims || error("force_centers[$k] has length $(length(item)); expected dims=$dims.")
        vals = vector_to_float(item, "force_centers[$k]")
        for d in 1:dims
            centers[d, k] = vals[d]
        end
    end
    return centers
end

function get_mu_bath(params, defaults)
    if haskey(params, "mu_bath")
        return to_float(params["mu_bath"], "mu_bath")
    elseif haskey(params, "mu_active")
        return to_float(params["mu_active"], "mu_active")
    end
    return to_float(defaults["mu_bath"], "mu_bath")
end

function build_param(params, defaults)
    dims = FPFluctuatingForceSDE.normalize_dimension(to_int(get_value(params, defaults, "dims"), "dims"))
    centers = force_centers_from_params(params, defaults, dims)
    radial_max = optional_float(params, "radial_max")
    return FluctuatingForceParam(
        dims=dims,
        L=to_float(get_value(params, defaults, "L"), "L"),
        N=max(to_int(get_value(params, defaults, "N"), "N"), 0),
        D_bath=to_float(get_value(params, defaults, "D_bath"), "D_bath"),
        dt=to_float(get_value(params, defaults, "dt"), "dt"),
        mu_bath=get_mu_bath(params, defaults),
        f0=to_float(get_value(params, defaults, "f0"), "f0"),
        sigma_f=to_float(get_value(params, defaults, "sigma_f"), "sigma_f"),
        profile_type=String(get_value(params, defaults, "profile_type")),
        force_centers=centers,
        mobile_forces=to_bool(get_value(params, defaults, "mobile_forces"), "mobile_forces"),
        force_mobility=to_float(get_value(params, defaults, "force_mobility"), "force_mobility"),
        force_diffusivity=to_float(get_value(params, defaults, "force_diffusivity"), "force_diffusivity"),
        n_steps=max(to_int(get_value(params, defaults, "n_steps"), "n_steps"), 0),
        warmup_steps=max(to_int(get_value(params, defaults, "warmup_steps"), "warmup_steps"), 0),
        sample_interval=max(to_int(get_value(params, defaults, "sample_interval"), "sample_interval"), 1),
        n_bins=max(to_int(get_value(params, defaults, "n_bins"), "n_bins"), 1),
        n_radial_bins=max(to_int(get_value(params, defaults, "n_radial_bins"), "n_radial_bins"), 1),
        radial_min=to_float(get_value(params, defaults, "radial_min"), "radial_min"),
        radial_max=radial_max,
        edge_bins_for_offset=max(to_int(get_value(params, defaults, "edge_bins_for_offset"), "edge_bins_for_offset"), 1),
        variance_floor=to_float(get_value(params, defaults, "variance_floor"), "variance_floor"),
        history_interval=max(to_int(get_value(params, defaults, "history_interval"), "history_interval"), 1),
        max_history_records=max(to_int(get_value(params, defaults, "max_history_records"), "max_history_records"), 0),
        save_force_history=to_bool(get_value(params, defaults, "save_force_history"), "save_force_history"),
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

function rng_from_param(param::FluctuatingForceParam)
    if param.seed > 0
        return MersenneTwister(param.seed)
    end
    return MersenneTwister(rand(1:2^30))
end

function sanitize_filename_token(value::AbstractString)
    token = replace(strip(value), r"[^A-Za-z0-9._-]+" => "-")
    token = replace(token, r"-{2,}" => "-")
    token = replace(token, r"^-+" => "")
    token = replace(token, r"-+$" => "")
    return isempty(token) ? "fluctuating_force_sde" : token
end

function compact_token(token::AbstractString; max_bytes::Int=120)
    safe = sanitize_filename_token(token)
    ncodeunits(safe) <= max_bytes && return safe
    hash = bytes2hex(sha1(codeunits(safe)))[1:10]
    keep = max(max_bytes - ncodeunits(hash) - 3, 12)
    return first(safe, min(keep, length(safe))) * "_h" * hash
end

function output_filename(save_dir::AbstractString, param::FluctuatingForceParam; save_tag=nothing)
    mkpath(save_dir)
    tag = isnothing(save_tag) ? Dates.format(now(), "yyyymmdd-HHMMSS") : String(save_tag)
    stem = compact_token(@sprintf(
        "fluctuating_force_sde_%dD_L%.6g_N%d_nf%d_id-%s",
        param.dims,
        param.L,
        param.N,
        size(param.force_centers, 2),
        tag,
    ))
    return joinpath(save_dir, stem * ".jld2")
end

function write_summary(path::AbstractString, result::Dict, output_path::AbstractString)
    open(path, "w") do io
        println(io, "output_path=$(output_path)")
        println(io, "result_type=$(result["result_type"])")
        println(io, "sample_count=$(result["sample_count"])")
        params = result["parameters"]
        for key in ["dims", "L", "N", "D_bath", "dt", "mu_bath", "f0", "sigma_f", "profile_type", "n_steps", "warmup_steps", "sample_interval", "n_bins", "mobile_forces", "force_mobility"]
            println(io, "$(key)=$(params[key])")
        end
        println(io, "n_forces=$(size(params["force_centers"], 2))")
        println(io, "thermal_offset=$(result["bins"]["thermal_offset"])")
        stability = result["stability"]
        for key in sort(collect(keys(stability)))
            println(io, "stability_$(key)=$(stability[key])")
        end
    end
    return path
end

function print_estimate(param::FluctuatingForceParam)
    total_steps = param.warmup_steps + param.n_steps
    println("Fluctuating-force SDE run estimate")
    println("  dims=$(param.dims)")
    println("  N=$(param.N)")
    println("  n_forces=$(size(param.force_centers, 2))")
    println("  warmup_steps=$(param.warmup_steps)")
    println("  production_steps=$(param.n_steps)")
    println("  total particle-force evaluations=$(total_steps * param.N * size(param.force_centers, 2))")
    println("  thermal_rms=$(sqrt(2.0 * param.D_bath * param.dt))")
    println("  thermal_rms/sigma_f=$(sqrt(2.0 * param.D_bath * param.dt) / param.sigma_f)")
    println("  mu_bath*f0*sqrt(dt)/sigma_f=$(abs(param.mu_bath * param.f0) * sqrt(param.dt) / param.sigma_f)")
end

function main()
    args = parse_commandline()
    defaults = get_default_params()
    params = YAML.load_file(String(args["config"]))
    param = build_param(params, defaults)
    performance_mode = performance_mode_from_params(args, params, defaults)

    if get(args, "estimate_only", false)
        print_estimate(param)
        return
    end

    save_dir = String(get(params, "save_dir", defaults["save_dir"]))
    rng = rng_from_param(param)
    state = initialize_state(param, rng)

    !performance_mode && print_estimate(param)
    start_time = time()
    stats = run!(state, param, rng)
    elapsed = time() - start_time

    result = result_dict(param, state, stats)
    metadata = Dict(
        "config_path" => abspath(String(args["config"])),
        "save_tag" => get(args, "save_tag", nothing),
        "wall_time_seconds" => elapsed,
        "saved_at" => Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
        "runner" => "run_fluctuating_force_sde.jl",
    )

    output_path = output_filename(save_dir, param; save_tag=get(args, "save_tag", nothing))
    jldsave(output_path; result=result, param=param, state=state, metadata=metadata)
    summary_path = replace(output_path, r"\.jld2$" => "_summary.txt")
    write_summary(summary_path, result, output_path)

    println("Saved fluctuating-force SDE result:")
    println("  $(output_path)")
    println("  $(summary_path)")
    println("  wall_time_seconds=$(round(elapsed, digits=3))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
