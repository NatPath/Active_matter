#!/usr/bin/env julia

using ArgParse
using Dates
using Printf
using Random
using Statistics

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

include(joinpath(REPO_ROOT, "src", "fluctuating_force_sde", "modules_fluctuating_force_sde.jl"))
using .FPFluctuatingForceSDE

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--distances"
            help = "Comma-separated fixed force separations d"
            arg_type = String
            default = "4,6,8,12,16,24,32"
        "--L"
            help = "Periodic domain length"
            arg_type = Float64
            default = 96.0
        "--N"
            help = "Number of bath particles"
            arg_type = Int
            default = 12_000
        "--D_bath"
            help = "Bath diffusivity"
            arg_type = Float64
            default = 1.0
        "--dt"
            help = "Euler-Maruyama time step"
            arg_type = Float64
            default = 0.002
        "--mu_bath"
            help = "Bath coupling to the fluctuating-force noise"
            arg_type = Float64
            default = 1.0
        "--f0"
            help = "Profile amplitude"
            arg_type = Float64
            default = 1.0
        "--sigma_f"
            help = "Profile width"
            arg_type = Float64
            default = 1.5
        "--profile_type"
            help = "Profile type: gaussian or compact_bump"
            arg_type = String
            default = "gaussian"
        "--warmup_steps"
            help = "Warmup steps before collecting density variance"
            arg_type = Int
            default = 20_000
        "--production_steps"
            help = "Production steps per replica"
            arg_type = Int
            default = 100_000
        "--chunk_steps"
            help = "Simulation steps between partial plot updates"
            arg_type = Int
            default = 2_000
        "--sample_interval"
            help = "Collect one density-count sample every this many production steps"
            arg_type = Int
            default = 20
        "--replicas"
            help = "Independent replicas per separation"
            arg_type = Int
            default = 2
        "--n_bins"
            help = "Number of 1D density bins"
            arg_type = Int
            default = 192
        "--plot_interval_seconds"
            help = "Minimum wall-clock seconds between plot refreshes"
            arg_type = Float64
            default = 8.0
        "--output_dir"
            help = "Directory for live plot and CSV"
            arg_type = String
            default = "analysis_outputs/fluctuating_force_sde/two_force_distance_sweep_live"
        "--save_tag"
            help = "Filename tag for outputs"
            arg_type = String
            default = "two_force_distance_sweep"
        "--seed"
            help = "Base random seed; zero uses random per run"
            arg_type = Int
            default = 12345
        "--display"
            help = "Also try to display the plot window while updating"
            action = :store_true
    end
    return parse_args(settings)
end

const CLI_ARGS = parse_commandline()

if !Bool(CLI_ARGS["display"]) && get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Plots

mutable struct SharedSweepState
    distances::Vector{Float64}
    metrics::Matrix{Float64}
    samples::Matrix{Int}
    done::Matrix{Bool}
    profiles::Vector{Vector{Float64}}
    lock::ReentrantLock
    plot_path::String
    csv_path::String
    start_time::DateTime
end

function parse_distances(raw::AbstractString)
    values = Float64[]
    for item in split(raw, ",")
        token = strip(item)
        isempty(token) && continue
        push!(values, parse(Float64, token))
    end
    isempty(values) && error("--distances produced no values.")
    any(<=(0.0), values) && error("All distances must be positive.")
    return sort(values)
end

function force_centers_for_distance(d::Real)
    return reshape([-0.5 * Float64(d), 0.5 * Float64(d)], 1, 2)
end

function force_bin_indices(param::FluctuatingForceParam)
    inv_dx = param.n_bins / param.L
    idxs = Int[]
    for k in axes(param.force_centers, 2)
        x = wrap_centered(param.force_centers[1, k], param.L)
        idx = clamp(floor(Int, (x + 0.5 * param.L) * inv_dx) + 1, 1, param.n_bins)
        push!(idxs, idx)
    end
    return unique(idxs)
end

function current_profile_and_metric(stats::FluctuatingForceStats, param::FluctuatingForceParam)
    _, variance = FPFluctuatingForceSDE.count_variance(stats)
    idxs = force_bin_indices(param)
    metric = mean(variance[idxs])
    return variance, metric
end

function base_param_for_distance(args, d::Real)
    return FluctuatingForceParam(
        dims=1,
        L=Float64(args["L"]),
        N=Int(args["N"]),
        D_bath=Float64(args["D_bath"]),
        dt=Float64(args["dt"]),
        mu_bath=Float64(args["mu_bath"]),
        f0=Float64(args["f0"]),
        sigma_f=Float64(args["sigma_f"]),
        profile_type=String(args["profile_type"]),
        force_centers=force_centers_for_distance(d),
        mobile_forces=false,
        force_mobility=0.0,
        force_diffusivity=0.0,
        warmup_steps=max(Int(args["warmup_steps"]), 0),
        n_steps=max(Int(args["production_steps"]), 0),
        sample_interval=max(Int(args["sample_interval"]), 1),
        n_bins=max(Int(args["n_bins"]), 1),
        save_force_history=false,
    )
end

function update_shared!(shared::SharedSweepState, d_idx::Int, replica_idx::Int, profile::Vector{Float64}, metric::Float64, sample_count::Int; done::Bool=false)
    profile_slot = (d_idx - 1) * size(shared.metrics, 2) + replica_idx
    lock(shared.lock)
    try
        shared.metrics[d_idx, replica_idx] = metric
        shared.samples[d_idx, replica_idx] = sample_count
        shared.done[d_idx, replica_idx] = done
        shared.profiles[profile_slot] .= profile
    finally
        unlock(shared.lock)
    end
    return shared
end

function run_one_replica!(shared::SharedSweepState, args, d_idx::Int, replica_idx::Int)
    d = shared.distances[d_idx]
    param = base_param_for_distance(args, d)
    seed = Int(args["seed"])
    rng_seed = seed > 0 ? seed + 10_000 * d_idx + replica_idx : rand(1:2^30)
    rng = MersenneTwister(rng_seed)
    state = initialize_state(param, rng)
    work = ForceSDEWork(param.dims, size(param.force_centers, 2))

    for _ in 1:param.warmup_steps
        step!(state, param, rng, work)
    end

    stats = FluctuatingForceStats(param)
    chunk_steps = max(Int(args["chunk_steps"]), 1)
    production_step = 0
    while production_step < param.n_steps
        local_steps = min(chunk_steps, param.n_steps - production_step)
        for _ in 1:local_steps
            production_step += 1
            obs = step!(state, param, rng, work)
            stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
            stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
            if production_step % param.sample_interval == 0
                FPFluctuatingForceSDE.accumulate_sample!(stats, state, param)
            end
        end
        profile, metric = current_profile_and_metric(stats, param)
        update_shared!(shared, d_idx, replica_idx, profile, metric, stats.sample_count; done=false)
        yield()
    end

    profile, metric = current_profile_and_metric(stats, param)
    update_shared!(shared, d_idx, replica_idx, profile, metric, stats.sample_count; done=true)
    return nothing
end

function finite_mean(values)
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    isempty(vals) && return NaN
    return mean(vals)
end

function finite_stderr(values)
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    length(vals) <= 1 && return NaN
    return std(vals) / sqrt(length(vals))
end

function snapshot(shared::SharedSweepState)
    lock(shared.lock)
    try
        return (
            distances=copy(shared.distances),
            metrics=copy(shared.metrics),
            samples=copy(shared.samples),
            done=copy(shared.done),
            profiles=[copy(p) for p in shared.profiles],
        )
    finally
        unlock(shared.lock)
    end
end

function metric_summary(metrics::Matrix{Float64})
    n_d = size(metrics, 1)
    means = fill(NaN, n_d)
    stderrs = fill(NaN, n_d)
    for i in 1:n_d
        means[i] = finite_mean(metrics[i, :])
        stderrs[i] = finite_stderr(metrics[i, :])
    end
    return means, stderrs
end

function offset_subtracted(distances::Vector{Float64}, metric_mean::Vector{Float64})
    finite_idxs = findall(isfinite, metric_mean)
    isempty(finite_idxs) && return fill(NaN, length(metric_mean)), NaN
    far_idx = finite_idxs[argmax(distances[finite_idxs])]
    offset = metric_mean[far_idx]
    shifted = metric_mean .- offset
    return shifted, offset
end

function representative_distance_indices(n_d::Int)
    n = min(n_d, 5)
    return unique(round.(Int, range(1, n_d; length=n)))
end

function mean_profile_for_distance(profiles::Vector{Vector{Float64}}, d_idx::Int, n_replicas::Int)
    selected = Vector{Vector{Float64}}()
    for r in 1:n_replicas
        slot = (d_idx - 1) * n_replicas + r
        if any(isfinite, profiles[slot])
            push!(selected, profiles[slot])
        end
    end
    isempty(selected) && return nothing
    out = zeros(Float64, length(selected[1]))
    for p in selected
        out .+= p
    end
    out ./= length(selected)
    return out
end

function write_csv(path::AbstractString, distances, metric_mean, metric_stderr, shifted, offset, samples, done)
    open(path, "w") do io
        println(io, "distance,variance_on_forces,stderr,variance_minus_farthest_offset,farthest_offset,min_samples,max_samples,completed_replicas,total_replicas")
        for i in eachindex(distances)
            row_samples = samples[i, :]
            completed = count(done[i, :])
            @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e,%d,%d,%d,%d\n",
                distances[i],
                metric_mean[i],
                metric_stderr[i],
                shifted[i],
                offset,
                minimum(row_samples),
                maximum(row_samples),
                completed,
                size(samples, 2))
        end
    end
    return path
end

function make_plot!(shared::SharedSweepState, args; final::Bool=false)
    snap = snapshot(shared)
    distances = snap.distances
    metrics = snap.metrics
    samples = snap.samples
    done = snap.done
    profiles = snap.profiles
    n_replicas = size(metrics, 2)
    metric_mean, metric_stderr = metric_summary(metrics)
    shifted, offset = offset_subtracted(distances, metric_mean)

    write_csv(shared.csv_path, distances, metric_mean, metric_stderr, shifted, offset, samples, done)

    p1 = plot(
        distances,
        metric_mean;
        yerr=metric_stderr,
        marker=:circle,
        lw=2,
        xlabel="force separation d",
        ylabel="variance on force bins",
        title="On-force density variance",
        label="mean over replicas",
        legend=:topright,
    )

    positive = findall(i -> isfinite(shifted[i]) && shifted[i] > 0.0 && distances[i] > 0.0, eachindex(distances))
    p2 = if length(positive) >= 2
        p = plot(
            xlabel="force separation d",
            ylabel="variance - farthest offset",
            title="Offset-subtracted scaling",
            xscale=:log10,
            yscale=:log10,
            legend=:topright,
        )
        plot!(p, distances[positive], shifted[positive]; marker=:circle, lw=2, label="measured")
        anchor = positive[1]
        ref = shifted[anchor] .* (distances[positive] ./ distances[anchor]).^(-2)
        plot!(p, distances[positive], ref; lw=2, ls=:dash, color=:black, label="slope -2")
        p
    else
        plot(
            xlabel="force separation d",
            ylabel="variance - farthest offset",
            title="Offset-subtracted scaling: waiting for positive points",
            legend=false,
        )
    end

    min_samples = [minimum(samples[i, :]) for i in eachindex(distances)]
    max_samples = [maximum(samples[i, :]) for i in eachindex(distances)]
    completed = [count(done[i, :]) for i in eachindex(distances)]
    p3 = plot(
        distances,
        min_samples;
        marker=:circle,
        lw=2,
        xlabel="force separation d",
        ylabel="density samples",
        title="Progress",
        label="min samples",
        legend=:topleft,
    )
    plot!(p3, distances, max_samples; marker=:diamond, lw=2, label="max samples")
    plot!(p3, distances, completed .* maximum(max_samples) ./ max(n_replicas, 1); marker=:rect, lw=2, label="completed replicas scaled")

    x_centers = collect(range(-0.5 * Float64(args["L"]) + 0.5 * Float64(args["L"]) / Int(args["n_bins"]), 0.5 * Float64(args["L"]) - 0.5 * Float64(args["L"]) / Int(args["n_bins"]); length=Int(args["n_bins"])))
    p4 = plot(
        xlabel="x",
        ylabel="density-count variance",
        title="Representative variance profiles",
        legend=:topright,
    )
    for idx in representative_distance_indices(length(distances))
        profile = mean_profile_for_distance(profiles, idx, n_replicas)
        isnothing(profile) && continue
        plot!(p4, x_centers, profile; lw=2, label=@sprintf("d=%.3g", distances[idx]))
    end

    elapsed_min = Dates.value(now() - shared.start_time) / 60_000
    status = final ? "final" : "live"
    title = @sprintf(
        "Two fixed fluctuating forces: %s, N=%d, replicas=%d, threads=%d, elapsed=%.1f min",
        status,
        Int(args["N"]),
        n_replicas,
        Threads.nthreads(),
        elapsed_min,
    )
    panel = plot(p1, p2, p3, p4; layout=(2, 2), size=(1500, 1050), plot_title=title)
    savefig(panel, shared.plot_path)
    if Bool(args["display"])
        display(panel)
    end
    return shared.plot_path
end

function main()
    args = CLI_ARGS
    distances = parse_distances(String(args["distances"]))
    replicas = max(Int(args["replicas"]), 1)
    n_bins = max(Int(args["n_bins"]), 1)
    output_dir = abspath(String(args["output_dir"]))
    mkpath(output_dir)
    tag = replace(String(args["save_tag"]), r"[^A-Za-z0-9._-]+" => "-")
    shared = SharedSweepState(
        distances,
        fill(NaN, length(distances), replicas),
        zeros(Int, length(distances), replicas),
        falses(length(distances), replicas),
        [fill(NaN, n_bins) for _ in 1:(length(distances) * replicas)],
        ReentrantLock(),
        joinpath(output_dir, "$(tag)_live.png"),
        joinpath(output_dir, "$(tag)_metrics.csv"),
        now(),
    )

    println("Starting local two-force distance sweep")
    println("  distances=$(join(distances, ","))")
    println("  replicas_per_distance=$(replicas)")
    println("  JULIA_NUM_THREADS=$(Threads.nthreads())")
    println("  plot=$(shared.plot_path)")
    println("  csv=$(shared.csv_path)")
    println("  note=This is a local diagnostic. Use cluster submit wrappers for heavy production.")

    tasks = Task[]
    for d_idx in eachindex(distances), replica_idx in 1:replicas
        push!(tasks, Threads.@spawn run_one_replica!(shared, args, d_idx, replica_idx))
    end

    plot_interval = max(Float64(args["plot_interval_seconds"]), 1.0)
    last_plot_time = time() - plot_interval
    while !all(istaskdone, tasks)
        if time() - last_plot_time >= plot_interval
            make_plot!(shared, args)
            last_plot_time = time()
            snap = snapshot(shared)
            completed = count(snap.done)
            total = length(snap.done)
            println(@sprintf("[%s] plot updated: %s (%d/%d replicas complete)",
                Dates.format(now(), dateformat"HH:MM:SS"),
                shared.plot_path,
                completed,
                total))
        end
        sleep(0.5)
    end

    for task in tasks
        fetch(task)
    end
    make_plot!(shared, args; final=true)
    println("Finished two-force distance sweep")
    println("  plot=$(shared.plot_path)")
    println("  csv=$(shared.csv_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
