#!/usr/bin/env julia

using ArgParse
using Dates
using Printf
using Random

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots
using JLD2

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

cd(REPO_ROOT)
include(joinpath(REPO_ROOT, "src", "fluctuating_force_sde", "modules_fluctuating_force_sde.jl"))
using .FPFluctuatingForceSDE

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--L"
            help = "System length"
            arg_type = Float64
            default = 32.0
        "--rho0"
            help = "Bath density used when --N is omitted"
            arg_type = Float64
            default = 100.0
        "--N"
            help = "Override particle count"
            arg_type = Int
        "--sigma_f"
            help = "Gaussian force width"
            arg_type = Float64
            default = 0.5
        "--f0"
            help = "Gaussian force amplitude"
            arg_type = Float64
            default = 1.5
        "--mu_bath"
            help = "Bath mobility coupling"
            arg_type = Float64
            default = 1.0
        "--D_bath"
            help = "Bath diffusivity"
            arg_type = Float64
            default = 1.0
        "--dt"
            help = "Time step"
            arg_type = Float64
            default = 0.001
        "--warmup_steps"
            help = "Warmup steps before accumulating variance"
            arg_type = Int
            default = 20_000
        "--n_steps"
            help = "Production steps"
            arg_type = Int
            default = 200_000
        "--sample_interval"
            help = "Production-step interval between density samples"
            arg_type = Int
            default = 10
        "--n_bins"
            help = "Number of 1D density bins"
            arg_type = Int
            default = 128
        "--edge_bins_for_offset"
            help = "Number of bins at each edge used for background offset"
            arg_type = Int
            default = 8
        "--plot_interval_samples"
            help = "Update live plot after this many new samples"
            arg_type = Int
            default = 1_000
        "--min_plot_interval_seconds"
            help = "Minimum wall seconds between plot writes"
            arg_type = Float64
            default = 15.0
        "--progress_interval_steps"
            help = "Print production progress every this many steps"
            arg_type = Int
            default = 10_000
        "--seed"
            help = "Random seed; use 0 for a generated seed"
            arg_type = Int
            default = 1
        "--out_dir"
            help = "Output directory for live PNG/CSV/JLD2"
            arg_type = String
        "--no_jld2"
            help = "Skip writing live_latest.jld2 on each plot update"
            action = :store_true
    end
    return parse_args(settings)
end

function build_param(args)
    L = Float64(args["L"])
    rho0 = Float64(args["rho0"])
    N = isnothing(args["N"]) ? Int(round(L * rho0)) : Int(args["N"])
    N > 0 || error("N must be positive.")

    return FluctuatingForceParam(
        dims=1,
        L=L,
        N=N,
        D_bath=Float64(args["D_bath"]),
        dt=Float64(args["dt"]),
        mu_bath=Float64(args["mu_bath"]),
        f0=Float64(args["f0"]),
        sigma_f=Float64(args["sigma_f"]),
        profile_type="gaussian",
        force_centers=zeros(Float64, 1, 1),
        mobile_forces=false,
        force_mobility=0.0,
        force_diffusivity=0.0,
        n_steps=max(Int(args["n_steps"]), 0),
        warmup_steps=max(Int(args["warmup_steps"]), 0),
        sample_interval=max(Int(args["sample_interval"]), 1),
        n_bins=max(Int(args["n_bins"]), 1),
        edge_bins_for_offset=max(Int(args["edge_bins_for_offset"]), 1),
        history_interval=typemax(Int),
        max_history_records=0,
        save_force_history=false,
        seed=Int(args["seed"]),
        description="live_fixed_origin_density_variance_sigma$(args["sigma_f"])",
    )
end

function default_out_dir(param::FluctuatingForceParam)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    sigma_token = replace(@sprintf("%.6g", param.sigma_f), "." => "p")
    return joinpath(
        REPO_ROOT,
        "analysis_outputs",
        "fluctuating_force_sde",
        "live_fixed_origin_sigma$(sigma_token)_$(stamp)",
    )
end

function rng_from_seed(seed::Integer)
    seed > 0 && return MersenneTwister(seed)
    return MersenneTwister(rand(1:2^30))
end

function live_arrays(stats::FluctuatingForceStats, param::FluctuatingForceParam)
    mean_counts, variance = FPFluctuatingForceSDE.count_variance(stats)
    offset = FPFluctuatingForceSDE.thermal_offset(variance, param)
    variance_active = FPFluctuatingForceSDE.active_variance(variance, offset, param)
    excess = variance .- offset
    return mean_counts, variance, offset, variance_active, excess
end

function write_bins_csv(path::AbstractString, stats::FluctuatingForceStats, param::FluctuatingForceParam)
    mean_counts, variance, offset, variance_active, excess = live_arrays(stats, param)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "x,mean_count,variance_total,edge_offset,variance_minus_offset,variance_active")
        for i in eachindex(stats.bin_centers)
            @printf(
                io,
                "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                stats.bin_centers[i],
                mean_counts[i],
                variance[i],
                offset,
                excess[i],
                variance_active[i],
            )
        end
    end
    mv(tmp, path; force=true)
    return path
end

function live_aggregate_dict(stats::FluctuatingForceStats, param::FluctuatingForceParam, production_step::Integer, elapsed::Real)
    mean_counts, variance, offset, variance_active, excess = live_arrays(stats, param)
    return Dict(
        "result_type" => "fluctuating_force_sde_live_density_variance",
        "production_step" => Int(production_step),
        "elapsed_seconds" => Float64(elapsed),
        "sample_count" => stats.sample_count,
        "parameters" => FPFluctuatingForceSDE.parameters_dict(param),
        "bins" => Dict(
            "edges" => stats.bin_edges,
            "centers" => stats.bin_centers,
            "mean_counts" => mean_counts,
            "variance_total" => variance,
            "thermal_offset" => offset,
            "variance_minus_offset" => excess,
            "variance_active" => variance_active,
        ),
        "stability" => Dict(
            "max_abs_bath_active_step" => stats.max_abs_bath_active_step,
            "max_abs_bath_thermal_step" => stats.max_abs_bath_thermal_step,
            "max_abs_force_center_step" => stats.max_abs_force_center_step,
        ),
    )
end

function write_summary(path::AbstractString, stats::FluctuatingForceStats, param::FluctuatingForceParam, production_step::Integer, elapsed::Real)
    _, _, offset, _, excess = live_arrays(stats, param)
    positive = filter(v -> isfinite(v) && v > 0, excess)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "result_type=fluctuating_force_sde_live_density_variance")
        println(io, "production_step=$(production_step)")
        println(io, "elapsed_seconds=$(elapsed)")
        println(io, "sample_count=$(stats.sample_count)")
        println(io, "L=$(param.L)")
        println(io, "N=$(param.N)")
        println(io, "rho0=$(param.N / param.L)")
        println(io, "sigma_f=$(param.sigma_f)")
        println(io, "f0=$(param.f0)")
        println(io, "dt=$(param.dt)")
        println(io, "n_bins=$(param.n_bins)")
        println(io, "bin_width=$(param.L / param.n_bins)")
        println(io, "edge_offset=$(offset)")
        println(io, "max_positive_excess=$(isempty(positive) ? NaN : maximum(positive))")
        println(io, "max_abs_bath_active_step=$(stats.max_abs_bath_active_step)")
        println(io, "max_abs_bath_thermal_step=$(stats.max_abs_bath_thermal_step)")
    end
    mv(tmp, path; force=true)
    return path
end

function write_html(path::AbstractString, png_name::AbstractString, stats::FluctuatingForceStats, production_step::Integer)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "<!doctype html>")
        println(io, "<meta charset=\"utf-8\">")
        println(io, "<meta http-equiv=\"refresh\" content=\"5\">")
        println(io, "<title>Fluctuating-force SDE live diagnostic</title>")
        println(io, "<body style=\"font-family: sans-serif; margin: 20px;\">")
        println(io, "<h2>Fluctuating-force SDE live diagnostic</h2>")
        println(io, "<p>production_step=$(production_step), sample_count=$(stats.sample_count)</p>")
        println(io, "<img src=\"$(png_name)?sample=$(stats.sample_count)\" style=\"max-width: 100%; height: auto;\">")
        println(io, "</body>")
    end
    mv(tmp, path; force=true)
    return path
end

function save_live_plot(path::AbstractString, stats::FluctuatingForceStats, param::FluctuatingForceParam, production_step::Integer, elapsed::Real)
    mean_counts, variance, offset, variance_active, excess = live_arrays(stats, param)
    x = Float64.(stats.bin_centers)
    uniform_count = param.N / param.n_bins
    profile2 = exp.(-x .^ 2 ./ (param.sigma_f * param.sigma_f))
    positive_excess = filter(v -> isfinite(v) && v > 0, excess)
    guide_scale = isempty(positive_excess) ? 1.0 : maximum(positive_excess)
    profile2_guide = maximum(profile2) > 0 ? guide_scale .* profile2 ./ maximum(profile2) : fill(NaN, length(x))
    title_suffix = @sprintf("samples=%d, step=%d, %.1fs", stats.sample_count, production_step, elapsed)

    p_mean = plot(
        x,
        mean_counts;
        lw=2,
        xlabel="x",
        ylabel="mean bin count",
        title="Mean density, $(title_suffix)",
        label="measured",
        framestyle=:box,
        grid=:y,
    )
    hline!(p_mean, [uniform_count]; lw=2, ls=:dash, color=:black, label=@sprintf("uniform %.3g", uniform_count))

    p_var = plot(
        x,
        variance;
        lw=2,
        xlabel="x",
        ylabel="count variance",
        title="Density-count variance",
        label="variance_total",
        framestyle=:box,
        grid=:y,
    )
    hline!(p_var, [offset]; lw=2, ls=:dash, color=:black, label=@sprintf("edge offset %.3g", offset))

    p_excess = plot(
        x,
        excess;
        lw=2,
        xlabel="x",
        ylabel="variance - edge offset",
        title="Offset-subtracted variance",
        label="measured excess",
        framestyle=:box,
        grid=:y,
    )
    plot!(p_excess, x, profile2_guide; lw=2, ls=:dash, label="f(x)^2 shape guide, scaled")
    hline!(p_excess, [0.0]; lw=1, ls=:dot, color=:black, label=false)

    tail_min = max(2.0 * param.sigma_f, param.L / param.n_bins)
    tail_mask = x .> tail_min
    tail_x = x[tail_mask]
    tail_active = variance_active[tail_mask]
    tail_reference = fill(NaN, length(tail_x))
    if !isempty(tail_x)
        anchor_idx = findfirst(v -> isfinite(v) && v > 1.0e-12, tail_active)
        if !isnothing(anchor_idx)
            anchor = tail_active[anchor_idx] * tail_x[anchor_idx]^2
            tail_reference .= anchor ./ (tail_x .^ 2)
        end
    end

    p_tail = plot(
        tail_x,
        tail_active;
        lw=2,
        xlabel="x",
        ylabel="active variance",
        xscale=:log10,
        yscale=:log10,
        title=@sprintf("Right-tail active variance, x > %.3g", tail_min),
        label="measured",
        framestyle=:box,
        grid=:both,
    )
    plot!(p_tail, tail_x, tail_reference; lw=2, ls=:dash, color=:black, label="anchored 1/x^2")

    tmp = replace(path, r"\.png$" => ".tmp.png")
    savefig(plot(p_mean, p_var, p_excess, p_tail; layout=(2, 2), size=(1400, 1000)), tmp)
    mv(tmp, path; force=true)
    return path
end

function write_live_outputs(out_dir::AbstractString, stats::FluctuatingForceStats, param::FluctuatingForceParam, production_step::Integer, elapsed::Real; write_jld2::Bool)
    png_path = joinpath(out_dir, "live_density_variance.png")
    csv_path = joinpath(out_dir, "live_bins.csv")
    summary_path = joinpath(out_dir, "live_summary.txt")
    html_path = joinpath(out_dir, "index.html")

    save_live_plot(png_path, stats, param, production_step, elapsed)
    write_bins_csv(csv_path, stats, param)
    write_summary(summary_path, stats, param, production_step, elapsed)
    write_html(html_path, basename(png_path), stats, production_step)

    if write_jld2
        aggregate = live_aggregate_dict(stats, param, production_step, elapsed)
        tmp = joinpath(out_dir, "live_latest.tmp.jld2")
        final = joinpath(out_dir, "live_latest.jld2")
        jldsave(tmp; aggregate=aggregate)
        mv(tmp, final; force=true)
    end

    return png_path
end

function main()
    args = parse_commandline()
    param = build_param(args)
    FPFluctuatingForceSDE.validate_param(param)

    out_dir = isnothing(args["out_dir"]) ? default_out_dir(param) : abspath(String(args["out_dir"]))
    mkpath(out_dir)

    rng = rng_from_seed(param.seed)
    state = initialize_state(param, rng)
    work = ForceSDEWork(param.dims, size(param.force_centers, 2))

    println("Local fluctuating-force SDE live diagnostic")
    println("  out_dir=$(out_dir)")
    println("  L=$(param.L), rho0=$(param.N / param.L), N=$(param.N)")
    println("  sigma_f=$(param.sigma_f), f0=$(param.f0), dt=$(param.dt), n_bins=$(param.n_bins), bin_width=$(param.L / param.n_bins)")
    println("  warmup_steps=$(param.warmup_steps), production_steps=$(param.n_steps), sample_interval=$(param.sample_interval)")
    println("  plot_interval_samples=$(args["plot_interval_samples"]), min_plot_interval_seconds=$(args["min_plot_interval_seconds"])")
    println("  live plot: $(joinpath(out_dir, "live_density_variance.png"))")
    println("  auto-refresh html: $(joinpath(out_dir, "index.html"))")
    flush(stdout)

    start_time = time()
    if param.warmup_steps > 0
        for step_idx in 1:param.warmup_steps
            step!(state, param, rng, work)
            if step_idx == param.warmup_steps || (step_idx % max(Int(args["progress_interval_steps"]), 1) == 0)
                @printf("warmup step %d / %d, elapsed %.1fs\n", step_idx, param.warmup_steps, time() - start_time)
                flush(stdout)
            end
        end
    end

    stats = FluctuatingForceStats(param)
    last_plot_sample = 0
    last_plot_time = -Inf
    plot_interval_samples = max(Int(args["plot_interval_samples"]), 1)
    min_plot_interval_seconds = max(Float64(args["min_plot_interval_seconds"]), 0.0)
    progress_interval_steps = max(Int(args["progress_interval_steps"]), 1)
    write_jld2 = !Bool(args["no_jld2"])

    for production_step in 1:param.n_steps
        obs = step!(state, param, rng, work)
        stats.max_abs_bath_active_step = max(stats.max_abs_bath_active_step, obs.max_abs_bath_active_step)
        stats.max_abs_bath_thermal_step = max(stats.max_abs_bath_thermal_step, obs.max_abs_bath_thermal_step)
        stats.max_abs_force_center_step = max(stats.max_abs_force_center_step, obs.max_abs_force_center_step)

        if production_step % param.sample_interval == 0
            FPFluctuatingForceSDE.accumulate_sample!(stats, state, param)
        end

        now_time = time()
        should_plot = stats.sample_count > 0 &&
            stats.sample_count - last_plot_sample >= plot_interval_samples &&
            now_time - last_plot_time >= min_plot_interval_seconds
        is_final = production_step == param.n_steps

        if should_plot || is_final
            png_path = write_live_outputs(out_dir, stats, param, production_step, now_time - start_time; write_jld2=write_jld2)
            last_plot_sample = stats.sample_count
            last_plot_time = now_time
            @printf("updated %s at step %d, samples %d, elapsed %.1fs\n", png_path, production_step, stats.sample_count, now_time - start_time)
            flush(stdout)
        elseif production_step % progress_interval_steps == 0
            @printf("production step %d / %d, samples %d, elapsed %.1fs\n", production_step, param.n_steps, stats.sample_count, now_time - start_time)
            flush(stdout)
        end
    end

    elapsed = time() - start_time
    result = result_dict(param, state, stats)
    final_path = joinpath(out_dir, "final_result.jld2")
    jldsave(final_path; result=result, param=param, state=state)
    println("Done.")
    println("  final_result=$(final_path)")
    println("  elapsed_seconds=$(round(elapsed, digits=3))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
