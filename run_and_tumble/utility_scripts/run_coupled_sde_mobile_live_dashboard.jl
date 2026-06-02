#!/usr/bin/env julia

using ArgParse
using Dates
using JLD2
using Printf
using Random

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

cd(REPO_ROOT)
include(joinpath(REPO_ROOT, "src", "active_objects_sde", "modules_coupled_sde_active_objects.jl"))
using .FPCoupledSDEActiveObjects

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--L"
            help = "Periodic system size"
            arg_type = Float64
            default = 32.0
        "--rho0"
            help = "Mean bath density; N defaults to round(rho0 * L)"
            arg_type = Float64
            default = 100.0
        "--N"
            help = "Optional explicit bath-particle count"
            arg_type = Int
        "--D0"
            help = "Bath diffusivity"
            arg_type = Float64
            default = 1.0
        "--dt"
            help = "Euler-Maruyama time step"
            arg_type = Float64
            default = 1.0e-3
        "--mu_bath"
            help = "Bath mobility multiplying the localized fluctuating force"
            arg_type = Float64
            default = 1.0
        "--mu_obj"
            help = "Mobile-object mobility"
            arg_type = Float64
            default = 4.0e-3
        "--f0"
            help = "Localized force-profile amplitude"
            arg_type = Float64
            default = 1.0
        "--sigma_f"
            help = "Localized force-profile width"
            arg_type = Float64
            default = 0.25
        "--separation"
            help = "Initial object separation"
            arg_type = Float64
            default = 0.5
        "--hard_min_separation"
            help = "Absolute hard minimum object distance; <=0 disables"
            arg_type = Float64
        "--hard_min_separation_sigma"
            help = "Hard minimum object distance in units of sigma_f; ignored when --hard_min_separation is set"
            arg_type = Float64
            default = 0.0
        "--n_steps"
            help = "Production steps to run"
            arg_type = Int
            default = 120_000
        "--warmup_steps"
            help = "Warmup steps before statistics start"
            arg_type = Int
            default = 1_000
        "--sample_interval"
            help = "Production-step interval between histogram samples"
            arg_type = Int
            default = 1
        "--plot_interval_steps"
            help = "Production steps between dashboard refresh attempts"
            arg_type = Int
            default = 250
        "--min_plot_interval_seconds"
            help = "Minimum wall-clock seconds between dashboard rewrites"
            arg_type = Float64
            default = 0.75
        "--progress_interval_steps"
            help = "Console progress interval"
            arg_type = Int
            default = 5_000
        "--n_bins"
            help = "Bins for density and probability-density diagnostics"
            arg_type = Int
            default = 64
        "--trace_points"
            help = "Maximum recent trajectory points shown in the live dashboard"
            arg_type = Int
            default = 900
        "--offset_pinf_tail_count"
            help = "Tail count used for live Pinf in Pinf - P(d) offset plots"
            arg_type = Int
            default = 12
        "--offset_reference_slope"
            help = "Reference slope for live offset-power-law plots"
            arg_type = Float64
            default = -2.0
        "--offset_fixed_slope"
            help = "Optional fixed slope for a best-offset live scan; default uses --offset_reference_slope"
            arg_type = Float64
        "--offset_scan_points"
            help = "Number of Pinf candidates in the live fixed-slope scan"
            arg_type = Int
            default = 1000
        "--offset_fixed_min_points"
            help = "Minimum number of positive offset points required in the fixed-slope scan"
            arg_type = Int
            default = 10
        "--seed"
            help = "Random seed; <=0 chooses a random seed"
            arg_type = Int
            default = 20260526
        "--out_dir"
            help = "Output directory for live_dashboard.png and index.html"
            arg_type = String
        "--resume_checkpoint"
            help = "Resume from a live_checkpoint.jld2 written by this script"
            arg_type = String
        "--checkpoint_path"
            help = "Checkpoint output path; defaults to <out_dir>/live_checkpoint.jld2"
            arg_type = String
        "--additional_steps"
            help = "When resuming, run this many more production steps beyond the checkpoint"
            arg_type = Int
    end
    return parse_args(settings)
end

function default_out_dir(param::CoupledSDEParam)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    rho_token = replace(@sprintf("%.4g", param.rho0), "." => "p")
    return joinpath(
        REPO_ROOT,
        "analysis_outputs",
        "coupled_sde_active_objects",
        "mobile_live",
        "local_L$(round(Int, param.L))_rho$(rho_token)_$(stamp)",
    )
end

function rng_from_seed(seed::Integer)
    if seed > 0
        return MersenneTwister(seed)
    end
    return MersenneTwister(rand(1:2^30))
end

function build_param(args)
    L = Float64(args["L"])
    rho0 = Float64(args["rho0"])
    N = isnothing(args["N"]) ? Int(round(rho0 * L)) : Int(args["N"])
    return CoupledSDEParam(
        mode=FPCoupledSDEActiveObjects.MOBILE_OBJECTS_MODE,
        L=L,
        rho0=rho0,
        N=N,
        D0=Float64(args["D0"]),
        dt=Float64(args["dt"]),
        mu_bath=Float64(args["mu_bath"]),
        mu_obj=Float64(args["mu_obj"]),
        f0=Float64(args["f0"]),
        sigma_f=Float64(args["sigma_f"]),
        profile_type="gaussian",
        separation=Float64(args["separation"]),
        n_steps=max(Int(args["n_steps"]), 0),
        warmup_steps=max(Int(args["warmup_steps"]), 0),
        sample_interval=max(Int(args["sample_interval"]), 1),
        history_interval=max(Int(args["plot_interval_steps"]), 1),
        n_bins=max(Int(args["n_bins"]), 1),
        max_history_records=0,
        save_raw_history=false,
        seed=Int(args["seed"]),
        description="local_live_mobile_objects",
    )
end

function validate_param(param::CoupledSDEParam)
    param.L > 0 || error("L must be positive.")
    param.N >= 0 || error("N must be nonnegative.")
    param.rho0 > 0 || error("rho0 must be positive.")
    param.D0 >= 0 || error("D0 must be nonnegative.")
    param.dt > 0 || error("dt must be positive.")
    param.sigma_f > 0 || error("sigma_f must be positive.")
    0.0 <= param.separation <= 0.5 * param.L || error("separation must be in [0, L/2].")
    return true
end

function resolve_hard_min_separation(args, param::CoupledSDEParam)
    hard_min = isnothing(args["hard_min_separation"]) ?
        Float64(args["hard_min_separation_sigma"]) * param.sigma_f :
        Float64(args["hard_min_separation"])
    hard_min >= 0.0 || error("hard minimum separation must be nonnegative.")
    hard_min <= 0.5 * param.L || error("hard minimum separation must be <= L/2.")
    return hard_min
end

function enforce_hard_minimum_separation!(state::CoupledSDEState, param::CoupledSDEParam, hard_min::Real)
    min_allowed = Float64(hard_min)
    min_allowed <= 0.0 && return 0.0, 0.0, false

    signed_sep = minimal_image(state.XA - state.XB, param.L)
    current = abs(signed_sep)
    current >= min_allowed && return 0.0, 0.0, false

    direction = current > eps(Float64) ? sign(signed_sep) : 1.0
    target = min_allowed + max(16.0 * eps(Float64) * max(param.L, min_allowed), 1.0e-12 * max(min_allowed, 1.0))
    correction = target - current
    dXA_hard = 0.5 * direction * correction
    dXB_hard = -0.5 * direction * correction

    state.XA_unwrapped += dXA_hard
    state.XB_unwrapped += dXB_hard
    state.XA = wrap_position(state.XA_unwrapped, param.L)
    state.XB = wrap_position(state.XB_unwrapped, param.L)
    state.last_dXA += dXA_hard
    state.last_dXB += dXB_hard
    state.last_drel = state.last_dXA - state.last_dXB

    return dXA_hard, dXB_hard, true
end

function step_mobile_objects_live!(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    rng::AbstractRNG,
    work::SDEWork,
    hard_min_separation::Real,
)
    obs = step_mobile_objects!(state, param, rng, work)
    dXA_hard, dXB_hard, hit = enforce_hard_minimum_separation!(state, param, hard_min_separation)
    hit || return obs

    dXA = obs.dXA + dXA_hard
    dXB = obs.dXB + dXB_hard
    return StepObservables(
        obs.step,
        obs.time,
        obs.XA,
        obs.XB,
        obs.separation_oriented,
        obs.separation_min,
        obs.SA,
        obs.SB,
        obs.Ssum,
        obs.dWA,
        obs.dWB,
        dXA,
        dXB,
        dXA - dXB,
        obs.max_abs_bath_active_step,
        obs.max_abs_bath_thermal_step,
        max(obs.max_abs_object_step, abs(dXA), abs(dXB)),
    )
end

function compatible_resume!(saved::CoupledSDEParam, current::CoupledSDEParam, path::AbstractString)
    saved.mode == current.mode || error("Checkpoint mode does not match current mode in $path.")
    saved.N == current.N || error("Checkpoint N=$(saved.N) does not match current N=$(current.N) in $path.")
    saved.n_bins == current.n_bins || error("Checkpoint n_bins=$(saved.n_bins) does not match current n_bins=$(current.n_bins) in $path.")
    saved.profile_type == current.profile_type || error("Checkpoint profile_type=$(saved.profile_type) does not match current profile_type=$(current.profile_type) in $path.")
    for key in (:L, :rho0, :D0, :dt, :mu_bath, :mu_obj, :f0, :sigma_f)
        a = Float64(getfield(saved, key))
        b = Float64(getfield(current, key))
        isapprox(a, b; rtol=1.0e-10, atol=1.0e-12) ||
            error("Checkpoint $key=$a does not match current $key=$b in $path.")
    end
    return true
end

mutable struct DensityFieldStats
    edges::Vector{Float64}
    centers::Vector{Float64}
    current_density::Vector{Float64}
    sum_density::Vector{Float64}
    sample_count::Int
end

function DensityFieldStats(param::CoupledSDEParam)
    edges = collect(range(0.0, param.L; length=param.n_bins + 1))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:param.n_bins]
    return DensityFieldStats(edges, centers, zeros(Float64, param.n_bins), zeros(Float64, param.n_bins), 0)
end

function density_snapshot(x::AbstractVector{Float64}, L::Real, n_bins::Integer)
    edges = collect(range(0.0, Float64(L); length=Int(n_bins) + 1))
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:Int(n_bins)]
    counts = zeros(Float64, Int(n_bins))
    width = edges[2] - edges[1]
    @inbounds for xi in x
        pos = wrap_position(xi, L)
        idx = floor(Int, pos / width) + 1
        idx = clamp(idx, 1, Int(n_bins))
        counts[idx] += 1.0
    end
    return edges, centers, counts ./ width
end

function accumulate_density_field!(density_stats::DensityFieldStats, state::CoupledSDEState, param::CoupledSDEParam)
    fill!(density_stats.current_density, 0.0)
    width = density_stats.edges[2] - density_stats.edges[1]
    inv_width = 1.0 / width
    @inbounds for xi in state.x
        pos = wrap_position(xi, param.L)
        idx = floor(Int, pos / width) + 1
        idx = clamp(idx, 1, length(density_stats.current_density))
        density_stats.current_density[idx] += inv_width
    end
    density_stats.sum_density .+= density_stats.current_density
    density_stats.sample_count += 1
    return density_stats
end

function average_density_field(density_stats::DensityFieldStats)
    density_stats.sample_count <= 0 && return fill(NaN, length(density_stats.sum_density))
    return density_stats.sum_density ./ density_stats.sample_count
end

function probability_density(counts::AbstractVector{Float64}, edges::AbstractVector{Float64})
    density = zeros(Float64, length(counts))
    total = sum(counts)
    total <= 0 && return fill(NaN, length(counts))
    widths = diff(edges)
    @inbounds for i in eachindex(counts)
        density[i] = counts[i] / total / widths[i]
    end
    return density
end

function distance_probability_density(stats::FPCoupledSDEActiveObjects.MobileStats)
    return probability_density(stats.histogram_counts, stats.bin_edges)
end

function vector_mean(values::AbstractVector{Float64})
    isempty(values) && return NaN
    return sum(values) / length(values)
end

function finite_max(values::AbstractVector{Float64}, fallback::Float64)
    best = -Inf
    for value in values
        if isfinite(value)
            best = max(best, value)
        end
    end
    return isfinite(best) ? best : fallback
end

function push_limited!(values::Vector{Float64}, value::Real, limit::Integer)
    push!(values, Float64(value))
    if length(values) > limit
        deleteat!(values, 1:(length(values) - limit))
    end
    return values
end

function finite_positive_distance_pdf(stats::FPCoupledSDEActiveObjects.MobileStats)
    d = Float64[]
    p = Float64[]
    P = distance_probability_density(stats)
    for i in eachindex(P)
        x = Float64(stats.bin_centers[i])
        y = Float64(P[i])
        c = Float64(stats.histogram_counts[i])
        if isfinite(x) && isfinite(y) && x > 0 && y > 0 && c > 0
            push!(d, x)
            push!(p, y)
        end
    end
    return d, p
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64})
    length(x) >= 2 || return nothing
    X = hcat(ones(length(x)), x)
    coeff = X \ y
    yhat = X * coeff
    ymean = sum(y) / length(y)
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- ymean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (intercept=coeff[1], slope=coeff[2], r2=r2, n=length(x), ss_res=ss_res)
end

function fixed_slope_fit(d::Vector{Float64}, p::Vector{Float64}, pinf::Float64, slope::Float64; min_points::Int=2)
    diff = pinf .- p
    keep = (d .> 0) .& (diff .> 0) .& isfinite.(diff)
    count(keep) >= max(min_points, 2) || return nothing
    x = log.(d[keep])
    y = log.(diff[keep])
    intercept = sum(y .- slope .* x) / length(x)
    yhat = intercept .+ slope .* x
    ymean = sum(y) / length(y)
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- ymean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (intercept=intercept, slope=slope, r2=r2, n=length(x), ss_res=ss_res, pinf=pinf)
end

function offset_fit(d::Vector{Float64}, p::Vector{Float64}, pinf::Float64)
    diff = pinf .- p
    keep = (d .> 0) .& (diff .> 0) .& isfinite.(diff)
    count(keep) >= 2 || return nothing
    return linear_fit(log.(d[keep]), log.(diff[keep]))
end

function tail_mean_pinf(p::Vector{Float64}, tail_count::Integer)
    isempty(p) && return NaN, 0
    k = min(max(Int(tail_count), 1), length(p))
    return sum(@view p[(end - k + 1):end]) / k, k
end

function scan_fixed_slope_pinf(d::Vector{Float64}, p::Vector{Float64}, slope::Float64; scan_points::Int=1000, min_points::Int=10)
    length(p) >= max(min_points, 2) || return nothing
    pmin = minimum(p)
    pmax = maximum(p)
    scale = max(pmax - pmin, eps(Float64) * max(abs(pmax), 1.0))
    lower = pmin + 1.0e-9 * scale
    upper = pmax + 2.0 * scale
    best = nothing
    for pinf in range(lower, upper; length=max(scan_points, 2))
        fit = fixed_slope_fit(d, p, pinf, slope; min_points=min_points)
        isnothing(fit) && continue
        if isnothing(best) || fit.ss_res < best.ss_res
            best = fit
        end
    end
    return best
end

function offset_diagnostics(stats::FPCoupledSDEActiveObjects.MobileStats; tail_count::Integer=12, fixed_slope=nothing, scan_points::Integer=1000, fixed_min_points::Integer=10)
    all_d, all_p = finite_positive_distance_pdf(stats)
    length(all_d) >= 2 || return nothing
    min_idx = argmin(all_p)
    d = all_d[min_idx:end]
    p = all_p[min_idx:end]
    length(d) >= 2 || return nothing
    pinf, used_tail_count = tail_mean_pinf(p, tail_count)
    free_fit = offset_fit(d, p, pinf)
    fixed_fit = isnothing(fixed_slope) ? nothing : scan_fixed_slope_pinf(
        d,
        p,
        Float64(fixed_slope);
        scan_points=max(Int(scan_points), 2),
        min_points=max(Int(fixed_min_points), 2),
    )
    return (
        all_d=all_d,
        all_p=all_p,
        d=d,
        p=p,
        min_d=all_d[min_idx],
        min_p=all_p[min_idx],
        pinf=pinf,
        tail_count=used_tail_count,
        free_fit=free_fit,
        fixed_fit=fixed_fit,
    )
end

function profile_on_grid(centers::AbstractVector{Float64}, X::Real, param::CoupledSDEParam)
    return [profile_value(minimal_image(x - X, param.L), param) for x in centers]
end

function plot_density_panel(state::CoupledSDEState, param::CoupledSDEParam, density_stats::DensityFieldStats)
    centers = density_stats.centers
    density = density_stats.sample_count > 0 ? density_stats.current_density : density_snapshot(state.x, param.L, param.n_bins)[3]
    density_avg = average_density_field(density_stats)
    profile_A = profile_on_grid(centers, state.XA, param)
    profile_B = profile_on_grid(centers, state.XB, param)
    profile_scale = param.f0 != 0 ? 0.28 * param.rho0 / abs(param.f0) : 1.0
    y_max = max(finite_max(density, param.rho0), finite_max(density_avg, param.rho0), 1.35 * param.rho0)

    p = plot(
        centers,
        density;
        seriestype=:steppost,
        lw=2,
        label="instant bath density",
        xlabel="position x",
        ylabel="particles / length",
        title="Instant and average bath density",
        xlim=(0, param.L),
        ylim=(0, y_max * 1.18),
        color=:steelblue,
        framestyle=:box,
        grid=:y,
        legend=:outertopright,
    )
    plot!(p, centers, density_avg; seriestype=:steppost, lw=2.4, color=:seagreen, label="running average density")
    hline!(p, [param.N / param.L]; lw=1.5, ls=:dash, color=:gray35, label="global mean density")
    plot!(p, centers, param.rho0 .+ profile_scale .* profile_A; lw=2, ls=:dash, color=:firebrick, label="profile A scaled")
    plot!(p, centers, param.rho0 .+ profile_scale .* profile_B; lw=2, ls=:dash, color=:darkorange, label="profile B scaled")
    vline!(p, [state.XA]; lw=3, color=:firebrick, label="object A")
    vline!(p, [state.XB]; lw=3, color=:darkorange, label="object B")
    return p
end

function plot_distance_pdf(
    stats::FPCoupledSDEActiveObjects.MobileStats,
    state::CoupledSDEState,
    param::CoupledSDEParam;
    hard_min_separation::Float64=0.0,
)
    d = stats.bin_centers
    P = distance_probability_density(stats)
    y_max = max(1.25 * finite_max(P, 1.0 / max(param.L / 2, eps())), 1.5 / param.L)
    current_d = minimum_separation(state.XA, state.XB, param.L)

    p = plot(
        d,
        P;
        lw=2,
        marker=:circle,
        markersize=2.5,
        xlabel="minimum object distance",
        ylabel="probability density",
        title="Running distance probability density",
        xlim=(0, param.L / 2),
        ylim=(0, y_max),
        color=:purple,
        label="P(d)",
        framestyle=:box,
        grid=:y,
    )
    if hard_min_separation > 0.0
        vline!(p, [hard_min_separation]; lw=2, ls=:dot, color=:red, label="hard min")
    end
    vline!(p, [current_d]; lw=2, ls=:dash, color=:black, label="current d")
    return p
end

function plot_distance_pdf_standalone(
    stats::FPCoupledSDEActiveObjects.MobileStats,
    state::CoupledSDEState,
    param::CoupledSDEParam;
    hard_min_separation::Float64=0.0,
)
    p = plot_distance_pdf(stats, state, param; hard_min_separation=hard_min_separation)
    title!(p, @sprintf("Live distance probability density, samples=%d", stats.sample_count))
    return p
end

function plot_offset_powerlaw_live(stats::FPCoupledSDEActiveObjects.MobileStats; tail_count::Integer=12, reference_slope::Float64=-2.0, fixed_slope=nothing, scan_points::Integer=1000, fixed_min_points::Integer=10)
    diagnostics = offset_diagnostics(
        stats;
        tail_count=tail_count,
        fixed_slope=fixed_slope,
        scan_points=scan_points,
        fixed_min_points=fixed_min_points,
    )
    if isnothing(diagnostics)
        return plot(
            title="Live offset power law unavailable",
            axis=false,
            legend=false,
            annotation=(0.5, 0.5, text("Need at least two positive distance-density bins", 12, :center)),
            size=(1100, 700),
        ), nothing
    end

    diff = diagnostics.pinf .- diagnostics.p
    keep = diff .> 0
    if count(keep) < 2
        return plot(
            title="Live offset power law unavailable",
            axis=false,
            legend=false,
            annotation=(0.5, 0.5, text("Need at least two bins with Pinf > P(d)", 12, :center)),
            size=(1100, 700),
        ), diagnostics
    end

    kept_d = diagnostics.d[keep]
    kept_diff = diff[keep]
    plt = plot(
        kept_d,
        kept_diff;
        xscale=:log10,
        yscale=:log10,
        lw=2,
        marker=:circle,
        markersize=3,
        xlabel="minimum separation d",
        ylabel="Pinf - P(d)",
        title=@sprintf("Live offset power law, tail%d Pinf=%.6g", diagnostics.tail_count, diagnostics.pinf),
        label="Pinf - P(d)",
        legend=:outertopright,
        framestyle=:box,
        grid=:both,
        size=(1100, 700),
    )
    if !isnothing(diagnostics.free_fit)
        free_curve = exp(diagnostics.free_fit.intercept) .* kept_d .^ diagnostics.free_fit.slope
        plot!(
            plt,
            kept_d,
            free_curve;
            lw=2,
            ls=:dash,
            color=:black,
            label=@sprintf("free slope %.4g, R2 %.3g", diagnostics.free_fit.slope, diagnostics.free_fit.r2),
        )
    end
    ref_curve = kept_diff[1] .* (kept_d ./ kept_d[1]) .^ reference_slope
    plot!(
        plt,
        kept_d,
        ref_curve;
        lw=2,
        ls=:dot,
        color=:gray,
        label=@sprintf("reference d^%.4g", reference_slope),
    )
    if !isnothing(diagnostics.fixed_fit)
        fixed_curve = exp(diagnostics.fixed_fit.intercept) .* kept_d .^ diagnostics.fixed_fit.slope
        plot!(
            plt,
            kept_d,
            fixed_curve;
            lw=2,
            ls=:dashdot,
            color=:purple,
            label=@sprintf("best fixed %.4g, R2 %.3g", diagnostics.fixed_fit.slope, diagnostics.fixed_fit.r2),
        )
    end
    vline!(plt, [diagnostics.min_d]; lw=1, ls=:dot, color=:gray35, label="minimum P")
    return plt, diagnostics
end

function plot_location_pdf(stats::FPCoupledSDEActiveObjects.MobileStats, state::CoupledSDEState, param::CoupledSDEParam)
    x = stats.position_bin_centers
    xa = probability_density(stats.XA_histogram_counts, stats.position_bin_edges)
    xb = probability_density(stats.XB_histogram_counts, stats.position_bin_edges)
    center = probability_density(stats.center_histogram_counts, stats.position_bin_edges)
    y_max = max(1.25 * maximum([
        finite_max(xa, 1.0 / param.L),
        finite_max(xb, 1.0 / param.L),
        finite_max(center, 1.0 / param.L),
    ]), 1.5 / param.L)

    p = plot(
        x,
        xa;
        lw=2,
        xlabel="position x",
        ylabel="probability density",
        title="Running location probability density",
        xlim=(0, param.L),
        ylim=(0, y_max),
        color=:firebrick,
        label="object A",
        framestyle=:box,
        grid=:y,
    )
    plot!(p, x, xb; lw=2, color=:darkorange, label="object B")
    plot!(p, x, center; lw=2, color=:seagreen, label="pair center")
    hline!(p, [1.0 / param.L]; lw=1.5, ls=:dash, color=:gray35, label="uniform")
    vline!(p, [state.XA]; lw=1.5, ls=:dot, color=:firebrick, label=false)
    vline!(p, [state.XB]; lw=1.5, ls=:dot, color=:darkorange, label=false)
    return p
end

function plot_position_trace(trace, param::CoupledSDEParam)
    p = plot(
        trace["time"],
        trace["XA"];
        lw=2,
        xlabel="time",
        ylabel="position x",
        title="Recent object locations",
        ylim=(0, param.L),
        color=:firebrick,
        label="object A",
        framestyle=:box,
        grid=:y,
    )
    plot!(p, trace["time"], trace["XB"]; lw=2, color=:darkorange, label="object B")
    return p
end

function plot_signal_trace(trace, param::CoupledSDEParam)
    p = plot(
        trace["time"],
        trace["dmin"];
        lw=2,
        xlabel="time",
        ylabel="distance",
        title="Recent distance and profile sums",
        ylim=(0, param.L / 2),
        color=:black,
        label="min distance",
        framestyle=:box,
        grid=:y,
    )
    if !isempty(trace["Ssum"])
        smax = max(finite_max(trace["Ssum"], 1.0), eps())
        scaled = (param.L / 2) .* (trace["Ssum"] ./ smax)
        plot!(p, trace["time"], scaled; lw=1.8, color=:teal, label="S_A + S_B scaled")
    end
    return p
end

function state_panel(
    state::CoupledSDEState,
    param::CoupledSDEParam,
    stats::FPCoupledSDEActiveObjects.MobileStats,
    density_stats::DensityFieldStats,
    production_step::Integer,
    elapsed::Real;
    hard_min_separation::Float64=0.0,
)
    d_oriented = oriented_separation(state.XA, state.XB, param.L)
    d_min = minimum_separation(state.XA, state.XB, param.L)
    steps_per_second = elapsed > 0 ? production_step / elapsed : NaN
    thermal_rms = sqrt(2.0 * param.D0 * param.dt)
    active_rms = abs(param.mu_bath * param.f0) * sqrt(param.dt)
    object_rms_A = abs(param.mu_obj * state.last_SA) * sqrt(param.dt)
    object_rms_B = abs(param.mu_obj * state.last_SB) * sqrt(param.dt)
    density_avg = average_density_field(density_stats)
    lines = [
        "Mobile coupled-SDE live state",
        @sprintf("step/time      %d / %.4f", production_step, state.time),
        @sprintf("samples/speed  %d / %.0f steps/s", stats.sample_count, steps_per_second),
        @sprintf("L, N, rho      %.3g, %d, %.3g", param.L, param.N, param.N / param.L),
        @sprintf("rho inst mean/max %.3g, %.3g", vector_mean(density_stats.current_density), finite_max(density_stats.current_density, NaN)),
        @sprintf("rho avg  mean/max %.3g, %.3g", vector_mean(density_avg), finite_max(density_avg, NaN)),
        @sprintf("dt, sigma_f    %.2e, %.3g", param.dt, param.sigma_f),
        @sprintf("mu_bath/obj    %.3g, %.3g", param.mu_bath, param.mu_obj),
        @sprintf("X_A, X_B       %.4f, %.4f", state.XA, state.XB),
        @sprintf("sep orient/min %.4f, %.4f", d_oriented, d_min),
        @sprintf("S_A, S_B       %.4g, %.4g", state.last_SA, state.last_SB),
        @sprintf("last dX_A/B    %.3e, %.3e", state.last_dXA, state.last_dXB),
        @sprintf("last drel      %.3e", state.last_drel),
        @sprintf("rms/sigma bath %.3g, %.3g", thermal_rms / param.sigma_f, active_rms / param.sigma_f),
        @sprintf("rms/sigma obj  %.3g, %.3g", object_rms_A / param.sigma_f, object_rms_B / param.sigma_f),
    ]
    if hard_min_separation > 0.0
        push!(lines, @sprintf("hard min d    %.4g", hard_min_separation))
    end

    p = plot(
        xlim=(0, 1),
        ylim=(0, 1);
        axis=false,
        grid=false,
        legend=false,
        title="",
        framestyle=:none,
    )
    annotate!(p, 0.03, 0.97, text(join(lines, "\n"), 7, :left, :top, "Courier"))
    return p
end

function write_summary(
    path::AbstractString,
    state::CoupledSDEState,
    param::CoupledSDEParam,
    stats::FPCoupledSDEActiveObjects.MobileStats,
    density_stats::DensityFieldStats,
    production_step::Integer,
    elapsed::Real;
    hard_min_separation::Float64=0.0,
)
    density_avg = average_density_field(density_stats)
    open(path, "w") do io
        println(io, "result_type=coupled_sde_mobile_live_dashboard")
        println(io, "production_step=$(production_step)")
        println(io, "state_step=$(state.step)")
        println(io, "time=$(state.time)")
        println(io, "elapsed_seconds=$(elapsed)")
        println(io, "sample_count=$(stats.sample_count)")
        println(io, "density_sample_count=$(density_stats.sample_count)")
        println(io, "L=$(param.L)")
        println(io, "N=$(param.N)")
        println(io, "rho0=$(param.N / param.L)")
        println(io, "instant_density_mean=$(vector_mean(density_stats.current_density))")
        println(io, "instant_density_max=$(finite_max(density_stats.current_density, NaN))")
        println(io, "average_density_mean=$(vector_mean(density_avg))")
        println(io, "average_density_max=$(finite_max(density_avg, NaN))")
        println(io, "dt=$(param.dt)")
        println(io, "D0=$(param.D0)")
        println(io, "mu_bath=$(param.mu_bath)")
        println(io, "mu_obj=$(param.mu_obj)")
        println(io, "f0=$(param.f0)")
        println(io, "sigma_f=$(param.sigma_f)")
        println(io, "hard_min_separation=$(hard_min_separation)")
        println(io, "hard_min_separation_over_sigma_f=$(hard_min_separation / param.sigma_f)")
        println(io, "XA=$(state.XA)")
        println(io, "XB=$(state.XB)")
        println(io, "minimum_separation=$(minimum_separation(state.XA, state.XB, param.L))")
        println(io, "SA=$(state.last_SA)")
        println(io, "SB=$(state.last_SB)")
        println(io, "last_dXA=$(state.last_dXA)")
        println(io, "last_dXB=$(state.last_dXB)")
        println(io, "last_drel=$(state.last_drel)")
    end
    return path
end

function write_distributions_csv(path::AbstractString, stats::FPCoupledSDEActiveObjects.MobileStats)
    distance_pdf = probability_density(stats.histogram_counts, stats.bin_edges)
    xa_pdf = probability_density(stats.XA_histogram_counts, stats.position_bin_edges)
    xb_pdf = probability_density(stats.XB_histogram_counts, stats.position_bin_edges)
    center_pdf = probability_density(stats.center_histogram_counts, stats.position_bin_edges)
    open(path, "w") do io
        println(io, "kind,bin_left,bin_right,bin_center,probability_density,count")
        for i in eachindex(stats.bin_centers)
            @printf(io, "distance,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                stats.bin_edges[i], stats.bin_edges[i + 1], stats.bin_centers[i],
                distance_pdf[i], stats.histogram_counts[i])
        end
        for i in eachindex(stats.position_bin_centers)
            @printf(io, "object_A_location,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                stats.position_bin_edges[i], stats.position_bin_edges[i + 1],
                stats.position_bin_centers[i], xa_pdf[i], stats.XA_histogram_counts[i])
            @printf(io, "object_B_location,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                stats.position_bin_edges[i], stats.position_bin_edges[i + 1],
                stats.position_bin_centers[i], xb_pdf[i], stats.XB_histogram_counts[i])
            @printf(io, "center_location,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                stats.position_bin_edges[i], stats.position_bin_edges[i + 1],
                stats.position_bin_centers[i], center_pdf[i], stats.center_histogram_counts[i])
        end
    end
    return path
end

function write_density_csv(path::AbstractString, density_stats::DensityFieldStats)
    density_avg = average_density_field(density_stats)
    open(path, "w") do io
        println(io, "bin_left,bin_right,bin_center,instant_density,average_density")
        for i in eachindex(density_stats.centers)
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g\n",
                density_stats.edges[i],
                density_stats.edges[i + 1],
                density_stats.centers[i],
                density_stats.current_density[i],
                density_avg[i])
        end
    end
    return path
end

function write_offset_summary(path::AbstractString, diagnostics, stats::FPCoupledSDEActiveObjects.MobileStats)
    open(path, "w") do io
        println(io, "result_type=coupled_sde_mobile_live_offset_powerlaw")
        println(io, "sample_count=$(stats.sample_count)")
        if isnothing(diagnostics)
            println(io, "status=insufficient_positive_distance_bins")
            return
        end
        println(io, "status=ok")
        println(io, "d_min_probability=$(diagnostics.min_d)")
        println(io, "P_min=$(diagnostics.min_p)")
        println(io, "fit_window_rows=$(length(diagnostics.d))")
        println(io, "Pinf=$(diagnostics.pinf)")
        println(io, "Pinf_source=tail_mean")
        println(io, "Pinf_tail_count=$(diagnostics.tail_count)")
        if isnothing(diagnostics.free_fit)
            println(io, "offset_loglog_fit_n=0")
        else
            println(io, "offset_loglog_slope=$(diagnostics.free_fit.slope)")
            println(io, "offset_loglog_intercept=$(diagnostics.free_fit.intercept)")
            println(io, "offset_loglog_r2=$(diagnostics.free_fit.r2)")
            println(io, "offset_loglog_n=$(diagnostics.free_fit.n)")
        end
        if isnothing(diagnostics.fixed_fit)
            println(io, "fixed_slope_fit_n=0")
        else
            println(io, "fixed_slope=$(diagnostics.fixed_fit.slope)")
            println(io, "fixed_slope_best_Pinf=$(diagnostics.fixed_fit.pinf)")
            println(io, "fixed_slope_best_intercept=$(diagnostics.fixed_fit.intercept)")
            println(io, "fixed_slope_best_r2=$(diagnostics.fixed_fit.r2)")
            println(io, "fixed_slope_best_ss_res=$(diagnostics.fixed_fit.ss_res)")
            println(io, "fixed_slope_best_fit_n=$(diagnostics.fixed_fit.n)")
        end
    end
    return path
end

function write_html(path::AbstractString, dashboard_name::AbstractString, distance_name::AbstractString, offset_name::AbstractString, summary_name::AbstractString, offset_summary_name::AbstractString)
    open(path, "w") do io
        println(io, """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Coupled-SDE mobile objects live dashboard</title>
  <style>
    body { margin: 0; background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; }
    main { max-width: 1500px; margin: 0 auto; padding: 16px; }
    img { width: 100%; height: auto; display: block; background: white; }
    section { margin-bottom: 18px; }
    h2 { font-size: 16px; margin: 12px 0 8px; font-weight: 650; color: #f8fafc; }
    .meta { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 10px; font-size: 13px; color: #cbd5e1; }
    a { color: #93c5fd; }
  </style>
</head>
<body>
  <main>
    <div class="meta">
      <span>Coupled-SDE mobile objects</span>
      <span><a href="$(summary_name)">summary</a> · <a href="$(offset_summary_name)">offset summary</a></span>
    </div>
    <section>
      <h2>Dashboard</h2>
      <img id="live-dashboard" src="$(dashboard_name)" alt="Live coupled-SDE mobile objects dashboard">
    </section>
    <section>
      <h2>Distance Probability Density</h2>
      <img id="live-distance" src="$(distance_name)" alt="Live distance probability density">
    </section>
    <section>
      <h2>Offset Power Law</h2>
      <img id="live-offset" src="$(offset_name)" alt="Live offset power-law plot">
    </section>
  </main>
  <script>
    const images = [
      [document.getElementById("live-dashboard"), "$(dashboard_name)"],
      [document.getElementById("live-distance"), "$(distance_name)"],
      [document.getElementById("live-offset"), "$(offset_name)"],
    ];
    setInterval(() => {
      for (const [img, src] of images) {
        img.src = src + "?t=" + Date.now();
      }
    }, 1000);
  </script>
</body>
</html>
""")
    end
    return path
end

function save_dashboard(
    path::AbstractString,
    state::CoupledSDEState,
    param::CoupledSDEParam,
    stats::FPCoupledSDEActiveObjects.MobileStats,
    density_stats::DensityFieldStats,
    trace,
    production_step::Integer,
    elapsed::Real;
    hard_min_separation::Float64=0.0,
)
    p_density = plot_density_panel(state, param, density_stats)
    p_state = state_panel(state, param, stats, density_stats, production_step, elapsed; hard_min_separation=hard_min_separation)
    p_distance = plot_distance_pdf(stats, state, param; hard_min_separation=hard_min_separation)
    p_location = plot_location_pdf(stats, state, param)
    p_positions = plot_position_trace(trace, param)
    p_signals = plot_signal_trace(trace, param)
    fig = plot(
        p_density,
        p_state,
        p_distance,
        p_location,
        p_positions,
        p_signals;
        layout=@layout([a{0.62w} b; c d; e f]),
        size=(1500, 1180),
        dpi=110,
        margin=5Plots.mm,
    )
    tmp = replace(path, r"\.png$" => ".tmp.png")
    savefig(fig, tmp)
    mv(tmp, path; force=true)
    return path
end

function record_trace!(trace, state::CoupledSDEState, limit::Integer)
    push_limited!(trace["time"], state.time, limit)
    push_limited!(trace["XA"], state.XA, limit)
    push_limited!(trace["XB"], state.XB, limit)
    push_limited!(trace["dmin"], minimum_separation(state.XA, state.XB, trace["L"]), limit)
    push_limited!(trace["Ssum"], state.last_SA + state.last_SB, limit)
    return trace
end

function fresh_trace(param::CoupledSDEParam)
    return Dict(
        "L" => param.L,
        "time" => Float64[],
        "XA" => Float64[],
        "XB" => Float64[],
        "dmin" => Float64[],
        "Ssum" => Float64[],
    )
end

function save_checkpoint(
    path::AbstractString,
    param::CoupledSDEParam,
    state::CoupledSDEState,
    stats::FPCoupledSDEActiveObjects.MobileStats,
    density_stats::DensityFieldStats,
    trace,
    rng::AbstractRNG,
    production_step::Integer,
)
    tmp = replace(path, r"\.jld2$" => ".tmp.jld2")
    jldsave(
        tmp;
        result_type="coupled_sde_mobile_live_checkpoint",
        param=param,
        state=state,
        stats=stats,
        density_stats=density_stats,
        trace=trace,
        rng=rng,
        production_step=Int(production_step),
    )
    mv(tmp, path; force=true)
    return path
end

function load_checkpoint(path::AbstractString, current_param::CoupledSDEParam)
    data = JLD2.load(path)
    get(data, "result_type", "") == "coupled_sde_mobile_live_checkpoint" ||
        error("Not a coupled-SDE mobile live checkpoint: $path")
    for key in ("param", "state", "stats", "density_stats", "rng", "production_step")
        haskey(data, key) || error("Checkpoint missing $key: $path")
    end
    saved_param = data["param"]
    compatible_resume!(saved_param, current_param, path)
    trace = get(data, "trace", fresh_trace(current_param))
    return data["state"], data["stats"], data["density_stats"], trace, data["rng"], Int(data["production_step"])
end

function write_live_outputs(
    out_dir::AbstractString,
    state::CoupledSDEState,
    param::CoupledSDEParam,
    stats::FPCoupledSDEActiveObjects.MobileStats,
    density_stats::DensityFieldStats,
    trace,
    production_step::Integer,
    elapsed::Real;
    offset_tail_count::Integer=12,
    offset_reference_slope::Float64=-2.0,
    offset_fixed_slope=nothing,
    offset_scan_points::Integer=1000,
    offset_fixed_min_points::Integer=10,
    hard_min_separation::Float64=0.0,
)
    png_path = joinpath(out_dir, "live_dashboard.png")
    distance_png_path = joinpath(out_dir, "live_distance_probability.png")
    offset_png_path = joinpath(out_dir, "live_offset_powerlaw.png")
    summary_path = joinpath(out_dir, "live_summary.txt")
    offset_summary_path = joinpath(out_dir, "live_offset_powerlaw_summary.txt")
    csv_path = joinpath(out_dir, "live_distributions.csv")
    density_csv_path = joinpath(out_dir, "live_density_field.csv")
    html_path = joinpath(out_dir, "index.html")
    save_dashboard(png_path, state, param, stats, density_stats, trace, production_step, elapsed; hard_min_separation=hard_min_separation)
    distance_plot = plot_distance_pdf_standalone(stats, state, param; hard_min_separation=hard_min_separation)
    offset_plot, offset_diagnostic = plot_offset_powerlaw_live(
        stats;
        tail_count=offset_tail_count,
        reference_slope=offset_reference_slope,
        fixed_slope=offset_fixed_slope,
        scan_points=offset_scan_points,
        fixed_min_points=offset_fixed_min_points,
    )
    tmp_distance = replace(distance_png_path, r"\.png$" => ".tmp.png")
    tmp_offset = replace(offset_png_path, r"\.png$" => ".tmp.png")
    savefig(distance_plot, tmp_distance)
    savefig(offset_plot, tmp_offset)
    mv(tmp_distance, distance_png_path; force=true)
    mv(tmp_offset, offset_png_path; force=true)
    write_summary(summary_path, state, param, stats, density_stats, production_step, elapsed; hard_min_separation=hard_min_separation)
    write_offset_summary(offset_summary_path, offset_diagnostic, stats)
    write_distributions_csv(csv_path, stats)
    write_density_csv(density_csv_path, density_stats)
    write_html(html_path, basename(png_path), basename(distance_png_path), basename(offset_png_path), basename(summary_path), basename(offset_summary_path))
    return png_path
end

function main()
    args = parse_commandline()
    param = build_param(args)
    validate_param(param)
    hard_min_separation = resolve_hard_min_separation(args, param)
    resume_checkpoint = args["resume_checkpoint"]

    out_dir = if !isnothing(args["out_dir"])
        abspath(String(args["out_dir"]))
    elseif !isnothing(resume_checkpoint)
        dirname(abspath(String(resume_checkpoint)))
    else
        default_out_dir(param)
    end
    mkpath(out_dir)
    checkpoint_path = isnothing(args["checkpoint_path"]) ?
        joinpath(out_dir, "live_checkpoint.jld2") :
        abspath(String(args["checkpoint_path"]))

    state = nothing
    stats = nothing
    density_stats = nothing
    trace = nothing
    rng = nothing
    start_production_step = 0

    if isnothing(resume_checkpoint)
        rng = rng_from_seed(param.seed)
        state = initialize_state(param, rng)
        stats = FPCoupledSDEActiveObjects.MobileStats(param)
        density_stats = DensityFieldStats(param)
        trace = fresh_trace(param)
    else
        state, stats, density_stats, trace, rng, start_production_step =
            load_checkpoint(abspath(String(resume_checkpoint)), param)
    end
    work = SDEWork(param.N)
    target_production_step = if !isnothing(resume_checkpoint) && !isnothing(args["additional_steps"])
        start_production_step + max(Int(args["additional_steps"]), 0)
    else
        max(param.n_steps, 0)
    end

    println("Local coupled-SDE mobile-object live dashboard")
    println("  out_dir=$(out_dir)")
    println("  checkpoint=$(checkpoint_path)")
    if !isnothing(resume_checkpoint)
        println("  resume_checkpoint=$(abspath(String(resume_checkpoint)))")
        println("  resumed_production_step=$(start_production_step)")
    end
    println("  L=$(param.L), rho0=$(param.N / param.L), N=$(param.N)")
    println("  sigma_f=$(param.sigma_f), f0=$(param.f0), dt=$(param.dt), n_bins=$(param.n_bins)")
    println("  hard_min_separation=$(hard_min_separation) ($(hard_min_separation / param.sigma_f) sigma_f)")
    println("  warmup_steps=$(isnothing(resume_checkpoint) ? param.warmup_steps : 0), target_production_step=$(target_production_step), sample_interval=$(param.sample_interval)")
    println("  plot_interval_steps=$(args["plot_interval_steps"]), min_plot_interval_seconds=$(args["min_plot_interval_seconds"])")
    println("  offset_tail_count=$(args["offset_pinf_tail_count"]), offset_reference_slope=$(args["offset_reference_slope"])")
    println("  offset_fixed_slope=$(isnothing(args["offset_fixed_slope"]) ? args["offset_reference_slope"] : args["offset_fixed_slope"]), offset_scan_points=$(args["offset_scan_points"])")
    println("  live plot: $(joinpath(out_dir, "live_dashboard.png"))")
    println("  live distance probability: $(joinpath(out_dir, "live_distance_probability.png"))")
    println("  live offset power law: $(joinpath(out_dir, "live_offset_powerlaw.png"))")
    println("  auto-refresh html: $(joinpath(out_dir, "index.html"))")
    flush(stdout)

    start_time = time()
    if isnothing(resume_checkpoint)
        for step_idx in 1:param.warmup_steps
            step_mobile_objects_live!(state, param, rng, work, hard_min_separation)
            if step_idx == param.warmup_steps || step_idx % max(Int(args["progress_interval_steps"]), 1) == 0
                @printf("warmup step %d / %d, elapsed %.1fs\n", step_idx, param.warmup_steps, time() - start_time)
                flush(stdout)
            end
        end
    end

    trace_limit = max(Int(args["trace_points"]), 1)
    plot_interval_steps = max(Int(args["plot_interval_steps"]), 1)
    min_plot_interval_seconds = max(Float64(args["min_plot_interval_seconds"]), 0.0)
    progress_interval_steps = max(Int(args["progress_interval_steps"]), 1)
    offset_tail_count = max(Int(args["offset_pinf_tail_count"]), 1)
    offset_reference_slope = Float64(args["offset_reference_slope"])
    offset_fixed_slope = isnothing(args["offset_fixed_slope"]) ? offset_reference_slope : Float64(args["offset_fixed_slope"])
    offset_scan_points = max(Int(args["offset_scan_points"]), 2)
    offset_fixed_min_points = max(Int(args["offset_fixed_min_points"]), 2)
    last_plot_step = start_production_step
    last_plot_time = -Inf
    wrote_final_outputs = false

    if target_production_step > start_production_step
        for production_step in (start_production_step + 1):target_production_step
            obs = step_mobile_objects_live!(state, param, rng, work, hard_min_separation)
            FPCoupledSDEActiveObjects.accumulate!(stats, obs, state, param, production_step)
            if production_step % param.sample_interval == 0
                accumulate_density_field!(density_stats, state, param)
            end
            record_trace!(trace, state, trace_limit)

            now_time = time()
            should_plot = production_step - last_plot_step >= plot_interval_steps &&
                now_time - last_plot_time >= min_plot_interval_seconds
            is_final = production_step == target_production_step

            if should_plot || is_final
                png_path = write_live_outputs(
                    out_dir,
                    state,
                    param,
                    stats,
                    density_stats,
                    trace,
                    production_step,
                    now_time - start_time;
                    offset_tail_count=offset_tail_count,
                    offset_reference_slope=offset_reference_slope,
                    offset_fixed_slope=offset_fixed_slope,
                    offset_scan_points=offset_scan_points,
                    offset_fixed_min_points=offset_fixed_min_points,
                    hard_min_separation=hard_min_separation,
                )
                save_checkpoint(checkpoint_path, param, state, stats, density_stats, trace, rng, production_step)
                wrote_final_outputs = is_final
                last_plot_step = production_step
                last_plot_time = now_time
                @printf("updated %s at step %d, samples %d, elapsed %.1fs\n", png_path, production_step, stats.sample_count, now_time - start_time)
                flush(stdout)
            elseif production_step % progress_interval_steps == 0
                @printf("production step %d / %d, samples %d, elapsed %.1fs\n", production_step, target_production_step, stats.sample_count, now_time - start_time)
                flush(stdout)
            end
        end
    end

    if !wrote_final_outputs
        elapsed_now = time() - start_time
        png_path = write_live_outputs(
            out_dir,
            state,
            param,
            stats,
            density_stats,
            trace,
            start_production_step,
            elapsed_now;
            offset_tail_count=offset_tail_count,
            offset_reference_slope=offset_reference_slope,
            offset_fixed_slope=offset_fixed_slope,
            offset_scan_points=offset_scan_points,
            offset_fixed_min_points=offset_fixed_min_points,
            hard_min_separation=hard_min_separation,
        )
        save_checkpoint(checkpoint_path, param, state, stats, density_stats, trace, rng, start_production_step)
        @printf("updated %s at step %d, samples %d, elapsed %.1fs\n", png_path, start_production_step, stats.sample_count, elapsed_now)
        flush(stdout)
    end

    elapsed = time() - start_time
    println("Done.")
    println("  out_dir=$(out_dir)")
    println("  checkpoint=$(checkpoint_path)")
    println("  elapsed_seconds=$(round(elapsed, digits=3))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
