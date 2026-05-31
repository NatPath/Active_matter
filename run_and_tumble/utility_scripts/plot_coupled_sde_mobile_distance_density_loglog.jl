#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

const OUTSIDE_LEGEND = :outertopright
const NORMAL_CI95 = 1.959963984540054

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--aggregate"
            help = "Path to one *_mobile_aggregate.jld2 file"
            arg_type = String
        "--analysis_dir"
            help = "Directory searched for newest *_mobile_aggregate.jld2"
            arg_type = String
        "--out_dir"
            help = "Directory for output PNGs"
            arg_type = String
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "coupled_sde_mobile"
    end
    return parse_args(settings)
end

function newest_mobile_aggregate(analysis_dir::AbstractString)
    files = [
        joinpath(analysis_dir, name)
        for name in readdir(analysis_dir)
        if occursin(r"_mobile_aggregate\.jld2$", name)
    ]
    isempty(files) && error("No *_mobile_aggregate.jld2 found under $(analysis_dir).")
    return sort(files; by=path -> stat(path).mtime, rev=true)[1]
end

function resolve_aggregate(args)
    has_aggregate = haskey(args, "aggregate") && !isnothing(args["aggregate"])
    has_analysis_dir = haskey(args, "analysis_dir") && !isnothing(args["analysis_dir"])
    xor(has_aggregate, has_analysis_dir) || error("Provide exactly one of --aggregate or --analysis_dir.")
    return has_aggregate ? abspath(String(args["aggregate"])) : newest_mobile_aggregate(abspath(String(args["analysis_dir"])))
end

function sanitize_tag(tag::AbstractString)
    safe = replace(String(tag), r"[^A-Za-z0-9._-]+" => "-")
    isempty(safe) ? "coupled_sde_mobile" : safe
end

function optional_error_value(row, key::String)
    value = Float64(get(row, key, NaN))
    return isfinite(value) && value >= 0 ? value : 0.0
end

function display_error_value(row, ci_key::String, sem_key::String)
    ci = optional_error_value(row, ci_key)
    ci > 0 && return ci
    return optional_error_value(row, sem_key)
end

function fit_sem_value(row, sem_key::String, ci_key::String)
    sem = optional_error_value(row, sem_key)
    sem > 0 && return sem
    ci = optional_error_value(row, ci_key)
    return ci > 0 ? ci / NORMAL_CI95 : 0.0
end

function log_safe_yerror(y::Vector{Float64}, err::Vector{Float64})
    safe = similar(err)
    for i in eachindex(err)
        upper = max(y[i] * (1.0 - 1.0e-9), 0.0)
        safe[i] = clamp(err[i], 0.0, upper)
    end
    return any(>(0.0), safe) ? safe : nothing
end

function present_error_vector(values::Vector{Float64})
    any(value -> isfinite(value) && value > 0, values) ? values : nothing
end

function finite_density(rows)
    d = Float64[]
    p = Float64[]
    counts = Float64[]
    p_display_error = Float64[]
    p_fit_sem = Float64[]
    has_ci = false
    has_sem = false
    for row in rows
        x = Float64(row["bin_center"])
        y = Float64(row["P_density"])
        c = Float64(row["count"])
        if isfinite(x) && isfinite(y) && isfinite(c) && x > 0 && y > 0 && c > 0
            push!(d, x)
            push!(p, y)
            push!(counts, c)
            display_err = display_error_value(row, "P_density_ci95", "P_density_sem")
            fit_sem = fit_sem_value(row, "P_density_sem", "P_density_ci95")
            push!(p_display_error, display_err)
            push!(p_fit_sem, fit_sem)
            has_ci |= optional_error_value(row, "P_density_ci95") > 0
            has_sem |= optional_error_value(row, "P_density_sem") > 0
        end
    end
    isempty(d) && error("No positive finite P_density rows found.")
    order = sortperm(d)
    ordered_p = p[order]
    ordered_display_error = present_error_vector(p_display_error[order])
    ordered_fit_sem = present_error_vector(p_fit_sem[order])
    ordered_plot_error = isnothing(ordered_display_error) ? nothing : log_safe_yerror(ordered_p, ordered_display_error)
    error_label = has_ci ? "95% CI" : (has_sem ? "replica SEM" : "")
    return d[order], ordered_p, counts[order], ordered_plot_error, ordered_fit_sem, error_label
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64}; sigma=nothing)
    length(x) >= 2 || return nothing
    if !isnothing(sigma)
        keep = [isfinite(x[i]) && isfinite(y[i]) && isfinite(sigma[i]) && sigma[i] > 0 for i in eachindex(x)]
        if count(keep) >= 2
            xw = x[keep]
            yw = y[keep]
            sw = sigma[keep]
            weights = 1.0 ./ (sw .^ 2)
            X = hcat(ones(length(xw)), xw)
            sqrtw = sqrt.(weights)
            coeff = (X .* reshape(sqrtw, :, 1)) \ (yw .* sqrtw)
            yhat = X * coeff
            ymean = sum(weights .* yw) / sum(weights)
            ss_res = sum(weights .* (yw .- yhat) .^ 2)
            ss_tot = sum(weights .* (yw .- ymean) .^ 2)
            r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
            reduced_chi2 = length(xw) > 2 ? ss_res / (length(xw) - 2) : NaN
            return (intercept=coeff[1], slope=coeff[2], r2=r2, n=length(xw), weighted=true, reduced_chi2=reduced_chi2)
        end
    end
    X = hcat(ones(length(x)), x)
    coeff = X \ y
    yhat = X * coeff
    ymean = sum(y) / length(y)
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- ymean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (intercept=coeff[1], slope=coeff[2], r2=r2, n=length(x), weighted=false, reduced_chi2=NaN)
end

function linear_fit_loglog(d::Vector{Float64}, p::Vector{Float64}; p_sem=nothing)
    length(d) >= 2 || return nothing
    x = log.(d)
    y = log.(p)
    sigma_logp = isnothing(p_sem) ? nothing : p_sem ./ p
    return linear_fit(x, y; sigma=sigma_logp)
end

function fit_label(fit)
    isnothing(fit) && return ""
    if fit.weighted
        return @sprintf("weighted fit: slope %.4g, wR2 %.3g", fit.slope, fit.r2)
    end
    return @sprintf("fit: slope %.4g, R2 %.3g", fit.slope, fit.r2)
end

function plot_density(d, p; title::String, min_d::Float64, min_p::Float64, p_error=nothing, error_label::String="", fit=nothing)
    plt = if isnothing(p_error)
        plot(
            d,
            p;
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="P_ss(d) density",
            title=title,
            label="P_ss(d)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    else
        plot(
            d,
            p;
            yerror=p_error,
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="P_ss(d) density",
            title=title,
            label="P_ss(d), $(error_label)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    end
    scatter!(plt, [min_d], [min_p]; markersize=6, color=:red, label="minimum P")
    if !isnothing(fit)
        fit_curve = exp(fit.intercept) .* d .^ fit.slope
        plot!(
            plt,
            d,
            fit_curve;
            lw=2,
            ls=:dash,
            color=:black,
            label=fit_label(fit),
        )
    end
    return plt
end

function main()
    args = parse_commandline()
    aggregate_path = resolve_aggregate(args)
    data = JLD2.load(aggregate_path)
    haskey(data, "rows") || error("Aggregate does not contain rows: $(aggregate_path)")

    rows = data["rows"]
    d, p, _, p_error, p_fit_sem, error_label = finite_density(rows)
    min_idx = argmin(p)
    min_d = d[min_idx]
    min_p = p[min_idx]

    tail_mask = d .>= min_d
    tail_d = d[tail_mask]
    tail_p = p[tail_mask]
    tail_error = isnothing(p_error) ? nothing : p_error[tail_mask]
    tail_fit_sem = isnothing(p_fit_sem) ? nothing : p_fit_sem[tail_mask]
    full_fit = linear_fit_loglog(d, p; p_sem=p_fit_sem)
    tail_fit = linear_fit_loglog(tail_d, tail_p; p_sem=tail_fit_sem)

    out_dir = if haskey(args, "out_dir") && !isnothing(args["out_dir"])
        abspath(String(args["out_dir"]))
    else
        dirname(aggregate_path)
    end
    mkpath(out_dir)
    tag = sanitize_tag(String(args["save_tag"]))

    full_plot = plot_density(
        d,
        p;
        title="Distance probability density, log-log",
        min_d=min_d,
        min_p=min_p,
        p_error=p_error,
        error_label=error_label,
        fit=full_fit,
    )
    tail_plot = plot_density(
        tail_d,
        tail_p;
        title=@sprintf("Distance probability density for d >= %.6g", min_d),
        min_d=min_d,
        min_p=min_p,
        p_error=tail_error,
        error_label=error_label,
        fit=tail_fit,
    )

    full_path = joinpath(out_dir, "$(tag)_distance_density_loglog.png")
    tail_path = joinpath(out_dir, "$(tag)_distance_density_loglog_from_minimum.png")
    savefig(full_plot, full_path)
    savefig(tail_plot, tail_path)

    println("Saved distance-density log-log plots:")
    println("  $(full_path)")
    println("  $(tail_path)")
    println(@sprintf("Minimum positive finite P_density: d=%.16g, P_density=%.16g", min_d, min_p))
    if !isnothing(full_fit)
        if full_fit.weighted
            println(@sprintf("Full weighted log-log fit: slope=%.8g, weighted_R2=%.8g, reduced_chi2=%.8g, n=%d", full_fit.slope, full_fit.r2, full_fit.reduced_chi2, full_fit.n))
        else
            println(@sprintf("Full log-log fit: slope=%.8g, R2=%.8g, n=%d", full_fit.slope, full_fit.r2, full_fit.n))
        end
    end
    if !isnothing(tail_fit)
        if tail_fit.weighted
            println(@sprintf("From-minimum weighted log-log fit: slope=%.8g, weighted_R2=%.8g, reduced_chi2=%.8g, n=%d", tail_fit.slope, tail_fit.r2, tail_fit.reduced_chi2, tail_fit.n))
        else
            println(@sprintf("From-minimum log-log fit: slope=%.8g, R2=%.8g, n=%d", tail_fit.slope, tail_fit.r2, tail_fit.n))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
