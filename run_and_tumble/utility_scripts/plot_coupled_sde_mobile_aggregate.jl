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
            help = "Directory searched for the newest *_mobile_aggregate.jld2"
            arg_type = String
        "--out_dir"
            help = "Directory for output PNG"
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

function finite_xy(rows, x_key::String, y_key::String)
    x = Float64[]
    y = Float64[]
    for r in rows
        xv = Float64(r[x_key])
        yv = Float64(r[y_key])
        if isfinite(xv) && isfinite(yv)
            push!(x, xv)
            push!(y, yv)
        end
    end
    return x, y
end

function finite_columns(rows, keys::Vector{String})
    cols = [Float64[] for _ in keys]
    for r in rows
        values = [Float64(r[key]) for key in keys]
        if all(isfinite, values)
            for i in eachindex(keys)
                push!(cols[i], values[i])
            end
        end
    end
    return cols
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

function present_error_vector(values::Vector{Float64})
    any(>(0.0), values) ? values : nothing
end

function display_error_label(rows, ci_key::String, sem_key::String)
    has_ci = any(row -> optional_error_value(row, ci_key) > 0, rows)
    has_ci && return "95% CI"
    has_sem = any(row -> optional_error_value(row, sem_key) > 0, rows)
    return has_sem ? "replica SEM" : ""
end

function log_safe_yerror(y::Vector{Float64}, err::Vector{Float64})
    safe = similar(err)
    for i in eachindex(err)
        upper = max(y[i] * (1.0 - 1.0e-9), 0.0)
        safe[i] = clamp(err[i], 0.0, upper)
    end
    return present_error_vector(safe)
end

function finite_distance_columns(rows)
    d = Float64[]
    measured = Float64[]
    expected = Float64[]
    flatness = Float64[]
    p_error = Float64[]
    ratio_error = Float64[]
    flatness_error = Float64[]
    for r in rows
        distance = Float64(r["bin_center"])
        measured_density = Float64(r["P_density"])
        expected_density = Float64(r["inv_D_proxy_density"])
        flatness_value = Float64(r["P_times_D_proxy"])
        dproxy = Float64(r["D_rel_proxy"])
        if isfinite(distance) && isfinite(measured_density) && isfinite(expected_density) && isfinite(flatness_value)
            err = display_error_value(r, "P_density_ci95", "P_density_sem")
            push!(d, distance)
            push!(measured, measured_density)
            push!(expected, expected_density)
            push!(flatness, flatness_value)
            push!(p_error, err)
            push!(ratio_error, expected_density > 0 ? err / expected_density : 0.0)
            push!(flatness_error, isfinite(dproxy) ? err * abs(dproxy) : 0.0)
        end
    end
    return d, measured, expected, flatness, present_error_vector(p_error), present_error_vector(ratio_error), present_error_vector(flatness_error), display_error_label(rows, "P_density_ci95", "P_density_sem")
end

function finite_location_columns(rows)
    x = Float64[]
    xa = Float64[]
    xb = Float64[]
    center = Float64[]
    uniform = Float64[]
    xa_error = Float64[]
    xb_error = Float64[]
    center_error = Float64[]
    for r in rows
        xv = Float64(r["x_center"])
        xav = Float64(r["XA_P_density"])
        xbv = Float64(r["XB_P_density"])
        cv = Float64(r["center_P_density"])
        uv = Float64(r["uniform_density"])
        if isfinite(xv) && isfinite(xav) && isfinite(xbv) && isfinite(cv) && isfinite(uv)
            push!(x, xv)
            push!(xa, xav)
            push!(xb, xbv)
            push!(center, cv)
            push!(uniform, uv)
            push!(xa_error, display_error_value(r, "XA_P_density_ci95", "XA_P_density_sem"))
            push!(xb_error, display_error_value(r, "XB_P_density_ci95", "XB_P_density_sem"))
            push!(center_error, display_error_value(r, "center_P_density_ci95", "center_P_density_sem"))
        end
    end
    label = display_error_label(rows, "XA_P_density_ci95", "XA_P_density_sem")
    isempty(label) && (label = display_error_label(rows, "XB_P_density_ci95", "XB_P_density_sem"))
    isempty(label) && (label = display_error_label(rows, "center_P_density_ci95", "center_P_density_sem"))
    return x, xa, xb, center, uniform, present_error_vector(xa_error), present_error_vector(xb_error), present_error_vector(center_error), label
end

function finite_ratio(rows)
    d = Float64[]
    ratio = Float64[]
    ratio_error = Float64[]
    ratio_fit_sem = Float64[]
    for r in rows
        distance = Float64(r["bin_center"])
        measured = Float64(r["P_density"])
        expected = Float64(r["inv_D_proxy_density"])
        if isfinite(distance) && isfinite(measured) && isfinite(expected) && distance > 0 && measured > 0 && expected > 0
            err = display_error_value(r, "P_density_ci95", "P_density_sem")
            sem = fit_sem_value(r, "P_density_sem", "P_density_ci95")
            push!(d, distance)
            push!(ratio, measured / expected)
            push!(ratio_error, err / expected)
            push!(ratio_fit_sem, sem / expected)
        end
    end
    return d, ratio, present_error_vector(ratio_error), present_error_vector(ratio_fit_sem), display_error_label(rows, "P_density_ci95", "P_density_sem")
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64}; sigma=nothing)
    n = length(x)
    n >= 2 || return (slope=NaN, intercept=NaN, r2=NaN, weighted=false, reduced_chi2=NaN)
    if !isnothing(sigma)
        keep = [isfinite(x[i]) && isfinite(y[i]) && isfinite(sigma[i]) && sigma[i] > 0 for i in eachindex(x)]
        if count(keep) >= 2
            xw = x[keep]
            yw = y[keep]
            sw = sigma[keep]
            weights = 1.0 ./ (sw .^ 2)
            mx = sum(weights .* xw) / sum(weights)
            my = sum(weights .* yw) / sum(weights)
            sxx = sum(weights .* (xw .- mx) .^ 2)
            sxy = sum(weights .* (xw .- mx) .* (yw .- my))
            if sxx <= 0
                return (slope=NaN, intercept=NaN, r2=NaN, weighted=true, reduced_chi2=NaN)
            end
            slope = sxy / sxx
            intercept = my - slope * mx
            yhat = intercept .+ slope .* xw
            ss_res = sum(weights .* (yw .- yhat) .^ 2)
            ss_tot = sum(weights .* (yw .- my) .^ 2)
            r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
            reduced_chi2 = length(xw) > 2 ? ss_res / (length(xw) - 2) : NaN
            return (slope=slope, intercept=intercept, r2=r2, weighted=true, reduced_chi2=reduced_chi2)
        end
    end
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx)^2 for xi in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in eachindex(x))
    if sxx <= 0
        return (slope=NaN, intercept=NaN, r2=NaN, weighted=false, reduced_chi2=NaN)
    end
    slope = sxy / sxx
    intercept = my - slope * mx
    yhat = [intercept + slope * xi for xi in x]
    ss_res = sum((y[i] - yhat[i])^2 for i in eachindex(y))
    ss_tot = sum((yi - my)^2 for yi in y)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (slope=slope, intercept=intercept, r2=r2, weighted=false, reduced_chi2=NaN)
end

function fit_title(prefix::AbstractString, fit)
    if fit.weighted
        return @sprintf("%s: slope %.3g, wR2 %.3g", prefix, fit.slope, fit.r2)
    end
    return @sprintf("%s: slope %.3g, R2 %.3g", prefix, fit.slope, fit.r2)
end

function fit_stdout_label(fit)
    fit.weighted ? "weighted_R2" : "R2"
end

function distance_plot(rows)
    d, measured, expected, flatness, p_error, ratio_error, flatness_error, error_label = finite_distance_columns(rows)

    p1 = if isnothing(p_error)
        plot(
            d,
            measured;
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="probability density",
            title="Steady-state pair distance",
            label="measured P_ss(d)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(
            d,
            measured;
            yerror=p_error,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="probability density",
            title="Steady-state pair distance",
            label="measured P_ss(d), $(error_label)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:y,
        )
    end
    plot!(p1, d, expected; lw=2, ls=:dash, label="expected, normalized 1 / D_proxy(d)")

    ratio = similar(measured)
    for i in eachindex(measured)
        ratio[i] = expected[i] > 0 ? measured[i] / expected[i] : NaN
    end
    p2 = if isnothing(ratio_error)
        plot(
            d,
            ratio;
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title="Distance-profile ratio",
            label=false,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(
            d,
            ratio;
            yerror=ratio_error,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title="Distance-profile ratio",
            label=false,
            framestyle=:box,
            grid=:y,
        )
    end
    hline!(p2, [1.0]; lw=2, ls=:dash, color=:black, label=false)

    p3 = if isnothing(flatness_error)
        plot(
            d,
            flatness;
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="P_ss(d) * D_proxy(d)",
            title="Zero-current flatness check",
            label=false,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(
            d,
            flatness;
            yerror=flatness_error,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="P_ss(d) * D_proxy(d)",
            title="Zero-current flatness check",
            label=false,
            framestyle=:box,
            grid=:y,
        )
    end

    return p1, p2, p3
end

function ratio_diagnostic_plot(rows)
    d, ratio, ratio_error, ratio_fit_sem, error_label = finite_ratio(rows)
    if length(d) < 2
        empty_plot = plot(title="Distance-ratio diagnostics unavailable", axis=false, legend=false)
        missing_fit = (slope=NaN, intercept=NaN, r2=NaN, weighted=false, reduced_chi2=NaN)
        return empty_plot, empty_plot, (powerlaw=missing_fit, exponential=missing_fit)
    end

    logd = log.(d)
    logr = log.(ratio)
    sigma_logr = isnothing(ratio_fit_sem) ? nothing : ratio_fit_sem ./ ratio
    powerlaw_fit = linear_fit(logd, logr; sigma=sigma_logr)
    exponential_fit = linear_fit(d, logr; sigma=sigma_logr)

    powerlaw_curve = exp(powerlaw_fit.intercept) .* d .^ powerlaw_fit.slope
    exponential_curve = exp.(exponential_fit.intercept .+ exponential_fit.slope .* d)
    ratio_plot_error = isnothing(ratio_error) ? nothing : log_safe_yerror(ratio, ratio_error)

    p_loglog = if isnothing(ratio_plot_error)
        plot(
            d,
            ratio;
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title=fit_title("Ratio log-log", powerlaw_fit),
            label="ratio",
            xscale=:log10,
            yscale=:log10,
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
        )
    else
        plot(
            d,
            ratio;
            yerror=ratio_plot_error,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title=fit_title("Ratio log-log", powerlaw_fit),
            label="ratio, propagated $(error_label)",
            xscale=:log10,
            yscale=:log10,
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
        )
    end
    plot!(p_loglog, d, powerlaw_curve; lw=2, ls=:dash, color=:black, label="power-law fit")

    p_semilog = if isnothing(ratio_plot_error)
        plot(
            d,
            ratio;
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title=fit_title("Ratio semi-log", exponential_fit),
            label="ratio",
            yscale=:log10,
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
        )
    else
        plot(
            d,
            ratio;
            yerror=ratio_plot_error,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation",
            ylabel="measured / expected",
            title=fit_title("Ratio semi-log", exponential_fit),
            label="ratio, propagated $(error_label)",
            yscale=:log10,
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
        )
    end
    plot!(p_semilog, d, exponential_curve; lw=2, ls=:dash, color=:black, label="exponential fit")

    return p_loglog, p_semilog, (powerlaw=powerlaw_fit, exponential=exponential_fit)
end

function location_plot(location_rows)
    x, xa, xb, center, uniform, xa_sem, xb_sem, center_sem, error_label = finite_location_columns(location_rows)

    p = if isnothing(xa_sem)
        plot(
            x,
            xa;
            lw=2,
            xlabel="position",
            ylabel="probability density",
            title="Steady-state object locations",
            label="object A",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:y,
        )
    else
        plot(
            x,
            xa;
            yerror=xa_sem,
            lw=2,
            xlabel="position",
            ylabel="probability density",
            title="Steady-state object locations",
            label="object A, $(error_label)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:y,
        )
    end
    if isnothing(xb_sem)
        plot!(p, x, xb; lw=2, label="object B")
    else
        plot!(p, x, xb; yerror=xb_sem, lw=2, label="object B, $(error_label)")
    end
    if isnothing(center_sem)
        plot!(p, x, center; lw=2, label="pair center")
    else
        plot!(p, x, center; yerror=center_sem, lw=2, label="pair center, $(error_label)")
    end
    plot!(p, x, uniform; lw=2, ls=:dash, color=:black, label="expected uniform 1/L")
    return p
end

function sanitize_tag(tag::AbstractString)
    safe = replace(String(tag), r"[^A-Za-z0-9._-]+" => "-")
    isempty(safe) ? "coupled_sde_mobile" : safe
end

function main()
    args = parse_commandline()
    aggregate_path = resolve_aggregate(args)
    data = JLD2.load(aggregate_path)
    haskey(data, "rows") || error("Aggregate does not contain rows: $(aggregate_path)")
    haskey(data, "location_rows") || error("Aggregate does not contain location_rows: $(aggregate_path)")

    rows = data["rows"]
    location_rows = data["location_rows"]
    p_distance, p_ratio, p_flatness = distance_plot(rows)
    p_ratio_loglog, p_ratio_semilog, ratio_fits = ratio_diagnostic_plot(rows)
    p_locations = location_plot(location_rows)

    out_dir = if haskey(args, "out_dir") && !isnothing(args["out_dir"])
        abspath(String(args["out_dir"]))
    else
        dirname(aggregate_path)
    end
    mkpath(out_dir)
    tag = sanitize_tag(String(args["save_tag"]))
    out_path = joinpath(out_dir, "$(tag)_mobile_steady_state_expected_comparison.png")
    savefig(plot(p_distance, p_ratio, p_flatness, p_locations; layout=(2, 2), size=(1650, 1050)), out_path)
    ratio_out_path = joinpath(out_dir, "$(tag)_mobile_distance_ratio_scaling.png")
    savefig(plot(p_ratio_loglog, p_ratio_semilog; layout=(1, 2), size=(1650, 620)), ratio_out_path)
    println("Saved coupled-SDE mobile steady-state comparison plot:")
    println("  $(out_path)")
    println("Saved coupled-SDE mobile distance-ratio scaling plot:")
    println("  $(ratio_out_path)")
    println(@sprintf(
        "Distance-ratio power-law fit: ratio ~ d^%.6g, %s=%.6g",
        ratio_fits.powerlaw.slope,
        fit_stdout_label(ratio_fits.powerlaw),
        ratio_fits.powerlaw.r2,
    ))
    println(@sprintf(
        "Distance-ratio exponential fit: ratio ~ exp(%.6g d), %s=%.6g",
        ratio_fits.exponential.slope,
        fit_stdout_label(ratio_fits.exponential),
        ratio_fits.exponential.r2,
    ))
    println("Expected distance profile: normalized 1 / D_proxy(d).")
    println("Expected location profile: uniform density 1/L.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
