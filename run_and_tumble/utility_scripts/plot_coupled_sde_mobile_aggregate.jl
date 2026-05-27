#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

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

function finite_ratio(rows)
    d = Float64[]
    ratio = Float64[]
    for r in rows
        distance = Float64(r["bin_center"])
        measured = Float64(r["P_density"])
        expected = Float64(r["inv_D_proxy_density"])
        if isfinite(distance) && isfinite(measured) && isfinite(expected) && distance > 0 && measured > 0 && expected > 0
            push!(d, distance)
            push!(ratio, measured / expected)
        end
    end
    return d, ratio
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n >= 2 || return (slope=NaN, intercept=NaN, r2=NaN)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx)^2 for xi in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in eachindex(x))
    if sxx <= 0
        return (slope=NaN, intercept=NaN, r2=NaN)
    end
    slope = sxy / sxx
    intercept = my - slope * mx
    yhat = [intercept + slope * xi for xi in x]
    ss_res = sum((y[i] - yhat[i])^2 for i in eachindex(y))
    ss_tot = sum((yi - my)^2 for yi in y)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (slope=slope, intercept=intercept, r2=r2)
end

function distance_plot(rows)
    d, measured, expected, flatness = finite_columns(
        rows,
        ["bin_center", "P_density", "inv_D_proxy_density", "P_times_D_proxy"],
    )

    p1 = plot(
        d,
        measured;
        lw=2,
        marker=:circle,
        markersize=3,
        xlabel="minimum separation",
        ylabel="probability density",
        title="Steady-state pair distance",
        label="measured P_ss(d)",
        framestyle=:box,
        grid=:y,
    )
    plot!(p1, d, expected; lw=2, ls=:dash, label="expected, normalized 1 / D_proxy(d)")

    ratio = similar(measured)
    for i in eachindex(measured)
        ratio[i] = expected[i] > 0 ? measured[i] / expected[i] : NaN
    end
    p2 = plot(
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
    hline!(p2, [1.0]; lw=2, ls=:dash, color=:black, label=false)

    p3 = plot(
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

    return p1, p2, p3
end

function ratio_diagnostic_plot(rows)
    d, ratio = finite_ratio(rows)
    if length(d) < 2
        empty_plot = plot(title="Distance-ratio diagnostics unavailable", axis=false, legend=false)
        return empty_plot, empty_plot, (powerlaw=(slope=NaN, intercept=NaN, r2=NaN), exponential=(slope=NaN, intercept=NaN, r2=NaN))
    end

    logd = log.(d)
    logr = log.(ratio)
    powerlaw_fit = linear_fit(logd, logr)
    exponential_fit = linear_fit(d, logr)

    powerlaw_curve = exp(powerlaw_fit.intercept) .* d .^ powerlaw_fit.slope
    exponential_curve = exp.(exponential_fit.intercept .+ exponential_fit.slope .* d)

    p_loglog = plot(
        d,
        ratio;
        lw=2,
        marker=:circle,
        markersize=3,
        xlabel="minimum separation",
        ylabel="measured / expected",
        title=@sprintf("Ratio log-log: slope %.3g, R2 %.3g", powerlaw_fit.slope, powerlaw_fit.r2),
        label="ratio",
        xscale=:log10,
        yscale=:log10,
        framestyle=:box,
        grid=:both,
    )
    plot!(p_loglog, d, powerlaw_curve; lw=2, ls=:dash, color=:black, label="power-law fit")

    p_semilog = plot(
        d,
        ratio;
        lw=2,
        marker=:circle,
        markersize=3,
        xlabel="minimum separation",
        ylabel="measured / expected",
        title=@sprintf("Ratio semi-log: exp slope %.3g, R2 %.3g", exponential_fit.slope, exponential_fit.r2),
        label="ratio",
        yscale=:log10,
        framestyle=:box,
        grid=:both,
    )
    plot!(p_semilog, d, exponential_curve; lw=2, ls=:dash, color=:black, label="exponential fit")

    return p_loglog, p_semilog, (powerlaw=powerlaw_fit, exponential=exponential_fit)
end

function location_plot(location_rows)
    x, xa, xb, center, uniform = finite_columns(
        location_rows,
        ["x_center", "XA_P_density", "XB_P_density", "center_P_density", "uniform_density"],
    )

    p = plot(
        x,
        xa;
        lw=2,
        xlabel="position",
        ylabel="probability density",
        title="Steady-state object locations",
        label="object A",
        framestyle=:box,
        grid=:y,
    )
    plot!(p, x, xb; lw=2, label="object B")
    plot!(p, x, center; lw=2, label="pair center")
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
    savefig(plot(p_distance, p_ratio, p_flatness, p_locations; layout=(2, 2), size=(1400, 1000)), out_path)
    ratio_out_path = joinpath(out_dir, "$(tag)_mobile_distance_ratio_scaling.png")
    savefig(plot(p_ratio_loglog, p_ratio_semilog; layout=(1, 2), size=(1400, 560)), ratio_out_path)
    println("Saved coupled-SDE mobile steady-state comparison plot:")
    println("  $(out_path)")
    println("Saved coupled-SDE mobile distance-ratio scaling plot:")
    println("  $(ratio_out_path)")
    println(@sprintf(
        "Distance-ratio power-law fit: ratio ~ d^%.6g, R2=%.6g",
        ratio_fits.powerlaw.slope,
        ratio_fits.powerlaw.r2,
    ))
    println(@sprintf(
        "Distance-ratio exponential fit: ratio ~ exp(%.6g d), R2=%.6g",
        ratio_fits.exponential.slope,
        ratio_fits.exponential.r2,
    ))
    println("Expected distance profile: normalized 1 / D_proxy(d).")
    println("Expected location profile: uniform density 1/L.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
