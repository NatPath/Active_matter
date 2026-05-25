#!/usr/bin/env julia

using ArgParse
using JLD2

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
    println("Saved coupled-SDE mobile steady-state comparison plot:")
    println("  $(out_path)")
    println("Expected distance profile: normalized 1 / D_proxy(d).")
    println("Expected location profile: uniform density 1/L.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
