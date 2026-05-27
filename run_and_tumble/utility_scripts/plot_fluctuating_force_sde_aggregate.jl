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
            help = "Path to one fluctuating-force SDE *_aggregate.jld2 file"
            arg_type = String
        "--analysis_dir"
            help = "Directory searched for the newest *_aggregate.jld2 file"
            arg_type = String
        "--out_dir"
            help = "Directory for output PNG"
            arg_type = String
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "fluctuating_force_sde"
    end
    return parse_args(settings)
end

function newest_aggregate(analysis_dir::AbstractString)
    files = [
        joinpath(analysis_dir, name)
        for name in readdir(analysis_dir)
        if occursin(r"_aggregate\.jld2$", name)
    ]
    isempty(files) && error("No *_aggregate.jld2 found under $(analysis_dir).")
    return sort(files; by=path -> stat(path).mtime, rev=true)[1]
end

function resolve_aggregate(args)
    has_aggregate = haskey(args, "aggregate") && !isnothing(args["aggregate"])
    has_analysis_dir = haskey(args, "analysis_dir") && !isnothing(args["analysis_dir"])
    xor(has_aggregate, has_analysis_dir) || error("Provide exactly one of --aggregate or --analysis_dir.")
    return has_aggregate ? abspath(String(args["aggregate"])) : newest_aggregate(abspath(String(args["analysis_dir"])))
end

function sanitize_tag(tag::AbstractString)
    safe = replace(String(tag), r"[^A-Za-z0-9._-]+" => "-")
    isempty(safe) ? "fluctuating_force_sde" : safe
end

function density_variance_plots(aggregate::Dict)
    get(aggregate, "result_type", "") == "fluctuating_force_sde_density_variance_aggregate" ||
        error("Expected fluctuating_force_sde_density_variance_aggregate, got $(get(aggregate, "result_type", "missing")).")

    params = aggregate["parameters"]
    dims = Int(params["dims"])
    dims == 1 || error("This diagnostic plotter currently supports 1D aggregates only; got dims=$(dims).")

    bins = aggregate["bins"]
    x = Float64.(bins["centers"])
    mean_count = Float64.(bins["mean_counts"])
    variance_total = Float64.(bins["variance_total"])
    variance_active = Float64.(bins["variance_active"])
    offset = Float64(bins["thermal_offset"])
    excess = variance_total .- offset

    L = Float64(params["L"])
    N = Int(params["N"])
    n_bins = Int(params["n_bins"])
    sigma_f = Float64(params["sigma_f"])
    uniform_count = N / n_bins

    profile2 = exp.(-x .^ 2 ./ (sigma_f * sigma_f))
    positive_excess = filter(v -> isfinite(v) && v > 0, excess)
    guide_scale = isempty(positive_excess) ? 1.0 : maximum(positive_excess)
    profile2_guide = maximum(profile2) > 0 ? guide_scale .* profile2 ./ maximum(profile2) : fill(NaN, length(x))

    p_mean = plot(
        x,
        mean_count;
        lw=2,
        xlabel="x",
        ylabel="mean bin count",
        title="Mean density around fixed force",
        label="measured",
        framestyle=:box,
        grid=:y,
    )
    hline!(p_mean, [uniform_count]; lw=2, ls=:dash, color=:black, label=@sprintf("uniform N/n_bins = %.3g", uniform_count))

    p_var = plot(
        x,
        variance_total;
        lw=2,
        xlabel="x",
        ylabel="count variance",
        title="Density-count variance",
        label="variance_total",
        framestyle=:box,
        grid=:y,
    )
    hline!(p_var, [offset]; lw=2, ls=:dash, color=:black, label=@sprintf("edge offset = %.3g", offset))

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

    p_active = plot(
        x,
        variance_active;
        lw=2,
        xlabel="x",
        ylabel="max(excess, floor)",
        yscale=:log10,
        title="Positive active variance",
        label=false,
        framestyle=:box,
        grid=:y,
    )

    return p_mean, p_var, p_excess, p_active, Dict(
        "L" => L,
        "N" => N,
        "n_bins" => n_bins,
        "sigma_f" => sigma_f,
        "sample_count" => Int(aggregate["sample_count"]),
        "n_replicas" => Int(aggregate["n_replicas"]),
        "thermal_offset" => offset,
        "max_excess" => isempty(positive_excess) ? NaN : maximum(positive_excess),
    )
end

function main()
    args = parse_commandline()
    aggregate_path = resolve_aggregate(args)
    data = JLD2.load(aggregate_path)
    haskey(data, "aggregate") || error("Aggregate file does not contain `aggregate`: $(aggregate_path)")

    p_mean, p_var, p_excess, p_active, summary = density_variance_plots(data["aggregate"])
    out_dir = if haskey(args, "out_dir") && !isnothing(args["out_dir"])
        abspath(String(args["out_dir"]))
    else
        dirname(aggregate_path)
    end
    mkpath(out_dir)

    tag = sanitize_tag(String(args["save_tag"]))
    out_path = joinpath(out_dir, "$(tag)_density_variance_diagnostic.png")
    savefig(plot(p_mean, p_var, p_excess, p_active; layout=(2, 2), size=(1400, 1000)), out_path)

    println("Saved fluctuating-force SDE density-variance diagnostic plot:")
    println("  $(out_path)")
    println(@sprintf(
        "Aggregate: replicas=%d, samples=%d, L=%.6g, N=%d, n_bins=%d, sigma_f=%.6g, edge_offset=%.6g, max_excess=%.6g",
        summary["n_replicas"],
        summary["sample_count"],
        summary["L"],
        summary["N"],
        summary["n_bins"],
        summary["sigma_f"],
        summary["thermal_offset"],
        summary["max_excess"],
    ))
    println("The dashed f(x)^2 curve is only a localization guide, not an asserted steady-state theory curve.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
