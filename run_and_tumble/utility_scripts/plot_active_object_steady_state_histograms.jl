#!/usr/bin/env julia

using ArgParse
using JLD2

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

include(joinpath(REPO_ROOT, "utility_scripts", "active_object_steady_state_histograms.jl"))

function parse_real_spec(spec::AbstractString)
    value = strip(String(spec))
    occursin("/", value) || return parse(Float64, value)

    parts = split(value, "/")
    length(parts) == 2 || error("Invalid rational offset: $(spec)")
    numerator = parse(Float64, strip(parts[1]))
    denominator = parse(Float64, strip(parts[2]))
    iszero(denominator) && error("Offset denominator cannot be zero: $(spec)")
    return numerator / denominator
end

function shifted_distance_ylabel(offset::Real)
    iszero(offset) && return "Probability"
    if offset < 0
        return "Probability - $(abs(offset))"
    end
    return "Probability + $(offset)"
end

function plot_distance_points(hist; title_prefix::AbstractString="", distance_offset::Real=0.0)
    n_objects = Int(hist["n_objects"])
    n_objects == 2 || error("Distance plotting is only defined for two-object histograms.")

    distance_support = Int.(hist["distance_support"])
    distance_probs = Float64.(hist["distance_probs"]) .+ distance_offset

    return scatter(
        distance_support,
        distance_probs;
        title=isempty(title_prefix) ? "Steady-state pair distance" : "$(title_prefix) pair distance",
        xlabel="Minimum edge distance",
        ylabel=shifted_distance_ylabel(distance_offset),
        framestyle=:box,
        grid=:y,
        legend=false,
        marker=:circle,
        markersize=4,
        color=:blue,
        size=(1100, 650),
    )
end

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "histograms"
            help = "One or more aggregated active-object histogram JLD2 files"
            nargs = '+'
        "--out_dir"
            help = "Directory for output PNGs"
            arg_type = String
            default = joinpath(REPO_ROOT, "tmp_active_object_hist_plots")
        "--distance_only"
            help = "Save only the pair-distance panel as a dot plot"
            action = :store_true
        "--distance_offset"
            help = "Add this offset to the plotted pair-distance probabilities"
            arg_type = String
            default = "0"
    end
    return parse_args(settings)
end

function main()
    args = parse_commandline()
    hist_paths = [abspath(path) for path in args["histograms"]]
    out_dir = abspath(String(args["out_dir"]))
    distance_only = args["distance_only"]
    distance_offset = parse_real_spec(args["distance_offset"])
    mkpath(out_dir)

    for hist_path in hist_paths
        data = JLD2.load(hist_path)
        haskey(data, "histogram") || error("Histogram file does not contain key 'histogram': $(hist_path)")
        hist = data["histogram"]
        base = replace(basename(hist_path), r"\.jld2$" => "")
        plot_obj = distance_only ?
            plot_distance_points(hist; title_prefix=base, distance_offset=distance_offset) :
            plot_histograms(hist; title_prefix=base)
        suffix = distance_only ? "_distance_points" : ""
        out_path = joinpath(out_dir, base * suffix * ".png")
        savefig(plot_obj, out_path)
        println(out_path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
