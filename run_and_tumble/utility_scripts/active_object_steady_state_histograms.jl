#!/usr/bin/env julia

using ArgParse
using DelimitedFiles
using JLD2
using Printf
using SHA

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

cd(REPO_ROOT)
include(joinpath(REPO_ROOT, "src", "diffusive", "modules_diffusive_no_activity.jl"))
include(joinpath(REPO_ROOT, "src", "active_objects", "modules_active_objects.jl"))

using .FPActiveObjects

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--state"
            help = "Path to one saved active-object state (.jld2)"
            arg_type = String
        "--state_list"
            help = "Path to a newline-delimited list of saved active-object states"
            arg_type = String
        "--output_dir"
            help = "Directory for histogram outputs"
            arg_type = String
        "--save_tag"
            help = "Output tag override for aggregate or single-run exports"
            arg_type = String
        "--base_aggregate"
            help = "Existing aggregated histogram JLD2 to use as the base when aggregating a state list"
            arg_type = String
        "--min_sweep"
            help = "Only include history from sweeps >= min_sweep"
            arg_type = Int
            default = 0
        "--max_sweep"
            help = "Only include history from sweeps <= max_sweep"
            arg_type = Int
        "--write_per_run"
            help = "When using --state_list, also export one histogram artifact per source state"
            action = :store_true
        "--plot_per_run"
            help = "When using --state_list and --write_per_run, also save one PNG per source state"
            action = :store_true
        "--no_plot"
            help = "Disable PNG output"
            action = :store_true
    end
    return parse_args(settings)
end

function sanitize_token(value::AbstractString)
    token = replace(strip(String(value)), r"[^A-Za-z0-9._+\-]+" => "-")
    token = replace(token, r"-{2,}" => "-")
    token = replace(token, r"^-+" => "")
    token = replace(token, r"-+$" => "")
    return isempty(token) ? "active_object_hist" : token
end

const MAX_FILENAME_COMPONENT_BYTES = 240
const LONGEST_HISTOGRAM_SUFFIX = "_steady_state_hist_positions.csv"

function compact_histogram_token(token::AbstractString)
    safe = sanitize_token(token)
    max_token_bytes = MAX_FILENAME_COMPONENT_BYTES - ncodeunits(LONGEST_HISTOGRAM_SUFFIX)
    ncodeunits(safe) <= max_token_bytes && return safe

    hash_token = bytes2hex(sha1(codeunits(safe)))[1:10]
    separator = "_h$(hash_token)_"
    remaining = max_token_bytes - ncodeunits(separator)
    remaining > 8 || return "ao_hist_h$(hash_token)"

    head_chars = max(4, fld(2 * remaining, 3))
    tail_chars = max(4, remaining - head_chars)
    head = first(safe, min(head_chars, length(safe)))
    tail = last(safe, min(tail_chars, length(safe)))
    head = replace(head, r"[-._]+$" => "")
    tail = replace(tail, r"^[-._]+" => "")
    compacted = "$(head)$(separator)$(tail)"
    ncodeunits(compacted) <= max_token_bytes && return compacted
    return "ao_hist_h$(hash_token)"
end

function state_id_tag(state_path::AbstractString)
    base = basename(state_path)
    m = match(r"_id-(.+)\.jld2$", base)
    return isnothing(m) ? nothing : String(m.captures[1])
end

function load_active_object_state(path::AbstractString)
    @load path state param potential
    state.potential = potential
    return state, param
end

function read_state_list(path::AbstractString)
    files = String[]
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        push!(files, line)
    end
    return files
end

function load_saved_histogram(path::AbstractString)
    @load path histogram
    return histogram
end

@inline function interval_overlap(a0::Int, a1::Int, b0::Int, b1::Int)
    return max(0, min(a1, b1) - max(a0, b0))
end

@inline max_min_edge_distance(L::Int) = max(fld(L, 2) - 1, 0)

function compute_time_weighted_histograms(state, param; min_sweep::Int=0, max_sweep::Union{Nothing,Int}=nothing)
    length(param.dims) == 1 || error("Only 1D active-object histograms are supported.")
    stats = state.object_stats
    haskey(stats, :history_sweeps) || error("Saved state does not contain active-object history.")
    haskey(stats, :history_left_sites) || error("Saved state does not contain active-object left-site history.")

    sweeps = Int.(stats[:history_sweeps])
    left_site_history = stats[:history_left_sites]
    isempty(sweeps) && error("Active-object history is empty.")
    length(sweeps) == length(left_site_history) || error("history_sweeps and history_left_sites length mismatch.")

    L = Int(param.dims[1])
    n_objects = length(left_site_history[1])
    position_counts = zeros(Float64, n_objects, L)
    distance_counts = n_objects == 2 ? zeros(Float64, max_min_edge_distance(L) + 1) : Float64[]

    final_sweep = Int(state.t)
    window_start = max(Int(min_sweep), 0)
    window_stop = isnothing(max_sweep) ? final_sweep : min(Int(max_sweep), final_sweep)
    window_stop >= window_start || error("max_sweep must be >= min_sweep after clipping to state.t.")

    total_time_weight = 0.0
    contributing_segments = 0

    for idx in eachindex(sweeps)
        seg_start = sweeps[idx]
        seg_end = idx < length(sweeps) ? sweeps[idx + 1] : final_sweep
        seg_end < seg_start && continue
        weight = interval_overlap(seg_start, seg_end, window_start, window_stop)
        weight <= 0 && continue

        left_sites = left_site_history[idx]
        length(left_sites) == n_objects || error("Inconsistent object count inside history.")
        for object_idx in 1:n_objects
            position_counts[object_idx, mod1(Int(left_sites[object_idx]), L)] += weight
        end
        if n_objects == 2
            distance_val = FPActiveObjects.object_min_edge_distance(left_sites, L)
            distance_counts[distance_val + 1] += weight
        end
        total_time_weight += weight
        contributing_segments += 1
    end

    total_time_weight > 0 || error("No positive dwell-time overlap remained after applying the sweep window.")
    position_probs = position_counts ./ total_time_weight
    distance_support = n_objects == 2 ? collect(0:(length(distance_counts) - 1)) : Int[]
    distance_probs = n_objects == 2 ? distance_counts ./ total_time_weight : Float64[]

    return Dict(
        "L" => L,
        "n_objects" => n_objects,
        "final_sweep" => final_sweep,
        "min_sweep" => window_start,
        "max_sweep" => window_stop,
        "total_time_weight" => total_time_weight,
        "num_history_records" => length(sweeps),
        "num_contributing_segments" => contributing_segments,
        "site_support" => collect(1:L),
        "position_counts" => position_counts,
        "position_probs" => position_probs,
        "distance_support" => distance_support,
        "distance_counts" => distance_counts,
        "distance_probs" => distance_probs,
    )
end

function init_aggregate_buffers(hist::Dict{String,Any})
    return Dict(
        "L" => Int(hist["L"]),
        "n_objects" => Int(hist["n_objects"]),
        "min_sweep" => Int(hist["min_sweep"]),
        "max_sweep" => Int(hist["max_sweep"]),
        "total_time_weight" => 0.0,
        "position_counts" => zeros(Float64, size(hist["position_counts"])),
        "distance_counts" => zeros(Float64, length(hist["distance_counts"])),
        "source_states" => String[],
    )
end

function init_aggregate_buffers_from_hist(hist::Dict{String,Any})
    source_states = haskey(hist, "source_states") ? String.(hist["source_states"]) : String[]
    return Dict(
        "L" => Int(hist["L"]),
        "n_objects" => Int(hist["n_objects"]),
        "min_sweep" => Int(hist["min_sweep"]),
        "max_sweep" => Int(hist["max_sweep"]),
        "total_time_weight" => Float64(hist["total_time_weight"]),
        "position_counts" => copy(hist["position_counts"]),
        "distance_counts" => copy(hist["distance_counts"]),
        "source_states" => source_states,
    )
end

function accumulate_histogram!(agg::Dict{String,Any}, hist::Dict{String,Any}, source_state::AbstractString)
    Int(hist["L"]) == Int(agg["L"]) || error("Cannot aggregate states with different L.")
    Int(hist["n_objects"]) == Int(agg["n_objects"]) || error("Cannot aggregate states with different object counts.")
    size(hist["position_counts"]) == size(agg["position_counts"]) || error("Position histogram shape mismatch.")
    length(hist["distance_counts"]) == length(agg["distance_counts"]) || error("Distance histogram shape mismatch.")
    agg["position_counts"] .+= hist["position_counts"]
    agg["distance_counts"] .+= hist["distance_counts"]
    agg["total_time_weight"] += Float64(hist["total_time_weight"])
    push!(agg["source_states"], String(source_state))
    agg["min_sweep"] = min(Int(agg["min_sweep"]), Int(hist["min_sweep"]))
    agg["max_sweep"] = max(Int(agg["max_sweep"]), Int(hist["max_sweep"]))
    return agg
end

function finalize_aggregate_histogram(agg::Dict{String,Any})
    total_time_weight = Float64(agg["total_time_weight"])
    total_time_weight > 0 || error("Aggregate total_time_weight is zero.")
    position_counts = agg["position_counts"]
    distance_counts = agg["distance_counts"]
    L = Int(agg["L"])
    n_objects = Int(agg["n_objects"])
    return Dict(
        "L" => L,
        "n_objects" => n_objects,
        "min_sweep" => Int(agg["min_sweep"]),
        "max_sweep" => Int(agg["max_sweep"]),
        "total_time_weight" => total_time_weight,
        "num_source_states" => length(agg["source_states"]),
        "source_states" => copy(agg["source_states"]),
        "site_support" => collect(1:L),
        "position_counts" => position_counts,
        "position_probs" => position_counts ./ total_time_weight,
        "distance_support" => n_objects == 2 ? collect(0:(length(distance_counts) - 1)) : Int[],
        "distance_counts" => distance_counts,
        "distance_probs" => n_objects == 2 ? distance_counts ./ total_time_weight : Float64[],
    )
end

function plot_histograms(hist::Dict{String,Any}; title_prefix::AbstractString="")
    site_support = hist["site_support"]
    position_probs = hist["position_probs"]
    n_objects = Int(hist["n_objects"])

    p_positions = plot(
        title=isempty(title_prefix) ? "Active-object steady-state positions" : "$(title_prefix) positions",
        xlabel="Left bond site",
        ylabel="Probability",
        framestyle=:box,
        grid=:y,
        legend=:outertopright,
    )
    for object_idx in 1:n_objects
        plot!(
            p_positions,
            site_support,
            vec(position_probs[object_idx, :]);
            lw=2.0,
            marker=:circle,
            markersize=2.5,
            label="object $(object_idx)",
        )
    end

    if n_objects != 2
        return p_positions
    end

    distance_support = hist["distance_support"]
    distance_probs = hist["distance_probs"]
    p_distance = bar(
        distance_support,
        distance_probs;
        title=isempty(title_prefix) ? "Steady-state pair distance" : "$(title_prefix) pair distance",
        xlabel="Minimum edge distance",
        ylabel="Probability",
        color=:purple,
        alpha=0.8,
        framestyle=:box,
        grid=:y,
        legend=false,
    )
    return plot(p_positions, p_distance, layout=(2, 1), size=(1200, 850))
end

function write_position_csv(path::AbstractString, hist::Dict{String,Any})
    site_support = hist["site_support"]
    position_probs = hist["position_probs"]
    n_objects = Int(hist["n_objects"])
    open(path, "w") do io
        headers = ["site"; ["P_object$(idx)_ss" for idx in 1:n_objects]]
        println(io, join(headers, ","))
        for site_idx in eachindex(site_support)
            row = [string(site_support[site_idx])]
            for object_idx in 1:n_objects
                push!(row, @sprintf("%.16e", position_probs[object_idx, site_idx]))
            end
            println(io, join(row, ","))
        end
    end
    return path
end

function write_distance_csv(path::AbstractString, hist::Dict{String,Any})
    Int(hist["n_objects"]) == 2 || return nothing
    distance_support = hist["distance_support"]
    distance_probs = hist["distance_probs"]
    open(path, "w") do io
        println(io, "distance,P_distance_ss")
        for idx in eachindex(distance_support)
            println(io, string(distance_support[idx], ",", @sprintf("%.16e", distance_probs[idx])))
        end
    end
    return path
end

function write_summary(path::AbstractString, hist::Dict{String,Any}; source_state::Union{Nothing,AbstractString}=nothing)
    open(path, "w") do io
        if !isnothing(source_state)
            println(io, "source_state=$(source_state)")
        elseif haskey(hist, "source_states")
            println(io, "num_source_states=$(length(hist["source_states"]))")
        end
        println(io, "L=$(hist["L"])")
        println(io, "n_objects=$(hist["n_objects"])")
        println(io, "min_sweep=$(hist["min_sweep"])")
        println(io, "max_sweep=$(hist["max_sweep"])")
        println(io, "total_time_weight=$(hist["total_time_weight"])")
        if haskey(hist, "num_history_records")
            println(io, "num_history_records=$(hist["num_history_records"])")
        end
        if haskey(hist, "num_contributing_segments")
            println(io, "num_contributing_segments=$(hist["num_contributing_segments"])")
        end
        if haskey(hist, "num_source_states")
            println(io, "num_source_states=$(hist["num_source_states"])")
        end
    end
    return path
end

function save_histogram_artifacts(output_prefix::AbstractString, hist::Dict{String,Any};
    source_state::Union{Nothing,AbstractString}=nothing,
    save_plot::Bool=true,
)
    mkpath(dirname(output_prefix))
    jld2_path = output_prefix * ".jld2"
    position_csv = output_prefix * "_positions.csv"
    distance_csv = output_prefix * "_distance.csv"
    summary_path = output_prefix * "_summary.txt"
    plot_path = output_prefix * ".png"

    jldsave(jld2_path; histogram=hist)
    write_position_csv(position_csv, hist)
    write_distance_csv(distance_csv, hist)
    write_summary(summary_path, hist; source_state=source_state)
    if save_plot
        p = plot_histograms(hist; title_prefix=basename(output_prefix))
        savefig(p, plot_path)
    end
    return Dict(
        "jld2" => jld2_path,
        "positions_csv" => position_csv,
        "distance_csv" => Int(hist["n_objects"]) == 2 ? distance_csv : "",
        "summary" => summary_path,
        "plot" => save_plot ? plot_path : "",
    )
end

function default_output_dir(args)
    if haskey(args, "output_dir") && !isnothing(args["output_dir"])
        return String(args["output_dir"])
    elseif haskey(args, "state") && !isnothing(args["state"])
        return dirname(abspath(String(args["state"])))
    end
    return pwd()
end

function output_prefix_from_state(state_path::AbstractString, output_dir::AbstractString; save_tag::Union{Nothing,AbstractString}=nothing)
    token = if !isnothing(save_tag)
        String(save_tag)
    else
        something(state_id_tag(state_path), replace(basename(state_path), r"\.jld2$" => ""))
    end
    token = compact_histogram_token(token)
    return joinpath(output_dir, "$(token)_steady_state_hist")
end

function aggregate_output_prefix(output_dir::AbstractString, aggregate_tag::AbstractString, aggregate_hist::Dict{String,Any})
    total_time_weight = Float64(aggregate_hist["total_time_weight"])
    tr_sweeps = Int(aggregate_hist["min_sweep"])
    tr_token = @sprintf("tr%d", tr_sweeps)
    total_time_token = @sprintf("ttot%d", round(Int, total_time_weight))
    tag_token = compact_histogram_token(aggregate_tag)
    return joinpath(output_dir, "$(tag_token)_$(tr_token)_$(total_time_token)_steady_state_hist")
end

function run_single_state(args)
    state_path = abspath(String(args["state"]))
    state, param = load_active_object_state(state_path)
    hist = compute_time_weighted_histograms(
        state,
        param;
        min_sweep=Int(args["min_sweep"]),
        max_sweep=haskey(args, "max_sweep") ? args["max_sweep"] : nothing,
    )
    output_dir = default_output_dir(args)
    prefix = output_prefix_from_state(state_path, output_dir; save_tag=get(args, "save_tag", nothing))
    artifacts = save_histogram_artifacts(prefix, hist; source_state=state_path, save_plot=!args["no_plot"])
    println("Saved active-object steady-state histogram artifacts:")
    println("  ", artifacts["jld2"])
    println("  ", artifacts["positions_csv"])
    if !isempty(artifacts["distance_csv"])
        println("  ", artifacts["distance_csv"])
    end
    println("  ", artifacts["summary"])
    !isempty(artifacts["plot"]) && println("  ", artifacts["plot"])
end

function run_state_list(args)
    state_list_path = abspath(String(args["state_list"]))
    state_paths = read_state_list(state_list_path)
    isempty(state_paths) && error("No state files found in $(state_list_path).")
    output_dir = default_output_dir(args)
    per_run_dir = joinpath(output_dir, "per_run")
    aggregate_dir = joinpath(output_dir, "aggregated")
    write_per_run = Bool(args["write_per_run"])
    plot_per_run = Bool(args["plot_per_run"])
    save_plot = !args["no_plot"]

    base_aggregate_path = haskey(args, "base_aggregate") && !isnothing(args["base_aggregate"]) ?
        abspath(String(args["base_aggregate"])) : nothing
    aggregate_buffers = nothing
    if !isnothing(base_aggregate_path)
        base_hist = load_saved_histogram(base_aggregate_path)
        Int(base_hist["min_sweep"]) == Int(args["min_sweep"]) ||
            error("Base aggregate min_sweep=$(base_hist["min_sweep"]) does not match requested min_sweep=$(args["min_sweep"]).")
        requested_max_sweep = haskey(args, "max_sweep") ? args["max_sweep"] : nothing
        if !isnothing(requested_max_sweep)
            Int(base_hist["max_sweep"]) == Int(requested_max_sweep) ||
                error("Base aggregate max_sweep=$(base_hist["max_sweep"]) does not match requested max_sweep=$(requested_max_sweep).")
        end
        aggregate_buffers = init_aggregate_buffers_from_hist(base_hist)
    end
    resolved_state_paths = String[]
    for state_path_raw in state_paths
        state_path = abspath(state_path_raw)
        state, param = load_active_object_state(state_path)
        hist = compute_time_weighted_histograms(
            state,
            param;
            min_sweep=Int(args["min_sweep"]),
            max_sweep=haskey(args, "max_sweep") ? args["max_sweep"] : nothing,
        )
        if isnothing(aggregate_buffers)
            aggregate_buffers = init_aggregate_buffers(hist)
        end
        accumulate_histogram!(aggregate_buffers, hist, state_path)
        push!(resolved_state_paths, state_path)

        if write_per_run
            prefix = output_prefix_from_state(state_path, per_run_dir)
            save_histogram_artifacts(prefix, hist; source_state=state_path, save_plot=plot_per_run)
        end
    end

    aggregate_hist = finalize_aggregate_histogram(aggregate_buffers)
    aggregate_tag = haskey(args, "save_tag") && !isnothing(args["save_tag"]) ? String(args["save_tag"]) : "active_object_hist_aggregate"
    aggregate_prefix = aggregate_output_prefix(aggregate_dir, aggregate_tag, aggregate_hist)
    artifacts = save_histogram_artifacts(aggregate_prefix, aggregate_hist; save_plot=save_plot)
    println("Aggregated ", length(resolved_state_paths), " active-object steady-state histograms")
    println("Saved aggregate artifacts:")
    println("  ", artifacts["jld2"])
    println("  ", artifacts["positions_csv"])
    if !isempty(artifacts["distance_csv"])
        println("  ", artifacts["distance_csv"])
    end
    println("  ", artifacts["summary"])
    !isempty(artifacts["plot"]) && println("  ", artifacts["plot"])
end

function main()
    args = parse_commandline()
    has_state = haskey(args, "state") && !isnothing(args["state"])
    has_state_list = haskey(args, "state_list") && !isnothing(args["state_list"])
    xor(has_state, has_state_list) || error("Provide exactly one of --state or --state_list.")
    if has_state
        run_single_state(args)
    else
        run_state_list(args)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
