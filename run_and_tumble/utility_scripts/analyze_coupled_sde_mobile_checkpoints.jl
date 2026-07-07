#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

const CHECKPOINT_SCRIPT_DIR = @__DIR__
const CHECKPOINT_REPO_ROOT = normpath(joinpath(CHECKPOINT_SCRIPT_DIR, ".."))

include(joinpath(CHECKPOINT_SCRIPT_DIR, "analyze_coupled_sde_mobile_objects.jl"))

function parse_checkpoint_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--checkpoint"
            help = "Path to one coupled-SDE mobile-object runner checkpoint JLD2"
            arg_type = String
        "--checkpoint_list"
            help = "Newline-delimited list of checkpoint JLD2 files"
            arg_type = String
        "--checkpoint_dir"
            help = "Directory searched non-recursively for *.checkpoint.jld2 files"
            arg_type = String
        "--output_dir"
            help = "Directory for checkpoint aggregate CSV, summary, JLD2, and plot"
            arg_type = String
            default = "analysis_outputs/coupled_sde_active_objects/mobile_checkpoints"
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "coupled_sde_mobile_latest_checkpoints"
        "--fit_min"
            help = "Minimum distance included in D_proxy fit"
            arg_type = Float64
        "--fit_max"
            help = "Maximum distance included in D_proxy fit"
            arg_type = Float64
        "--tail_count"
            help = "Number of largest nonempty bins averaged for fallback Dinf"
            arg_type = Int
            default = 5
        "--periodic_fit"
            help = "Fit D = Dinf + A*(1/d^2 + 1/(L-d)^2)"
            action = :store_true
        "--no_plot"
            help = "Disable PNG output"
            action = :store_true
        "--allow_skips"
            help = "Write aggregate even if some checkpoints cannot be loaded"
            action = :store_true
    end
    return parse_args(settings)
end

function read_checkpoint_list(path::AbstractString)
    files = String[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        push!(files, line)
    end
    return files
end

function find_checkpoint_files(root::AbstractString)
    files = String[]
    for name in readdir(root)
        endswith(name, ".checkpoint.jld2") || continue
        push!(files, joinpath(root, name))
    end
    return sort(files)
end

function resolve_checkpoint_inputs(args)
    sources = count(k -> haskey(args, k) && !isnothing(args[k]), ["checkpoint", "checkpoint_list", "checkpoint_dir"])
    sources == 1 || error("Provide exactly one of --checkpoint, --checkpoint_list, or --checkpoint_dir.")
    if haskey(args, "checkpoint") && !isnothing(args["checkpoint"])
        return [abspath(String(args["checkpoint"]))]
    elseif haskey(args, "checkpoint_list") && !isnothing(args["checkpoint_list"])
        return abspath.(read_checkpoint_list(String(args["checkpoint_list"])))
    end
    return abspath.(find_checkpoint_files(String(args["checkpoint_dir"])))
end

function checkpoint_result(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "checkpoint") || error("Checkpoint file does not contain checkpoint metadata: $path")
    haskey(data, "param") || error("Checkpoint file does not contain param: $path")
    haskey(data, "state") || error("Checkpoint file does not contain state: $path")
    haskey(data, "stats") || error("Checkpoint file does not contain stats: $path")
    checkpoint = data["checkpoint"]
    get(checkpoint, "checkpoint_type", "") == "coupled_sde_mobile_objects" ||
        error("Expected coupled_sde_mobile_objects checkpoint in $path, got $(get(checkpoint, "checkpoint_type", "missing")).")
    return FPCoupledSDEActiveObjects.mobile_result_dict(data["param"], data["state"], data["stats"])
end

function aggregate_checkpoint_results(paths)
    isempty(paths) && error("No checkpoint JLD2 files found.")
    buffer = nothing
    skipped = String[]
    for path in paths
        result = try
            checkpoint_result(path)
        catch err
            push!(skipped, "$(path): $(sprint(showerror, err))")
            continue
        end
        if isnothing(buffer)
            buffer = init_buffer(result)
        end
        accumulate!(buffer, result, path)
    end
    isnothing(buffer) && error("No coupled-SDE mobile checkpoints could be loaded.")
    return buffer, skipped
end

function checkpoint_main()
    args = parse_checkpoint_commandline()
    paths = resolve_checkpoint_inputs(args)
    buffer, skipped = aggregate_checkpoint_results(paths)
    if !isempty(skipped) && !Bool(args["allow_skips"])
        error("Failed to load $(length(skipped)) checkpoint(s). Rerun with --allow_skips to write a partial aggregate.\n" * join(skipped, "\n"))
    end

    rows = rows_from_buffer(buffer)
    location_rows = location_rows_from_buffer(buffer)
    fit_rows = select_fit_rows(rows, args)
    fit = fit_proxy(fit_rows, Float64(buffer["L"]); periodic_fit=Bool(args["periodic_fit"]))
    Dinf_for_slope = isnothing(fit) ? tail_dinf(rows, Int(args["tail_count"])) : fit["Dinf"]
    slope = log_slope(fit_rows, Dinf_for_slope)

    output_dir = abspath(String(args["output_dir"]))
    mkpath(output_dir)
    tag = replace(String(args["save_tag"]), r"[^A-Za-z0-9._-]+" => "-")
    csv_path = joinpath(output_dir, "$(tag)_mobile_aggregate.csv")
    location_csv_path = joinpath(output_dir, "$(tag)_mobile_locations.csv")
    jld2_path = joinpath(output_dir, "$(tag)_mobile_aggregate.jld2")
    summary_path = joinpath(output_dir, "$(tag)_mobile_summary.txt")
    plot_path = joinpath(output_dir, "$(tag)_mobile.png")

    checkpoint_note = [
        "checkpoint_input_count=$(length(paths))",
        "checkpoint_loaded_count=$(Int(buffer["n_replicas"]))",
        "checkpoint_skipped_count=$(length(skipped))",
    ]
    skipped_with_note = vcat(checkpoint_note, skipped)

    write_csv(csv_path, rows)
    write_location_csv(location_csv_path, location_rows)
    jldsave(
        jld2_path;
        rows=rows,
        location_rows=location_rows,
        fit=fit,
        slope=slope,
        skipped=skipped,
        source_checkpoints=paths,
        error_metadata=aggregate_error_metadata(),
    )
    write_summary(summary_path, buffer, rows, location_rows, fit, slope, skipped_with_note, args)
    maybe_plot(plot_path, rows, fit, args)

    println("Saved mobile-object coupled-SDE checkpoint aggregate:")
    println("  $(csv_path)")
    println("  $(location_csv_path)")
    println("  $(jld2_path)")
    println("  $(summary_path)")
    if !args["no_plot"]
        println("  $(plot_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    checkpoint_main()
end
