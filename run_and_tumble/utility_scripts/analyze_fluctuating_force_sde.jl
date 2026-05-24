#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

cd(REPO_ROOT)
include(joinpath(REPO_ROOT, "src", "fluctuating_force_sde", "modules_fluctuating_force_sde.jl"))
using .FPFluctuatingForceSDE

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--state"
            help = "Path to one fluctuating-force SDE JLD2 result"
            arg_type = String
        "--state_list"
            help = "Newline-delimited list of fluctuating-force SDE result files"
            arg_type = String
        "--state_dir"
            help = "Directory searched recursively for fluctuating-force SDE JLD2 result files"
            arg_type = String
        "--output_dir"
            help = "Directory for CSV, summary, and aggregate JLD2"
            arg_type = String
            default = "analysis_outputs/fluctuating_force_sde"
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "fluctuating_force_sde"
        "--no_plot"
            help = "Accepted for submit-wrapper compatibility; this analyzer currently writes CSV/JLD2 only"
            action = :store_true
    end
    return parse_args(settings)
end

function read_state_list(path::AbstractString)
    files = String[]
    for raw in eachline(path)
        line = strip(raw)
        isempty(line) && continue
        startswith(line, "#") && continue
        push!(files, line)
    end
    return files
end

function find_jld2_files(root::AbstractString)
    files = String[]
    for (dir, _, names) in walkdir(root)
        for name in names
            endswith(name, ".jld2") || continue
            push!(files, joinpath(dir, name))
        end
    end
    return sort(files)
end

function resolve_inputs(args)
    sources = count(k -> haskey(args, k) && !isnothing(args[k]), ["state", "state_list", "state_dir"])
    sources == 1 || error("Provide exactly one of --state, --state_list, or --state_dir.")
    if haskey(args, "state") && !isnothing(args["state"])
        return [abspath(String(args["state"]))]
    elseif haskey(args, "state_list") && !isnothing(args["state_list"])
        return abspath.(read_state_list(String(args["state_list"])))
    end
    return abspath.(find_jld2_files(String(args["state_dir"])))
end

function load_result(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "result") || error("File does not contain a result dataset: $path")
    result = data["result"]
    get(result, "result_type", "") == "fluctuating_force_sde_density_variance" ||
        error("Expected fluctuating_force_sde_density_variance result in $path, got $(get(result, "result_type", "missing")).")
    haskey(data, "param") || error("File does not contain saved FluctuatingForceParam: $path")
    return data["param"], result
end

function compatible!(base::FluctuatingForceParam, param::FluctuatingForceParam, path::AbstractString)
    for key in (:dims, :N, :n_bins, :n_radial_bins)
        getfield(base, key) == getfield(param, key) || error("Incompatible $key for $path.")
    end
    for key in (:L, :D_bath, :dt, :mu_bath, :f0, :sigma_f, :radial_min, :edge_bins_for_offset, :variance_floor)
        a = Float64(getfield(base, key))
        b = Float64(getfield(param, key))
        isapprox(a, b; rtol=1.0e-10, atol=1.0e-12) || error("Incompatible $key for $path: aggregate has $a, file has $b")
    end
    base.profile_type == param.profile_type || error("Incompatible profile_type for $path.")
    size(base.force_centers) == size(param.force_centers) || error("Incompatible force_centers shape for $path.")
    all(isapprox.(base.force_centers, param.force_centers; rtol=1.0e-10, atol=1.0e-12)) ||
        error("Incompatible force_centers for $path.")
    return true
end

function aggregate_results(paths)
    isempty(paths) && error("No JLD2 files found.")
    base_param = nothing
    stats = nothing
    skipped = String[]
    source_files = String[]
    for path in paths
        loaded = try
            load_result(path)
        catch err
            push!(skipped, "$(path): $(sprint(showerror, err))")
            continue
        end
        param, result = loaded
        if isnothing(base_param)
            base_param = param
            stats = FluctuatingForceStats(base_param)
        end
        compatible!(base_param, param, path)
        bins = result["bins"]
        stats.sum_counts .+= Float64.(bins["sum_counts"])
        stats.sum_counts2 .+= Float64.(bins["sum_counts2"])
        stats.sample_count += Int(result["sample_count"])
        if haskey(result, "forces")
            forces = result["forces"]
            if haskey(forces, "sum_profile_sums")
                stats.sum_force_sums .+= Float64.(forces["sum_profile_sums"])
            end
            if haskey(forces, "sum_profile_sums2")
                stats.sum_force_sums2 .+= Float64.(forces["sum_profile_sums2"])
            end
        end
        push!(source_files, path)
    end
    isnothing(base_param) && error("No fluctuating-force SDE results could be loaded.")
    return base_param, stats, source_files, skipped
end

function aggregate_dict(param::FluctuatingForceParam, stats::FluctuatingForceStats, source_files, skipped)
    mean_counts, variance = FPFluctuatingForceSDE.count_variance(stats)
    offset = FPFluctuatingForceSDE.thermal_offset(variance, param)
    variance_active = FPFluctuatingForceSDE.active_variance(variance, offset, param)
    force_count = max(stats.sample_count, 1)
    mean_force_sums = stats.sum_force_sums ./ force_count
    variance_force_sums = stats.sum_force_sums2 ./ force_count .- mean_force_sums .^ 2
    radial = param.dims == 2 ? FPFluctuatingForceSDE.radial_summary(variance, param) : nothing
    return Dict(
        "result_type" => "fluctuating_force_sde_density_variance_aggregate",
        "parameters" => FPFluctuatingForceSDE.parameters_dict(param),
        "sample_count" => stats.sample_count,
        "n_replicas" => length(source_files),
        "source_files" => source_files,
        "skipped" => skipped,
        "bins" => Dict(
            "edges" => stats.bin_edges,
            "centers" => stats.bin_centers,
            "grid_shape" => param.dims == 1 ? (param.n_bins,) : (param.n_bins, param.n_bins),
            "sum_counts" => stats.sum_counts,
            "sum_counts2" => stats.sum_counts2,
            "mean_counts" => mean_counts,
            "variance_total" => variance,
            "thermal_offset" => offset,
            "variance_active" => variance_active,
        ),
        "radial" => radial,
        "forces" => Dict(
            "sum_profile_sums" => stats.sum_force_sums,
            "sum_profile_sums2" => stats.sum_force_sums2,
            "mean_profile_sums" => mean_force_sums,
            "variance_profile_sums" => variance_force_sums,
        ),
    )
end

function sanitize_tag(tag::AbstractString)
    safe = replace(String(tag), r"[^A-Za-z0-9._-]+" => "-")
    isempty(safe) ? "fluctuating_force_sde" : safe
end

function write_bins_csv(path::AbstractString, aggregate::Dict)
    params = aggregate["parameters"]
    bins = aggregate["bins"]
    dims = Int(params["dims"])
    centers = Float64.(bins["centers"])
    variance_total = Float64.(bins["variance_total"])
    variance_active = Float64.(bins["variance_active"])
    mean_counts = Float64.(bins["mean_counts"])
    n_bins = Int(params["n_bins"])
    open(path, "w") do io
        if dims == 1
            println(io, "x,mean_count,variance_total,variance_active")
            for i in 1:n_bins
                @printf(io, "%.16e,%.16e,%.16e,%.16e\n", centers[i], mean_counts[i], variance_total[i], variance_active[i])
            end
        else
            println(io, "x,y,mean_count,variance_total,variance_active")
            for iy in 1:n_bins, ix in 1:n_bins
                flat = ix + (iy - 1) * n_bins
                @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e\n", centers[ix], centers[iy], mean_counts[flat], variance_total[flat], variance_active[flat])
            end
        end
    end
    return path
end

function write_radial_csv(path::AbstractString, aggregate::Dict)
    radial = aggregate["radial"]
    isnothing(radial) && return nothing
    open(path, "w") do io
        println(io, "r_left,r_right,r_center,cell_count,variance_total,variance_active")
        edges = Float64.(radial["edges"])
        centers = Float64.(radial["centers"])
        counts = Int.(radial["cell_counts"])
        total = Float64.(radial["variance_total"])
        active = Float64.(radial["variance_active"])
        for i in eachindex(centers)
            @printf(io, "%.16e,%.16e,%.16e,%d,%.16e,%.16e\n", edges[i], edges[i + 1], centers[i], counts[i], total[i], active[i])
        end
    end
    return path
end

function write_summary(path::AbstractString, aggregate::Dict)
    open(path, "w") do io
        println(io, "result_type=$(aggregate["result_type"])")
        println(io, "n_replicas=$(aggregate["n_replicas"])")
        println(io, "sample_count=$(aggregate["sample_count"])")
        params = aggregate["parameters"]
        for key in ["dims", "L", "N", "D_bath", "dt", "mu_bath", "f0", "sigma_f", "profile_type", "n_bins", "mobile_forces", "force_mobility"]
            println(io, "$(key)=$(params[key])")
        end
        println(io, "thermal_offset=$(aggregate["bins"]["thermal_offset"])")
        if !isempty(aggregate["skipped"])
            println(io, "skipped_files=$(length(aggregate["skipped"]))")
            for item in aggregate["skipped"]
                println(io, "skipped=$(item)")
            end
        end
    end
    return path
end

function main()
    args = parse_commandline()
    paths = resolve_inputs(args)
    param, stats, source_files, skipped = aggregate_results(paths)
    aggregate = aggregate_dict(param, stats, source_files, skipped)

    output_dir = abspath(String(args["output_dir"]))
    mkpath(output_dir)
    tag = sanitize_tag(String(args["save_tag"]))
    bins_csv = joinpath(output_dir, "$(tag)_bins.csv")
    radial_csv = joinpath(output_dir, "$(tag)_radial.csv")
    jld2_path = joinpath(output_dir, "$(tag)_aggregate.jld2")
    summary_path = joinpath(output_dir, "$(tag)_summary.txt")

    write_bins_csv(bins_csv, aggregate)
    write_radial_csv(radial_csv, aggregate)
    jldsave(jld2_path; aggregate=aggregate)
    write_summary(summary_path, aggregate)

    println("Saved fluctuating-force SDE aggregate:")
    println("  $(bins_csv)")
    if !isnothing(aggregate["radial"])
        println("  $(radial_csv)")
    end
    println("  $(jld2_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
