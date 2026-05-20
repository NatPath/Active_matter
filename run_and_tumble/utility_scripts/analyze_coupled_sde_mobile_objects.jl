#!/usr/bin/env julia

using ArgParse
using JLD2
using Printf

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end
using Plots

const SCRIPT_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

cd(REPO_ROOT)
include(joinpath(REPO_ROOT, "src", "active_objects_sde", "modules_coupled_sde_active_objects.jl"))

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--state"
            help = "Path to one coupled-SDE mobile-object JLD2 result"
            arg_type = String
        "--state_list"
            help = "Newline-delimited list of coupled-SDE mobile-object result files"
            arg_type = String
        "--state_dir"
            help = "Directory searched recursively for coupled-SDE JLD2 result files"
            arg_type = String
        "--output_dir"
            help = "Directory for CSV, summary, and plots"
            arg_type = String
            default = "analysis_outputs/coupled_sde_active_objects/mobile_objects"
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "coupled_sde_mobile"
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
    get(result, "result_type", "") == "coupled_sde_mobile_objects" ||
        error("Expected coupled_sde_mobile_objects result in $path, got $(get(result, "result_type", "missing")).")
    return result
end

function init_buffer(result)
    params = result["parameters"]
    bins = result["bins"]
    n_bins = length(bins["centers"])
    return Dict(
        "L" => Float64(params["L"]),
        "rho0" => Float64(params["rho0"]),
        "N" => Int(params["N"]),
        "D0" => Float64(params["D0"]),
        "dt" => Float64(params["dt"]),
        "mu_bath" => Float64(params["mu_bath"]),
        "mu_obj" => Float64(params["mu_obj"]),
        "f0" => Float64(params["f0"]),
        "sigma_f" => Float64(params["sigma_f"]),
        "edges" => Float64.(bins["edges"]),
        "centers" => Float64.(bins["centers"]),
        "histogram_counts" => zeros(Float64, n_bins),
        "bin_counts" => zeros(Float64, n_bins),
        "sum_delta_rel_sq" => zeros(Float64, n_bins),
        "sum_Ssum" => zeros(Float64, n_bins),
        "sum_separation_min" => zeros(Float64, n_bins),
        "sample_count" => 0,
        "n_replicas" => 0,
        "history_records" => 0,
        "history_truncated_replicas" => 0,
        "source_files" => String[],
    )
end

function compatible!(buf, result, path)
    params = result["parameters"]
    for key in ["L", "rho0", "N", "D0", "dt", "mu_bath", "mu_obj", "f0", "sigma_f"]
        a = Float64(buf[key])
        b = Float64(params[key])
        isapprox(a, b; rtol=1.0e-10, atol=1.0e-12) ||
            error("Incompatible $key for $path: aggregate has $a, file has $b")
    end
    edges = Float64.(result["bins"]["edges"])
    length(edges) == length(buf["edges"]) || error("Bin-edge length mismatch for $path.")
    all(isapprox.(edges, buf["edges"]; rtol=1.0e-10, atol=1.0e-12)) || error("Bin edges differ for $path.")
end

function accumulate!(buf, result, path)
    compatible!(buf, result, path)
    bins = result["bins"]
    buf["histogram_counts"] .+= Float64.(bins["histogram_counts"])
    buf["bin_counts"] .+= Float64.(bins["bin_counts"])
    buf["sum_delta_rel_sq"] .+= Float64.(bins["sum_delta_rel_sq"])
    buf["sum_Ssum"] .+= Float64.(bins["sum_Ssum"])
    buf["sum_separation_min"] .+= Float64.(bins["sum_separation_min"])
    buf["sample_count"] += Int(result["sample_count"])
    buf["n_replicas"] += 1
    history = get(result, "history", Dict())
    if haskey(history, "time")
        buf["history_records"] += length(history["time"])
    end
    if get(history, "truncated", false) == true
        buf["history_truncated_replicas"] += 1
    end
    push!(buf["source_files"], path)
    return buf
end

function aggregate_results(paths)
    isempty(paths) && error("No JLD2 files found.")
    buffer = nothing
    skipped = String[]
    for path in paths
        result = try
            load_result(path)
        catch err
            push!(skipped, "$(path): $(sprint(showerror, err))")
            continue
        end
        if isnothing(buffer)
            buffer = init_buffer(result)
        end
        accumulate!(buffer, result, path)
    end
    isnothing(buffer) && error("No mobile-object coupled-SDE results could be loaded.")
    return buffer, skipped
end

function safe_divide(numer::AbstractVector{Float64}, denom::AbstractVector{Float64})
    out = similar(numer)
    for i in eachindex(numer)
        out[i] = denom[i] > 0 ? numer[i] / denom[i] : NaN
    end
    return out
end

function rows_from_buffer(buf)
    edges = buf["edges"]
    centers = buf["centers"]
    counts = buf["bin_counts"]
    hist = buf["histogram_counts"]
    widths = diff(edges)
    mu_obj = Float64(buf["mu_obj"])
    dt = Float64(buf["dt"])
    total_hist = sum(hist)
    mean_delta_rel_sq = safe_divide(buf["sum_delta_rel_sq"], counts)
    mean_Ssum = safe_divide(buf["sum_Ssum"], counts)
    mean_sep = safe_divide(buf["sum_separation_min"], counts)
    D_traj = mean_delta_rel_sq ./ (2.0 * dt)
    D_proxy = 0.5 * mu_obj^2 .* mean_Ssum
    P_mass = total_hist > 0 ? hist ./ total_hist : fill(NaN, length(hist))
    P_density = P_mass ./ widths

    invD_raw = similar(D_proxy)
    for i in eachindex(D_proxy)
        invD_raw[i] = isfinite(D_proxy[i]) && D_proxy[i] > 0 ? 1.0 / D_proxy[i] : NaN
    end
    norm = sum((isfinite(invD_raw[i]) ? invD_raw[i] * widths[i] : 0.0) for i in eachindex(invD_raw))
    invD_density = norm > 0 ? invD_raw ./ norm : fill(NaN, length(invD_raw))
    flatness = P_density .* D_proxy

    rows = Vector{Dict{String,Any}}()
    for i in eachindex(centers)
        push!(rows, Dict(
            "bin_left" => edges[i],
            "bin_right" => edges[i + 1],
            "bin_center" => centers[i],
            "mean_separation_min" => mean_sep[i],
            "count" => counts[i],
            "histogram_count" => hist[i],
            "P_mass" => P_mass[i],
            "P_density" => P_density[i],
            "mean_delta_rel_sq" => mean_delta_rel_sq[i],
            "mean_Ssum" => mean_Ssum[i],
            "D_rel_traj" => D_traj[i],
            "D_rel_proxy" => D_proxy[i],
            "inv_D_proxy_density" => invD_density[i],
            "P_times_D_proxy" => flatness[i],
        ))
    end
    return rows
end

function select_fit_rows(rows, args)
    fit_rows = [r for r in rows if r["count"] > 0 && isfinite(r["D_rel_proxy"]) && r["bin_center"] > 0]
    if haskey(args, "fit_min") && !isnothing(args["fit_min"])
        fit_rows = [r for r in fit_rows if r["bin_center"] >= Float64(args["fit_min"])]
    end
    if haskey(args, "fit_max") && !isnothing(args["fit_max"])
        fit_rows = [r for r in fit_rows if r["bin_center"] <= Float64(args["fit_max"])]
    end
    return fit_rows
end

function fit_proxy(rows, L::Real; periodic_fit::Bool=false)
    length(rows) >= 2 || return nothing
    x = Float64[]
    y = Float64[]
    for r in rows
        d = Float64(r["bin_center"])
        push!(x, periodic_fit ? (1.0 / d^2 + 1.0 / (Float64(L) - d)^2) : 1.0 / d^2)
        push!(y, Float64(r["D_rel_proxy"]))
    end
    X = hcat(ones(length(x)), x)
    coeff = X \ y
    yhat = X * coeff
    ss_res = sum((y .- yhat).^2)
    ybar = sum(y) / length(y)
    ss_tot = sum((y .- ybar).^2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return Dict("Dinf" => coeff[1], "A" => coeff[2], "r2" => r2, "n_fit" => length(y))
end

function tail_dinf(rows, tail_count::Int)
    nonempty = [r for r in rows if r["count"] > 0 && isfinite(r["D_rel_proxy"])]
    isempty(nonempty) && return NaN
    sorted_rows = sort(nonempty; by=r -> r["bin_center"])
    n = min(max(tail_count, 1), length(sorted_rows))
    tail = sorted_rows[(end - n + 1):end]
    return sum(Float64(r["D_rel_proxy"]) for r in tail) / n
end

function log_slope(rows, Dinf::Real)
    xs = Float64[]
    ys = Float64[]
    for r in rows
        y = Float64(r["D_rel_proxy"]) - Float64(Dinf)
        if r["bin_center"] > 0 && y > 0
            push!(xs, log(Float64(r["bin_center"])))
            push!(ys, log(y))
        end
    end
    length(xs) >= 2 || return nothing
    X = hcat(ones(length(xs)), xs)
    coeff = X \ ys
    return Dict("intercept" => coeff[1], "slope" => coeff[2], "n_fit" => length(xs))
end

function finite_std(values)
    vals = [Float64(v) for v in values if isfinite(Float64(v))]
    length(vals) >= 2 || return NaN
    m = sum(vals) / length(vals)
    return sqrt(sum((vals .- m).^2) / (length(vals) - 1))
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "bin_left,bin_right,bin_center,mean_separation_min,count,histogram_count,P_mass,P_density,mean_delta_rel_sq,mean_Ssum,D_rel_traj,D_rel_proxy,inv_D_proxy_density,P_times_D_proxy")
        for r in rows
            @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                r["bin_left"], r["bin_right"], r["bin_center"], r["mean_separation_min"], r["count"], r["histogram_count"],
                r["P_mass"], r["P_density"], r["mean_delta_rel_sq"], r["mean_Ssum"], r["D_rel_traj"], r["D_rel_proxy"],
                r["inv_D_proxy_density"], r["P_times_D_proxy"])
        end
    end
    return path
end

function write_summary(path, buf, rows, fit, slope, skipped, args)
    fit_rows = select_fit_rows(rows, args)
    flatness_std = finite_std([r["P_times_D_proxy"] for r in fit_rows])
    open(path, "w") do io
        println(io, "n_replicas=$(buf["n_replicas"])")
        println(io, "sample_count=$(buf["sample_count"])")
        println(io, "L=$(buf["L"])")
        println(io, "rho0=$(buf["rho0"])")
        println(io, "N=$(buf["N"])")
        println(io, "dt=$(buf["dt"])")
        println(io, "mu_obj=$(buf["mu_obj"])")
        println(io, "history_records=$(buf["history_records"])")
        println(io, "history_truncated_replicas=$(buf["history_truncated_replicas"])")
        println(io, "periodic_fit=$(args["periodic_fit"])")
        println(io, "P_times_D_proxy_std_in_fit_window=$(flatness_std)")
        if !isnothing(fit)
            println(io, "fit_Dinf=$(fit["Dinf"])")
            println(io, "fit_A=$(fit["A"])")
            println(io, "fit_r2=$(fit["r2"])")
            println(io, "fit_n=$(fit["n_fit"])")
        end
        if !isnothing(slope)
            println(io, "loglog_slope_after_Dinf=$(slope["slope"])")
            println(io, "loglog_slope_n=$(slope["n_fit"])")
        end
        if !isempty(skipped)
            println(io, "skipped_files=$(length(skipped))")
            for item in skipped
                println(io, "skipped=$(item)")
            end
        end
    end
    return path
end

function maybe_plot(output_path, rows, fit, args)
    args["no_plot"] && return nothing
    centers = [Float64(r["bin_center"]) for r in rows]
    counts = [Float64(r["count"]) for r in rows]
    nonempty = findall(>(0.0), counts)
    c = centers[nonempty]
    Dtraj = [Float64(rows[i]["D_rel_traj"]) for i in nonempty]
    Dproxy = [Float64(rows[i]["D_rel_proxy"]) for i in nonempty]
    P = [Float64(rows[i]["P_density"]) for i in nonempty]
    invD = [Float64(rows[i]["inv_D_proxy_density"]) for i in nonempty]
    flat = [Float64(rows[i]["P_times_D_proxy"]) for i in nonempty]

    p1 = plot(c, Dproxy; marker=:circle, lw=2, xlabel="minimum separation", ylabel="D_rel", title="Conditional diffusivity", label="proxy")
    plot!(p1, c, Dtraj; marker=:diamond, lw=2, label="trajectory")

    p2 = plot(c, P; marker=:circle, lw=2, xlabel="minimum separation", ylabel="density", title="Stationary separation", label="P_ss")
    plot!(p2, c, invD; marker=:diamond, lw=2, label="normalized 1/D_proxy")

    p3 = plot(c, flat; marker=:circle, lw=2, xlabel="minimum separation", ylabel="P_ss * D_proxy", title="Ito flatness diagnostic", legend=false)

    if !isnothing(fit)
        Dinf = fit["Dinf"]
        ycorr = Dproxy .- Dinf
        positive = findall(>(0.0), ycorr)
        p4 = plot(c[positive], ycorr[positive]; xscale=:log10, yscale=:log10, marker=:circle, lw=2, xlabel="minimum separation", ylabel="D_proxy - Dinf", title="Scaling diagnostic", label="proxy")
        if !isempty(positive)
            ref = ycorr[positive][1] .* (c[positive] ./ c[positive][1]).^(-2)
            plot!(p4, c[positive], ref; lw=2, ls=:dash, label="slope -2")
        end
    else
        p4 = plot(title="Scaling diagnostic unavailable", axis=false, legend=false)
    end

    savefig(plot(p1, p2, p3, p4, layout=(2, 2), size=(1300, 950)), output_path)
    return output_path
end

function main()
    args = parse_commandline()
    paths = resolve_inputs(args)
    buffer, skipped = aggregate_results(paths)
    rows = rows_from_buffer(buffer)
    fit_rows = select_fit_rows(rows, args)
    fit = fit_proxy(fit_rows, Float64(buffer["L"]); periodic_fit=Bool(args["periodic_fit"]))
    Dinf_for_slope = isnothing(fit) ? tail_dinf(rows, Int(args["tail_count"])) : fit["Dinf"]
    slope = log_slope(fit_rows, Dinf_for_slope)

    output_dir = abspath(String(args["output_dir"]))
    mkpath(output_dir)
    tag = replace(String(args["save_tag"]), r"[^A-Za-z0-9._-]+" => "-")
    csv_path = joinpath(output_dir, "$(tag)_mobile_aggregate.csv")
    summary_path = joinpath(output_dir, "$(tag)_mobile_summary.txt")
    plot_path = joinpath(output_dir, "$(tag)_mobile.png")
    write_csv(csv_path, rows)
    write_summary(summary_path, buffer, rows, fit, slope, skipped, args)
    maybe_plot(plot_path, rows, fit, args)

    println("Saved mobile-object coupled-SDE aggregate:")
    println("  $(csv_path)")
    println("  $(summary_path)")
    if !args["no_plot"]
        println("  $(plot_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
