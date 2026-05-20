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
            help = "Path to one coupled-SDE fixed-separation JLD2 result"
            arg_type = String
        "--state_list"
            help = "Newline-delimited list of coupled-SDE fixed-separation result files"
            arg_type = String
        "--state_dir"
            help = "Directory searched recursively for coupled-SDE JLD2 result files"
            arg_type = String
        "--output_dir"
            help = "Directory for CSV, summary, and plots"
            arg_type = String
            default = "analysis_outputs/coupled_sde_active_objects/fixed_separation"
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "coupled_sde_fixed"
        "--fit_min"
            help = "Minimum separation included in fit"
            arg_type = Float64
        "--fit_max"
            help = "Maximum separation included in fit"
            arg_type = Float64
        "--tail_count"
            help = "Number of largest separations averaged for a fallback Dinf estimate"
            arg_type = Int
            default = 3
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
    get(result, "result_type", "") == "coupled_sde_fixed_separation" ||
        error("Expected coupled_sde_fixed_separation result in $path, got $(get(result, "result_type", "missing")).")
    return result
end

function init_buffer(result)
    params = result["parameters"]
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
        "n_replicas" => 0,
        "sample_count" => 0,
        "sum_SA" => 0.0,
        "sum_SB" => 0.0,
        "sum_SA2" => 0.0,
        "sum_SB2" => 0.0,
        "sum_SASB" => 0.0,
        "sum_Ssum" => 0.0,
        "sum_Ssum2" => 0.0,
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
end

function accumulate!(buf, result, path)
    compatible!(buf, result, path)
    sums = result["sums"]
    buf["n_replicas"] += 1
    buf["sample_count"] += Int(result["sample_count"])
    buf["sum_SA"] += Float64(sums["SA"])
    buf["sum_SB"] += Float64(sums["SB"])
    buf["sum_SA2"] += Float64(sums["SA2"])
    buf["sum_SB2"] += Float64(sums["SB2"])
    buf["sum_SASB"] += Float64(sums["SASB"])
    buf["sum_Ssum"] += Float64(sums["Ssum"])
    buf["sum_Ssum2"] += Float64(sums["Ssum2"])
    push!(buf["source_files"], path)
    return buf
end

function aggregate_results(paths)
    isempty(paths) && error("No JLD2 files found.")
    buffers = Dict{Float64,Dict{String,Any}}()
    skipped = String[]
    for path in paths
        result = try
            load_result(path)
        catch err
            push!(skipped, "$(path): $(sprint(showerror, err))")
            continue
        end
        sep = Float64(result["separation_min"])
        if !haskey(buffers, sep)
            buffers[sep] = init_buffer(result)
        end
        accumulate!(buffers[sep], result, path)
    end
    isempty(buffers) && error("No fixed-separation coupled-SDE results could be loaded.")
    return buffers, skipped
end

function rows_from_buffers(buffers)
    rows = Vector{Dict{String,Any}}()
    for sep in sort(collect(keys(buffers)))
        buf = buffers[sep]
        count = Int(buf["sample_count"])
        mean_value(sum_key) = count > 0 ? Float64(buf[sum_key]) / count : NaN
        mean_Ssum = mean_value("sum_Ssum")
        mean_SA = mean_value("sum_SA")
        mean_SB = mean_value("sum_SB")
        mean_SASB = mean_value("sum_SASB")
        cov_SA_SB = mean_SASB - mean_SA * mean_SB
        mu_obj = Float64(buf["mu_obj"])
        D_rel_proxy = 0.5 * mu_obj^2 * mean_Ssum
        push!(rows, Dict(
            "separation" => sep,
            "L" => Float64(buf["L"]),
            "rho0" => Float64(buf["rho0"]),
            "N" => Int(buf["N"]),
            "mu_obj" => mu_obj,
            "n_replicas" => Int(buf["n_replicas"]),
            "sample_count" => count,
            "mean_SA" => mean_SA,
            "mean_SB" => mean_SB,
            "mean_SA2" => mean_value("sum_SA2"),
            "mean_SB2" => mean_value("sum_SB2"),
            "mean_SASB" => mean_SASB,
            "cov_SA_SB" => cov_SA_SB,
            "mean_Ssum" => mean_Ssum,
            "D_rel_proxy" => D_rel_proxy,
            "D_rel_proxy_over_mu_obj2" => mu_obj > 0 ? D_rel_proxy / mu_obj^2 : NaN,
        ))
    end
    return rows
end

function select_fit_rows(rows, args)
    fit_rows = [r for r in rows if isfinite(r["D_rel_proxy"]) && r["separation"] > 0]
    if haskey(args, "fit_min") && !isnothing(args["fit_min"])
        fit_rows = [r for r in fit_rows if r["separation"] >= Float64(args["fit_min"])]
    end
    if haskey(args, "fit_max") && !isnothing(args["fit_max"])
        fit_rows = [r for r in fit_rows if r["separation"] <= Float64(args["fit_max"])]
    end
    return fit_rows
end

function fit_proxy(rows; periodic_fit::Bool=false)
    length(rows) >= 2 || return nothing
    x = Float64[]
    y = Float64[]
    for r in rows
        d = Float64(r["separation"])
        L = Float64(r["L"])
        push!(x, periodic_fit ? (1.0 / d^2 + 1.0 / (L - d)^2) : 1.0 / d^2)
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
    sorted_rows = sort(rows; by=r -> r["separation"])
    n = min(max(tail_count, 1), length(sorted_rows))
    tail = sorted_rows[(end - n + 1):end]
    return sum(Float64(r["D_rel_proxy"]) for r in tail) / n
end

function log_slope(rows, Dinf::Real)
    xs = Float64[]
    ys = Float64[]
    for r in rows
        y = Float64(r["D_rel_proxy"]) - Float64(Dinf)
        if r["separation"] > 0 && y > 0
            push!(xs, log(Float64(r["separation"])))
            push!(ys, log(y))
        end
    end
    length(xs) >= 2 || return nothing
    X = hcat(ones(length(xs)), xs)
    coeff = X \ ys
    return Dict("intercept" => coeff[1], "slope" => coeff[2], "n_fit" => length(xs))
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, "separation,L,rho0,N,mu_obj,n_replicas,sample_count,mean_SA,mean_SB,mean_SA2,mean_SB2,mean_SASB,cov_SA_SB,mean_Ssum,D_rel_proxy,D_rel_proxy_over_mu_obj2")
        for r in rows
            @printf(io, "%.16e,%.16e,%.16e,%d,%.16e,%d,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                r["separation"], r["L"], r["rho0"], r["N"], r["mu_obj"], r["n_replicas"], r["sample_count"],
                r["mean_SA"], r["mean_SB"], r["mean_SA2"], r["mean_SB2"], r["mean_SASB"], r["cov_SA_SB"],
                r["mean_Ssum"], r["D_rel_proxy"], r["D_rel_proxy_over_mu_obj2"])
        end
    end
    return path
end

function write_summary(path, rows, fit, slope, skipped, args)
    open(path, "w") do io
        println(io, "num_separations=$(length(rows))")
        println(io, "num_input_files=$(sum(r["n_replicas"] for r in rows))")
        println(io, "periodic_fit=$(args["periodic_fit"])")
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
    d = [Float64(r["separation"]) for r in rows]
    D = [Float64(r["D_rel_proxy"]) for r in rows]
    p1 = plot(d, D; marker=:circle, lw=2, xlabel="separation", ylabel="D_rel_proxy", title="Fixed-object proxy diffusivity", legend=false)
    if !isnothing(fit)
        Dinf = fit["Dinf"]
        ycorr = D .- Dinf
        positive = findall(>(0.0), ycorr)
        p2 = plot(d[positive], ycorr[positive]; xscale=:log10, yscale=:log10, marker=:circle, lw=2, xlabel="separation", ylabel="D_rel_proxy - Dinf", title="Scaling diagnostic", label="data")
        if !isempty(positive)
            ref = ycorr[positive][1] .* (d[positive] ./ d[positive][1]).^(-2)
            plot!(p2, d[positive], ref; lw=2, ls=:dash, label="slope -2")
        end
        savefig(plot(p1, p2, layout=(2, 1), size=(1100, 850)), output_path)
    else
        savefig(p1, output_path)
    end
    return output_path
end

function main()
    args = parse_commandline()
    paths = resolve_inputs(args)
    buffers, skipped = aggregate_results(paths)
    rows = rows_from_buffers(buffers)
    fit_rows = select_fit_rows(rows, args)
    fit = fit_proxy(fit_rows; periodic_fit=Bool(args["periodic_fit"]))
    Dinf_for_slope = isnothing(fit) ? tail_dinf(rows, Int(args["tail_count"])) : fit["Dinf"]
    slope = log_slope(fit_rows, Dinf_for_slope)

    output_dir = abspath(String(args["output_dir"]))
    mkpath(output_dir)
    tag = replace(String(args["save_tag"]), r"[^A-Za-z0-9._-]+" => "-")
    csv_path = joinpath(output_dir, "$(tag)_fixed_separation_aggregate.csv")
    jld2_path = joinpath(output_dir, "$(tag)_fixed_separation_aggregate.jld2")
    summary_path = joinpath(output_dir, "$(tag)_fixed_separation_summary.txt")
    plot_path = joinpath(output_dir, "$(tag)_fixed_separation.png")
    write_csv(csv_path, rows)
    jldsave(jld2_path; rows=rows, fit=fit, slope=slope, skipped=skipped)
    write_summary(summary_path, rows, fit, slope, skipped, args)
    maybe_plot(plot_path, rows, fit, args)

    println("Saved fixed-separation coupled-SDE aggregate:")
    println("  $(csv_path)")
    println("  $(jld2_path)")
    println("  $(summary_path)")
    if !args["no_plot"]
        println("  $(plot_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
