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

function load_result_and_metadata(path::AbstractString)
    data = JLD2.load(path)
    haskey(data, "result") || error("File does not contain a result dataset: $path")
    result = data["result"]
    get(result, "result_type", "") == "coupled_sde_mobile_objects" ||
        error("Expected coupled_sde_mobile_objects result in $path, got $(get(result, "result_type", "missing")).")
    metadata = get(data, "metadata", Dict{String,Any}())
    return result, metadata
end

function load_result(path::AbstractString)
    result, _ = load_result_and_metadata(path)
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
        "replica_P_mass_sum" => zeros(Float64, n_bins),
        "replica_P_mass_sumsq" => zeros(Float64, n_bins),
        "replica_P_density_sum" => zeros(Float64, n_bins),
        "replica_P_density_sumsq" => zeros(Float64, n_bins),
        "replica_P_density_count" => 0,
        "position_edges" => Float64.(get(result, "locations", Dict("edges" => bins["edges"]))["edges"]),
        "position_centers" => Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]),
        "XA_histogram_counts" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "XB_histogram_counts" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "center_histogram_counts" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_XA_P_density_sum" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_XA_P_density_sumsq" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_XB_P_density_sum" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_XB_P_density_sumsq" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_center_P_density_sum" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_center_P_density_sumsq" => zeros(Float64, length(Float64.(get(result, "locations", Dict("centers" => bins["centers"]))["centers"]))),
        "replica_location_density_count" => 0,
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
    if haskey(result, "locations")
        position_edges = Float64.(result["locations"]["edges"])
        length(position_edges) == length(buf["position_edges"]) || error("Location-bin edge length mismatch for $path.")
        all(isapprox.(position_edges, buf["position_edges"]; rtol=1.0e-10, atol=1.0e-12)) || error("Location bin edges differ for $path.")
    end
end

function accumulate!(buf, result, path)
    compatible!(buf, result, path)
    bins = result["bins"]
    buf["histogram_counts"] .+= Float64.(bins["histogram_counts"])
    buf["bin_counts"] .+= Float64.(bins["bin_counts"])
    buf["sum_delta_rel_sq"] .+= Float64.(bins["sum_delta_rel_sq"])
    buf["sum_Ssum"] .+= Float64.(bins["sum_Ssum"])
    buf["sum_separation_min"] .+= Float64.(bins["sum_separation_min"])
    replica_hist = Float64.(bins["histogram_counts"])
    replica_total = sum(replica_hist)
    if replica_total > 0
        widths = diff(buf["edges"])
        replica_mass = replica_hist ./ replica_total
        replica_density = replica_mass ./ widths
        buf["replica_P_mass_sum"] .+= replica_mass
        buf["replica_P_mass_sumsq"] .+= replica_mass .^ 2
        buf["replica_P_density_sum"] .+= replica_density
        buf["replica_P_density_sumsq"] .+= replica_density .^ 2
        buf["replica_P_density_count"] += 1
    end
    if haskey(result, "locations")
        locations = result["locations"]
        histA = Float64.(locations["XA_histogram_counts"])
        histB = Float64.(locations["XB_histogram_counts"])
        histC = Float64.(locations["center_histogram_counts"])
        buf["XA_histogram_counts"] .+= histA
        buf["XB_histogram_counts"] .+= histB
        buf["center_histogram_counts"] .+= histC
        totalA = sum(histA)
        totalB = sum(histB)
        totalC = sum(histC)
        if totalA > 0 && totalB > 0 && totalC > 0
            position_widths = diff(buf["position_edges"])
            densityA = (histA ./ totalA) ./ position_widths
            densityB = (histB ./ totalB) ./ position_widths
            densityC = (histC ./ totalC) ./ position_widths
            buf["replica_XA_P_density_sum"] .+= densityA
            buf["replica_XA_P_density_sumsq"] .+= densityA .^ 2
            buf["replica_XB_P_density_sum"] .+= densityB
            buf["replica_XB_P_density_sumsq"] .+= densityB .^ 2
            buf["replica_center_P_density_sum"] .+= densityC
            buf["replica_center_P_density_sumsq"] .+= densityC .^ 2
            buf["replica_location_density_count"] += 1
        end
    end
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
    initial_rows = Vector{Dict{String,Any}}()
    for path in paths
        result, metadata = try
            load_result_and_metadata(path)
        catch err
            push!(skipped, "$(path): $(sprint(showerror, err))")
            continue
        end
        if isnothing(buffer)
            buffer = init_buffer(result)
        end
        accumulate!(buffer, result, path)
        push!(initial_rows, initial_condition_row(path, result, metadata))
    end
    isnothing(buffer) && error("No mobile-object coupled-SDE results could be loaded.")
    return buffer, skipped, initial_rows
end

function dict_value(data, key::String, default=missing)
    data isa AbstractDict || return default
    return haskey(data, key) ? data[key] : default
end

function nested_initial_condition(result, metadata)
    result_initial = dict_value(result, "initial_condition", missing)
    result_initial isa AbstractDict && return result_initial
    metadata_initial = dict_value(metadata, "initial_condition", missing)
    metadata_initial isa AbstractDict && return metadata_initial
    return Dict{String,Any}()
end

function initial_condition_row(path::AbstractString, result, metadata)
    params = dict_value(result, "parameters", Dict{String,Any}())
    initial = nested_initial_condition(result, metadata)
    return Dict(
        "source_file" => path,
        "save_tag" => dict_value(metadata, "save_tag", ""),
        "seed" => dict_value(metadata, "seed", dict_value(params, "seed", "")),
        "random_initial_objects" => dict_value(params, "random_initial_objects", ""),
        "initial_min_separation_config" => dict_value(params, "initial_min_separation", missing),
        "initial_max_separation_config" => dict_value(params, "initial_max_separation", missing),
        "hard_min_separation" => dict_value(params, "hard_min_separation", dict_value(metadata, "hard_min_separation", missing)),
        "initial_source" => dict_value(initial, "source", dict_value(metadata, "initial_condition_source", "")),
        "initial_step" => dict_value(initial, "step", missing),
        "initial_time" => dict_value(initial, "time", missing),
        "initial_XA" => dict_value(initial, "XA", missing),
        "initial_XB" => dict_value(initial, "XB", missing),
        "initial_XA_unwrapped" => dict_value(initial, "XA_unwrapped", missing),
        "initial_XB_unwrapped" => dict_value(initial, "XB_unwrapped", missing),
        "initial_separation_oriented" => dict_value(initial, "separation_oriented", missing),
        "initial_separation_min" => dict_value(initial, "separation_min", missing),
        "initial_pair_center" => dict_value(initial, "pair_center", missing),
    )
end

function safe_divide(numer::AbstractVector{Float64}, denom::AbstractVector{Float64})
    out = similar(numer)
    for i in eachindex(numer)
        out[i] = denom[i] > 0 ? numer[i] / denom[i] : NaN
    end
    return out
end

function sample_sem_from_sums(sum_values::AbstractVector{Float64}, sumsq_values::AbstractVector{Float64}, n::Integer)
    sem = similar(sum_values)
    if n <= 1
        fill!(sem, NaN)
        return sem
    end
    for i in eachindex(sum_values)
        mean_value = sum_values[i] / n
        variance = (sumsq_values[i] - n * mean_value^2) / (n - 1)
        variance = max(variance, 0.0)
        sem[i] = sqrt(variance / n)
    end
    return sem
end

function ci95_from_sem(sem::AbstractVector{Float64}, n::Integer)
    n > 1 || return fill(NaN, length(sem))
    return NORMAL_CI95 .* sem
end

function ci95_from_se(se::AbstractVector{Float64})
    return NORMAL_CI95 .* se
end

function aggregate_error_metadata()
    return Dict(
        "P_mass_sem" => "standard error across replica-normalized distance probability masses",
        "P_density_sem" => "standard error across replica-normalized distance probability densities",
        "P_mass_ci95" => "normal-approximate 95% confidence half-width across replica-normalized distance probability masses",
        "P_density_ci95" => "normal-approximate 95% confidence half-width across replica-normalized distance probability densities",
        "P_mass_multinomial_se" => "pooled multinomial standard error from aggregate distance counts; ignores autocorrelation and replica-to-replica variation",
        "P_density_multinomial_se" => "pooled multinomial density standard error from aggregate distance counts; ignores autocorrelation and replica-to-replica variation",
        "P_mass_multinomial_ci95" => "normal-approximate 95% confidence half-width from pooled multinomial distance counts; ignores autocorrelation and replica-to-replica variation",
        "P_density_multinomial_ci95" => "normal-approximate 95% confidence half-width from pooled multinomial distance counts; ignores autocorrelation and replica-to-replica variation",
        "location_density_sem" => "standard error across replica-normalized object-location densities",
        "location_density_ci95" => "normal-approximate 95% confidence half-width across replica-normalized object-location densities",
        "location_density_multinomial_se" => "pooled multinomial density standard error from aggregate location counts; ignores autocorrelation and replica-to-replica variation",
        "location_density_multinomial_ci95" => "normal-approximate 95% confidence half-width from pooled multinomial location counts; ignores autocorrelation and replica-to-replica variation",
    )
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
    replica_density_count = Int(get(buf, "replica_P_density_count", 0))
    P_mass_sem = sample_sem_from_sums(buf["replica_P_mass_sum"], buf["replica_P_mass_sumsq"], replica_density_count)
    P_density_sem = sample_sem_from_sums(buf["replica_P_density_sum"], buf["replica_P_density_sumsq"], replica_density_count)
    P_mass_ci95 = ci95_from_sem(P_mass_sem, replica_density_count)
    P_density_ci95 = ci95_from_sem(P_density_sem, replica_density_count)
    P_mass_multinomial_se = total_hist > 0 ? sqrt.(max.(P_mass .* (1.0 .- P_mass), 0.0) ./ total_hist) : fill(NaN, length(hist))
    P_density_multinomial_se = P_mass_multinomial_se ./ widths
    P_mass_multinomial_ci95 = ci95_from_se(P_mass_multinomial_se)
    P_density_multinomial_ci95 = ci95_from_se(P_density_multinomial_se)

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
            "P_mass_sem" => P_mass_sem[i],
            "P_density_sem" => P_density_sem[i],
            "P_mass_ci95" => P_mass_ci95[i],
            "P_density_ci95" => P_density_ci95[i],
            "P_mass_multinomial_se" => P_mass_multinomial_se[i],
            "P_density_multinomial_se" => P_density_multinomial_se[i],
            "P_mass_multinomial_ci95" => P_mass_multinomial_ci95[i],
            "P_density_multinomial_ci95" => P_density_multinomial_ci95[i],
            "P_error_replicas" => replica_density_count,
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

function location_rows_from_buffer(buf)
    edges = buf["position_edges"]
    centers = buf["position_centers"]
    widths = diff(edges)
    histA = buf["XA_histogram_counts"]
    histB = buf["XB_histogram_counts"]
    histC = buf["center_histogram_counts"]
    totalA = sum(histA)
    totalB = sum(histB)
    totalC = sum(histC)
    replica_location_density_count = Int(get(buf, "replica_location_density_count", 0))
    densityA_sem = sample_sem_from_sums(buf["replica_XA_P_density_sum"], buf["replica_XA_P_density_sumsq"], replica_location_density_count)
    densityB_sem = sample_sem_from_sums(buf["replica_XB_P_density_sum"], buf["replica_XB_P_density_sumsq"], replica_location_density_count)
    densityC_sem = sample_sem_from_sums(buf["replica_center_P_density_sum"], buf["replica_center_P_density_sumsq"], replica_location_density_count)
    densityA_ci95 = ci95_from_sem(densityA_sem, replica_location_density_count)
    densityB_ci95 = ci95_from_sem(densityB_sem, replica_location_density_count)
    densityC_ci95 = ci95_from_sem(densityC_sem, replica_location_density_count)
    rows = Vector{Dict{String,Any}}()
    uniform_density = 1.0 / Float64(buf["L"])
    for i in eachindex(centers)
        massA = totalA > 0 ? histA[i] / totalA : NaN
        massB = totalB > 0 ? histB[i] / totalB : NaN
        massC = totalC > 0 ? histC[i] / totalC : NaN
        densityA = isfinite(massA) ? massA / widths[i] : NaN
        densityB = isfinite(massB) ? massB / widths[i] : NaN
        densityC = isfinite(massC) ? massC / widths[i] : NaN
        densityA_multinomial_se = totalA > 0 ? sqrt(max(massA * (1.0 - massA), 0.0) / totalA) / widths[i] : NaN
        densityB_multinomial_se = totalB > 0 ? sqrt(max(massB * (1.0 - massB), 0.0) / totalB) / widths[i] : NaN
        densityC_multinomial_se = totalC > 0 ? sqrt(max(massC * (1.0 - massC), 0.0) / totalC) / widths[i] : NaN
        densityA_multinomial_ci95 = NORMAL_CI95 * densityA_multinomial_se
        densityB_multinomial_ci95 = NORMAL_CI95 * densityB_multinomial_se
        densityC_multinomial_ci95 = NORMAL_CI95 * densityC_multinomial_se
        push!(rows, Dict(
            "x_left" => edges[i],
            "x_right" => edges[i + 1],
            "x_center" => centers[i],
            "XA_count" => histA[i],
            "XB_count" => histB[i],
            "center_count" => histC[i],
            "XA_P_mass" => massA,
            "XB_P_mass" => massB,
            "center_P_mass" => massC,
            "XA_P_density" => densityA,
            "XB_P_density" => densityB,
            "center_P_density" => densityC,
            "XA_P_density_sem" => densityA_sem[i],
            "XB_P_density_sem" => densityB_sem[i],
            "center_P_density_sem" => densityC_sem[i],
            "XA_P_density_ci95" => densityA_ci95[i],
            "XB_P_density_ci95" => densityB_ci95[i],
            "center_P_density_ci95" => densityC_ci95[i],
            "XA_P_density_multinomial_se" => densityA_multinomial_se,
            "XB_P_density_multinomial_se" => densityB_multinomial_se,
            "center_P_density_multinomial_se" => densityC_multinomial_se,
            "XA_P_density_multinomial_ci95" => densityA_multinomial_ci95,
            "XB_P_density_multinomial_ci95" => densityB_multinomial_ci95,
            "center_P_density_multinomial_ci95" => densityC_multinomial_ci95,
            "location_error_replicas" => replica_location_density_count,
            "uniform_density" => uniform_density,
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
        println(io, "bin_left,bin_right,bin_center,mean_separation_min,count,histogram_count,P_mass,P_density,P_mass_sem,P_density_sem,P_mass_ci95,P_density_ci95,P_mass_multinomial_se,P_density_multinomial_se,P_mass_multinomial_ci95,P_density_multinomial_ci95,P_error_replicas,mean_delta_rel_sq,mean_Ssum,D_rel_traj,D_rel_proxy,inv_D_proxy_density,P_times_D_proxy")
        for r in rows
            @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n",
                r["bin_left"], r["bin_right"], r["bin_center"], r["mean_separation_min"], r["count"], r["histogram_count"],
                r["P_mass"], r["P_density"], r["P_mass_sem"], r["P_density_sem"], r["P_mass_ci95"], r["P_density_ci95"],
                r["P_mass_multinomial_se"], r["P_density_multinomial_se"], r["P_mass_multinomial_ci95"], r["P_density_multinomial_ci95"],
                Int(r["P_error_replicas"]), r["mean_delta_rel_sq"], r["mean_Ssum"], r["D_rel_traj"], r["D_rel_proxy"],
                r["inv_D_proxy_density"], r["P_times_D_proxy"])
        end
    end
    return path
end

function write_location_csv(path, rows)
    open(path, "w") do io
        println(io, "x_left,x_right,x_center,XA_count,XB_count,center_count,XA_P_mass,XB_P_mass,center_P_mass,XA_P_density,XB_P_density,center_P_density,XA_P_density_sem,XB_P_density_sem,center_P_density_sem,XA_P_density_ci95,XB_P_density_ci95,center_P_density_ci95,XA_P_density_multinomial_se,XB_P_density_multinomial_se,center_P_density_multinomial_se,XA_P_density_multinomial_ci95,XB_P_density_multinomial_ci95,center_P_density_multinomial_ci95,location_error_replicas,uniform_density")
        for r in rows
            @printf(io, "%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%d,%.16e\n",
                r["x_left"], r["x_right"], r["x_center"], r["XA_count"], r["XB_count"], r["center_count"],
                r["XA_P_mass"], r["XB_P_mass"], r["center_P_mass"], r["XA_P_density"], r["XB_P_density"],
                r["center_P_density"], r["XA_P_density_sem"], r["XB_P_density_sem"], r["center_P_density_sem"],
                r["XA_P_density_ci95"], r["XB_P_density_ci95"], r["center_P_density_ci95"],
                r["XA_P_density_multinomial_se"], r["XB_P_density_multinomial_se"], r["center_P_density_multinomial_se"],
                r["XA_P_density_multinomial_ci95"], r["XB_P_density_multinomial_ci95"], r["center_P_density_multinomial_ci95"],
                Int(r["location_error_replicas"]), r["uniform_density"])
        end
    end
    return path
end

function csv_field(value)
    if value === missing || isnothing(value)
        return ""
    end
    text = string(value)
    return "\"" * replace(text, "\"" => "\"\"") * "\""
end

function finite_or_nan(value)
    if value === missing || isnothing(value)
        return NaN
    elseif value isa Number
        return Float64(value)
    elseif value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return NaN
        try
            return parse(Float64, stripped)
        catch
            return NaN
        end
    end
    return NaN
end

function number_field(value)
    return @sprintf("%.16e", finite_or_nan(value))
end

function write_initial_conditions_csv(path, rows)
    open(path, "w") do io
        println(io, "source_file,save_tag,seed,random_initial_objects,initial_min_separation_config,initial_max_separation_config,hard_min_separation,initial_source,initial_step,initial_time,initial_XA,initial_XB,initial_XA_unwrapped,initial_XB_unwrapped,initial_separation_oriented,initial_separation_min,initial_pair_center")
        for r in rows
            fields = [
                csv_field(r["source_file"]),
                csv_field(r["save_tag"]),
                csv_field(r["seed"]),
                csv_field(r["random_initial_objects"]),
                number_field(r["initial_min_separation_config"]),
                number_field(r["initial_max_separation_config"]),
                number_field(r["hard_min_separation"]),
                csv_field(r["initial_source"]),
                number_field(r["initial_step"]),
                number_field(r["initial_time"]),
                number_field(r["initial_XA"]),
                number_field(r["initial_XB"]),
                number_field(r["initial_XA_unwrapped"]),
                number_field(r["initial_XB_unwrapped"]),
                number_field(r["initial_separation_oriented"]),
                number_field(r["initial_separation_min"]),
                number_field(r["initial_pair_center"]),
            ]
            println(io, join(fields, ","))
        end
    end
    return path
end

function max_abs_location_density_deviation(location_rows, key::String)
    vals = Float64[]
    for r in location_rows
        density = Float64(r[key])
        uniform_density = Float64(r["uniform_density"])
        if isfinite(density) && uniform_density > 0
            push!(vals, abs(density - uniform_density) / uniform_density)
        end
    end
    isempty(vals) && return NaN
    return maximum(vals)
end

function write_summary(path, buf, rows, location_rows, fit, slope, skipped, args, initial_rows)
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
        println(io, "initial_condition_records=$(length(initial_rows))")
        println(io, "P_error_replicas=$(get(buf, "replica_P_density_count", 0))")
        println(io, "location_error_replicas=$(get(buf, "replica_location_density_count", 0))")
        println(io, "P_error_method=replica_sem_over_normalized_histograms")
        println(io, "P_ci95_method=normal_approx_1.96_times_replica_sem")
        println(io, "P_count_error_method=pooled_multinomial_se_ignores_autocorrelation")
        println(io, "P_count_ci95_method=normal_approx_1.96_times_pooled_multinomial_se")
        println(io, "location_error_method=replica_sem_over_normalized_location_histograms")
        println(io, "location_ci95_method=normal_approx_1.96_times_replica_sem")
        println(io, "location_count_error_method=pooled_multinomial_se_ignores_autocorrelation")
        println(io, "location_count_ci95_method=normal_approx_1.96_times_pooled_multinomial_se")
        println(io, "periodic_fit=$(args["periodic_fit"])")
        println(io, "P_times_D_proxy_std_in_fit_window=$(flatness_std)")
        println(io, "XA_location_max_abs_relative_deviation_from_uniform=$(max_abs_location_density_deviation(location_rows, "XA_P_density"))")
        println(io, "XB_location_max_abs_relative_deviation_from_uniform=$(max_abs_location_density_deviation(location_rows, "XB_P_density"))")
        println(io, "center_location_max_abs_relative_deviation_from_uniform=$(max_abs_location_density_deviation(location_rows, "center_P_density"))")
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

function optional_error_vector(rows, indices, key::String)
    errors = Float64[]
    has_positive_error = false
    for i in indices
        value = Float64(get(rows[i], key, NaN))
        if isfinite(value) && value >= 0
            push!(errors, value)
            has_positive_error |= value > 0
        else
            push!(errors, 0.0)
        end
    end
    return has_positive_error ? errors : nothing
end

function optional_ci95_or_sem_vector(rows, indices, ci_key::String, sem_key::String)
    ci = optional_error_vector(rows, indices, ci_key)
    if !isnothing(ci)
        return ci, "95% CI"
    end
    sem = optional_error_vector(rows, indices, sem_key)
    if !isnothing(sem)
        return sem, "replica SEM"
    end
    return nothing, ""
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
    Perr, Perr_label = optional_ci95_or_sem_vector(rows, nonempty, "P_density_ci95", "P_density_sem")
    invD = [Float64(rows[i]["inv_D_proxy_density"]) for i in nonempty]
    flat = [Float64(rows[i]["P_times_D_proxy"]) for i in nonempty]

    p1 = plot(c, Dproxy; marker=:circle, lw=2, xlabel="minimum separation", ylabel="D_rel", title="Conditional diffusivity", label="proxy", legend=OUTSIDE_LEGEND)
    plot!(p1, c, Dtraj; marker=:diamond, lw=2, label="trajectory")

    p2 = if isnothing(Perr)
        plot(c, P; marker=:circle, lw=2, xlabel="minimum separation", ylabel="density", title="Stationary separation", label="P_ss", legend=OUTSIDE_LEGEND)
    else
        plot(c, P; yerror=Perr, marker=:circle, lw=2, xlabel="minimum separation", ylabel="density", title="Stationary separation", label="P_ss $(Perr_label)", legend=OUTSIDE_LEGEND)
    end
    plot!(p2, c, invD; marker=:diamond, lw=2, label="normalized 1/D_proxy")

    p3 = plot(c, flat; marker=:circle, lw=2, xlabel="minimum separation", ylabel="P_ss * D_proxy", title="Ito flatness diagnostic", legend=false)

    if !isnothing(fit)
        Dinf = fit["Dinf"]
        ycorr = Dproxy .- Dinf
        positive = findall(>(0.0), ycorr)
        p4 = plot(c[positive], ycorr[positive]; xscale=:log10, yscale=:log10, marker=:circle, lw=2, xlabel="minimum separation", ylabel="D_proxy - Dinf", title="Scaling diagnostic", label="proxy", legend=OUTSIDE_LEGEND)
        if !isempty(positive)
            ref = ycorr[positive][1] .* (c[positive] ./ c[positive][1]).^(-2)
            plot!(p4, c[positive], ref; lw=2, ls=:dash, label="slope -2")
        end
    else
        p4 = plot(title="Scaling diagnostic unavailable", axis=false, legend=false)
    end

    savefig(plot(p1, p2, p3, p4, layout=(2, 2), size=(1550, 980)), output_path)
    return output_path
end

function main()
    args = parse_commandline()
    paths = resolve_inputs(args)
    buffer, skipped, initial_rows = aggregate_results(paths)
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
    initial_csv_path = joinpath(output_dir, "$(tag)_mobile_initial_conditions.csv")
    jld2_path = joinpath(output_dir, "$(tag)_mobile_aggregate.jld2")
    summary_path = joinpath(output_dir, "$(tag)_mobile_summary.txt")
    plot_path = joinpath(output_dir, "$(tag)_mobile.png")
    write_csv(csv_path, rows)
    write_location_csv(location_csv_path, location_rows)
    write_initial_conditions_csv(initial_csv_path, initial_rows)
    jldsave(jld2_path; rows=rows, location_rows=location_rows, initial_rows=initial_rows, fit=fit, slope=slope, skipped=skipped, error_metadata=aggregate_error_metadata())
    write_summary(summary_path, buffer, rows, location_rows, fit, slope, skipped, args, initial_rows)
    maybe_plot(plot_path, rows, fit, args)

    println("Saved mobile-object coupled-SDE aggregate:")
    println("  $(csv_path)")
    println("  $(location_csv_path)")
    println("  $(initial_csv_path)")
    println("  $(jld2_path)")
    println("  $(summary_path)")
    if !args["no_plot"]
        println("  $(plot_path)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
