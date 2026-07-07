#!/usr/bin/env julia

using ArgParse
using Printf

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--replica_root"
            help = "Directory containing one output directory per ghost-probe replica."
            arg_type = String
            required = true
        "--output_dir"
            help = "Directory for aggregate CSV outputs."
            arg_type = String
            required = true
        "--save_tag"
            help = "Tag written into the aggregate summary."
            arg_type = String
            default = "ghost_probe_aggregate"
    end
    return parse_args(settings)
end

function parse_csv_value(raw::AbstractString)
    text = strip(raw)
    isempty(text) && return NaN
    value = tryparse(Float64, text)
    isnothing(value) ? text : value
end

function read_csv_rows(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && return Dict{String, Any}[]
    header = split(lines[1], ",")
    rows = Dict{String, Any}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ",")
        length(parts) == length(header) || error("Malformed CSV row in $(path): expected $(length(header)) columns, got $(length(parts)).")
        row = Dict{String, Any}()
        for (key, raw) in zip(header, parts)
            row[key] = parse_csv_value(raw)
        end
        row["_source_file"] = path
        push!(rows, row)
    end
    return rows
end

function find_replica_csvs(replica_root::AbstractString, filename::AbstractString)
    paths = String[]
    for (root, _, files) in walkdir(replica_root)
        filename in files || continue
        push!(paths, joinpath(root, filename))
    end
    return sort(paths)
end

function finite_float(row, key::String)
    value = get(row, key, NaN)
    value isa Real || return NaN
    out = Float64(value)
    return isfinite(out) ? out : NaN
end

function finite_values(rows, key::String)
    values = Float64[]
    for row in rows
        value = finite_float(row, key)
        isfinite(value) && push!(values, value)
    end
    return values
end

function mean_and_sem(values)
    n = length(values)
    n > 0 || return NaN, NaN
    mu = sum(values) / n
    n > 1 || return mu, NaN
    var = sum((v - mu)^2 for v in values) / (n - 1)
    return mu, sqrt(var / n)
end

function aggregate_distance_rows(rows)
    sample_counts = [max(finite_float(row, "sample_count"), 0.0) for row in rows]
    total_samples = sum(sample_counts)
    total_samples > 0 || return nothing
    weighted(key) = sum(sample_counts[i] * finite_float(rows[i], key) for i in eachindex(rows)) / total_samples
    function weighted_optional(key, weights=sample_counts)
        keep = [i for i in eachindex(rows) if weights[i] > 0 && isfinite(finite_float(rows[i], key))]
        isempty(keep) && return NaN
        denom = sum(weights[i] for i in keep)
        return sum(weights[i] * finite_float(rows[i], key) for i in keep) / denom
    end

    mean_source = weighted("mean_source")
    Q_source = weighted("Q_source")
    mean_baseline = finite_float(rows[1], "mean_baseline")
    Q_baseline = finite_float(rows[1], "Q_baseline")
    centered_Q_baseline = finite_float(rows[1], "centered_Q_baseline")
    centered_Q_source = Q_source - mean_source^2
    mean_excess = mean_source - mean_baseline
    delta_Q_raw = Q_source - Q_baseline
    delta_Q_centered = centered_Q_source - centered_Q_baseline
    mean_cross_term = 2.0 * mean_baseline * mean_excess
    mean_square_term = mean_excess^2
    probe_noise_counts = [
        isfinite(finite_float(row, "probe_noise_sample_count")) ? max(finite_float(row, "probe_noise_sample_count"), 0.0) : 0.0
        for row in rows
    ]
    total_probe_noise_samples = sum(probe_noise_counts)

    row = Dict{String, Any}(
        "block_id" => round(Int, finite_float(rows[1], "block_id")),
        "production_step" => round(Int, finite_float(rows[1], "production_step")),
        "t_start" => finite_float(rows[1], "t_start"),
        "t_end" => finite_float(rows[1], "t_end"),
        "distance" => finite_float(rows[1], "distance"),
        "distance_over_probe_sigma_f" => finite_float(rows[1], "distance_over_probe_sigma_f"),
        "probe_center_count" => finite_float(rows[1], "probe_center_count"),
        "replica_count" => length(unique(String(row["_source_file"]) for row in rows)),
        "total_sample_count" => total_samples,
        "total_effective_probe_samples" => sum(finite_float(row, "effective_probe_samples") for row in rows),
        "total_probe_noise_sample_count" => total_probe_noise_samples,
        "total_effective_probe_noise_samples" => sum(
            isfinite(finite_float(row, "effective_probe_noise_samples")) ? max(finite_float(row, "effective_probe_noise_samples"), 0.0) : 0.0
            for row in rows
        ),
        "probe_integral" => finite_float(rows[1], "probe_integral"),
        "probe_square_integral" => finite_float(rows[1], "probe_square_integral"),
        "uniform_signal" => finite_float(rows[1], "uniform_signal"),
        "poisson_centered_Q" => finite_float(rows[1], "poisson_centered_Q"),
        "mean_source" => mean_source,
        "mean_baseline" => mean_baseline,
        "mean_excess" => mean_excess,
        "Q_source" => Q_source,
        "Q_baseline" => Q_baseline,
        "centered_Q_source" => centered_Q_source,
        "centered_Q_baseline" => centered_Q_baseline,
        "delta_Q_raw" => delta_Q_raw,
        "delta_Q_centered" => delta_Q_centered,
        "mean_cross_term" => mean_cross_term,
        "mean_square_term" => mean_square_term,
        "reconstructed_delta_Q_raw" => delta_Q_centered + mean_cross_term + mean_square_term,
        "mean_probe_dX" => weighted_optional("mean_probe_dX", probe_noise_counts),
        "mean_probe_dX2" => weighted_optional("mean_probe_dX2", probe_noise_counts),
        "D_probe_noise_sampled" => weighted_optional("D_probe_noise_sampled", probe_noise_counts),
        "D_probe_proxy_raw" => weighted_optional("D_probe_proxy_raw"),
        "D_probe_baseline" => weighted_optional("D_probe_baseline"),
        "delta_D_probe_noise_sampled" => weighted_optional("delta_D_probe_noise_sampled", probe_noise_counts),
        "delta_D_probe_proxy_raw" => weighted_optional("delta_D_probe_proxy_raw"),
        "delta_D_probe_noise_sampled_over_mu_obj2" => weighted_optional("delta_D_probe_noise_sampled_over_mu_obj2", probe_noise_counts),
        "D_probe_noise_sampled_to_proxy_ratio" => weighted_optional("D_probe_noise_sampled_to_proxy_ratio", probe_noise_counts),
        "B_fit" => NaN,
        "A_fit" => NaN,
        "alpha_fit" => NaN,
        "raw_minus_B_fit" => NaN,
    )

    for key in ("mean_source", "Q_source", "delta_Q_raw_uniform", "delta_Q_centered_uniform")
        values = finite_values(rows, key)
        _, sem = mean_and_sem(values)
        out_key = key == "delta_Q_raw_uniform" ? "delta_Q_raw_sem" :
            key == "delta_Q_centered_uniform" ? "delta_Q_centered_sem" : "$(key)_sem"
        row[out_key] = sem
    end
    for key in ("D_probe_noise_sampled", "delta_D_probe_noise_sampled", "delta_D_probe_noise_sampled_over_mu_obj2")
        values = finite_values(rows, key)
        _, sem = mean_and_sem(values)
        row["$(key)_sem"] = sem
    end
    return row
end

function group_rows(rows, keys)
    groups = Dict{Tuple, Vector{Dict{String, Any}}}()
    for row in rows
        key = Tuple(finite_float(row, String(k)) for k in keys)
        push!(get!(groups, key, Dict{String, Any}[]), row)
    end
    return groups
end

function fit_raw_baseline_fixed(rows; alpha::Float64=2.0)
    fit_rows = [row for row in rows if finite_float(row, "distance") > 0 && isfinite(finite_float(row, "Q_source"))]
    length(fit_rows) >= 2 || return nothing
    d = [finite_float(row, "distance") for row in fit_rows]
    y = [finite_float(row, "Q_source") for row in fit_rows]
    X0 = ones(length(d))
    X1 = 1.0 ./ (d .^ alpha)
    coeff = hcat(X0, X1) \ y
    yhat = coeff[1] .* X0 .+ coeff[2] .* X1
    ss_res = sum((y .- yhat).^2)
    ybar = sum(y) / length(y)
    ss_tot = sum((y .- ybar).^2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return Dict(
        "B_fit" => coeff[1],
        "A_fit" => coeff[2],
        "alpha_fit" => alpha,
        "R2_linear" => r2,
        "fit_n" => length(y),
    )
end

function add_fit_to_block!(rows)
    fit = fit_raw_baseline_fixed(rows)
    isnothing(fit) && return nothing
    B = Float64(fit["B_fit"])
    for row in rows
        row["B_fit"] = fit["B_fit"]
        row["A_fit"] = fit["A_fit"]
        row["alpha_fit"] = fit["alpha_fit"]
        row["raw_minus_B_fit"] = finite_float(row, "Q_source") - B
    end
    return fit
end

function aggregate_rows(rows)
    distance_groups = group_rows(rows, ("block_id", "distance"))
    aggregate = Dict{String, Any}[]
    for grouped in values(distance_groups)
        row = aggregate_distance_rows(grouped)
        isnothing(row) || push!(aggregate, row)
    end
    sort!(aggregate; by=row -> (finite_float(row, "block_id"), finite_float(row, "distance")))

    fit_rows = Dict{String, Any}[]
    for (block_id, block_rows) in group_rows(aggregate, ("block_id",))
        fit = add_fit_to_block!(block_rows)
        isnothing(fit) && continue
        first_row = block_rows[1]
        push!(fit_rows, Dict{String, Any}(
            "block_id" => round(Int, block_id[1]),
            "production_step" => round(Int, finite_float(first_row, "production_step")),
            "t_start" => finite_float(first_row, "t_start"),
            "t_end" => finite_float(first_row, "t_end"),
            "B_fit" => fit["B_fit"],
            "A_fit" => fit["A_fit"],
            "alpha_fit" => fit["alpha_fit"],
            "R2_linear" => fit["R2_linear"],
            "fit_n" => fit["fit_n"],
        ))
    end
    sort!(fit_rows; by=row -> finite_float(row, "block_id"))
    return aggregate, fit_rows
end

function final_rows_from_files(paths)
    rows = Dict{String, Any}[]
    for path in paths
        file_rows = read_csv_rows(path)
        isempty(file_rows) && continue
        final_step = maximum(finite_float(row, "production_step") for row in file_rows)
        append!(rows, [row for row in file_rows if finite_float(row, "production_step") == final_step])
    end
    return rows
end

function csv_value(value)
    if value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? @sprintf("%.16e", value) : "NaN"
    end
    return string(value)
end

function write_csv(path::AbstractString, rows, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((csv_value(get(row, col, NaN)) for col in columns), ","))
        end
    end
    return path
end

const AGG_COLUMNS = [
    "block_id", "production_step", "t_start", "t_end", "distance",
    "distance_over_probe_sigma_f", "probe_center_count", "replica_count",
    "total_sample_count", "total_effective_probe_samples", "total_probe_noise_sample_count",
    "total_effective_probe_noise_samples", "probe_integral",
    "probe_square_integral", "uniform_signal", "poisson_centered_Q",
    "mean_source", "mean_source_sem", "mean_baseline", "mean_excess",
    "Q_source", "Q_source_sem", "Q_baseline", "centered_Q_source",
    "centered_Q_baseline", "delta_Q_raw", "delta_Q_raw_sem",
    "delta_Q_centered", "delta_Q_centered_sem", "mean_cross_term",
    "mean_square_term", "reconstructed_delta_Q_raw", "B_fit", "A_fit",
    "alpha_fit", "raw_minus_B_fit", "mean_probe_dX", "mean_probe_dX2",
    "D_probe_noise_sampled", "D_probe_noise_sampled_sem", "D_probe_proxy_raw",
    "D_probe_baseline", "delta_D_probe_noise_sampled", "delta_D_probe_noise_sampled_sem",
    "delta_D_probe_proxy_raw", "delta_D_probe_noise_sampled_over_mu_obj2",
    "delta_D_probe_noise_sampled_over_mu_obj2_sem", "D_probe_noise_sampled_to_proxy_ratio",
]

const FIT_COLUMNS = [
    "block_id", "production_step", "t_start", "t_end", "B_fit", "A_fit",
    "alpha_fit", "R2_linear", "fit_n",
]

function main()
    args = parse_commandline()
    replica_root = abspath(String(args["replica_root"]))
    output_dir = abspath(String(args["output_dir"]))
    isdir(replica_root) || error("replica_root does not exist: $(replica_root)")
    mkpath(output_dir)

    cumulative_paths = find_replica_csvs(replica_root, "ghost_probe_cumulative.csv")
    isempty(cumulative_paths) && error("No ghost_probe_cumulative.csv files found under $(replica_root).")

    all_cumulative_rows = Dict{String, Any}[]
    for path in cumulative_paths
        append!(all_cumulative_rows, read_csv_rows(path))
    end
    cumulative_rows, cumulative_fits = aggregate_rows(all_cumulative_rows)

    final_input_rows = final_rows_from_files(cumulative_paths)
    final_rows, final_fits = aggregate_rows(final_input_rows)

    write_csv(joinpath(output_dir, "ghost_probe_replica_aggregate_cumulative.csv"), cumulative_rows, AGG_COLUMNS)
    write_csv(joinpath(output_dir, "ghost_probe_replica_aggregate_cumulative_fits.csv"), cumulative_fits, FIT_COLUMNS)
    write_csv(joinpath(output_dir, "ghost_probe_replica_aggregate_final.csv"), final_rows, AGG_COLUMNS)
    write_csv(joinpath(output_dir, "ghost_probe_replica_aggregate_final_fits.csv"), final_fits, FIT_COLUMNS)

    open(joinpath(output_dir, "ghost_probe_replica_aggregate_summary.txt"), "w") do io
        println(io, "result_type=coupled_sde_ghost_probe_replica_aggregate")
        println(io, "save_tag=$(args["save_tag"])")
        println(io, "replica_root=$(replica_root)")
        println(io, "output_dir=$(output_dir)")
        println(io, "replica_csv_count=$(length(cumulative_paths))")
        println(io, "cumulative_rows=$(length(cumulative_rows))")
        println(io, "final_rows=$(length(final_rows))")
    end

    println("Aggregated $(length(cumulative_paths)) ghost-probe replica CSVs")
    println("  output_dir=$(output_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
