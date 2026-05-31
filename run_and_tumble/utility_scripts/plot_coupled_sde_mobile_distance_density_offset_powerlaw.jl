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

function parse_commandline()
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--aggregate"
            help = "Path to one *_mobile_aggregate.jld2 file"
            arg_type = String
        "--analysis_dir"
            help = "Directory searched for newest *_mobile_aggregate.jld2"
            arg_type = String
        "--out_dir"
            help = "Directory for output PNG and summary"
            arg_type = String
        "--save_tag"
            help = "Output filename tag"
            arg_type = String
            default = "coupled_sde_mobile"
        "--pinf"
            help = "Use this Pinf instead of scanning for a best offset"
            arg_type = Float64
        "--pinf_tail_count"
            help = "Use the mean P_density over the largest N distances in the fitted window as Pinf"
            arg_type = Int
        "--scan_points"
            help = "Number of candidate Pinf values in the offset scan"
            arg_type = Int
            default = 4000
        "--scan_width_factor"
            help = "Scan Pinf from max(P)+eps to max(P)+factor*(max(P)-min(P))"
            arg_type = Float64
            default = 2.0
        "--tail_counts"
            help = "Comma-separated tail-window sizes for Pinf sensitivity diagnostics"
            arg_type = String
            default = "4,8,12,16,24,32"
        "--reference_slope"
            help = "Reference power-law slope plotted through the first fitted point"
            arg_type = Float64
            default = -2.0
        "--fixed_slope"
            help = "Also fit a fixed-slope power law by scanning Pinf; useful for testing a hypothesized exponent such as -2"
            arg_type = Float64
        "--fixed_slope_scan_min_points"
            help = "Minimum points required in the fixed-slope Pinf scan. Default: 10"
            arg_type = Int
            default = 10
        "--no_reference"
            help = "Disable the reference slope line"
            action = :store_true
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

function sanitize_tag(tag::AbstractString)
    safe = replace(String(tag), r"[^A-Za-z0-9._-]+" => "-")
    isempty(safe) ? "coupled_sde_mobile" : safe
end

function optional_error_value(row, key::String)
    value = Float64(get(row, key, NaN))
    return isfinite(value) && value >= 0 ? value : 0.0
end

function display_error_value(row, ci_key::String, sem_key::String)
    ci = optional_error_value(row, ci_key)
    ci > 0 && return ci
    return optional_error_value(row, sem_key)
end

function fit_sem_value(row, sem_key::String, ci_key::String)
    sem = optional_error_value(row, sem_key)
    sem > 0 && return sem
    ci = optional_error_value(row, ci_key)
    return ci > 0 ? ci / NORMAL_CI95 : 0.0
end

function log_safe_yerror(y::Vector{Float64}, err::Vector{Float64})
    safe = similar(err)
    for i in eachindex(err)
        upper = max(y[i] * (1.0 - 1.0e-9), 0.0)
        safe[i] = clamp(err[i], 0.0, upper)
    end
    return any(>(0.0), safe) ? safe : nothing
end

function present_error_vector(values::Vector{Float64})
    any(value -> isfinite(value) && value > 0, values) ? values : nothing
end

function finite_density(rows)
    d = Float64[]
    p = Float64[]
    p_display_error = Float64[]
    p_fit_sem = Float64[]
    has_ci = false
    has_sem = false
    for row in rows
        x = Float64(row["bin_center"])
        y = Float64(row["P_density"])
        c = Float64(row["count"])
        if isfinite(x) && isfinite(y) && isfinite(c) && x > 0 && y > 0 && c > 0
            push!(d, x)
            push!(p, y)
            push!(p_display_error, display_error_value(row, "P_density_ci95", "P_density_sem"))
            push!(p_fit_sem, fit_sem_value(row, "P_density_sem", "P_density_ci95"))
            has_ci |= optional_error_value(row, "P_density_ci95") > 0
            has_sem |= optional_error_value(row, "P_density_sem") > 0
        end
    end
    isempty(d) && error("No positive finite P_density rows found.")
    order = sortperm(d)
    ordered_p = p[order]
    ordered_display_error = present_error_vector(p_display_error[order])
    ordered_fit_sem = present_error_vector(p_fit_sem[order])
    error_label = has_ci ? "95% CI" : (has_sem ? "replica SEM" : "")
    return d[order], ordered_p, ordered_display_error, ordered_fit_sem, error_label
end

function linear_fit(x::Vector{Float64}, y::Vector{Float64}; sigma=nothing)
    length(x) >= 2 || return nothing
    if !isnothing(sigma)
        keep = [isfinite(x[i]) && isfinite(y[i]) && isfinite(sigma[i]) && sigma[i] > 0 for i in eachindex(x)]
        if count(keep) >= 2
            xw = x[keep]
            yw = y[keep]
            sw = sigma[keep]
            weights = 1.0 ./ (sw .^ 2)
            X = hcat(ones(length(xw)), xw)
            sqrtw = sqrt.(weights)
            coeff = (X .* reshape(sqrtw, :, 1)) \ (yw .* sqrtw)
            yhat = X * coeff
            ymean = sum(weights .* yw) / sum(weights)
            ss_res = sum(weights .* (yw .- yhat) .^ 2)
            ss_tot = sum(weights .* (yw .- ymean) .^ 2)
            r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
            reduced_chi2 = length(xw) > 2 ? ss_res / (length(xw) - 2) : NaN
            return (intercept=coeff[1], slope=coeff[2], r2=r2, n=length(xw), weighted=true, reduced_chi2=reduced_chi2)
        end
    end
    X = hcat(ones(length(x)), x)
    coeff = X \ y
    yhat = X * coeff
    ymean = sum(y) / length(y)
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- ymean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (intercept=coeff[1], slope=coeff[2], r2=r2, n=length(x), weighted=false, reduced_chi2=NaN)
end

function fixed_slope_fit(
    d::Vector{Float64},
    p::Vector{Float64},
    pinf::Float64,
    slope::Float64;
    p_sem=nothing,
    min_points::Int=2,
    dof_params::Int=1,
)
    diff = pinf .- p
    keep = (d .> 0) .& (diff .> 0) .& isfinite.(diff)
    count(keep) >= max(min_points, 2) || return nothing
    x = log.(d[keep])
    y = log.(diff[keep])
    n = length(x)

    if !isnothing(p_sem)
        sigma_logdiff = p_sem[keep] ./ diff[keep]
        valid = [isfinite(x[i]) && isfinite(y[i]) && isfinite(sigma_logdiff[i]) && sigma_logdiff[i] > 0 for i in eachindex(x)]
        if count(valid) >= max(min_points, 2)
            xw = x[valid]
            yw = y[valid]
            sw = sigma_logdiff[valid]
            weights = 1.0 ./ (sw .^ 2)
            intercept = sum(weights .* (yw .- slope .* xw)) / sum(weights)
            yhat = intercept .+ slope .* xw
            ymean = sum(weights .* yw) / sum(weights)
            ss_res = sum(weights .* (yw .- yhat) .^ 2)
            ss_tot = sum(weights .* (yw .- ymean) .^ 2)
            r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
            dof = max(length(xw) - dof_params, 1)
            reduced_chi2 = ss_res / dof
            return (
                intercept=intercept,
                slope=slope,
                r2=r2,
                n=length(xw),
                weighted=true,
                reduced_chi2=reduced_chi2,
                ss_res=ss_res,
                ss_tot=ss_tot,
                pinf=pinf,
            )
        end
    end

    intercept = sum(y .- slope .* x) / n
    yhat = intercept .+ slope .* x
    ymean = sum(y) / n
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- ymean) .^ 2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : NaN
    return (
        intercept=intercept,
        slope=slope,
        r2=r2,
        n=n,
        weighted=false,
        reduced_chi2=NaN,
        ss_res=ss_res,
        ss_tot=ss_tot,
        pinf=pinf,
    )
end

function offset_fit(d::Vector{Float64}, p::Vector{Float64}, pinf::Float64; p_sem=nothing)
    diff = pinf .- p
    keep = (d .> 0) .& (diff .> 0) .& isfinite.(diff)
    count(keep) >= 2 || return nothing
    sigma_logdiff = isnothing(p_sem) ? nothing : p_sem[keep] ./ diff[keep]
    return linear_fit(log.(d[keep]), log.(diff[keep]); sigma=sigma_logdiff)
end

function fixed_fit_score(fit)
    fit.weighted && isfinite(fit.reduced_chi2) && return fit.reduced_chi2
    return fit.ss_res
end

function scan_pinf_fixed_slope(
    d::Vector{Float64},
    p::Vector{Float64},
    slope::Float64;
    p_sem=nothing,
    scan_points::Int=4000,
    scan_width_factor::Float64=2.0,
    min_points::Int=10,
)
    scan_points >= 2 || error("--scan_points must be at least 2.")
    min_points >= 2 || error("--fixed_slope_scan_min_points must be at least 2.")
    pmin = minimum(p)
    pmax = maximum(p)
    scale = max(pmax - pmin, eps(Float64) * max(abs(pmax), 1.0))
    lower = pmin + 1.0e-9 * scale
    upper = pmax + max(scan_width_factor, 1.0e-9) * scale
    best = nothing
    for pinf in range(lower, upper; length=scan_points)
        fit = fixed_slope_fit(d, p, pinf, slope; p_sem=p_sem, min_points=min_points, dof_params=2)
        isnothing(fit) && continue
        if isnothing(best) || fixed_fit_score(fit) < fixed_fit_score(best.fit)
            best = (pinf=pinf, fit=fit)
        end
    end
    isnothing(best) && error("Could not fit any fixed-slope Pinf candidate with at least $(min_points) points.")
    return best
end

function scan_pinf(d::Vector{Float64}, p::Vector{Float64}; p_sem=nothing, scan_points::Int=4000, scan_width_factor::Float64=2.0)
    scan_points >= 2 || error("--scan_points must be at least 2.")
    pmin = minimum(p)
    pmax = maximum(p)
    scale = max(pmax - pmin, eps(Float64) * max(abs(pmax), 1.0))
    lower = pmax + 1.0e-9 * scale
    upper = pmax + max(scan_width_factor, 1.0e-9) * scale
    best = nothing
    for pinf in range(lower, upper; length=scan_points)
        fit = offset_fit(d, p, pinf; p_sem=p_sem)
        isnothing(fit) && continue
        if isnothing(best) || fit.r2 > best.fit.r2
            best = (pinf=pinf, fit=fit)
        end
    end
    isnothing(best) && error("Could not fit any scanned Pinf candidate.")
    return best
end

function parse_tail_counts(spec::AbstractString)
    counts = Int[]
    for token in split(spec, ",")
        stripped = strip(token)
        isempty(stripped) && continue
        value = parse(Int, stripped)
        value > 0 || error("Tail counts must be positive. Got $(value).")
        push!(counts, value)
    end
    isempty(counts) && error("No valid tail counts in $(spec).")
    return counts
end

function tail_diagnostics(d::Vector{Float64}, p::Vector{Float64}, counts::Vector{Int}; p_sem=nothing)
    rows = []
    for count in counts
        k = min(count, length(p))
        pinf = sum(@view p[(end - k + 1):end]) / k
        fit = offset_fit(d, p, pinf; p_sem=p_sem)
        push!(rows, (tail_count=k, pinf=pinf, fit=fit))
    end
    return rows
end

function tail_mean_pinf(p::Vector{Float64}, count::Int)
    count > 0 || error("--pinf_tail_count must be positive. Got $(count).")
    k = min(count, length(p))
    return sum(@view p[(end - k + 1):end]) / k, k
end

function save_summary(path::AbstractString, diagnostics)
    open(path, "w") do io
        for (key, value) in diagnostics
            println(io, "$(key)=$(value)")
        end
    end
    return path
end

function fit_label(fit)
    isnothing(fit) && return ""
    if fit.weighted
        return @sprintf("weighted slope %.4g, wR2 %.4g", fit.slope, fit.r2)
    end
    return @sprintf("slope %.4g, R2 %.4g", fit.slope, fit.r2)
end

function fixed_fit_label(fit)
    if fit.weighted
        return @sprintf("fixed slope %.4g, wR2 %.4g, chi2 %.3g", fit.slope, fit.r2, fit.reduced_chi2)
    end
    return @sprintf("fixed slope %.4g, R2 %.4g", fit.slope, fit.r2)
end

function slope_filename_label(slope::Float64)
    label = slope < 0 ? "minus$(abs(slope))" : "$(slope)"
    return replace(label, "." => "p", "+" => "")
end

function fixed_slope_plot(
    d::Vector{Float64},
    p::Vector{Float64},
    p_error,
    error_label::String,
    fit,
    d_min::Float64,
    fixed_slope::Float64,
)
    diff = fit.pinf .- p
    keep = diff .> 0
    diff_error = isnothing(p_error) ? nothing : log_safe_yerror(diff[keep], p_error[keep])
    fit_curve = exp(fit.intercept) .* d[keep] .^ fixed_slope

    title = @sprintf("Best fixed-slope offset: slope %.4g, Pinf=%.8g", fixed_slope, fit.pinf)
    plt = if isnothing(diff_error)
        plot(
            d[keep],
            diff[keep];
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="Pinf - P_ss(d)",
            title=title,
            label="Pinf - P_ss(d)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    else
        plot(
            d[keep],
            diff[keep];
            yerror=diff_error,
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="Pinf - P_ss(d)",
            title=title,
            label="Pinf - P_ss(d), $(error_label)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    end
    plot!(plt, d[keep], fit_curve; lw=2, ls=:dash, color=:black, label=fixed_fit_label(fit))
    vline!(plt, [d_min]; lw=1, ls=:dot, color=:gray, label="minimum P")
    return plt
end

function main()
    args = parse_commandline()
    aggregate_path = resolve_aggregate(args)
    data = JLD2.load(aggregate_path)
    haskey(data, "rows") || error("Aggregate does not contain rows: $(aggregate_path)")

    all_d, all_p, all_error, all_fit_sem, error_label = finite_density(data["rows"])
    min_idx = argmin(all_p)
    d_min = all_d[min_idx]
    p_min = all_p[min_idx]
    d = all_d[min_idx:end]
    p = all_p[min_idx:end]
    p_error = isnothing(all_error) ? nothing : all_error[min_idx:end]
    p_fit_sem = isnothing(all_fit_sem) ? nothing : all_fit_sem[min_idx:end]

    has_fixed_pinf = haskey(args, "pinf") && !isnothing(args["pinf"])
    has_tail_pinf = haskey(args, "pinf_tail_count") && !isnothing(args["pinf_tail_count"])
    if has_fixed_pinf && has_tail_pinf
        error("Use only one of --pinf or --pinf_tail_count.")
    end

    pinf_source = "scan"
    pinf_tail_count_used = 0
    chosen = if has_fixed_pinf
        pinf = Float64(args["pinf"])
        fit = offset_fit(d, p, pinf; p_sem=p_fit_sem)
        isnothing(fit) && error("Could not fit supplied Pinf=$(pinf).")
        pinf_source = "fixed"
        (pinf=pinf, fit=fit)
    elseif has_tail_pinf
        pinf, pinf_tail_count_used = tail_mean_pinf(p, Int(args["pinf_tail_count"]))
        fit = offset_fit(d, p, pinf; p_sem=p_fit_sem)
        isnothing(fit) && error("Could not fit tail-mean Pinf=$(pinf) from N=$(pinf_tail_count_used).")
        pinf_source = "tail_mean"
        (pinf=pinf, fit=fit)
    else
        scan_pinf(
            d,
            p;
            p_sem=p_fit_sem,
            scan_points=Int(args["scan_points"]),
            scan_width_factor=Float64(args["scan_width_factor"]),
        )
    end

    diff = chosen.pinf .- p
    keep = diff .> 0
    fit_curve = exp(chosen.fit.intercept) .* d[keep] .^ chosen.fit.slope
    diff_error = isnothing(p_error) ? nothing : log_safe_yerror(diff[keep], p_error[keep])
    fixed_slope = haskey(args, "fixed_slope") && !isnothing(args["fixed_slope"]) ? Float64(args["fixed_slope"]) : nothing
    fixed_at_chosen = isnothing(fixed_slope) ? nothing : fixed_slope_fit(
        d,
        p,
        chosen.pinf,
        fixed_slope;
        p_sem=p_fit_sem,
        min_points=max(2, Int(args["fixed_slope_scan_min_points"])),
        dof_params=1,
    )
    fixed_best = isnothing(fixed_slope) ? nothing : scan_pinf_fixed_slope(
        d,
        p,
        fixed_slope;
        p_sem=p_fit_sem,
        scan_points=Int(args["scan_points"]),
        scan_width_factor=Float64(args["scan_width_factor"]),
        min_points=max(2, Int(args["fixed_slope_scan_min_points"])),
    )

    out_dir = if haskey(args, "out_dir") && !isnothing(args["out_dir"])
        abspath(String(args["out_dir"]))
    else
        dirname(aggregate_path)
    end
    mkpath(out_dir)
    tag = sanitize_tag(String(args["save_tag"]))

    plt = if isnothing(diff_error)
        plot(
            d[keep],
            diff[keep];
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="Pinf - P_ss(d)",
            title=@sprintf("Offset power law from d_min=%.6g, Pinf=%.8g", d_min, chosen.pinf),
            label="Pinf - P_ss(d)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    else
        plot(
            d[keep],
            diff[keep];
            yerror=diff_error,
            xscale=:log10,
            yscale=:log10,
            lw=2,
            marker=:circle,
            markersize=3,
            xlabel="minimum separation d",
            ylabel="Pinf - P_ss(d)",
            title=@sprintf("Offset power law from d_min=%.6g, Pinf=%.8g", d_min, chosen.pinf),
            label="Pinf - P_ss(d), $(error_label)",
            legend=OUTSIDE_LEGEND,
            framestyle=:box,
            grid=:both,
            size=(1100, 700),
        )
    end
    plot!(
        plt,
        d[keep],
        fit_curve;
        lw=2,
        ls=:dash,
        color=:black,
        label=fit_label(chosen.fit),
    )
    if !isnothing(fixed_at_chosen)
        fixed_curve = exp(fixed_at_chosen.intercept) .* d[keep] .^ fixed_slope
        plot!(
            plt,
            d[keep],
            fixed_curve;
            lw=2,
            ls=:dashdot,
            color=:purple,
            label=fixed_fit_label(fixed_at_chosen),
        )
    end
    if !Bool(args["no_reference"])
        reference_slope = Float64(args["reference_slope"])
        kept_d = d[keep]
        kept_diff = diff[keep]
        ref_curve = kept_diff[1] .* (kept_d ./ kept_d[1]) .^ reference_slope
        plot!(
            plt,
            kept_d,
            ref_curve;
            lw=2,
            ls=:dot,
            color=:gray,
            label=@sprintf("reference d^%.4g", reference_slope),
        )
    end

    plot_path = joinpath(out_dir, "$(tag)_distance_density_offset_powerlaw.png")
    summary_path = joinpath(out_dir, "$(tag)_distance_density_offset_powerlaw_summary.txt")
    savefig(plt, plot_path)
    fixed_plot_path = ""
    if !isnothing(fixed_best)
        fixed_label = slope_filename_label(fixed_slope)
        fixed_plot_path = joinpath(out_dir, "$(tag)_fixed_slope_$(fixed_label)_best_offset_distance_density_offset_powerlaw.png")
        savefig(
            fixed_slope_plot(d, p, p_error, error_label, fixed_best.fit, d_min, fixed_slope),
            fixed_plot_path,
        )
    end

    tail_rows = tail_diagnostics(d, p, parse_tail_counts(String(args["tail_counts"])); p_sem=p_fit_sem)
    diagnostics = Pair{String,Any}[
        "aggregate_path" => aggregate_path,
        "d_min_probability" => d_min,
        "P_min" => p_min,
        "fit_window_rows" => length(d),
        "P_max_in_fit_window" => maximum(p),
        "Pinf" => chosen.pinf,
        "Pinf_source" => pinf_source,
        "Pinf_tail_count" => pinf_tail_count_used,
        "plot_error_bar" => isempty(error_label) ? "none" : error_label,
        "weighted_fit_sigma" => isnothing(p_fit_sem) ? "none" : "P_density_sem_or_P_density_ci95_div_1.96",
        "offset_loglog_slope" => chosen.fit.slope,
        "offset_loglog_intercept" => chosen.fit.intercept,
        "offset_loglog_r2" => chosen.fit.r2,
        "offset_loglog_fit_weighted" => chosen.fit.weighted,
        "offset_loglog_reduced_chi2" => chosen.fit.reduced_chi2,
        "offset_loglog_n" => chosen.fit.n,
    ]
    if !isnothing(fixed_slope)
        push!(diagnostics, "fixed_slope" => fixed_slope)
        push!(diagnostics, "fixed_slope_scan_min_points" => Int(args["fixed_slope_scan_min_points"]))
        push!(diagnostics, "fixed_slope_selection" => "min_reduced_chi2_if_weighted_else_min_sse")
        if isnothing(fixed_at_chosen)
            push!(diagnostics, "fixed_slope_at_selected_Pinf_fit_n" => 0)
        else
            push!(diagnostics, "fixed_slope_at_selected_Pinf" => fixed_at_chosen.pinf)
            push!(diagnostics, "fixed_slope_at_selected_intercept" => fixed_at_chosen.intercept)
            push!(diagnostics, "fixed_slope_at_selected_r2" => fixed_at_chosen.r2)
            push!(diagnostics, "fixed_slope_at_selected_fit_weighted" => fixed_at_chosen.weighted)
            push!(diagnostics, "fixed_slope_at_selected_reduced_chi2" => fixed_at_chosen.reduced_chi2)
            push!(diagnostics, "fixed_slope_at_selected_ss_res" => fixed_at_chosen.ss_res)
            push!(diagnostics, "fixed_slope_at_selected_fit_n" => fixed_at_chosen.n)
        end
        push!(diagnostics, "fixed_slope_best_Pinf" => fixed_best.pinf)
        push!(diagnostics, "fixed_slope_best_intercept" => fixed_best.fit.intercept)
        push!(diagnostics, "fixed_slope_best_r2" => fixed_best.fit.r2)
        push!(diagnostics, "fixed_slope_best_fit_weighted" => fixed_best.fit.weighted)
        push!(diagnostics, "fixed_slope_best_reduced_chi2" => fixed_best.fit.reduced_chi2)
        push!(diagnostics, "fixed_slope_best_ss_res" => fixed_best.fit.ss_res)
        push!(diagnostics, "fixed_slope_best_fit_n" => fixed_best.fit.n)
        push!(diagnostics, "fixed_slope_best_plot_path" => fixed_plot_path)
    end
    for row in tail_rows
        prefix = "tail$(row.tail_count)"
        push!(diagnostics, "$(prefix)_Pinf" => row.pinf)
        if isnothing(row.fit)
            push!(diagnostics, "$(prefix)_fit_n" => 0)
        else
            push!(diagnostics, "$(prefix)_slope" => row.fit.slope)
            push!(diagnostics, "$(prefix)_r2" => row.fit.r2)
            push!(diagnostics, "$(prefix)_fit_weighted" => row.fit.weighted)
            push!(diagnostics, "$(prefix)_reduced_chi2" => row.fit.reduced_chi2)
            push!(diagnostics, "$(prefix)_fit_n" => row.fit.n)
        end
    end
    save_summary(summary_path, diagnostics)

    println("Saved offset-powerlaw plot:")
    println("  $(plot_path)")
    if !isempty(fixed_plot_path)
        println("Saved fixed-slope best-offset plot:")
        println("  $(fixed_plot_path)")
    end
    println("Saved offset-powerlaw summary:")
    println("  $(summary_path)")
    println(@sprintf("Minimum P_density: d=%.16g, P=%.16g", d_min, p_min))
    println(@sprintf("Chosen Pinf=%.16g (%s)", chosen.pinf, pinf_source))
    if chosen.fit.weighted
        println(@sprintf("Offset weighted fit: Pinf - P ~ d^%.8g, weighted_R2=%.8g, reduced_chi2=%.8g, n=%d", chosen.fit.slope, chosen.fit.r2, chosen.fit.reduced_chi2, chosen.fit.n))
    else
        println(@sprintf("Offset fit: Pinf - P ~ d^%.8g, R2=%.8g, n=%d", chosen.fit.slope, chosen.fit.r2, chosen.fit.n))
    end
    if !isnothing(fixed_slope)
        if !isnothing(fixed_at_chosen)
            println(@sprintf(
                "Fixed slope at selected Pinf: slope=%.8g, Pinf=%.16g, weighted_R2=%.8g, reduced_chi2=%.8g, n=%d",
                fixed_slope,
                fixed_at_chosen.pinf,
                fixed_at_chosen.r2,
                fixed_at_chosen.reduced_chi2,
                fixed_at_chosen.n,
            ))
        end
        println(@sprintf(
            "Best fixed-slope offset: slope=%.8g, Pinf=%.16g, weighted_R2=%.8g, reduced_chi2=%.8g, n=%d",
            fixed_slope,
            fixed_best.pinf,
            fixed_best.fit.r2,
            fixed_best.fit.reduced_chi2,
            fixed_best.fit.n,
        ))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
