using JLD2
using Printf
using Statistics

const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask
const BOND_PASS_SAMPLE_COUNT_KEY = :bond_pass_sample_count
const AGG_TWO_FORCE_REPLICA_COUNT_KEY = :agg_two_force_replica_count
const AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY = :agg_two_force_var_slot_mean
const AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY = :agg_two_force_var_raw_slot_mean
const AGG_TWO_FORCE_J2_SLOT_MEAN_KEY = :agg_two_force_j2_slot_mean

_fielddict(x) = hasproperty(x, :fields) ? getproperty(x, :fields) : nothing

function get_prop(x, key::Symbol, default=nothing)
    if x isa AbstractDict
        return get(x, key, default)
    end
    fd = _fielddict(x)
    if fd !== nothing
        return get(fd, key, default)
    end
    try
        if Base.hasproperty(x, key)
            return getproperty(x, key)
        end
    catch
        return default
    end
    return default
end

function forcing_bonds_1d(state, L::Int)
    forcing_raw = get_prop(state, :forcing, nothing)
    forcing_raw === nothing && return Tuple{Int,Int}[], Float64[]
    forcings = forcing_raw isa AbstractVector ? forcing_raw : [forcing_raw]
    bonds = Tuple{Int,Int}[]
    magnitudes = Float64[]
    for force in forcings
        bond_indices = get_prop(force, :bond_indices, nothing)
        magnitude = get_prop(force, :magnitude, nothing)
        bond_indices === nothing && continue
        magnitude === nothing && continue
        if length(bond_indices) >= 2 && length(bond_indices[1]) == 1 && length(bond_indices[2]) == 1
            b1 = mod1(Int(round(Float64(bond_indices[1][1]))), L)
            b2 = mod1(Int(round(Float64(bond_indices[2][1]))), L)
            push!(bonds, (b1, b2))
            push!(magnitudes, Float64(magnitude))
        end
    end
    return bonds, magnitudes
end

function tracked_force_indices(stats::AbstractDict, magnitudes::Vector{Float64})
    n_forces = length(magnitudes)
    if haskey(stats, BOND_PASS_TRACK_MASK_KEY)
        mask = Float64.(stats[BOND_PASS_TRACK_MASK_KEY])
        if length(mask) == n_forces
            tracked = [i for i in 1:n_forces if mask[i] > 0.5]
            !isempty(tracked) && return tracked
        end
    end
    tracked = [i for i in 1:n_forces if abs(magnitudes[i]) > 0]
    return isempty(tracked) ? collect(1:n_forces) : tracked
end

function connected_correlation_fix_term(param)
    dims = get_prop(param, :dims)
    n_sites = Int(prod(dims))
    n_particles = Int(round(Float64(get_prop(param, :N))))
    return Float64(n_particles) / (Float64(n_sites)^2)
end

function connected_corr_mat_1d(state, param)
    rho_avg = Float64.(get_prop(state, :ρ_avg))
    rho_cuts = get_prop(state, :ρ_matrix_avg_cuts, nothing)
    full_corr = Float64.(get_prop(rho_cuts, :full, nothing))
    full_corr isa AbstractMatrix || error("State does not contain a 1D :full correlation matrix.")
    return full_corr .- (rho_avg * transpose(rho_avg))
end

function bond_centered_cut_1d_from_corr(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    L = size(corr_mat, 2)
    cut = 0.5 .* (Float64.(corr_mat[b1, :]) .+ Float64.(corr_mat[b2, :]))
    if smooth_diagonal
        b1_left = mod1(b1 - 1, L)
        b1_right = mod1(b1 + 1, L)
        b2_left = mod1(b2 - 1, L)
        b2_right = mod1(b2 + 1, L)
        cut[b1] = 0.5 * (cut[b1_left] + cut[b1_right])
        cut[b2] = 0.5 * (cut[b2_left] + cut[b2_right])
    end
    return cut
end

function bond_center_value_from_corr(corr_mat::AbstractMatrix{<:Real}, b1::Int, b2::Int; smooth_diagonal::Bool=true)
    cut = bond_centered_cut_1d_from_corr(corr_mat, b1, b2; smooth_diagonal=smooth_diagonal)
    return 0.5 * (cut[b1] + cut[b2])
end

function summarize_file(path::String)
    payload = load(path)
    state = payload["state"]
    param = payload["param"]
    stats = get_prop(state, :bond_pass_stats, Dict{Symbol,Vector{Float64}}())

    rho_avg = Float64.(get_prop(state, :ρ_avg, nothing))
    rho_avg === nothing && error("Missing ρ_avg in state for $path")
    dims = get_prop(param, :dims, nothing)
    L = if dims === nothing
        length(rho_avg)
    else
        length(dims) == 1 || error("This diagnostic only supports 1D states.")
        Int(dims[1])
    end

    bonds, magnitudes = forcing_bonds_1d(state, L)
    length(bonds) >= 2 || error("Expected at least two forcing bonds in $path")
    tracked = tracked_force_indices(stats, magnitudes)

    corr_mat = connected_corr_mat_1d(state, param)
    var_direct_smoothed = [
        bond_center_value_from_corr(corr_mat, bonds[i][1], bonds[i][2]; smooth_diagonal=true)
        for i in tracked
    ]
    var_direct_raw = [
        bond_center_value_from_corr(corr_mat, bonds[i][1], bonds[i][2]; smooth_diagonal=false)
        for i in tracked
    ]

    var_exact_smoothed = haskey(stats, AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY) ?
        Float64.(stats[AGG_TWO_FORCE_VAR_SLOT_MEAN_KEY]) : Float64[]
    var_exact_raw = haskey(stats, AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY) ?
        Float64.(stats[AGG_TWO_FORCE_VAR_RAW_SLOT_MEAN_KEY]) : Float64[]
    j2_direct = haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) ?
        [Float64.(stats[BOND_PASS_TOTAL_SQ_AVG_KEY])[i] for i in tracked] : Float64[]
    j2_exact = haskey(stats, AGG_TWO_FORCE_J2_SLOT_MEAN_KEY) ?
        Float64.(stats[AGG_TWO_FORCE_J2_SLOT_MEAN_KEY]) : Float64[]

    sample_count = haskey(stats, BOND_PASS_SAMPLE_COUNT_KEY) ? Float64.(stats[BOND_PASS_SAMPLE_COUNT_KEY])[1] : NaN
    replica_count = haskey(stats, AGG_TWO_FORCE_REPLICA_COUNT_KEY) ? Float64.(stats[AGG_TWO_FORCE_REPLICA_COUNT_KEY])[1] : NaN
    tracked_indices_str = join(tracked, ",")
    tracked_bonds_str = join(["($(bonds[i][1]),$(bonds[i][2]))" for i in tracked], ", ")
    fmt_vec(values) = join([@sprintf("%.10g", v) for v in values], ", ")

    println("FILE=$path")
    println("  t=$(get_prop(state, :t, missing))")
    println("  bond_pass_sample_count=$sample_count")
    println("  replica_count=$replica_count")
    println("  tracked_force_indices=$tracked_indices_str")
    println("  tracked_bonds=$tracked_bonds_str")

    if !isempty(var_exact_smoothed)
        println("  bond_center_variance_smoothed_direct = $(fmt_vec(var_direct_smoothed))")
        println("  bond_center_variance_smoothed_exact  = $(fmt_vec(var_exact_smoothed))")
        println("  bond_center_variance_smoothed_diff   = $(fmt_vec(var_direct_smoothed .- var_exact_smoothed))")
    else
        println("  bond_center_variance_smoothed_exact  = <missing>")
    end

    if !isempty(var_exact_raw)
        println("  bond_center_variance_raw_direct      = $(fmt_vec(var_direct_raw))")
        println("  bond_center_variance_raw_exact       = $(fmt_vec(var_exact_raw))")
        println("  bond_center_variance_raw_diff        = $(fmt_vec(var_direct_raw .- var_exact_raw))")
    else
        println("  bond_center_variance_raw_exact       = <missing>")
    end

    if !isempty(j2_exact)
        println("  j2_direct                            = $(fmt_vec(j2_direct))")
        println("  j2_exact                             = $(fmt_vec(j2_exact))")
        println("  j2_diff                              = $(fmt_vec(j2_direct .- j2_exact))")
    else
        println("  j2_exact                             = <missing>")
    end
    println()
end

if isempty(ARGS)
    println("Usage: julia --startup-file=no cluster_scripts/debug_two_force_aggregate_file.jl <aggregate_file.jld2> [more_files...]")
    exit(1)
end

for path in ARGS
    summarize_file(path)
end
