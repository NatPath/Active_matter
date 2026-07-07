# softcrowded_rtp_wall_measurements.jl
#
# Standalone 2D lattice RTP simulation near a flat wall with soft-crowding
# interactions, plus measurements for the effective stochastic wall-current
# description:
#   rho_profile(y)
#   zeta(x,t) = wall-layer tangential current
#   C_zzeta(dx,dt)
#   S_zeta(k,omega) if FFTW is installed
#   C_rho_zeta(dx,y,tau)
#   C_rho_rho(dx;y1,y2)
#
# Model:
#   - 2D lattice, periodic in x, reflecting walls in y.
#   - RTP orientations are ±x, ±y.
#   - Multiple occupancy is allowed.
#   - Soft crowding energy: E = (u/2) * sum_sites n(n-1).
#   - Active move attempts are Metropolis accepted with ΔE from soft crowding.
#   - One sweep = N randomly selected particle updates.
#
# Usage:
#   julia softcrowded_rtp_wall_measurements.jl
#
# Optional for spectral output:
#   julia -e 'using Pkg; Pkg.add("FFTW")'

using Random
using Statistics
using DelimitedFiles
using Printf

const HAS_FFTW = try
    @eval using FFTW
    true
catch err
    @warn "FFTW not available; S_zeta(k,omega) will be skipped. Install with: julia -e 'using Pkg; Pkg.add(\"FFTW\")'"
    false
end

Base.@kwdef mutable struct Params
    # Geometry
    Lx::Int = 128
    Ly::Int = 64

    # Particles
    rho0::Float64 = 0.50          # N = round(rho0 * Lx * Ly)
    alpha::Float64 = 0.05         # tumble probability per chosen update; approx tumble rate per sweep

    # Soft crowding: E_site(n)=u/2*n*(n-1); beta=1/T for Metropolis acceptance
    u::Float64 = 1.0
    beta::Float64 = 1.0

    # Runtime
    total_sweeps::Int = 200_000
    burn_in::Int = 50_000
    sample_interval::Int = 10
    seed::Int = 1

    # Wall-layer current measurement
    # If wall_width <= 0, use ceil(1/alpha), clipped to [1,Ly].
    wall_width::Int = 0

    # Correlation windows
    max_dx::Int = 64
    max_dt::Int = 200
    max_tau_rho_zeta::Int = 50

    # Selected y values for C_{rho,zeta}. If empty, chosen automatically.
    selected_y::Vector{Int} = Int[]

    # Selected y-pairs for C_{rho,rho}. If empty, chosen automatically.
    selected_pairs::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[]

    # Output
    outdir::String = "rtp_wall_softcrowding_output"
end

mutable struct State
    x::Vector{Int}
    y::Vector{Int}
    dir::Vector{Int}          # 1:+x, 2:-x, 3:+y, 4:-y
    occ::Matrix{Int}
    jx_sweep::Matrix{Int}     # accepted horizontal current per x-bond and row y during one sweep
end

wrapx(x::Int, Lx::Int) = mod1(x, Lx)

function dxdy(dir::Int)
    if dir == 1
        return 1, 0
    elseif dir == 2
        return -1, 0
    elseif dir == 3
        return 0, 1
    elseif dir == 4
        return 0, -1
    else
        error("invalid direction")
    end
end

function init_state(p::Params)
    rng = MersenneTwister(p.seed)
    N = round(Int, p.rho0 * p.Lx * p.Ly)
    x = Vector{Int}(undef, N)
    y = Vector{Int}(undef, N)
    dir = Vector{Int}(undef, N)
    occ = zeros(Int, p.Lx, p.Ly)

    for i in 1:N
        xi = rand(rng, 1:p.Lx)
        yi = rand(rng, 1:p.Ly)
        x[i] = xi
        y[i] = yi
        dir[i] = rand(rng, 1:4)
        occ[xi, yi] += 1
    end

    jx_sweep = zeros(Int, p.Lx, p.Ly)
    return State(x, y, dir, occ, jx_sweep), rng
end

# Soft-crowding energy change for moving one particle a -> b.
# E(n)=u/2*n*(n-1), so ΔE = u*n_b - u*(n_a-1).
function deltaE_softcrowd(occ::Matrix{Int}, xa::Int, ya::Int, xb::Int, yb::Int, u::Float64)
    na = occ[xa, ya]
    nb = occ[xb, yb]
    return u * (nb - (na - 1))
end

function microstep!(state::State, p::Params, rng::AbstractRNG)
    N = length(state.x)
    i = rand(rng, 1:N)

    # RTP tumble. This convention gives approximately alpha tumbles per particle per sweep.
    if rand(rng) < p.alpha
        state.dir[i] = rand(rng, 1:4)
    end

    dx, dy = dxdy(state.dir[i])
    xa = state.x[i]
    ya = state.y[i]
    xb = wrapx(xa + dx, p.Lx)
    yb = ya + dy

    # Reflecting bottom/top walls. Rejected moves into y=0 or y=Ly+1.
    if yb < 1 || yb > p.Ly
        return
    end

    dE = deltaE_softcrowd(state.occ, xa, ya, xb, yb, p.u)
    accept = (dE <= 0.0) || (rand(rng) < exp(-p.beta * dE))
    if !accept
        return
    end

    # Accepted horizontal current. Bond index b denotes bond b -> b+1 in row y.
    if dx == 1
        state.jx_sweep[xa, ya] += 1
    elseif dx == -1
        state.jx_sweep[xb, ya] -= 1
    end

    state.occ[xa, ya] -= 1
    state.occ[xb, yb] += 1
    state.x[i] = xb
    state.y[i] = yb
end

function one_sweep!(state::State, p::Params, rng::AbstractRNG)
    fill!(state.jx_sweep, 0)
    N = length(state.x)
    for _ in 1:N
        microstep!(state, p, rng)
    end
end

function choose_wall_width(p::Params)
    if p.wall_width > 0
        return min(max(p.wall_width, 1), p.Ly)
    end
    # v_lattice ~ 1 site/sweep in this convention, so persistence length ~ 1/alpha.
    return min(p.Ly, max(1, ceil(Int, 1 / max(p.alpha, 1e-12))))
end

function choose_selected_y(p::Params, w::Int)
    if !isempty(p.selected_y)
        return [y for y in p.selected_y if 1 <= y <= p.Ly]
    end
    cand = unique([w + 1, 2w, 4w, p.Ly ÷ 2])
    return [y for y in cand if 1 <= y <= p.Ly]
end

function choose_selected_pairs(p::Params, ys::Vector{Int})
    if !isempty(p.selected_pairs)
        return [(a,b) for (a,b) in p.selected_pairs if 1 <= a <= p.Ly && 1 <= b <= p.Ly]
    end
    pairs = Tuple{Int,Int}[]
    for y in ys
        push!(pairs, (y,y))
    end
    if length(ys) >= 2
        push!(pairs, (ys[1], ys[end]))
    end
    return unique(pairs)
end

function simulate(p::Params)
    mkpath(p.outdir)
    state, rng = init_state(p)
    w = choose_wall_width(p)

    nsamples = fld(p.total_sweeps - p.burn_in, p.sample_interval)
    @assert nsamples > 2 "Need total_sweeps > burn_in + 2*sample_interval"

    rho_samples = Array{Float32}(undef, nsamples, p.Lx, p.Ly)
    zeta_samples = Array{Float32}(undef, nsamples, p.Lx)
    zeta_window = zeros(Float64, p.Lx)

    sample_idx = 0

    @info "Starting simulation" Lx=p.Lx Ly=p.Ly N=length(state.x) nsamples=nsamples wall_width=w u=p.u alpha=p.alpha

    for sweep in 1:p.total_sweeps
        one_sweep!(state, p, rng)

        if sweep > p.burn_in
            # Integrated accepted tangential current in the wall layer during this sweep.
            # Later divided by sample_interval to produce current per sweep.
            for x in 1:p.Lx
                s = 0
                @inbounds for y in 1:w
                    s += state.jx_sweep[x,y]
                end
                zeta_window[x] += s
            end

            if (sweep - p.burn_in) % p.sample_interval == 0
                sample_idx += 1
                @inbounds for x in 1:p.Lx, y in 1:p.Ly
                    rho_samples[sample_idx, x, y] = Float32(state.occ[x,y])
                end
                @inbounds for x in 1:p.Lx
                    zeta_samples[sample_idx, x] = Float32(zeta_window[x] / p.sample_interval)
                    zeta_window[x] = 0.0
                end
            end
        end

        if sweep % max(1, p.total_sweeps ÷ 20) == 0
            @info "progress" sweep=sweep total=p.total_sweeps sample_idx=sample_idx
        end
    end

    @assert sample_idx == nsamples
    return rho_samples, zeta_samples, w
end

function mean_profile(rho_samples)
    ns, Lx, Ly = size(rho_samples)
    prof = zeros(Float64, Ly)
    for y in 1:Ly
        prof[y] = mean(@view rho_samples[:,:,y])
    end
    return prof
end

function centered_zeta(zeta_samples)
    zmean = mean(zeta_samples)
    return Float64.(zeta_samples) .- zmean, zmean
end

function centered_rho(rho_samples, rho_mean_y::Vector{Float64})
    ns, Lx, Ly = size(rho_samples)
    drho = Array{Float32}(undef, ns, Lx, Ly)
    @inbounds for t in 1:ns, x in 1:Lx, y in 1:Ly
        drho[t,x,y] = Float32(rho_samples[t,x,y] - rho_mean_y[y])
    end
    return drho
end

function corr_zeta_zeta(dzeta::Matrix{Float64}, max_dx::Int, max_dt::Int)
    ns, Lx = size(dzeta)
    max_dx = min(max_dx, Lx - 1)
    max_dt = min(max_dt, ns - 1)
    C = zeros(Float64, max_dt + 1, max_dx + 1)

    @inbounds for dt in 0:max_dt
        nt = ns - dt
        for dx in 0:max_dx
            acc = 0.0
            count = 0
            for t in 1:nt
                for x in 1:Lx
                    x2 = mod1(x + dx, Lx)
                    acc += dzeta[t,x] * dzeta[t+dt,x2]
                    count += 1
                end
            end
            C[dt+1, dx+1] = acc / count
        end
    end
    return C
end

function estimate_sigma_from_Czz(Czz::Matrix{Float64}, sample_interval::Int)
    # Approximate 1/2 * integral dtdx C(x,t), using the stored nonnegative dx,dt quadrant
    # and mirror symmetry in dx and dt. Use this only as a diagnostic amplitude.
    ntd, ndx = size(Czz)
    total = 0.0
    for it in 1:ntd
        time_weight = (it == 1) ? 1.0 : 2.0
        space_sum = Czz[it,1]
        if ndx > 1
            space_sum += 2.0 * sum(@view Czz[it,2:ndx])
        end
        total += time_weight * space_sum
    end
    return 0.5 * sample_interval * total
end

function spectrum_zeta(dzeta::Matrix{Float64}, sample_interval::Int)
    if !HAS_FFTW
        return nothing, nothing, nothing
    end
    ns, Lx = size(dzeta)
    F = fft(dzeta)  # dims: time, x
    S = abs2.(F) ./ (ns * Lx)
    freqs_omega = fftfreq(ns, sample_interval) .* (2π) # angular frequency, rad/sweep
    ks = fftfreq(Lx, 1.0) .* (2π)                     # lattice wavenumber, rad/site
    return S, freqs_omega, ks
end

function fftfreq(n::Int, dt::Real)
    if iseven(n)
        vals = vcat(0:(n÷2-1), -n÷2:-1)
    else
        vals = vcat(0:((n-1)÷2), -((n-1)÷2):-1)
    end
    return collect(vals) ./ (n * dt)
end

function corr_rho_zeta(drho::Array{Float32,3}, dzeta::Matrix{Float64}, ys::Vector{Int}, max_dx::Int, max_tau::Int)
    ns, Lx, Ly = size(drho)
    max_dx = min(max_dx, Lx - 1)
    max_tau = min(max_tau, ns - 1)

    rows = Vector{NTuple{4,Float64}}()
    @inbounds for y in ys
        for tau in 0:max_tau
            nt = ns - tau
            for dx in 0:max_dx
                acc = 0.0
                count = 0
                for t in 1:nt
                    for x in 1:Lx
                        x2 = mod1(x + dx, Lx)
                        acc += Float64(drho[t+tau,x2,y]) * dzeta[t,x]
                        count += 1
                    end
                end
                push!(rows, (Float64(y), Float64(tau), Float64(dx), acc / count))
            end
        end
    end
    return rows
end

function corr_rho_rho(drho::Array{Float32,3}, pairs::Vector{Tuple{Int,Int}}, max_dx::Int)
    ns, Lx, Ly = size(drho)
    max_dx = min(max_dx, Lx - 1)

    rows = Vector{NTuple{4,Float64}}()
    @inbounds for (y1,y2) in pairs
        for dx in 0:max_dx
            acc = 0.0
            count = 0
            for t in 1:ns
                for x in 1:Lx
                    x2 = mod1(x + dx, Lx)
                    acc += Float64(drho[t,x,y1]) * Float64(drho[t,x2,y2])
                    count += 1
                end
            end
            push!(rows, (Float64(y1), Float64(y2), Float64(dx), acc / count))
        end
    end
    return rows
end

function write_rows(path::String, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(row, ","))
        end
    end
end

function save_outputs(p::Params, rho_samples, zeta_samples, w::Int)
    mkpath(p.outdir)

    rho_mean_y = mean_profile(rho_samples)
    dzeta, zmean = centered_zeta(zeta_samples)
    drho = centered_rho(rho_samples, rho_mean_y)

    ys = choose_selected_y(p, w)
    pairs = choose_selected_pairs(p, ys)

    @info "Computing C_zeta_zeta"
    Czz = corr_zeta_zeta(dzeta, p.max_dx, p.max_dt)
    sigma_eff = estimate_sigma_from_Czz(Czz, p.sample_interval)

    @info "Computing C_rho_zeta" ys=ys
    Crz_rows = corr_rho_zeta(drho, dzeta, ys, p.max_dx, p.max_tau_rho_zeta)

    @info "Computing C_rho_rho" pairs=pairs
    Crr_rows = corr_rho_rho(drho, pairs, p.max_dx)

    # Save scalar/info file.
    open(joinpath(p.outdir, "run_info.txt"), "w") do io
        println(io, "Lx=$(p.Lx)")
        println(io, "Ly=$(p.Ly)")
        println(io, "rho0=$(p.rho0)")
        println(io, "alpha=$(p.alpha)")
        println(io, "u=$(p.u)")
        println(io, "beta=$(p.beta)")
        println(io, "total_sweeps=$(p.total_sweeps)")
        println(io, "burn_in=$(p.burn_in)")
        println(io, "sample_interval=$(p.sample_interval)")
        println(io, "wall_width=$w")
        println(io, "zeta_mean=$zmean")
        println(io, "sigma_eff_truncated=$sigma_eff")
        println(io, "selected_y=$(ys)")
        println(io, "selected_pairs=$(pairs)")
        println(io, "NOTE: sigma_eff_truncated uses max_dx=$(p.max_dx), max_dt=$(p.max_dt) and is diagnostic only.")
    end

    # Density profile.
    prof_rows = [(Float64(y), rho_mean_y[y]) for y in 1:p.Ly]
    write_rows(joinpath(p.outdir, "rho_profile.csv"), ["y", "rho_mean"], prof_rows)

    # Mean wall current per x and global mean.
    zeta_mean_x = vec(mean(zeta_samples, dims=1))
    zeta_rows = [(Float64(x), Float64(zeta_mean_x[x])) for x in 1:p.Lx]
    write_rows(joinpath(p.outdir, "zeta_mean_x.csv"), ["x", "zeta_mean"], zeta_rows)

    # C_zeta_zeta as long table.
    Czz_rows = Vector{NTuple{3,Float64}}()
    for dt in 0:size(Czz,1)-1
        for dx in 0:size(Czz,2)-1
            push!(Czz_rows, (Float64(dt), Float64(dx), Czz[dt+1,dx+1]))
        end
    end
    write_rows(joinpath(p.outdir, "C_zeta_zeta.csv"), ["dt", "dx", "C"], Czz_rows)

    # C_rho_zeta.
    write_rows(joinpath(p.outdir, "C_rho_zeta.csv"), ["y", "tau", "dx", "C"], Crz_rows)

    # C_rho_rho.
    write_rows(joinpath(p.outdir, "C_rho_rho.csv"), ["y1", "y2", "dx", "C"], Crr_rows)

    # Save sampled zeta time series; this is useful for independent analysis.
    writedlm(joinpath(p.outdir, "zeta_samples.csv"), zeta_samples, ',')

    # Optional spectrum.
    if HAS_FFTW
        @info "Computing S_zeta(k,omega)"
        S, omegas, ks = spectrum_zeta(dzeta, p.sample_interval)
        writedlm(joinpath(p.outdir, "S_zeta_matrix.csv"), S, ',')
        writedlm(joinpath(p.outdir, "S_zeta_omega_axis.csv"), omegas, ',')
        writedlm(joinpath(p.outdir, "S_zeta_k_axis.csv"), ks, ',')
    end

    @info "Finished outputs" outdir=p.outdir sigma_eff=sigma_eff zeta_mean=zmean
end

function main()
    p = Params(
        # Increase these after the pipeline works.
        Lx = 128,
        Ly = 64,
        rho0 = 0.50,
        alpha = 0.05,
        u = 1.0,
        beta = 1.0,
        total_sweeps = 200_000,
        burn_in = 50_000,
        sample_interval = 10,
        seed = 1,
        wall_width = 0,
        max_dx = 64,
        max_dt = 200,
        max_tau_rho_zeta = 50,
        outdir = "rtp_wall_softcrowding_output"
    )

    rho_samples, zeta_samples, w = simulate(p)
    save_outputs(p, rho_samples, zeta_samples, w)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
