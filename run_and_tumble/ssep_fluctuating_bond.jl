using Random
using Statistics
using Base.Threads
using Plots
using Dates

# ============================================================
# USER PARAMETERS  <<< change these first
# ============================================================
const RHO    = 0.5      # particle density
const ALPHA  = 1.0      # flip rate of the fluctuating bond
const TBURN  = 2.0e3    # burn-in time
const TMEAS  = 2.0e4    # measurement time
const NRUNS  = 1      # number of independent runs
const SEED   = 1        # random seed

const N      = 50       # system size
const XVALS  = [3, 6, 9]   # fixed x values for C(x,y)
# ============================================================


# ------------------------------------------------------------
# Map physical coordinate y to internal site index
# Bond is between physical sites 0 and 1.
# Internal sites are 1,...,N with:
#   1 <-> physical 1
#   2 <-> physical 2
#   ...
#   N <-> physical 0
#   N-1 <-> physical -1
# etc.
# ------------------------------------------------------------
function phys_to_site(y::Int, N::Int)
    if y > 0
        return y
    elseif y < 0
        return N + y
    else
        return N
    end
end

# ------------------------------------------------------------
# One Gillespie trajectory
# ------------------------------------------------------------
function one_run_ssep(
    N::Int,
    M::Int,
    alpha::Float64,
    Tburn::Float64,
    Tmeas::Float64,
    xvals_phys::Vector{Int},
    yvals_phys::Vector{Int},
    seed::Int,
)
    rng = MersenneTwister(seed)

    nx = length(xvals_phys)
    ny = length(yvals_phys)

    xsites = [phys_to_site(x, N) for x in xvals_phys]
    ysites = [phys_to_site(y, N) for y in yvals_phys]

    n = zeros(Int, N)
    occ = randperm(rng, N)[1:M]
    n[occ] .= 1

    # sigma = +1 means 0 -> 1 allowed, i.e. internal N -> 1
    # sigma = -1 means 1 -> 0 allowed, i.e. internal 1 -> N
    sigma = 1

    densInt = zeros(Float64, N)
    pairInt = zeros(Float64, nx, ny)

    t = 0.0
    Ttot = Tburn + Tmeas
    rates = zeros(Float64, 2N)

    while t < Ttot
        fill!(rates, 0.0)

        # Regular bonds
        @inbounds for i in 1:N-1
            rates[i] = n[i] * (1 - n[i + 1])
            rates[(N - 1) + i] = n[i + 1] * (1 - n[i])
        end

        # Special fluctuating bond
        if sigma == 1
            rates[2N - 1] = n[N] * (1 - n[1])   # 0 -> 1
        else
            rates[2N - 1] = n[1] * (1 - n[N])   # 1 -> 0
        end

        # Flip event
        rates[2N] = alpha

        R = sum(rates)
        R <= 0 && break

        dt = -log(rand(rng)) / R
        tnext = t + dt

        a = max(t, Tburn)
        b = min(tnext, Ttot)
        if b > a
            dtt = b - a
            @inbounds for i in 1:N
                densInt[i] += n[i] * dtt
            end
            @inbounds for k in 1:nx
                nxk = n[xsites[k]]
                for j in 1:ny
                    pairInt[k, j] += nxk * n[ysites[j]] * dtt
                end
            end
        end

        tnext >= Ttot && break

        u = rand(rng) * R
        s = 0.0
        ev = 0
        @inbounds for j in 1:2N
            s += rates[j]
            if u <= s
                ev = j
                break
            end
        end

        if ev <= N - 1
            i = ev
            n[i] = 0
            n[i + 1] = 1

        elseif ev <= 2N - 2
            i = ev - (N - 1)
            n[i + 1] = 0
            n[i] = 1

        elseif ev == 2N - 1
            if sigma == 1
                n[N] = 0
                n[1] = 1
            else
                n[1] = 0
                n[N] = 1
            end

        else
            println("fliped")
            sigma = -sigma
        end

        t = tnext
    end

    rhoAvg = densInt ./ Tmeas
    pairAvg = pairInt ./ Tmeas

    corr = zeros(Float64, nx, ny)
    @inbounds for k in 1:nx
        xsite = xsites[k]
        for j in 1:ny
            ysite = ysites[j]
            corr[k, j] = pairAvg[k, j] - rhoAvg[xsite] * rhoAvg[ysite]
        end
    end

    return rhoAvg, corr
end

# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
function ssep_fluctuating_bond(;
    rho::Float64 = RHO,
    alpha::Float64 = ALPHA,
    Tburn::Float64 = TBURN,
    Tmeas::Float64 = TMEAS,
    nRuns::Int = NRUNS,
    seed::Int = SEED,
)
    M = round(Int, rho * N)
    yhalf = div(N, 2)

    xvals_phys = filter(x -> x <= yhalf, XVALS)
    yvals_phys = vcat(collect(-(yhalf - 1):-1), collect(1:yhalf))

    nx = length(xvals_phys)
    ny = length(yvals_phys)

    rho_runs = [zeros(Float64, N) for _ in 1:nRuns]
    corr_runs = [zeros(Float64, nx, ny) for _ in 1:nRuns]

    @threads for run in 1:nRuns
        rhoOne, corrOne = one_run_ssep(
            N, M, alpha, Tburn, Tmeas,
            xvals_phys, yvals_phys, seed + run
        )
        rho_runs[run] .= rhoOne
        corr_runs[run] .= corrOne
        println("run $run done")
    end

    rhoAvg = zeros(Float64, N)
    corrAvg = zeros(Float64, nx, ny)

    for run in 1:nRuns
        println("run $run done")
        rhoAvg .+= rho_runs[run]
        corrAvg .+= corr_runs[run]
    end

    rhoAvg ./= nRuns
    corrAvg ./= nRuns

    println("Done.")
    println("Julia threads available: ", Threads.nthreads())
    println("N = ", N)
    println("rho = ", rho)
    println("M = ", M)
    println("alpha = ", alpha)
    println("Tburn = ", Tburn)
    println("Tmeas = ", Tmeas)
    println("nRuns = ", nRuns)
    println("x values = ", xvals_phys)
    println("Mean density = ", mean(rhoAvg))
    println("Max density deviation = ", maximum(abs.(rhoAvg .- mean(rhoAvg))))

    p1 = plot(
        1:N,
        rhoAvg;
        marker = :circle,
        linewidth = 2,
        label = "simulation",
        xlabel = "internal site index",
        ylabel = "⟨n⟩",
        title = "Average density, N=50, ρ=$(rho), α=$(alpha), runs=$(nRuns)",
        legend = :best,
    )

    plot!(
        p1,
        [1, N],
        [rho, rho];
        linestyle = :dash,
        linewidth = 2,
        label = "target density ρ",
    )

    display(p1)

    p2 = plot(
        xlabel = "y (distance from fluctuating bond between 0 and 1)",
        ylabel = "C(x,y)",
        title = "Connected correlations, negative and positive y, N=50, runs=$(nRuns)",
        legend = :best,
    )

    for (k, x) in enumerate(xvals_phys)
        idx = yvals_phys .!= x
        plot!(
            p2,
            yvals_phys[idx],
            corrAvg[k, idx];
            marker = :circle,
            linewidth = 2,
            label = "x = $x",
        )
    end

    display(p2)

    plot_dir = joinpath("results_figures", "ssep_fluctuating_bond")
    mkpath(plot_dir)
    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    run_tag = "N$(N)_rho$(rho)_alpha$(alpha)_runs$(nRuns)_$(timestamp)"
    density_plot_path = joinpath(plot_dir, "density_" * run_tag * ".png")
    corr_plot_path = joinpath(plot_dir, "correlation_" * run_tag * ".png")
    savefig(p1, density_plot_path)
    savefig(p2, corr_plot_path)
    println("Saved density plot to $density_plot_path")
    println("Saved correlation plot to $corr_plot_path")

    return rhoAvg, corrAvg, xvals_phys, yvals_phys
end

rhoAvg, corrAvg, xvals_phys, yvals_phys = ssep_fluctuating_bond()
