# At the top of the file, outside the FP module
using Statistics
using FFTW
using LsqFit

#Wrap everything with a module to allow redefinition of type
module FP

    # Add these imports at the top of the file
    using Statistics
    using FFTW
    using LsqFit

    mutable struct Param
        D::Float64  # diffusion constant
        # Lx::Int64   # system size along x
        # Ly::Int64   # system size along y
        dims::Vector{Int} # system sizes for each dimension

        ρ₀::Float64  # density
        N::Int64    # number of particles
    end

    #constructor
    function setParam(D, dims::Vector{Int}, ρ₀)

        N = Int(round( ρ₀*prod(dims)))       # number of particles
        param = Param(D, dims, ρ₀, N)
        # param = Param(D, Lx, Ly, ρ₀, N)
        return param
    end


    mutable struct State
        t::Float64              # time
        pos::Array{Int64, 2}    # position vector
        ρ::Array{Int64}      # density field
    end

    function setState(t, param, pos)

        ρ = zeros(Int64, param.dims...)
        # ρ = zeros(Int64, param.Lx, param.Ly)

        for n in 1:param.N
            idxs = ntuple(i -> pos[n,i], length(param.dims))
            ρ[idxs...] += 1
            # x, y = pos[n, 1], pos[n,2]
            # ρ[x,y] += 1
        end

        state = State(t, pos, ρ)
        return state
    end

    function update!(Δt, param, state, rng)
        D = param.D

        t_end = state.t + Δt
        while state.t <= t_end
            n = rand(rng, 1:param.N)

            # Define weights for each dimension (assuming equal weights for simplicity)
            weights = fill(D, length(param.dims) * 2)  # Each dimension has two directions
            w_sum = sum(weights)

            # Use tower_sampling to select a move
            move_idx = tower_sampling(weights, w_sum, rng)
            d = div(move_idx + 1, 2)
            move = (move_idx % 2 == 1) ? 1 : -1

            # Update position with periodic boundary conditions
            state.pos[n, d] = mod1(state.pos[n, d] + move, param.dims[d])

            state.t += 1 / (param.N * D)
        end

        # Update the density field
        fill!(state.ρ, 0)
        for n in 1:param.N
            idxs = ntuple(i -> state.pos[n, i], length(param.dims))
            state.ρ[idxs...] += 1
        end
    end
    # function update!(Δt, param, state, rng)
    #     D = param.D

    #     t_end = state.t + Δt
    #     while state.t ≤ t_end
    #         n = rand(rng, 1:param.N)

    #         w = [D, D, D, D]  # right, up, left, down
    #         w_sum=sum(w)
    #         move = tower_sampling(w, w_sum, rng)

    #         if move==1
    #             state.pos[n,1] = mod1( state.pos[n,1] + 1, param.Lx)
    #         elseif move==2
    #             state.pos[n,2] = mod1( state.pos[n,2] + 1, param.Ly)
    #         elseif move==3
    #             state.pos[n,1] = mod1( state.pos[n,1] - 1, param.Lx)
    #         else
    #             state.pos[n,2] = mod1( state.pos[n,2] - 1, param.Ly)
    #         end

    #         state.t += 1/(param.N*w_sum)
    #     end

    #     # update the density field
    #     fill!(state.ρ, 0.0)
    #     for n in 1:param.N
    #         x, y = state.pos[n, 1], state.pos[n,2]
    #         state.ρ[x,y] += 1
    #     end
    # end

    function tower_sampling(weights, w_sum, rng)
        key = w_sum*rand(rng)

        selected = 1
        gathered = weights[selected]
        while gathered < key
            selected += 1
            gathered += weights[selected]
        end

        return selected
    end
end


function compute_spatial_correlation(ρ)
    F = fft(ρ)
    power_spectrum = F .* conj(F)
    corr = real(ifft(power_spectrum))
    return fftshift(corr) / (size(ρ, 1) * size(ρ, 2))
end

function compute_time_correlation(ρ_history)
    n_frames = size(ρ_history, ndims(ρ_history))
    corr = zeros(n_frames)
    ρ_mean = mean(ρ_history)
    ρ_var = var(ρ_history)

    for dt in 0:(n_frames-1)
        c = 0.0
        for t in 1:(n_frames-dt)
            c += mean((ρ_history[..., t] .- ρ_mean) .* (ρ_history[..., t+dt] .- ρ_mean))
        end
        corr[dt+1] = c / ((n_frames-dt) * ρ_var)
    end

    return corr
end
# function compute_time_correlation(ρ_history)
#     n_frames = size(ρ_history, 3)
#     corr = zeros(n_frames)
#     ρ_mean = mean(ρ_history)
#     ρ_var = var(ρ_history)
    
#     for dt in 0:(n_frames-1)
#         c = 0.0
#         for t in 1:(n_frames-dt)
#             c += mean((ρ_history[:,:,t] .- ρ_mean) .* (ρ_history[:,:,t+dt] .- ρ_mean))
#         end
#         corr[dt+1] = c / ((n_frames-dt) * ρ_var)
#     end
    
#     return corr
# end

function fit_exponential(t, y)
    model(t, p) = p[1] .* exp.(-t ./ p[2]) .+ p[3]  
    p0 = [1.0, 1.0, 0.0]  # Initial guess for [amplitude, decay time, offset]
    fit = curve_fit(model, t, y, p0)
    return fit.param
end

function initialize_simulation(state, param, n_frame)
    state.t = 0
    prg = Progress(n_frame)
    ρ_history = zeros(param.Lx, param.Ly, n_frame)
    decay_times = Float64[]
    return prg, ρ_history, decay_times
end

function update_and_compute_correlations(state, param, t_gap, ρ_history, frame, rng)
    FP.update!(t_gap, param, state, rng)
    ρ_history[:,:,frame] = state.ρ
    spatial_corr = compute_spatial_correlation(state.ρ)
    time_corr = compute_time_correlation(ρ_history[:,:,1:frame])
    return spatial_corr, time_corr
end

function plot_density(state, param)
    x_range = range(1, param.Lx, length = param.Lx)
    y_range = range(1, param.Ly, length = param.Ly)
    heatmap(x_range, y_range, transpose(state.ρ), 
            title="Density", 
            c=cgrad(:inferno), xlims=(1, param.Lx), ylims=(1, param.Ly), 
            clims=(0,3), aspect_ratio=1, xlabel="x", ylabel="y")
end

function plot_spatial_correlation(spatial_corr, param)
    dx_range = range(-param.Lx÷2, param.Lx÷2-1, length=param.Lx)
    dy_range = range(-param.Ly÷2, param.Ly÷2-1, length=param.Ly)
    heatmap(dx_range, dy_range, transpose(spatial_corr), 
            title="Spatial Correlation", 
            c=cgrad(:viridis), 
            aspect_ratio=1, xlabel="Δx", ylabel="Δy")
end

function plot_time_correlation(time_corr, frame, fit_params=nothing)
    t = 0:(frame-1)
    p = plot(t, time_corr, 
             title="Time Correlation", 
             xlabel="Δt", ylabel="C(Δt)", 
             legend=false, lw=2, label="Data")
    
    if !isnothing(fit_params)
        plot!(p, t, fit_params[1] .* exp.(-t ./ fit_params[2]) .+ fit_params[3], 
              lw=2, ls=:dash, label="Fit")
        annotate!(p, [(frame/2, 0.8, text("τ = $(round(fit_params[2], digits=2))", 10))])
    end
    
    return p
end

function make_movie!(state, param, t_gap, n_frame, rng, file_name, in_fps)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_frame)

    anim = @animate for frame in 1:n_frame
        spatial_corr, time_corr = update_and_compute_correlations(state, param, t_gap, ρ_history, frame, rng)
        
        p1 = plot_density(state, param)
        p2 = plot_spatial_correlation(spatial_corr, param)
        
        if frame > 10
            fit_params = fit_exponential(0:(frame-1), time_corr)
            push!(decay_times, fit_params[2])
            p3 = plot_time_correlation(time_corr, frame, fit_params)
        else
            p3 = plot_time_correlation(time_corr, frame)
        end
        
        plot(p1, p2, p3, size=(1800,600), layout=(1,3))
        next!(prg)
    end

    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)

    plot_decay_time_evolution(decay_times)
end

function plot_decay_time_evolution(decay_times)
    p_decay = plot(decay_times, 
                   title="Evolution of Decay Time", 
                   xlabel="Frame", ylabel="τ", 
                   legend=false, lw=2)
    savefig(p_decay, "decay_time_evolution.png")
end
