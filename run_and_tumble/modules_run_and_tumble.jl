# At the top of the file, outside the FP module
using Statistics
using FFTW
using LsqFit

#Wrap everything with a module to allow redefinition of type
module FP
    using LinearAlgebra
    # dim_num::Int = 1

    struct Param # model parameters
        α::Float64  # rate of tumbling
        dims::Tuple{Vararg{Int}} # systen's dimensions
        ρ₀::Float64  # density
        N::Int64    # number of particles
        D::Float64  #diffusion coefficient
    end

    #constructor
    function setParam(α, dims, ρ₀, D)

        N = Int(round( ρ₀*prod(dims)))       # number of particles

        param = Param(α, dims, ρ₀, N, D)
        return param
    end

    mutable struct Particle
        position::Array{Int64}
        direction::Array{Float64}
    end

    function setParticle(sys_params,rng)
        dim_num = length(sys_params.dims)
        position = zeros(Int64, dim_num)
        direction = zeros(Float64, dim_num)
        for (i,dim) in enumerate(sys_params.dims)
            position[i] = rand(rng, 1:dim)
            direction[i] = rand(rng,1:dim)
        end
        normalize!(direction)
        Particle(position,direction)
    end

    mutable struct State
        t::Float64              # time
        #pos::Array{Int64, 1+dim_num}    # position vector
        particles::Array{Particle}
        ρ::Array{Int64}      # density field
        T::Float64                      # temperature
        V::Array{Float64}      # potential
    end

    function setState(t, rng, param, T, V=zeros(Float64,param.dims))
        println(V)
        N = param.N
        # initialize particles
        particles = Array{Particle}(undef,N)
        for n in 1:N
            particles[n]= setParticle(param, rng)
        end

        # Initialize the matrix with dimensions specified by a tuple
        ρ = zeros(Int64, param.dims...)

        # Iterate over the positions and update the matrix
        for n in 1:N
            indices = Tuple(particles[n].position[i] for i in 1:length(param.dims))
            ρ[CartesianIndex(indices...)] += 1
        end
        state = State(t, particles, ρ, T, V)
        return state
    end

    function calculate_jump_probability(direction,D,ΔV,T,ϵ=0.1)
        p=(D+ϵ*direction)*min(1,exp(-ΔV*T))
        return p
    end

    function update!(Δt, param, state, rng)

        if length(param.dims) == 1
            V = state.V
            T = state.T
            α = param.α
            t_end = state.t + Δt
            while state.t ≤ t_end
                n = rand(rng,1:param.N)
                particle = state.particles[n]
                spot_index = particle.position[1]
                left_index= mod1(spot_index-1,param.dims[1])
                right_index = mod1(spot_index+1,param.dims[1])
                p_right = calculate_jump_probability(particle.direction[1], param.D, V[right_index]-V[spot_index],T)
                p_left = calculate_jump_probability(particle.direction[1], param.D, V[left_index]-V[spot_index],T)
                p_tumble = α/2
                # p_stay = α/2

                state.ρ[spot_index] -= 1

                possible_moves = [p_left, p_right, p_tumble ]  # right, left, tumble
                w_sum=sum(possible_moves)
                choice = tower_sampling(possible_moves, w_sum, rng)
                if choice==1
                    particle.position[1]=mod1(particle.position[1]-1,param.dims[1])
                elseif choice==2
                    particle.position[1]=mod1(particle.position[1]+1,param.dims[1])
                elseif choice==3
                    particle.direction[1]*=-1
                end
                new_position = particle.position[1]

                state.ρ[new_position] += 1

                
                state.t += 1/(param.N*w_sum) # find correct time increment , in the previous simulation divided by w_sum
            end
            
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end

    end

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
    return fftshift(corr) / prod(size(ρ))
end

function compute_time_correlation(ρ_history)
    n_frames = size(ρ_history, ndims(ρ_history))
    corr = zeros(n_frames)
    ρ_mean = mean(ρ_history)
    ρ_var = var(ρ_history)
    dim_num=length(size(ρ_history))-1
    if dim_num==1 
        for dt in 0:(n_frames-1)
            c = 0.0
            for t in 1:(n_frames-dt)
                c += mean((ρ_history[:,t] .- ρ_mean) .* (ρ_history[:,t+dt] .- ρ_mean))
            end
            corr[dt+1] = c / ((n_frames-dt) * ρ_var)
        end
    elseif dim_num==2
        for dt in 0:(n_frames-1)
            c = 0.0
            for t in 1:(n_frames-dt)
                c += mean((ρ_history[:,:,t] .- ρ_mean) .* (ρ_history[:,:,t+dt] .- ρ_mean))
            end
            corr[dt+1] = c / ((n_frames-dt) * ρ_var)
        end
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
    
    return corr
end

function fit_exponential(t, y)
    model(t, p) = p[1] .* exp.(-t ./ p[2]) .+ p[3]  
    p0 = [1.0, 1.0, 0.0]  # Initial guess for [amplitude, decay time, offset]
    fit = curve_fit(model, t, y, p0)
    return fit.param
end

function initialize_simulation(state, param, n_frame)
    state.t = 0
    prg = Progress(n_frame)
    ρ_history = zeros((param.dims..., n_frame))
    decay_times = Float64[]
    return prg, ρ_history, decay_times
end

function update_and_compute_correlations(state, param, t_gap, ρ_history, frame, rng)
    FP.update!(t_gap, param, state, rng)
    dim_num= length(param.dims)
    if dim_num==1
        ρ_history[:,frame] = state.ρ
        spatial_corr = compute_spatial_correlation(state.ρ)
        time_corr = compute_time_correlation(ρ_history[:,1:frame])
    elseif dim_num==2
        ρ_history[:,:,frame] = state.ρ
        spatial_corr = compute_spatial_correlation(state.ρ)
        time_corr = compute_time_correlation(ρ_history[:,:,1:frame])
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
    return spatial_corr, time_corr
end

function plot_density(state, param)
    dim_num = length(param.dims)
    if dim_num==1
        x_range = 1:param.dims[1]
        plot(x_range, state.ρ, 
             title="1D Density", 
             xlabel="Position", ylabel="Density", 
             legend=false, lw=2)
    elseif dim_num == 2
        x_range = range(1, param.Lx, length = param.Lx)
        y_range = range(1, param.Ly, length = param.Ly)
        heatmap(x_range, y_range, transpose(state.ρ), 
                title="Density", 
                c=cgrad(:inferno), xlims=(1, param.Lx), ylims=(1, param.Ly), 
                clims=(0,3), aspect_ratio=1, xlabel="x", ylabel="y")
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
end

function plot_spatial_correlation(spatial_corr, param)
    dim_num = length(param.dims)
    if dim_num == 1
        # Adjust dx_range to match the length of spatial_corr
        dx_range = range(-param.dims[1] ÷ 2, length=length(spatial_corr))
        plot(dx_range, spatial_corr,
             title="1D Spatial Correlation",
             xlabel="Δx", ylabel="Correlation",
             legend=false, lw=2)
        # Define the range for Δx
        # dx_range = range(-param.dims[1] ÷ 2, param.dims[1] ÷ 2 - 1, length=param.dims[1])
        # # Plot the 1D spatial correlation
        # plot(dx_range, spatial_corr,
        #      title="1D Spatial Correlation",
        #      xlabel="Δx", ylabel="Correlation",
        #      legend=false, lw=2)
    elseif dim_num == 2
        # Define the ranges for Δx and Δy
        dx_range = range(-param.dims[1] ÷ 2, param.dims[1] ÷ 2 - 1, length=param.dims[1])
        dy_range = range(-param.dims[2] ÷ 2, param.dims[2] ÷ 2 - 1, length=param.dims[2])
        # Plot the 2D spatial correlation as a heatmap
        heatmap(dx_range, dy_range, transpose(spatial_corr),
                title="2D Spatial Correlation",
                c=cgrad(:viridis),
                aspect_ratio=1, xlabel="Δx", ylabel="Δy")
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
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

    # Initialize the animation
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
    # for frame in 1:n_frame
    #     spatial_corr, time_corr = update_and_compute_correlations(state, param, t_gap, ρ_history, frame, rng)
    #     p1 = plot_density(state, param)
    #     p2 = plot_spatial_correlation(spatial_corr, param)
        
    #     if frame > 10
    #         fit_params = fit_exponential(0:(frame-1), time_corr)
    #         push!(decay_times, fit_params[2])
    #         p3 = plot_time_correlation(time_corr, frame, fit_params)
    #     else
    #         p3 = plot_time_correlation(time_corr, frame)
    #     end
        
    #     plot_object=plot(p1, p2, p3, size=(1800,600), layout=(1,3))
    #     next!(prg)
    # end
    # display(plot_object)

    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)

    another_plot=plot_decay_time_evolution(decay_times)
    display(another_plot)
end

function plot_decay_time_evolution(decay_times)
    p_decay = plot(decay_times, 
                   title="Evolution of Decay Time", 
                   xlabel="Frame", ylabel="τ", 
                   legend=false, lw=2)
    savefig(p_decay, "decay_time_evolution.png")
end
