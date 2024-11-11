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
        ρ_avg::Array{Float64} #time averaged density field
        T::Float64                      # temperature
        V::Array{Float64}      # potential
    end

    function setState(t, rng, param, T, V=zeros(Float64,param.dims))
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
        ρ_avg = Float64.(ρ)
        state = State(t, particles, ρ, ρ_avg, T, V)
        return state
    end

    function calculate_jump_probability(direction,D,ΔV,T,ϵ=0.9)
        p=(D+ϵ*direction)*min(1,exp(-ΔV*T))/(D+ϵ)
        return p
    end

    function update!(Δt, param, state, rng)

        if length(param.dims) == 1
            V = state.V
            T = state.T
            α = param.α
            t_end = state.t + Δt
            while state.t < t_end
                n_and_a = rand(rng,1:3*param.N)
                n = (n_and_a-mod1(n_and_a,3)) ÷ 3 +1
                particle = state.particles[n]
                spot_index = particle.position[1]
                left_index= mod1(spot_index-1,param.dims[1])
                right_index = mod1(spot_index+1,param.dims[1])

                action_index = mod1(n_and_a,3)
                state.ρ[spot_index] -= 1
                candidate_spot_index = 0
                if action_index == 1 # left
                    candidate_spot_index = left_index
                elseif action_index == 2 # right
                    candidate_spot_index = right_index
                elseif action_index == 3 # tumble
                    candidate_spot_index = spot_index
                end
                if action_index==3
                    p_candidate=α
                else
                    p_candidate= calculate_jump_probability(particle.direction[1], param.D, V[candidate_spot_index]-V[spot_index],T)
                end
                p_stay = 1-p_candidate
                p_arr = [p_candidate, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr),rng)
                if choice == 1
                    particle.position[1] =  candidate_spot_index
                    if action_index==3
                        particle.direction[1]*=-1
                    end
                end
                new_position = particle.position[1]
                state.ρ[new_position] += 1

                # p_left = calculate_jump_probability(particle.direction[1], param.D, V[left_index]-V[spot_index],T)
                # p_right = calculate_jump_probability(particle.direction[1], param.D, V[right_index]-V[spot_index],T)
                # p_stay = calculate_jump_probability(particle.direction[1], param.D, 0, T)
                # p_tumble = 0 #α/2
                # # p_stay = α/2


                # possible_moves = [p_left, p_right, p_stay ,p_tumble ]  # right, left, tumble
                # w_sum=sum(possible_moves)
                # choice = tower_sampling(possible_moves, w_sum, rng)
                # if choice==1
                #     # particle.position[1]=mod1(particle.position[1]-1,param.dims[1])
                #     particle.position[1] = left_index
                #     # println("choice left")
                # elseif choice==2
                #     # particle.position[1]=mod1(particle.position[1]+1,param.dims[1])
                #     particle.position[1] = right_index
                #     # println("choice right")
                # elseif choice==3
                #     particle.direction[1]*=-1
                #     # println("choice tumble")
                # end

                
                # state.t += 1/(param.N*w_sum) # find correct time increment , in the previous simulation divided by w_sum
                state.t += 1/param.N
                #state.t += Δt
            end
            
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end

    end

    function tower_sampling(weights, w_sum, rng)
        #key = w_sum*rand(rng)
        key = w_sum*rand()

        selected = 1
        gathered = weights[selected]
        while gathered < key
            selected += 1
            gathered += weights[selected]
        end

        return selected
    end
end



function fit_exponential(t, y)
    model(t, p) = p[1] .* exp.(-t ./ p[2]) .+ p[3]  
    p0 = [1.0, 1.0, 0.0]  # Initial guess for [amplitude, decay time, offset]
    fit = curve_fit(model, t, y, p0)
    return fit.param
end

function update_time_averaged_density_field(density_field_history)

end
function calculate_time_averaged_density_field(density_field_history)
    n_dims = length(size(density_field_history))-1
    return mean(density_field_history,dims=n_dims+1)[:,1]

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

function update_and_compute_correlations!(state, param, t_gap, ρ_history, frame, rng, calc_correlations=false)
    FP.update!(t_gap, param, state, rng)
    dim_num= length(param.dims)
    if dim_num==1
        state.ρ_avg = (state.ρ_avg * (frame-1)+state.ρ)/frame
        # time_averaged_desnity_field = calculate_time_averaged_density_field(ρ_history[:,1:frame])
        if !calc_correlations
            return nothing, nothing
        else
            ρ_history[:,frame] = state.ρ
            spatial_corr = compute_spatial_correlation(state.ρ)
            time_corr = compute_time_correlation(ρ_history[:,1:frame])
        end
        # spatial_corr = compute_spatial_correlation(zeros(size(state.ρ)))
        # time_corr = compute_spatial_correlation(zeros(size(ρ_history[:,1:frame])))
    elseif dim_num==2
        time_averaged_desnity_field = calculate_time_averaged_density_field(ρ_history[:,:,1:frame])
        if !calc_correlations
            return nothing, nothing
        else
            ρ_history[:,:,frame] = state.ρ
            spatial_corr = compute_spatial_correlation(state.ρ)
            time_corr = compute_time_correlation(ρ_history[:,:,1:frame])
        end
    else
        throw(DomainError("Invalid input - dimension not supported yet"))
    end
    return spatial_corr, time_corr
end

# function plot_density(state, param)
#     dim_num = length(param.dims)
#     if dim_num==1
#         x_range = 1:param.dims[1]
#         plot(x_range, state.ρ, 
#              title="1D Density", 
#              xlabel="Position", ylabel="Density", 
#              legend=false, lw=2)
#     elseif dim_num == 2
#         x_range = range(1, param.Lx, length = param.Lx)
#         y_range = range(1, param.Ly, length = param.Ly)
#         heatmap(x_range, y_range, transpose(state.ρ), 
#                 title="Density", 
#                 c=cgrad(:inferno), xlims=(1, param.Lx), ylims=(1, param.Ly), 
#                 clims=(0,3), aspect_ratio=1, xlabel="x", ylabel="y")
#     else
#         throw(DomainError("Invalid input - dimension not supported yet"))
#     end
# end
function plot_density(density,param; title = "Density")
    dim_num = length(size(density))
    if dim_num==1
        x_range = 1:param.dims[1]
        plot(x_range, density, 
             title=title, 
             xlabel="Position", ylabel=title, 
             legend=false, lw=2,seriestype=:scatter)
    elseif dim_num == 2
        Lx= param.dims[1]
        Ly= param.dims[2]
        x_range = range(1, Lx, length = Lx)
        y_range = range(1, Ly, length = Ly)
        heatmap(x_range, y_range, transpose(density), 
                title=title, 
                c=cgrad(:inferno), xlims=(1, Lx), ylims=(1, Ly), 
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

function initialize_simulation(state, param, n_frame)
    state.t = 0
    prg = Progress(n_frame)
    ρ_history = zeros((param.dims..., n_frame))
    decay_times = Float64[]
    return prg, ρ_history, decay_times
end


function run_simulation!(state, param, t_gap, n_sweeps, rng, calc_correlations = false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_sweeps)
    # Initialize the animation
    for sweep in 1:n_sweeps
        spatial_corr, time_corr = update_and_compute_correlations!(state, param, t_gap, ρ_history, sweep, rng)
        # println(size(time_averaged_desnity_field))
        # println(size(state.ρ))
        # p = plot_density(state.particles[1].position,param)
        # p0 = plot_density(state.ρ_avg, param; title="time averaged density")
        # p1 = plot_density(state.ρ, param)

        if calc_correlations
            p2 = plot_spatial_correlation(spatial_corr, param)
            
            if frame > 10
                fit_params = fit_exponential(0:(sweep-1), time_corr)
                push!(decay_times, fit_params[2])
                p3 = plot_time_correlation(time_corr, sweep, fit_params)
            else
                p3 = plot_time_correlation(time_corr, sweep)
            end
            plot(p0,p1, p2, p3, size=(1200,1200), layout=(2,2))
        else
            # plot(p0, p1, size=(1200,600), layout=(2,1))
        end
        next!(prg)
    end
    normalized_dist = state.ρ_avg/ sum(state.ρ_avg)
    # p0 = plot_density(state.ρ_avg, param; title="time averaged density")
    p0 = plot_density(normalized_dist, param; title="time averaged density")
    p1 = plot_density(state.ρ, param)
    p=plot(p0, p1, size=(1200,600), layout=(2,1))
    display(p)

    #
    println("Simulation complete")
    
    # proportion_vec = abs.(state.ρ_avg-(exp.(-state.V/state.T)))./exp.(-state.V/state.T)
    exp_expression= exp.(-state.V/state.T)
    boltzman_dist= exp_expression/ sum(exp_expression)
    proportion_vec = normalized_dist - boltzman_dist
    plot(proportion_vec; title="proportion_vec")

end


function make_movie!(state, param, t_gap, n_frame, rng, file_name, in_fps, calc_correlations = false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_frame)
    # Initialize the animation
    anim = @animate for frame in 1:n_frame
        spatial_corr, time_corr = update_and_compute_correlations!(state, param, t_gap, ρ_history, frame, rng)
        # println(size(time_averaged_desnity_field))
        # println(size(state.ρ))
        # p = plot_density(state.particles[1].position,param)
        p0 = plot_density(state.ρ_avg, param; title="time averaged density")
        p1 = plot_density(state.ρ, param)

        if calc_correlations
            p2 = plot_spatial_correlation(spatial_corr, param)
            
            if frame > 10
                fit_params = fit_exponential(0:(frame-1), time_corr)
                push!(decay_times, fit_params[2])
                p3 = plot_time_correlation(time_corr, frame, fit_params)
            else
                p3 = plot_time_correlation(time_corr, frame)
            end
            plot(p0,p1, p2, p3, size=(1200,1200), layout=(2,2))
        else
            plot(p0, p1, size=(1200,600), layout=(2,1))
        end
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

    #plot final configuration
    p0 = plot_density(state.ρ_avg, param; title="time averaged density")
    p1 = plot_density(state.ρ, param)
    p=plot(p0, p1, size=(1200,600), layout=(2,1))
    display(p)

    #
    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)



    # another_plot=plot_decay_time_evolution(decay_times)
    # display(another_plot)
end

function plot_decay_time_evolution(decay_times)
    p_decay = plot(decay_times, 
                   title="Evolution of Decay Time", 
                   xlabel="Frame", ylabel="τ", 
                   legend=false, lw=2)
    savefig(p_decay, "decay_time_evolution.png")
end
