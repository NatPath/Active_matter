# At the top of the file, outside the FP module
using Statistics
using FFTW
using LsqFit

#Wrap everything with a module to allow redefinition of type
module FP
    using LinearAlgebra

    struct Param # model parameters
        α::Float64  # rate of tumbling
        β::Float64  # rate of potential fluctuation
        dims::Tuple{Vararg{Int}} # systen's dimensions
        ρ₀::Float64  # density
        N::Int64    # number of particles
        D::Float64  #diffusion coefficient
    end

    #constructor
    function setParam(α ,β, dims, ρ₀, D)

        N = Int(round( ρ₀*prod(dims)))       # number of particles

        param = Param(α, β, dims, ρ₀, N, D)
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
            direction[i] = rand(rng,1:dim)*rand(rng,[-1,1])
        end
        direction = direction ./ norm(direction)
        print(direction)
        Particle(position,direction)
    end

    mutable struct Potential 
        V::Array{Float64}      # potential
        fluctuation_mask::Array{Float64}
        fluctuation_sign::Int64
        
    end
    function setPotential(V, fluctuation_mask)
        return Potential(V, fluctuation_mask, 1)
    end

    mutable struct State
        t::Float64              # time
        #pos::Array{Int64, 1+dim_num}    # position vector
        particles::Array{Particle}
        ρ::Array{Int64}      # density field
        ρ₊::Array{Int64}      # density field
        ρ₋::Array{Int64}      # density field
        ρ_avg::Array{Float64} #time averaged density field
        ρ_matrix_avg::Array{Float64}
        T::Float64                      # temperature
        # V::Array{Float64}      # potential
        potential::Potential
    end

    function setState(t, rng, param, T, potential=setPotential(zeros(Float64,param.dims),zeros(Float64,param.dims)))
        N = param.N
        # initialize particles
        particles = Array{Particle}(undef,N)
        for n in 1:N
            particles[n]= setParticle(param, rng)
        end

        # Initialize the matrix with dimensions specified by a tuple
        ρ = zeros(Int64, param.dims...)
        ρ₊ = zeros(Int64, param.dims...)
        ρ₋ = zeros(Int64, param.dims...)

        # Iterate over the positions and update the matrix
        for n in 1:N
            indices = Tuple(particles[n].position[i] for i in 1:length(param.dims))
            ρ[CartesianIndex(indices...)] += 1
            if particles[n].direction[1]==1
                ρ₊[CartesianIndex(indices...)] += 1
            elseif particles[n].direction[1] ==-1
                ρ₋[CartesianIndex(indices...)] += 1
            end
        end
        ρ_avg = Float64.(ρ)
        ρ_matrix_avg = ρ_avg * transpose(ρ_avg) 
        state = State(t, particles, ρ,ρ₊,ρ₋, ρ_avg, ρ_matrix_avg, T, potential)
        return state
    end

    function calculate_jump_probability(particle_direction,choice_direction,D,ΔV,T,ϵ=0.6, ΔV_max=0.4)
        relative_direction = particle_direction*choice_direction
        
        # p =(D+ϵ*relative_direction-ΔV)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
        # if ΔV!=0
        #     p =(D+ϵ*relative_direction)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
        # else
        #     p =(D+ϵ*relative_direction/2)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
        # end
        if relative_direction==1
            p = (D+ ϵ*relative_direction- ΔV/T) / ( D+ ϵ + ΔV_max/T)
        else
            p=0
        end
        return p
    end

    function update!(Δt, param, state, rng)

        if length(param.dims) == 1
            V = state.potential.V
            T = state.T
            α = param.α
            β = param.β
            t_end = state.t + Δt
            while state.t < t_end
                n_and_a = rand(rng,1:4*param.N)
                n = (n_and_a-mod1(n_and_a,4)) ÷ 4 +1
                particle = state.particles[n]
                spot_index = particle.position[1]
                left_index= mod1(spot_index-1,param.dims[1])
                right_index = mod1(spot_index+1,param.dims[1])

                action_index = mod1(n_and_a,4)
                state.ρ[spot_index] -= 1
                
                candidate_spot_index = 0
                if action_index == 1 # left
                    candidate_spot_index = left_index
                    choice_direction = -1
                elseif action_index == 2 # right
                    candidate_spot_index = right_index
                    choice_direction = 1
                else  # tumble or fluctuate potential
                    candidate_spot_index = spot_index
                    choice_direction = 0
                end

                if action_index==3
                    p_candidate=α
                elseif action_index==4
                    p_candidate=β
                else
                    p_candidate= calculate_jump_probability(particle.direction[1], choice_direction, param.D, V[candidate_spot_index]-V[spot_index],T)
                end
                p_stay = 1-p_candidate
                p_arr = [p_candidate, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr),rng)
                if choice == 1
                    particle.position[1] =  candidate_spot_index

                    if particle.direction[1] == 1
                        state.ρ₊[spot_index]-=1
                        state.ρ₊[candidate_spot_index]+=1
                    elseif particle.direction[1] ==-1
                        state.ρ₋[spot_index]-=1
                        state.ρ₋[candidate_spot_index]+=1
                    end

                    if action_index==3
                        state.ρ₊[spot_index]-= particle.direction[1]
                        state.ρ₋[spot_index]+= particle.direction[1]
                        particle.direction[1]*=-1
                        # println("tumbled")
                    end
                    if action_index == 4
                        state.potential.V += state.potential.fluctuation_mask*state.potential.fluctuation_sign
                        state.potential.fluctuation_sign*=-1
                    end
                end
                new_position = particle.position[1]
                state.ρ[new_position] += 1

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
        ρ_matrix = state.ρ*transpose(state.ρ)
        state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-1)+ρ_matrix)/frame
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
function plot_density(density, param, state; title="Density", show_directions=false)
    dim_num = length(size(density))
    if dim_num==1
        if !show_directions
            # Original plotting code
            x_range = 1:param.dims[1]
            # p = plot(x_range, density, 
            #      title=title, 
            #      xlabel="Position", ylabel=title, 
            #      legend=false, lw=2, seriestype=:scatter)
            p = plot(x_range, density,
                title=title,
                xlabel="Position",
                ylabel="Density",
                label="Density",
                lw=2,
                seriestype=:scatter,
                color=:blue,
                legend=:topleft)
            
            # Secondary axis for potential
            plot!(twinx(), x_range, state.potential.V,
                  ylabel="Potential",
                  label="Potential",
                  color=:red,
                  alpha=0.3,
                  linestyle=:dash,
                  legend=:topright)
        else
            # Create subplot with total density, right-moving and left-moving particles
            x_range = 1:param.dims[1]
            p1 = plot(x_range, density, 
                     title="Total Density", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter)
            
            p2 = plot(x_range, state.ρ₊,
                     title="Right-moving (ρ₊)", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter,
                     color=:red)
            
            p3 = plot(x_range, state.ρ₋,
                     title="Left-moving (ρ₋)", 
                     xlabel="Position", ylabel="Density",
                     legend=false, lw=2, seriestype=:scatter,
                     color=:blue)
            
            p = plot(p1, p2, p3, layout=(3,1), size=(600,600))
        end
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
    return p
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

function initialize_simulation(state, param, n_frame, calc_correlations)
    state.t = 0
    prg = Progress(n_frame)
    decay_times = Float64[]
    if calc_correlations
        ρ_history = zeros((param.dims..., n_frame))
        return prg, ρ_history, decay_times
    else
        ρ_history = zeros((param.dims..., 1))
        return prg, ρ_history, decay_times
    end
end


function run_simulation!(state, param, t_gap, n_sweeps, rng; calc_correlations = false, show_times= [])
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_sweeps, calc_correlations)


    
    # Initialize the animation
    for sweep in 1:n_sweeps
        spatial_corr, time_corr = update_and_compute_correlations!(state, param, t_gap, ρ_history, sweep, rng)
        if sweep in show_times
            normalized_dist = state.ρ_avg / sum(state.ρ_avg)
            p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
            # p1 = plot_density(state.ρ, param, state; title="Current density", show_directions=false)
            outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
            corr_mat = state.ρ_matrix_avg-outer_prod_ρ
            p4 = heatmap(corr_mat, xlabel="x", ylabel="y", 
                        title="Correlation Matrix Heatmap", color=:viridis)
            middle_spot = param.dims[1]÷2
            p5 = plot(corr_mat[middle_spot,:],title="correlation matrix cut for x=$(middle_spot)")
            point_to_look_at = middle_spot+5
            vline!(p4,[point_to_look_at],label="x=$(point_to_look_at)")
            left_value=corr_mat[point_to_look_at,point_to_look_at-1]
            right_value=corr_mat[point_to_look_at,point_to_look_at+1]
            left_side=corr_mat[point_to_look_at, 1:point_to_look_at-1]
            right_side=corr_mat[point_to_look_at, point_to_look_at+1:end]

            p6 = plot(vcat(left_side,[(left_value+right_value)/2],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
            vline!(p6,[point_to_look_at],label="x=$(point_to_look_at)")
            # p6 = plot(vcat(left_side,[corr_mat[point_to_look_at,point_to_look_at]],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
            # p6= plot(corr_mat[point_to_look_at,:])

            p_final=plot(p0,p4,p5,p6, size=(1800,800),plot_title="sweep $(sweep)")
            display(p_final)

        end
        next!(prg)
    end
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
    # p1 = plot_density(state.ρ, param, state; title="Current density", show_directions=false)
    outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
    corr_mat = state.ρ_matrix_avg-outer_prod_ρ
    p4 = heatmap(corr_mat, xlabel="x", ylabel="y", 
                title="Correlation Matrix Heatmap", color=:viridis)
    middle_spot = param.dims[1]÷2
    p5 = plot(corr_mat[middle_spot,:],title="correlation matrix cut for x=$(middle_spot)")
    point_to_look_at = middle_spot+5
    vline!(p4,[point_to_look_at],label="x=$(point_to_look_at)")
    left_value=corr_mat[point_to_look_at,point_to_look_at-1]
    right_value=corr_mat[point_to_look_at,point_to_look_at+1]
    left_side=corr_mat[point_to_look_at, 1:point_to_look_at-1]
    right_side=corr_mat[point_to_look_at, point_to_look_at+1:end]

    p6 = plot(vcat(left_side,[(left_value+right_value)/2],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
    vline!(p6,[point_to_look_at],label="x=$(point_to_look_at)")
    # p6 = plot(vcat(left_side,[corr_mat[point_to_look_at,point_to_look_at]],right_side),title="correlation matrix cut for x=$(point_to_look_at) ")
    # p6= plot(corr_mat[point_to_look_at,:])

    p_final=plot(p0,p4,p5,p6, size=(1800,800),plot_title="sweep $(n_sweeps)")
    display(p_final)

    #
    println("Simulation complete")
    return normalized_dist, corr_mat  
    # proportion_vec = abs.(state.ρ_avg-(exp.(-state.V/state.T)))./exp.(-state.V/state.T)
    # exp_expression= exp.(-state.V/state.T)
    # boltzman_dist= exp_expression/ sum(exp_expression)
    # proportion_vec = normalized_dist - boltzman_dist
    # plot(proportion_vec; title="proportion_vec")

end


function make_movie!(state, param, t_gap, n_frame, rng, file_name, in_fps, show_directions = false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_frame, true)
    
    # Initialize the animation
    anim = @animate for frame in 1:n_frame
        spatial_corr, time_corr = update_and_compute_correlations!(state, param, t_gap, ρ_history, frame, rng)
        
        if show_directions
            # Create two subplots side by side
            p1 = plot(title="Particle Densities by Direction",
                    xlabel="Position", ylabel="Density")
            
            # Plot right-moving particles
            plot!(p1, 1:param.dims[1], state.ρ₊, 
                label="Right-moving", color=:red, 
                marker=:circle, markersize=4)
            
            # Plot left-moving particles on the same graph
            plot!(p1, 1:param.dims[1], state.ρ₋, 
                label="Left-moving", color=:blue, 
                marker=:circle, markersize=4)
            
            # Plot total density
            p2 = plot(1:param.dims[1], state.ρ,
                    title="Total Density",
                    xlabel="Position", ylabel="Density",
                    label="Total", color=:black,
                    marker=:circle, markersize=4)
            
            # Combine plots
            plot(p1, p2, layout=(2,1), size=(800,800))
        else

            normalized_dist = state.ρ_avg / sum(state.ρ_avg)
            p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
            # p1 = plot_density(state.ρ, param, state; title="Current density", show_directions=false)
            outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
            p4 = heatmap(state.ρ_matrix_avg - outer_prod_ρ, xlabel="x", ylabel="y", 
                        title="Correlation Matrix Heatmap", color=:viridis)

            plot(p0,p4, size=(1200,600))

        end
        
        next!(prg)
    end
    
    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)

    # After movie is complete, show final statistics
    println("Generating final statistics...")
    
    # Calculate and display final statistics
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    p0 = plot_density(normalized_dist, param, state; title="Time averaged density")
    outer_prod_ρ = state.ρ_avg*transpose(state.ρ_avg)
    p4 = heatmap(state.ρ_matrix_avg - outer_prod_ρ, xlabel="x", ylabel="y", 
                 title="Correlation Matrix Heatmap", color=:viridis)

    # Display final plots
    final_plots = plot(p0, p4, layout=(1,2), size=(1200,600))
    display(final_plots)
    
    # Save final statistics plot
    savefig(final_plots, replace(file_name, ".gif" => "_final_stats.png"))
end

function plot_decay_time_evolution(decay_times)
    p_decay = plot(decay_times, 
                   title="Evolution of Decay Time", 
                   xlabel="Frame", ylabel="τ", 
                   legend=false, lw=2)
    savefig(p_decay, "decay_time_evolution.png")
end
