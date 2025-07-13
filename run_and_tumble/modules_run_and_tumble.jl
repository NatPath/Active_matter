using Statistics
using FFTW
using LsqFit
include("plot_utils.jl") 
using .PlotUtils
include("potentials.jl")

#Wrap everything with a module to allow redefinition of type
module FP
    # using ..PlotUtils: plot_sweep 
    using ..Potentials: AbstractPotential, potential_update!, Potential, MultiPotential, IndependentFluctuatingPoints, BondForce, bondforce_update!
    using LinearAlgebra
    export Param, setParam, Particle, setParticle, setDummyState, setState, calculate_statistics
    struct Param # model parameters
        α::Float64  # rate of tumbling
        γ::Float64  # rate of potential fluctuation
        ϵ::Float64  # activity
        dims::Tuple{Vararg{Int}} # systen's dimensions
        ρ₀::Float64  # density
        N::Int64    # number of particles
        D::Float64  #diffusion coefficient
        potential_type::String 
        fluctuation_type::String
        potential_magnitude::Float64
        forcing_fluctuation_rate::Float64 # rate of forcing fluctuation
    end

    #constructor
    function setParam(α, γ, ϵ, dims, ρ₀, D, potential_type,fluctuation_type, potential_magnitude,forcing_fluctuation_rate=0.0)

        N = Int(round( ρ₀*prod(dims)))       # number of particles
        println(" params set with N = $N, γ = $γ, γ′=$(γ*N)")

        param = Param(α, γ, ϵ, dims, ρ₀, N, D, potential_type, fluctuation_type, potential_magnitude,forcing_fluctuation_rate)
        return param
    end

    mutable struct Particle
        position::Array{Int64}
        direction::Array{Float64}
    end

    function setParticle(sys_params,rng; ic="random",ic_specific=[])
        dim_num = length(sys_params.dims)
        position = zeros(Int64, dim_num)
        direction = zeros(Float64, dim_num)
        if ic == "random"
            # Randomly initialize position and direction
            for (i,dim) in enumerate(sys_params.dims)
                position[i] = rand(rng, 1:dim)
                direction[i] = rand(rng,[-1,1])
            end
        elseif ic == "center"
            # Initialize position at the center and direction randomly
            for (i,dim) in enumerate(sys_params.dims)
                position[i] = div(dim, 2)
                direction[i] = rand(rng,[-1,1])
            end
        elseif ic == "specific"
            # Initialize position and direction based on specific input
            if length(ic_specific) == dim_num + 1
                position = ic_specific[1:dim_num]
            else
                throw(DomainError("Invalid input - specific initial condition must have length $(dim_num+1)"))
            end
            for i in 1:dim_num
                direction[i] = rand(rng,[-1,1])
            end
        else
            throw(DomainError("Invalid input - initial condition not supported yet"))
        end
        
        direction = direction ./ norm(direction)
        Particle(position,direction)
    end

    mutable struct State
        t::Int64 # time
        #pos::Array{Int64, 1+dim_num}    # position vector
        particles::Array{Particle}
        ρ::AbstractArray{Int64}      # density field
        ρ₊::AbstractArray{Int64}      # density field
        ρ₋::AbstractArray{Int64}      # density field
        #ρ_polarization_arr::Array{Int64}
        ρ_avg::AbstractArray{Float64} #time averaged density field
        # ρ_matrix_avg::AbstractArray{Float64}
        # matrix cuts
        ρ_matrix_avg_cuts::Dict{Symbol,AbstractArray{Float64}} # Dict with keys :full, :x_cut, :y_cut, :diag_cut
        # ρ_matrix_avg_x_cut::AbstractArray{Float64}
        # ρ_matrix_avg_y_cut::AbstractArray{Float64}
        # ρ_matrix_avg_diag_cut::AbstractArray{Float64}
        T::Float64                      # temperature
        # V::Array{Float64}      # potential
        potential::AbstractPotential
        forcing:: BondForce

    end

    function setDummyState(state_to_imitate, ρ_avg, ρ_matrix_avg_cuts, t)
        dummy_state = State(t, state_to_imitate.particles, state_to_imitate.ρ,state_to_imitate.ρ₊,state_to_imitate.ρ₋, ρ_avg, ρ_matrix_avg_cuts, state_to_imitate.T, state_to_imitate.potential, state_to_imitate.forcing)
        #dummy_state = State(t, state_to_imitate.particles, state_to_imitate.ρ,state_to_imitate.ρ₊,state_to_imitate.ρ₋, state_to_imitate.ρ_polarization_arr, ρ_avg, ρ_matrix_avg, state_to_imitate.T, state_to_imitate.potential)
        return dummy_state
    end

    function populate_densities!(
        ρ::AbstractArray{<:Integer},
        ρ₊::AbstractArray{<:Integer},
        ρ₋::AbstractArray{<:Integer},
        particles::AbstractVector{Particle}
    )
        fill!(ρ,  0)
        fill!(ρ₊, 0)
        fill!(ρ₋, 0)
    
        for p in particles
            pos = CartesianIndex(p.position...)            # CartesianIndex
            ρ[pos] += 1
            if p.direction[1] > 0
                ρ₊[pos] += 1
            elseif p.direction[1] < 0
                ρ₋[pos] += 1
            end
        end
    end

    """
    outer_density_2D(ρ::AbstractMatrix{T}) where T<:Number

    Return the 4D tensor C[i,j,k,ℓ] = ρ[i,j] * ρ[k,ℓ], laid out
    as an Array of size (Nx, Ny, Nx, Ny).
    Internally does the vec–outer–reshape trick.
    """
    function outer_density_2D(ρ::AbstractMatrix{T}) where T<:Number
        Nx, Ny = size(ρ)
        v = vec(ρ)                  # length N = Nx*Ny
        M = v * v'                  # N×N outer product
        # reshape to a 4D Array (i,j,k,ℓ)
        return reshape(M, Nx, Ny, Nx, Ny)
    end

    function full_corr_tensor(ρ::AbstractMatrix{T}) where T<:Number
        Nx, Ny = size(ρ)
        v = vec(ρ)          # length N = Nx*Ny
        M = v * transpose(v)  # N×N outer product
        return reshape(M, Nx, Ny, Nx, Ny)
    end


    
    function setState(t, rng, param, T, potential=Potentials.setPotential(zeros(Float64,param.dims)),bond_force=Potentials.setBondForce((1,2),true,0.0); ic ="random", full_corr_tensor=false)
        N = param.N
        # initialize particles
        particles = Array{Particle}(undef,N)
        for n in 1:N
            particles[n]= setParticle(param, rng)
        end
        dim = length(param.dims)
        # Initialize the matrix with dimensions specified by a tuple
        ρ = zeros(Int64, param.dims...)
        ρ₊ = zeros(Int64, param.dims...)
        ρ₋ = zeros(Int64, param.dims...)
        populate_densities!(ρ,ρ₊,ρ₋,particles)
        # Iterate over the positions and update the matrix
        # for n in 1:N
        #     indices = Tuple(particles[n].position[i] for i in 1:length(param.dims))
        #     ρ[CartesianIndex(indices...)] += 1
        #     if particles[n].direction[1]==1
        #         ρ₊[CartesianIndex(indices...)] += 1
        #     elseif particles[n].direction[1] ==-1
        #         ρ₋[CartesianIndex(indices...)] += 1
        #     end
        # end
        ρ_avg = Float64.(ρ)
        if dim == 1
            ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                :full => ρ_avg * transpose(ρ_avg)
            )
            # ρ_matrix_avg = ρ_avg * transpose(ρ_avg) 
        elseif dim ==2
            if full_corr_tensor
                x_middle = div(param.dims[1],2)
                y_middle = div(param.dims[2],2)
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :full => outer_density_2D(ρ_avg),
                )
            else
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :x_cut => ρ_avg[x_middle,:] * transpose(ρ_avg[x_middle,:]),
                    :y_cut => ρ_avg[:,y_middle] * transpose(ρ_avg[:,y_middle]),
                    :diag_cut => ρ_avg[diagind(ρ_avg)] * transpose(ρ_avg[diagind(ρ_avg)])
                )
            end
            # ρ_matrix_avg = ρ .* reshape(ρ, 1, 1, size(ρ, 1), size(ρ, 2))
            # ρ_matrix_avg = permutedims(ρ_matrix_avg, (3, 4, 1, 2))
        end
        state = State(t, particles, ρ,ρ₊,ρ₋, ρ_avg, ρ_matrix_avg_cuts, T, potential, bond_force)
        return state
    end

    function calculate_jump_probability(particle_direction,choice_direction,D,ΔV,T; ϵ=0.0, ΔV_max=0.4,bond_forcing=0.0)
        # relative_direction = particle_direction*choice_direction
        relative_direction = dot(particle_direction,choice_direction)
        p = (D-bond_forcing)*min(1,exp(-(ΔV-relative_direction*ϵ)/T))
        return p
    end
    # p =(D+ϵ*relative_direction-ΔV)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
    # if ΔV!=0
    #     p =(D+ϵ*relative_direction)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
    # else
    #     p =(D+ϵ*relative_direction/2)*min(1,exp(-ΔV/T))/(D+ϵ+ΔV_max)
    # end
    # if relative_direction==1
    #     p = (D+ ϵ*relative_direction- ΔV/T) / ( D+ ϵ + ΔV_max/T)
    # else
    #     p=0
    #     p = (D- (ϵ*relative_direction- ΔV/T)) / ( D+ ϵ + ΔV_max/T)
    # end

    function update!(param, state, rng )

        if length(param.dims) == 1
            V = state.potential.V
            T = state.T
            α = param.α
            γ = param.γ
            Δt=1
            t_end = state.t + Δt
            t= state.t
            while t < t_end
                n_and_a = rand(rng,1:3*(param.N+2))
                action_index = mod1(n_and_a,3)
                n = (n_and_a-action_index) ÷ 3 +1
                if n<=param.N
                    particle = state.particles[n]
                    spot_index = particle.position[1]
                    left_index= mod1(spot_index-1,param.dims[1])
                    right_index = mod1(spot_index+1,param.dims[1])

                    state.ρ[spot_index] -= 1
                    
                    candidate_spot_index = 0
                    if action_index == 1 # left
                        candidate_spot_index = left_index
                        choice_direction = -1
                    elseif action_index == 2 # right
                        candidate_spot_index = right_index
                        choice_direction = 1
                    else  # tumble ,fluctuate potential, fluctuate forcing
                        candidate_spot_index = spot_index
                        choice_direction = 0
                    end
                    if action_index==3 # tumble
                        p_candidate=α
                    else
                        if ([spot_index] == state.forcing.bond_indices[1] && [candidate_spot_index] == state.forcing.bond_indices[2] && state.forcing.direction_flag) || ([spot_index] == state.forcing.bond_indices[2] && [candidate_spot_index] == state.forcing.bond_indices[1] && !state.forcing.direction_flag)
                            p_candidate= calculate_jump_probability(particle.direction[1], choice_direction, param.D, V[candidate_spot_index]-V[spot_index],T; ϵ=param.ϵ,bond_forcing=state.forcing.magnitude)
                        else
                            p_candidate= calculate_jump_probability(particle.direction[1], choice_direction, param.D, V[candidate_spot_index]-V[spot_index],T; ϵ=param.ϵ,bond_forcing=0.0)
                        end
                    end
                else
                    if n == param.N+1 # potential fluctuation
                        action_index = 4
                        p_candidate=γ
                    elseif n == param.N+2 # bond fluctuation
                        action_index = 5
                        p_candidate = param.forcing_fluctuation_rate
                    end
                end

                p_stay = 1-p_candidate
                p_arr = [p_candidate, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr),rng)
                if choice == 1
                    if n<=param.N
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
                        end
                        new_position = particle.position[1]
                        state.ρ[new_position] += 1

                    elseif action_index == 4
                        potential_update!(state.potential,rng)
                    elseif action_index == 5
                        bondforce_update!(state.forcing)
                    end
                end
                t += 1/param.N
                #state.t += Δt
            end
            
        elseif length(param.dims) == 2
            V = state.potential.V
            T = state.T
            α = param.α
            γ = param.γ
            Δt = 1
            t_end = state.t + Δt
            t = state.t
            while t < t_end
                n_and_a = rand(rng, 1:5*(param.N+2))  # 6 actions: left, right, up, down, tumble, fluctuate
                action_index = mod1(n_and_a, 5)
                n = (n_and_a - action_index) ÷ 5 + 1
                if n<=param.N
                    particle = state.particles[n]
                    i,j = particle.position
                    spot_index = [i,j]

                    state.ρ[i,j] -= 1

                    # Calculate neighboring indices
                    Lx,Ly = param.dims
                    left = (mod1(i-1, Lx), j)
                    right = (mod1(i+1, Lx), j)
                    down = (i, mod1(j-1, Ly))
                    up = (i, mod1(j+1, Ly))

                    if action_index==1  # left
                        cand = left
                        dirvec = [-1.0, 0.0]
                    elseif action_index==2  # right
                        cand = right
                        dirvec = [1.0, 0.0]
                    elseif action_index==3  # down
                        cand = down
                        dirvec = [0.0, -1.0]
                    elseif action_index==4  # up
                        cand = up
                        dirvec = [0.0, 1.0]
                    end
                    
                    if action_index == 5  # tumble
                        cand = (i,j)
                        p_cand = α
                    else
                        candidate_spot_index = collect(cand)
                        if (spot_index == state.forcing.bond_indices[1] && candidate_spot_index == state.forcing.bond_indices[2] && state.forcing.direction_flag) || (spot_index == state.forcing.bond_indices[2] && candidate_spot_index == state.forcing.bond_indices[1] && !state.forcing.direction_flag)
                            p_cand = calculate_jump_probability(particle.direction,dirvec,param.D,V[cand...]-V[i,j],T;bond_forcing=state.forcing.magnitude)
                        else
                            p_cand = calculate_jump_probability(particle.direction,dirvec,param.D,V[cand...]-V[i,j],T)
                        end
                    end
                elseif n == param.N+1  # potential fluctuation
                    action_index = 6
                    p_cand = γ
                elseif n == param.N+2  # bond fluctuation
                    action_index = 7
                    p_cand = param.forcing_fluctuation_rate
                end

                p_stay = 1 - p_cand
                p_arr = [p_cand, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr), rng)

                if choice == 1
                    if n<=param.N
                        particle.position .= cand
                        if action_index == 5  # tumble
                            v = randn(rng,2)
                            particle.direction .= v/norm(v)
                        end
                    elseif action_index == 6  # fluctuate potential
                        potential_update!(state.potential, rng)
                    elseif action_index == 7  # fluctuate forcing
                        bondforce_update!(state.forcing)
                    end
                end
                if n<=param.N
                    state.ρ[particle.position...] += 1
                    if state.ρ[particle.position...]<0
                        println("Negative density encountered at position $(particle.position)...")
                    end
                end
                t += 1 / param.N
            end
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
        state.t += Δt
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

function update_and_compute_correlations!(state, param,  ρ_history, frame, rng, calc_var_frequency=1, calc_correlations=false)
    FP.update!( param, state, rng)
    if frame%calc_var_frequency==0
    
        dim_num= length(param.dims)
        if dim_num==1
            state.ρ_avg = (state.ρ_avg * (frame-calc_var_frequency)+state.ρ*calc_var_frequency)/frame
            # @. state.ρ_avg += (state.ρ-state.ρ_avg)*calc_var_frequency/frame
            ρ_matrix = state.ρ*transpose(state.ρ)
            # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
            state.ρ_matrix_avg_cuts[:full] = (state.ρ_matrix_avg_cuts[:full]*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
            # @. state.ρ_matrix_avg_cuts[:full] += (ρ_matrix-state.ρ_matrix_avg_cuts[:full])*calc_var_frequency/frame
            # time_averaged_desnity_field = calculate_time_averaged_density_field(ρ_history[:,1:frame])
        elseif dim_num==2
            # state.ρ_avg = (state.ρ_avg * (frame-calc_var_frequency)+state.ρ*calc_var_frequency)/frame
            @. state.ρ_avg += (state.ρ-state.ρ_avg)*calc_var_frequency/frame
            
            x_middle = div(param.dims[1],2)
            y_middle = div(param.dims[2],2)
            if haskey(state.ρ_matrix_avg_cuts, :full)
                ρ_matrix = FP.outer_density_2D(state.ρ) 
                # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
                @. state.ρ_matrix_avg[:full] += (ρ_matrix-state.ρ_matrix_avg[:full])*calc_var_frequency/frame
            else

                @. state.ρ_matrix_avg_cuts[:x_cut] += (state.ρ[x_middle,:] * transpose(state.ρ[x_middle,:])-state.ρ_matrix_avg_cuts[:x_cut])*calc_var_frequency/frame
                @. state.ρ_matrix_avg_cuts[:y_cut] += (state.ρ[:,y_middle] * transpose(state.ρ[:,y_middle]) - state.ρ_matrix_avg_cuts[:y_cut])*calc_var_frequency/frame
                @. state.ρ_matrix_avg_cuts[:diag_cut] += (state.ρ[diagind(state.ρ)] * transpose(state.ρ[diagind(state.ρ)]) - state.ρ_matrix_avg_cuts[:diag_cut])*calc_var_frequency/frame
            end
            # x_middle = div(param.dims[1],2)
            # y_middle = div(param.dims[2],2)
            # ρ_matrix_x_cut = state.ρ[x_middle,:] .* transpose(state.ρ[x_middle,:])
            # ρ_matrix_y_cut = state.ρ[:,y_middle] .* transpose(state.ρ[:,y_middle])
            # # extract the main diagonal from the ρ matrix
            # ρ_matrix_diag_cut = state.ρ[diagind(state.ρ)]
            # ρ_matrix_diag_cut = state.ρ
            # ρ_matrix_avg_cut = (ρ_matrix_x_cut + ρ_matrix_y_cut) / 2
            # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
            # time_averaged_desnity_field = calculate_time_averaged_density_field(ρ_history[:,:,1:frame])
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
    end
end


function initialize_simulation(state, param, n_frame, calc_correlations)
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

function calculate_statistics(state)
    normalized_dist = state.ρ_avg / sum(state.ρ_avg)
    
    # Handle different dimensions properly
    if ndims(state.ρ_avg) == 1
        outer_prod_ρ = state.ρ_avg * transpose(state.ρ_avg)
    elseif ndims(state.ρ_avg) == 2
        outer_prod_ρ = FP.outer_density_2D(state.ρ_avg)
    else
        throw(DomainError("Unsupported dimension: $(ndims(state.ρ_avg))"))
    end
    
    corr_mat = state.ρ_matrix_avg - outer_prod_ρ
    return normalized_dist, corr_mat
end
function run_simulation!(state, param, n_sweeps, rng; 
                        calc_correlations = false, 
                        show_times = [], 
                        save_times = [],plot_flag = false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_sweeps, calc_correlations)

    # Initialize the animation
    t_init = state.t+1
    t_end = t_init + n_sweeps-1

    #V statistics
    # counts = zeros(Int,length(state.potential.potentials))
    for sweep in t_init:t_end
        update_and_compute_correlations!(state, param, ρ_history, sweep, rng)

        # Save state at specified times
        if sweep in save_times
            save_dir = "saved_states"
            save_state(state,param,save_dir)
            println("State saved at sweep $sweep")
        end

        # Your existing show_times code
        if sweep in show_times && plot_flag
            PlotUtils.plot_sweep(sweep, state, param)
            
        end
        
        next!(prg)
    end
    # normalized_dist, corr_mat = PlotUtils.plot_sweep(n_sweeps, state, param)
    # normalized_dist, corr_mat = calculate_statistics(state)

    
    println("Simulation complete")
    # return normalized_dist, corr_mat  
    return state.ρ_avg, state.ρ_matrix_avg_cuts
end
