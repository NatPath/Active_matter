using Statistics
using FFTW
using LsqFit
# using BenchmarkTools
include("plot_utils.jl") 
using .PlotUtils
include("potentials.jl")

#Wrap everything with a module to allow redefinition of type
module FPDiffusive
    # using ..PlotUtils: plot_sweep 
    using ..Potentials: AbstractPotential, potential_update!, Potential, MultiPotential, IndependentFluctuatingPoints, BondForce, bondforce_update!
    using LinearAlgebra
    export Param, setParam, Particle, setParticle, setDummyState, setState, calculate_statistics, reset_statistics!
    
    # Add exponential look-up table structure
    struct ExpLookupTable
        values::Vector{Float64}
        min_val::Float64
        max_val::Float64
        step::Float64
        inv_step::Float64
    end
    
    function create_exp_lookup(min_val::Float64, max_val::Float64, n_points::Int=10000)
        step = (max_val - min_val) / (n_points - 1)
        inv_step = 1.0 / step
        values = [exp(min_val + i * step) for i in 0:(n_points-1)]
        return ExpLookupTable(values, min_val, max_val, step, inv_step)
    end
    
    function lookup_exp(table::ExpLookupTable, x::Float64)
        if x < table.min_val
            return exp(x)  # fallback for values outside range
        elseif x > table.max_val
            return exp(x)  # fallback for values outside range
        else
            idx = Int(floor((x - table.min_val) * table.inv_step)) + 1
            return table.values[idx]
        end
    end

    struct Param # model parameters
        γ::Float64  # rate of potential fluctuation
        dims::Tuple{Vararg{Int}} # systen's dimensions
        ρ₀::Float64  # density
        N::Int64    # number of particles
        D::Float64  #diffusion coefficient
        potential_type::String 
        fluctuation_type::String
        potential_magnitude::Float64
        ffr::Union{Float64,Vector{Float64}} # forcing fluctuation rate(s)
    end

    #constructor
    function setParam(γ, dims, ρ₀, D, potential_type,fluctuation_type, potential_magnitude,ffr=0.0)
        ffrs = if ffr isa AbstractVector
            Float64.(collect(ffr))
        else
            [Float64(ffr)]
        end
        if isempty(ffrs)
            ffrs = [0.0]
        end
        N = Int(round( ρ₀*prod(dims)))       # number of particles
        println(" params set with N = $N, γ = $γ ")

        ffr_value = length(ffrs) == 1 ? ffrs[1] : ffrs
        param = Param(γ, dims, ρ₀, N, D, potential_type, fluctuation_type, potential_magnitude, ffr_value)
        return param
    end

    mutable struct Particle{D}
        position::NTuple{D,Int64}
    end

    function setParticle(sys_params,rng; ic="random",ic_specific=[])
        dim_num = length(sys_params.dims)
        if ic == "random"
            position = ntuple(i -> rand(rng, 1:sys_params.dims[i]), dim_num)
        elseif ic == "center"
            position = ntuple(i -> div(sys_params.dims[i], 2), dim_num)
        elseif ic == "specific"
            if length(ic_specific) == dim_num
                position = Tuple(ic_specific[1:dim_num])
            else
                throw(DomainError("Invalid input - specific initial condition must have length $(dim_num)"))
            end
        else
            throw(DomainError("Invalid input - initial condition not supported yet"))
        end
        
        Particle{dim_num}(position)
    end

    # mutable struct State{N}
    #     t::Int64 # time
    #     #pos::Array{Int64, 1+dim_num}    # position vector
    #     particles::Array{Particle}
    #     ρ::AbstractArray{Int64,N}      # density field
    #     ρ₊::AbstractArray{Int64}      # density field
    #     ρ₋::AbstractArray{Int64}      # density field
    #     #ρ_polarization_arr::Array{Int64}
    #     ρ_avg::AbstractArray{Float64} #time averaged density field
    #     # ρ_matrix_avg::AbstractArray{Float64}
    #     # matrix cuts
    #     ρ_matrix_avg_cuts::Dict{Symbol,AbstractArray{Float64}} # Dict with keys :full, :x_cut, :y_cut, :diag_cut
    #     # ρ_matrix_avg_x_cut::AbstractArray{Float64}
    #     # ρ_matrix_avg_y_cut::AbstractArray{Float64}
    #     # ρ_matrix_avg_diag_cut::AbstractArray{Float64}
    #     T::Float64                      # temperature
    #     # V::Array{Float64}      # potential
    #     potential::AbstractPotential
    #     forcing:: BondForce
    #     exp_table::ExpLookupTable  # Add exponential lookup table
    # end

    mutable struct State{N, C, B, D}
        t::Int64
        particles::Vector{Particle{D}}
        ρ::Array{Int64, N}           
        ρ₊::Array{Int64, N}          
        ρ₋::Array{Int64, N}          
        ρ_avg::Array{Float64, N}     
        ρ_matrix_avg_cuts::C
        bond_pass_stats::B
        max_site_occupancy::Int64
        T::Float64
        potential::AbstractPotential
        forcing::Union{BondForce, Vector{BondForce}}
        exp_table::ExpLookupTable
    end

    const BOND_PASS_FORWARD_AVG_KEY = :bond_pass_forward_avg
    const BOND_PASS_REVERSE_AVG_KEY = :bond_pass_reverse_avg
    const BOND_PASS_TOTAL_AVG_KEY = :bond_pass_total_avg
    const BOND_PASS_TOTAL_SQ_AVG_KEY = :bond_pass_total_sq_avg
    const BOND_PASS_SAMPLE_COUNT_KEY = :bond_pass_sample_count
    const BOND_PASS_TRACK_MASK_KEY = :bond_pass_track_mask
    const BOND_PASS_SPATIAL_F_AVG_KEY = :bond_pass_spatial_f_avg
    const BOND_PASS_SPATIAL_F2_AVG_KEY = :bond_pass_spatial_f2_avg
    const BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY = :bond_pass_spatial_sample_count

    function bond_pass_track_mask_for_forcings(forcings::Vector{BondForce}, mode::AbstractString)
        if mode == "all_forcing_bonds"
            return ones(Float64, length(forcings))
        elseif mode == "nonzero_magnitude"
            return [abs(force.magnitude) > 0.0 ? 1.0 : 0.0 for force in forcings]
        else
            throw(ArgumentError("Unsupported bond_pass_count_mode: $mode. Use \"nonzero_magnitude\" or \"all_forcing_bonds\"."))
        end
    end

    function initialize_bond_passage_stats!(bond_pass_stats, n_forces::Int; track_mask=nothing)
        bond_pass_stats[BOND_PASS_FORWARD_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_REVERSE_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_SQ_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_SAMPLE_COUNT_KEY] = [0.0]
        if track_mask === nothing
            if haskey(bond_pass_stats, BOND_PASS_TRACK_MASK_KEY) && length(bond_pass_stats[BOND_PASS_TRACK_MASK_KEY]) == n_forces
                bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = Float64.(bond_pass_stats[BOND_PASS_TRACK_MASK_KEY])
            else
                bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = ones(Float64, n_forces)
            end
        else
            if length(track_mask) != n_forces
                throw(ArgumentError("track_mask must have length $n_forces."))
            end
            bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = Float64.(track_mask)
        end
        return nothing
    end

    function initialize_spatial_bond_passage_stats!(bond_pass_stats, L::Int)
        bond_pass_stats[BOND_PASS_SPATIAL_F_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_F2_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY] = [0.0]
        return nothing
    end

    function setDummyState(state_to_imitate, ρ_avg, ρ_matrix_avg_cuts, t, bond_pass_stats=nothing)
        stats_to_use = if isnothing(bond_pass_stats)
            if hasfield(typeof(state_to_imitate), :bond_pass_stats)
                getfield(state_to_imitate, :bond_pass_stats)
            else
                Dict{Symbol,Vector{Float64}}()
            end
        else
            bond_pass_stats
        end
        max_site_occupancy = hasfield(typeof(state_to_imitate), :max_site_occupancy) ? state_to_imitate.max_site_occupancy : maximum(state_to_imitate.ρ)
        dummy_state = State(
            t, 
            state_to_imitate.particles, 
            state_to_imitate.ρ, 
            state_to_imitate.ρ₊, 
            state_to_imitate.ρ₋, 
            ρ_avg, 
            ρ_matrix_avg_cuts,
            stats_to_use,
            max_site_occupancy,
            state_to_imitate.T, 
            state_to_imitate.potential, 
            state_to_imitate.forcing, 
            state_to_imitate.exp_table
        )
        return dummy_state
    end

    function populate_densities!(
        ρ::AbstractArray{<:Integer},
        ρ₊::AbstractArray{<:Integer},
        ρ₋::AbstractArray{<:Integer},
        particles::AbstractVector{<:Particle}
    )
        fill!(ρ,  0)
        fill!(ρ₊, 0)
        fill!(ρ₋, 0)
    
        for p in particles
            pos = CartesianIndex(p.position...)            # CartesianIndex
            ρ[pos] += 1
            # Keep directional arrays neutral/finite for legacy plotting paths.
            ρ₊[pos] += 1
            ρ₋[pos] += 1
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

    function setState(t, rng, param, T, potential=Potentials.setPotential(zeros(Float64,param.dims)),bond_force=Potentials.setBondForce(([1],[2]),true,0.0); ic ="random", full_corr_tensor=false, int_type::Type{<:Integer}=Int32, bond_pass_count_mode::AbstractString="nonzero_magnitude")
        N = param.N
        dim = length(param.dims)

        # initialize particles
        particles = Vector{Particle{dim}}(undef, N)
        if ic == "flat"
            print("flat ic initialized \n")
            num_sites = prod(param.dims)
            cart_inds = CartesianIndices(param.dims)
            for n in 1:N
                lin_idx = mod(n - 1, num_sites) + 1
                pos_tuple = Tuple(cart_inds[lin_idx])
                particles[n] = Particle{dim}(pos_tuple)
            end
        else
            for n in 1:N
                particles[n] = setParticle(param, rng; ic=ic)
            end
        end
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
            x_middle = div(param.dims[1],2)
            y_middle = div(param.dims[2],2)
            if full_corr_tensor
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :full => outer_density_2D(ρ_avg),
                )
            else
                ρ_matrix_avg_cuts = Dict{Symbol,AbstractArray{Float64}}(
                    :x_cut => ρ_avg[:,y_middle] * transpose(ρ_avg[:,y_middle]),
                    :y_cut => ρ_avg[x_middle,:] * transpose(ρ_avg[x_middle,:]),
                    # :diag_cut => ρ_avg[diagind(ρ_avg)] * transpose(ρ_avg[diagind(ρ_avg)])
                    :diag_cut => diag(ρ_avg)* transpose(diag(ρ_avg))
                )
            end
            # ρ_matrix_avg = ρ .* reshape(ρ, 1, 1, size(ρ, 1), size(ρ, 2))
            # ρ_matrix_avg = permutedims(ρ_matrix_avg, (3, 4, 1, 2))
        end
        # Create exponential lookup table with reasonable range for jump probabilities
        exp_table = create_exp_lookup(-20.0, 20.0, 10000)
        bond_pass_stats = Dict{Symbol,Vector{Float64}}()
        bond_forces = if bond_force isa BondForce
            [bond_force]
        elseif bond_force isa AbstractVector && all(f -> f isa BondForce, bond_force)
            BondForce[f for f in bond_force]
        else
            throw(ArgumentError("bond_force must be BondForce or Vector{BondForce}"))
        end
        track_mask = bond_pass_track_mask_for_forcings(bond_forces, String(bond_pass_count_mode))
        initialize_bond_passage_stats!(bond_pass_stats, length(bond_forces); track_mask=track_mask)
        if dim == 1
            initialize_spatial_bond_passage_stats!(bond_pass_stats, param.dims[1])
        end
        max_site_occupancy = maximum(ρ)
        state = State(t, particles, ρ,ρ₊,ρ₋, ρ_avg, ρ_matrix_avg_cuts, bond_pass_stats, max_site_occupancy, T, potential, bond_forces, exp_table)
        return state
    end

    function reset_statistics!(state)
        dim = ndims(state.ρ)
        state.t = 0
        state.ρ_avg .= state.ρ
        state.max_site_occupancy = maximum(state.ρ)

        if dim == 1
            ρf = float(state.ρ)
            state.ρ_matrix_avg_cuts[:full] .= ρf * transpose(ρf)
        elseif dim == 2
            x_middle = div(size(state.ρ, 1), 2)
            y_middle = div(size(state.ρ, 2), 2)
            if haskey(state.ρ_matrix_avg_cuts, :full)
                state.ρ_matrix_avg_cuts[:full] .= FPDiffusive.outer_density_2D(float(state.ρ))
            else
                ρf = float(state.ρ)
                state.ρ_matrix_avg_cuts[:x_cut] .= ρf[:, y_middle] * transpose(ρf[:, y_middle])
                state.ρ_matrix_avg_cuts[:y_cut] .= ρf[x_middle, :] * transpose(ρf[x_middle, :])
                state.ρ_matrix_avg_cuts[:diag_cut] .= diag(ρf) * transpose(diag(ρf))
            end
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
        forcings = get_state_forcings!(state)
        existing_mask = if haskey(state.bond_pass_stats, BOND_PASS_TRACK_MASK_KEY) &&
                           length(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY]) == length(forcings)
            Float64.(state.bond_pass_stats[BOND_PASS_TRACK_MASK_KEY])
        else
            bond_pass_track_mask_for_forcings(forcings, "nonzero_magnitude")
        end
        initialize_bond_passage_stats!(state.bond_pass_stats, length(forcings); track_mask=existing_mask)
        if dim == 1
            initialize_spatial_bond_passage_stats!(state.bond_pass_stats, size(state.ρ, 1))
        end
        return state
    end

    dot_like(a::Number, b::Number) = a * b
    dot_like(a, b) = sum(x * y for (x, y) in zip(a, b))

    function param_ffrs(param)
        if hasfield(typeof(param), :ffr)
            if param.ffr isa AbstractVector
                return Float64.(collect(param.ffr))
            end
            return [Float64(param.ffr)]
        elseif hasfield(typeof(param), :ffrs)
            return param.ffrs
        end
        return [0.0]
    end

    function get_state_forcings!(state)
        if state.forcing isa BondForce
            state.forcing = BondForce[state.forcing]
        elseif !(state.forcing isa Vector{BondForce})
            if state.forcing isa AbstractVector && all(f -> f isa BondForce, state.forcing)
                state.forcing = BondForce[f for f in state.forcing]
            else
                throw(ArgumentError("state.forcing must be BondForce or Vector{BondForce}"))
            end
        end
        return state.forcing
    end

    function ensure_bond_passage_stats!(state, n_forces::Int; forcings=nothing)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_FORWARD_AVG_KEY) || length(stats[BOND_PASS_FORWARD_AVG_KEY]) != n_forces
            stats[BOND_PASS_FORWARD_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_REVERSE_AVG_KEY) || length(stats[BOND_PASS_REVERSE_AVG_KEY]) != n_forces
            stats[BOND_PASS_REVERSE_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_TOTAL_AVG_KEY) || length(stats[BOND_PASS_TOTAL_AVG_KEY]) != n_forces
            stats[BOND_PASS_TOTAL_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_TOTAL_SQ_AVG_KEY) || length(stats[BOND_PASS_TOTAL_SQ_AVG_KEY]) != n_forces
            stats[BOND_PASS_TOTAL_SQ_AVG_KEY] = zeros(Float64, n_forces)
        end
        if !haskey(stats, BOND_PASS_SAMPLE_COUNT_KEY) || length(stats[BOND_PASS_SAMPLE_COUNT_KEY]) != 1
            stats[BOND_PASS_SAMPLE_COUNT_KEY] = [0.0]
        end
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            if forcings === nothing
                stats[BOND_PASS_TRACK_MASK_KEY] = ones(Float64, n_forces)
            else
                stats[BOND_PASS_TRACK_MASK_KEY] = bond_pass_track_mask_for_forcings(forcings, "nonzero_magnitude")
            end
        end
        return nothing
    end

    function tracked_force_indices_from_state(state, n_forces::Int)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            return collect(1:n_forces)
        end
        mask = stats[BOND_PASS_TRACK_MASK_KEY]
        return [i for i in 1:n_forces if mask[i] > 0.5]
    end

    function ensure_spatial_bond_passage_stats!(state, L::Int)
        stats = state.bond_pass_stats
        if !haskey(stats, BOND_PASS_SPATIAL_F_AVG_KEY) || length(stats[BOND_PASS_SPATIAL_F_AVG_KEY]) != L
            stats[BOND_PASS_SPATIAL_F_AVG_KEY] = zeros(Float64, L)
        end
        if !haskey(stats, BOND_PASS_SPATIAL_F2_AVG_KEY) || length(stats[BOND_PASS_SPATIAL_F2_AVG_KEY]) != L
            stats[BOND_PASS_SPATIAL_F2_AVG_KEY] = zeros(Float64, L)
        end
        if !haskey(stats, BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY) || length(stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY]) != 1
            stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY] = [0.0]
        end
        return nothing
    end

    function record_bond_passage_1d!(forward_counts::Vector{Int64}, reverse_counts::Vector{Int64},
                                     forcings::Vector{BondForce}, tracked_force_indices::Vector{Int},
                                     from_idx::Int, to_idx::Int)
        for force_idx in tracked_force_indices
            force = forcings[force_idx]
            b1 = force.bond_indices[1][1]
            b2 = force.bond_indices[2][1]
            if from_idx == b1 && to_idx == b2
                forward_counts[force_idx] += 1
            elseif from_idx == b2 && to_idx == b1
                reverse_counts[force_idx] += 1
            end
        end
        return nothing
    end

    function record_bond_passage_2d!(forward_counts::Vector{Int64}, reverse_counts::Vector{Int64},
                                     forcings::Vector{BondForce}, tracked_force_indices::Vector{Int},
                                     from_pos::NTuple{2,Int}, to_pos::NTuple{2,Int})
        for force_idx in tracked_force_indices
            force = forcings[force_idx]
            b1 = (force.bond_indices[1][1], force.bond_indices[1][2])
            b2 = (force.bond_indices[2][1], force.bond_indices[2][2])
            if from_pos == b1 && to_pos == b2
                forward_counts[force_idx] += 1
            elseif from_pos == b2 && to_pos == b1
                reverse_counts[force_idx] += 1
            end
        end
        return nothing
    end

    function update_bond_passage_averages!(state, sweep_forward_counts::Vector{Int64}, sweep_reverse_counts::Vector{Int64})
        n_forces = length(sweep_forward_counts)
        if length(sweep_reverse_counts) != n_forces
            throw(ArgumentError("Forward and reverse bond passage arrays must have the same length."))
        end

        ensure_bond_passage_stats!(state, n_forces)
        stats = state.bond_pass_stats

        n_prev = stats[BOND_PASS_SAMPLE_COUNT_KEY][1]
        n_new = n_prev + 1.0

        forward = Float64.(sweep_forward_counts)
        # Signed convention: right-to-left passage contributes negatively to F_right.
        reverse = -Float64.(sweep_reverse_counts)
        total = forward .+ reverse
        total_sq = total .^ 2

        forward_avg = stats[BOND_PASS_FORWARD_AVG_KEY]
        reverse_avg = stats[BOND_PASS_REVERSE_AVG_KEY]
        total_avg = stats[BOND_PASS_TOTAL_AVG_KEY]
        total_sq_avg = stats[BOND_PASS_TOTAL_SQ_AVG_KEY]

        @. forward_avg += (forward - forward_avg) / n_new
        @. reverse_avg += (reverse - reverse_avg) / n_new
        @. total_avg += (total - total_avg) / n_new
        @. total_sq_avg += (total_sq - total_sq_avg) / n_new

        stats[BOND_PASS_SAMPLE_COUNT_KEY][1] = n_new
        return nothing
    end

    function update_spatial_bond_passage_averages!(state, sweep_forward_counts::Vector{Int64}, sweep_reverse_counts::Vector{Int64})
        L = length(sweep_forward_counts)
        if length(sweep_reverse_counts) != L
            throw(ArgumentError("Spatial forward and reverse arrays must have the same length."))
        end

        ensure_spatial_bond_passage_stats!(state, L)
        stats = state.bond_pass_stats

        n_prev = stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY][1]
        n_new = n_prev + 1.0

        f = Float64.(sweep_forward_counts) .- Float64.(sweep_reverse_counts)
        f2 = f .^ 2

        f_avg = stats[BOND_PASS_SPATIAL_F_AVG_KEY]
        f2_avg = stats[BOND_PASS_SPATIAL_F2_AVG_KEY]
        @. f_avg += (f - f_avg) / n_new
        @. f2_avg += (f2 - f2_avg) / n_new

        stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY][1] = n_new
        return nothing
    end

    @inline function forcing_rate(ffrs::AbstractVector{<:Real}, force_idx::Int)
        return force_idx <= length(ffrs) ? Float64(ffrs[force_idx]) : 0.0
    end

    function active_bond_forcing_1d(forcings::Vector{BondForce}, from_idx::Int, to_idx::Int)
        total_forcing = 0.0
        for force in forcings
            from_bond = force.bond_indices[1][1]
            to_bond = force.bond_indices[2][1]
            if (from_idx == from_bond && to_idx == to_bond && force.direction_flag) ||
               (from_idx == to_bond && to_idx == from_bond && !force.direction_flag)
                total_forcing += force.magnitude
            end
        end
        return total_forcing
    end

    function active_bond_forcing_2d(forcings::Vector{BondForce}, from_pos::NTuple{2,Int}, to_pos::NTuple{2,Int})
        total_forcing = 0.0
        for force in forcings
            from_bond = (force.bond_indices[1][1], force.bond_indices[1][2])
            to_bond = (force.bond_indices[2][1], force.bond_indices[2][2])
            if (from_pos == from_bond && to_pos == to_bond && force.direction_flag) ||
               (from_pos == to_bond && to_pos == from_bond && !force.direction_flag)
                total_forcing += force.magnitude
            end
        end
        return total_forcing
    end

    function calculate_jump_probability(D,ΔV,T,exp_table::ExpLookupTable; bond_forcing=0.0)
        exp_arg = -ΔV / T
        exp_val = lookup_exp(exp_table, exp_arg)
        p = (D-bond_forcing)*min(1,exp_val)
        return p
    end
    mutable struct BenchmarkResults
        action_selection_time::Float64
        jump_probability_time::Float64
        tower_sampling_time::Float64
        state_update_time::Float64
        density_update_time::Float64
        total_samples::Int
        
        BenchmarkResults() = new(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    end
    
    function reset_benchmarks!(bench::BenchmarkResults)
        bench.action_selection_time = 0.0
        bench.jump_probability_time = 0.0
        bench.tower_sampling_time = 0.0
        bench.state_update_time = 0.0
        bench.density_update_time = 0.0
        bench.total_samples = 0
    end
    
    function print_benchmark_summary(bench::BenchmarkResults)
        if bench.total_samples > 0
            println("\n=== Benchmark Results (per update) ===")
            println("Action selection: $(round(bench.action_selection_time/bench.total_samples*1e6, digits=2)) μs")
            println("Jump probability: $(round(bench.jump_probability_time/bench.total_samples*1e6, digits=2)) μs")
            println("Tower sampling: $(round(bench.tower_sampling_time/bench.total_samples*1e6, digits=2)) μs")
            println("State update: $(round(bench.state_update_time/bench.total_samples*1e6, digits=2)) μs")
            println("Density update: $(round(bench.density_update_time/bench.total_samples*1e6, digits=2)) μs")
            println("Total samples: $(bench.total_samples)")
            println("=====================================\n")
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

    function update!(param, state, rng; benchmark=false)
        bench_results = benchmark ? BenchmarkResults() : nothing

        if length(param.dims) == 1
            V = state.potential.V
            T = state.T
            γ = param.γ
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            L = param.dims[1]
            ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = zeros(Int64, n_forces)
            sweep_reverse_counts = zeros(Int64, n_forces)
            ensure_spatial_bond_passage_stats!(state, L)
            sweep_spatial_forward_counts = zeros(Int64, L)
            sweep_spatial_reverse_counts = zeros(Int64, L)
            Δt=1
            t_end = state.t + Δt
            t= state.t
            while t < t_end
                # Benchmark action selection
                if benchmark
                    t1 = time_ns()
                end
                
                n_and_a = rand(rng,1:2*(param.N + 1))
                action_index = mod1(n_and_a,2)
                n = (n_and_a-action_index) ÷ 2 +1
                
                if benchmark
                    bench_results.action_selection_time += (time_ns() - t1) / 1e9
                end
                
                if n<=param.N
                    particle = state.particles[n]
                    spot_index = particle.position[1]
                    left_index= mod1(spot_index-1,param.dims[1])
                    right_index = mod1(spot_index+1,param.dims[1])

                    
                    if action_index == 1 # left
                        candidate_spot_index = left_index
                    else # right
                        candidate_spot_index = right_index
                    end

                    # Benchmark jump probability calculation
                    if benchmark
                        t2 = time_ns()
                    end

                    bond_forcing = active_bond_forcing_1d(forcings, spot_index, candidate_spot_index)
                    p_candidate = calculate_jump_probability(param.D, V[candidate_spot_index]-V[spot_index], T, state.exp_table; bond_forcing=bond_forcing)

                    if benchmark
                        bench_results.jump_probability_time += (time_ns() - t2) / 1e9
                    end
                else
                    if n == param.N+1 # potential fluctuation
                        p_candidate = γ
                    end
                end

                # Benchmark tower sampling
                if benchmark
                    t3 = time_ns()
                end
                
                p_stay = 1-p_candidate
                p_arr = [p_candidate, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr),rng)
                
                if benchmark
                    bench_results.tower_sampling_time += (time_ns() - t3) / 1e9
                end
                
                # Benchmark state updates
                if benchmark
                    t4 = time_ns()
                end
                
                if choice == 1
                    if n<=param.N
                        # Benchmark density updates
                        if benchmark
                            t5 = time_ns()
                        end

                        if !isempty(tracked_force_indices) && candidate_spot_index != spot_index
                            record_bond_passage_1d!(sweep_forward_counts, sweep_reverse_counts,
                                                    forcings, tracked_force_indices,
                                                    spot_index, candidate_spot_index)
                        end
                        if action_index == 2 && candidate_spot_index == right_index
                            sweep_spatial_forward_counts[spot_index] += 1
                        elseif action_index == 1 && candidate_spot_index == left_index
                            # Left jump i->i-1 crosses bond (i-1,i) in reverse orientation.
                            sweep_spatial_reverse_counts[left_index] += 1
                        end

                        particle.position = (candidate_spot_index,)
                        new_position = particle.position[1]
                        state.ρ[spot_index] -= 1
                        state.ρ[new_position] += 1
                        state.ρ₊[spot_index] -= 1
                        state.ρ₊[new_position] += 1
                        state.ρ₋[spot_index] -= 1
                        state.ρ₋[new_position] += 1
                        if state.ρ[new_position] > state.max_site_occupancy
                            state.max_site_occupancy = state.ρ[new_position]
                        end
                        
                        if benchmark
                            bench_results.density_update_time += (time_ns() - t5) / 1e9
                        end

                    elseif n == param.N + 1
                        potential_update!(state.potential,rng)
                    end
                end

                # Independent force fluctuations (each force has its own channel)
                for force_idx in 1:n_forces
                    # Calibrated so ffr is "expected flips per sweep per force".
                    # One sweep has param.N micro-steps.
                    p_force = forcing_rate(ffrs, force_idx) / max(param.N, 1)
                    if p_force > 1.0
                        p_force = 1.0
                    elseif p_force < 0.0
                        p_force = 0.0
                    end
                    if rand(rng) < p_force
                        bondforce_update!(forcings[force_idx])
                    end
                end
                
                if benchmark
                    bench_results.state_update_time += (time_ns() - t4) / 1e9
                    bench_results.total_samples += 1
                end
                
                t += 1/param.N
            end
            update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
            update_spatial_bond_passage_averages!(state, sweep_spatial_forward_counts, sweep_spatial_reverse_counts)
            
        elseif length(param.dims) == 2
            V = state.potential.V
            T = state.T
            γ = param.γ
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = zeros(Int64, n_forces)
            sweep_reverse_counts = zeros(Int64, n_forces)
            Δt = 1
            t_end = state.t + Δt
            t = state.t
            while t < t_end
                # Benchmark action selection
                if benchmark
                    t1 = time_ns()
                end
                
                n_and_a = rand(rng, 1:4*(param.N + 1))
                action_index = mod1(n_and_a, 4)
                n = (n_and_a - action_index) ÷ 4 + 1
                
                if benchmark
                    bench_results.action_selection_time += (time_ns() - t1) / 1e9
                end
                
                if n<=param.N
                    particle = state.particles[n]
                    i,j = particle.position
                    spot_index = (i, j)


                    # Calculate neighboring indices
                    Lx,Ly = param.dims
                    left = (mod1(i-1, Lx), j)
                    right = (mod1(i+1, Lx), j)
                    down = (i, mod1(j-1, Ly))
                    up = (i, mod1(j+1, Ly))

                    if action_index==1  # left
                        cand = left
                    elseif action_index==2  # right
                        cand = right
                    elseif action_index==3  # down
                        cand = down
                    elseif action_index==4  # up
                        cand = up
                    end

                    # Benchmark jump probability calculation
                    if benchmark
                        t2 = time_ns()
                    end

                    bond_forcing = active_bond_forcing_2d(forcings, spot_index, cand)
                    p_cand = calculate_jump_probability(param.D, V[cand...]-V[i,j], T, state.exp_table; bond_forcing=bond_forcing)

                    if benchmark
                        bench_results.jump_probability_time += (time_ns() - t2) / 1e9
                    end
                elseif n == param.N+1  # potential fluctuation
                    p_cand = γ
                end

                # Benchmark tower sampling
                if benchmark
                    t3 = time_ns()
                end
                
                p_stay = 1 - p_cand
                p_arr = [p_cand, p_stay]
                choice = tower_sampling(p_arr, sum(p_arr), rng)
                
                if benchmark
                    bench_results.tower_sampling_time += (time_ns() - t3) / 1e9
                end

                # Benchmark state updates
                if benchmark
                    t4 = time_ns()
                end
                
                if choice == 1
                    if n<=param.N
                        if !isempty(tracked_force_indices) && cand != spot_index
                            record_bond_passage_2d!(sweep_forward_counts, sweep_reverse_counts,
                                                    forcings, tracked_force_indices,
                                                    spot_index, cand)
                        end
                        particle.position = cand
                    elseif n == param.N + 1  # fluctuate potential
                        potential_update!(state.potential, rng)
                    end
                end

                # Independent force fluctuations (each force has its own channel)
                for force_idx in 1:n_forces
                    # Calibrated so ffr is "expected flips per sweep per force".
                    # One sweep has param.N micro-steps.
                    p_force = forcing_rate(ffrs, force_idx) / max(param.N, 1)
                    if p_force > 1.0
                        p_force = 1.0
                    elseif p_force < 0.0
                        p_force = 0.0
                    end
                    if rand(rng) < p_force
                        bondforce_update!(forcings[force_idx])
                    end
                end
                
                # Benchmark density updates
                if benchmark
                    t5 = time_ns()
                end
                
                if n<=param.N
                    state.ρ[i,j] -= 1
                    state.ρ[particle.position...] += 1
                    state.ρ₊[i,j] -= 1
                    state.ρ₊[particle.position...] += 1
                    state.ρ₋[i,j] -= 1
                    state.ρ₋[particle.position...] += 1
                    if state.ρ[particle.position...]<0
                        println("Negative density encountered at position $(particle.position)...")
                    elseif state.ρ[particle.position...] > state.max_site_occupancy
                        state.max_site_occupancy = state.ρ[particle.position...]
                    end
                end
                
                if benchmark
                    bench_results.density_update_time += (time_ns() - t5) / 1e9
                    bench_results.state_update_time += (time_ns() - t4) / 1e9
                    bench_results.total_samples += 1
                end
                
                t += 1 / param.N
            end
            update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
        state.t += Δt
        
        if benchmark
            print_benchmark_summary(bench_results)
            return bench_results
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

function update_and_compute_correlations!(state, param,  ρ_history, frame, rng, calc_var_frequency=1, calc_correlations=false)
    FPDiffusive.update!( param, state, rng)
    calc_flag = true
    if frame%calc_var_frequency==0 && calc_flag 
        #println("Calculating correlations at frame $frame")

        dim_num= length(param.dims)
        if dim_num==1
            state.ρ_avg = (state.ρ_avg * (frame-calc_var_frequency)+state.ρ*calc_var_frequency)/frame
            # @. state.ρ_avg += (state.ρ-state.ρ_avg)*calc_var_frequency/frame
            ρf = float(state.ρ)
            ρ_matrix = ρf * transpose(ρf)
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
                ρ_matrix = FPDiffusive.outer_density_2D(float(state.ρ)) 
                # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
                @. state.ρ_matrix_avg_cuts[:full] += (ρ_matrix-state.ρ_matrix_avg_cuts[:full])*calc_var_frequency/frame
            else
                # @. state.ρ_matrix_avg_cuts[:x_cut] += (state.ρ[:,y_middle] * transpose(state.ρ[:,y_middle])-state.ρ_matrix_avg_cuts[:x_cut])*calc_var_frequency/frame
                # @. state.ρ_matrix_avg_cuts[:y_cut] += (state.ρ[x_middle,:] * transpose(state.ρ[x_middle,:]) - state.ρ_matrix_avg_cuts[:y_cut])*calc_var_frequency/frame
                ρf = float(state.ρ)
                state.ρ_matrix_avg_cuts[:x_cut] .+= (ρf[:, y_middle] * transpose(ρf[:, y_middle]) .- state.ρ_matrix_avg_cuts[:x_cut]) * calc_var_frequency / frame
                state.ρ_matrix_avg_cuts[:y_cut] .+= (ρf[x_middle, :] * transpose(ρf[x_middle, :]) .- state.ρ_matrix_avg_cuts[:y_cut]) * calc_var_frequency / frame

                # state.ρ_matrix_avg_cuts[:diag_cut] += (state.ρ[diagind(state.ρ)] * transpose(state.ρ[diagind(state.ρ)]) - state.ρ_matrix_avg_cuts[:diag_cut])*calc_var_frequency/frame
                state.ρ_matrix_avg_cuts[:diag_cut] += (diag(ρf) * transpose(diag(ρf)) - state.ρ_matrix_avg_cuts[:diag_cut])*calc_var_frequency/frame
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
        outer_prod_ρ = FPDiffusive.outer_density_2D(state.ρ_avg)
    else
        throw(DomainError("Unsupported dimension: $(ndims(state.ρ_avg))"))
    end
    
    corr_mat = state.ρ_matrix_avg - outer_prod_ρ
    return normalized_dist, corr_mat
end
function run_simulation!(state, param, n_sweeps, rng; 
                        calc_correlations = false, 
                        show_times = [], 
                        save_times = [],
                        plot_flag = false,
                        benchmark_frequency = 0,
                        relaxed_ic::Bool=false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_sweeps, calc_correlations)

    # Initialize the animation
    t_init = state.t+1
    t_end = t_init + n_sweeps-1
    println("with t_init = $t_init , t_end = $t_end")

    # # Benchmark variables
    # benchmark_results = nothing
    # if benchmark_frequency > 0
    #     println("Benchmarking enabled - will benchmark every $benchmark_frequency sweeps")
    # end

    for sweep in t_init:t_end
        # Run benchmarking periodically
        # run_benchmark = benchmark_frequency > 0 && (sweep - t_init + 1) % benchmark_frequency == 0
        
        # if run_benchmark
        #     println("\n--- Benchmarking at sweep $sweep ---")
        #     FPDiffusive.update!(param, state, rng; benchmark=true)
        # else
        #     FPDiffusive.update!(param, state, rng)
        # end
        
        update_and_compute_correlations!(state, param, ρ_history, sweep, rng)

        # Save state at specified times
        if sweep in save_times
            save_dir = "saved_states"
            save_state(state,param,save_dir; relaxed_ic=relaxed_ic)
            println("State saved at sweep $sweep")
        end

        # Your existing show_times code
        if (sweep in show_times ) && plot_flag
            PlotUtils.plot_sweep(sweep, state, param)
        end
        
        next!(prg)
    end

    println("Simulation complete")
    return state.ρ_avg, state.ρ_matrix_avg_cuts
end
