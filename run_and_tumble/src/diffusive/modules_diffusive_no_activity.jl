using Statistics
# using BenchmarkTools
include(joinpath(@__DIR__, "..", "common", "potentials.jl"))

#Wrap everything with a module to allow redefinition of type
module FPDiffusive
    # using ..PlotUtils: plot_sweep 
    using ..Potentials: AbstractPotential, potential_update!, Potential, MultiPotential, IndependentFluctuatingPoints, BondForce, bondforce_update!
    using LinearAlgebra
    export Param, setParam, Particle, setParticle, setDummyState, setState, calculate_statistics, reset_statistics!, configure_bond_passage_tracking!, latest_bond_passage_counts, normalize_simulation_backend, loaded_state_backend, occupancy_sampler_mode_name

    const PARTICLE_SIMULATION_BACKEND = "particles"
    const OCCUPANCY_SIMULATION_BACKEND = "occupancy"

    function normalize_simulation_backend(raw_backend)
        backend = lowercase(strip(String(raw_backend)))
        if backend in ("particles", "particle")
            return PARTICLE_SIMULATION_BACKEND
        elseif backend in ("occupancy", "occupancies", "lumped", "lumps")
            return OCCUPANCY_SIMULATION_BACKEND
        end
        throw(ArgumentError("Unsupported simulation_backend: $raw_backend. Use \"particles\" or \"occupancy\"."))
    end

    const STATE_RUNTIME_SCRATCH = IdDict{Any, Dict{Symbol,Any}}()
    
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
        forcing_rate_scheme::String
    end

    const LEGACY_FORCING_RATE_SCHEME = "legacy_penalty"
    const SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME = "symmetric_normalized"

    function normalize_forcing_rate_scheme(raw_scheme)
        scheme = lowercase(strip(String(raw_scheme)))
        if scheme in ("legacy_penalty", "legacy", "current")
            return LEGACY_FORCING_RATE_SCHEME
        elseif scheme in ("symmetric_normalized", "symmetric", "normalized_symmetric")
            return SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
        end
        throw(ArgumentError("Unsupported forcing_rate_scheme: $raw_scheme. Use \"legacy_penalty\" (or \"current\") or \"symmetric_normalized\"."))
    end

    #constructor
    function setParam(γ, dims, ρ₀, D, potential_type,fluctuation_type, potential_magnitude,ffr=0.0;
                      forcing_rate_scheme=LEGACY_FORCING_RATE_SCHEME)
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
        scheme = normalize_forcing_rate_scheme(forcing_rate_scheme)
        param = Param(γ, dims, ρ₀, N, D, potential_type, fluctuation_type, potential_magnitude, ffr_value, scheme)
        return param
    end

    @inline function forcing_rate_scheme(param)
        if hasfield(typeof(param), :forcing_rate_scheme)
            return normalize_forcing_rate_scheme(getfield(param, :forcing_rate_scheme))
        end
        return LEGACY_FORCING_RATE_SCHEME
    end

    struct Particle{D,I<:Signed}
        position::NTuple{D,I}
    end

    @inline function particle_position_type(particle)
        return typeof(particle.position[1])
    end

    function particle_position_type_from_particles(particles, fallback::Type{<:Signed}=Int32)
        particles === nothing && return fallback
        if isempty(particles)
            return fallback
        end
        raw_position = getfield(first(particles), :position)
        if isempty(raw_position)
            return fallback
        end
        candidate = typeof(raw_position[1])
        return candidate <: Signed ? candidate : fallback
    end

    @inline function convert_position_tuple(raw_position, ::Type{I}, dim_num::Int) where {I<:Signed}
        return ntuple(i -> convert(I, Int(raw_position[i])), dim_num)
    end

    function loaded_object_field(obj, field_name::Symbol)
        if hasfield(typeof(obj), field_name)
            return getfield(obj, field_name)
        end
        if hasfield(typeof(obj), :fields)
            field_names = Tuple(typeof(obj).parameters[2])
            idx = findfirst(==(field_name), field_names)
            idx === nothing && error("Loaded object of type $(typeof(obj)) does not contain field $(field_name).")
            return getfield(obj, :fields)[idx]
        end
        error("Unsupported object type $(typeof(obj)); cannot read field $(field_name).")
    end

    @inline function move_particle(particle::Particle{D,I}, new_position) where {D,I<:Signed}
        return Particle{D,I}(convert_position_tuple(new_position, I, D))
    end

    Base.convert(::Type{Particle{D}}, x) where {D} =
        Particle{D,Int32}(convert_position_tuple(loaded_object_field(x, :position), Int32, D))

    function convert_particles(particles, ::Type{I}, dim_num::Int) where {I<:Signed}
        converted = Vector{Particle{dim_num,I}}(undef, length(particles))
        for idx in eachindex(particles)
            raw_position = loaded_object_field(particles[idx], :position)
            converted[idx] = Particle{dim_num,I}(convert_position_tuple(raw_position, I, dim_num))
        end
        return converted
    end

    function convert_density_array(arr::AbstractArray{<:Integer,N}, ::Type{T}) where {N,T<:Signed}
        converted = Array{T,N}(undef, size(arr))
        @inbounds for idx in eachindex(arr)
            converted[idx] = convert(T, arr[idx])
        end
        return converted
    end

    function particles_from_density_array(ρ::AbstractArray{<:Integer,N}, ::Type{I}) where {N,I<:Signed}
        total_particles = Int(sum(Int64.(ρ)))
        particles = Vector{Particle{N,I}}(undef, total_particles)
        next_idx = 1
        for cart_idx in CartesianIndices(ρ)
            count = Int(ρ[cart_idx])
            count < 0 && throw(ArgumentError("Density array contains a negative occupancy at $(Tuple(cart_idx))."))
            if count == 0
                continue
            end
            pos_tuple = ntuple(i -> convert(I, cart_idx[i]), N)
            @inbounds for _ in 1:count
                particles[next_idx] = Particle{N,I}(pos_tuple)
                next_idx += 1
            end
        end
        return particles
    end

    loaded_state_backend(state) = state isa OccupancyState ? OCCUPANCY_SIMULATION_BACKEND : PARTICLE_SIMULATION_BACKEND

    @inline function has_directional_densities(state)
        return !(getfield(state, :ρ₊) === nothing || getfield(state, :ρ₋) === nothing)
    end

    function setParticle(sys_params,rng; ic="random",ic_specific=[], position_int_type::Type{<:Signed}=Int32)
        dim_num = length(sys_params.dims)
        if ic == "random"
            position = ntuple(i -> convert(position_int_type, rand(rng, 1:sys_params.dims[i])), dim_num)
        elseif ic == "center"
            position = ntuple(i -> convert(position_int_type, div(sys_params.dims[i], 2)), dim_num)
        elseif ic == "specific"
            if length(ic_specific) == dim_num
                position = ntuple(i -> convert(position_int_type, Int(ic_specific[i])), dim_num)
            else
                throw(DomainError("Invalid input - specific initial condition must have length $(dim_num)"))
            end
        else
            throw(DomainError("Invalid input - initial condition not supported yet"))
        end
        
        Particle{dim_num,position_int_type}(position)
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

    mutable struct State{N, P, R, C, B}
        t::Int64
        particles::P
        ρ::R
        ρ₊::Union{Nothing,R}
        ρ₋::Union{Nothing,R}
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
    const BOND_PASS_COLLECTION_ENABLED_KEY = :bond_pass_collection_enabled
    const BOND_PASS_SPATIAL_F_AVG_KEY = :bond_pass_spatial_f_avg
    const BOND_PASS_SPATIAL_F2_AVG_KEY = :bond_pass_spatial_f2_avg
    const BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY = :bond_pass_spatial_sample_count

    function bond_pass_track_mask_for_forcings(forcings::Vector{BondForce}, mode::AbstractString)
        if mode == "all_forcing_bonds"
            return ones(Float64, length(forcings))
        elseif mode == "nonzero_magnitude"
            return [abs(force.magnitude) > 0.0 ? 1.0 : 0.0 for force in forcings]
        elseif mode == "none"
            return zeros(Float64, length(forcings))
        else
            throw(ArgumentError("Unsupported bond_pass_count_mode: $mode. Use \"none\", \"nonzero_magnitude\", or \"all_forcing_bonds\"."))
        end
    end

    @inline function bond_passage_collection_enabled(stats)
        return !haskey(stats, BOND_PASS_COLLECTION_ENABLED_KEY) ||
               isempty(stats[BOND_PASS_COLLECTION_ENABLED_KEY]) ||
               stats[BOND_PASS_COLLECTION_ENABLED_KEY][1] > 0.5
    end

    function initialize_bond_passage_stats!(bond_pass_stats, n_forces::Int; track_mask=nothing,
                                            collection_enabled::Bool=true)
        bond_pass_stats[BOND_PASS_FORWARD_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_REVERSE_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_TOTAL_SQ_AVG_KEY] = zeros(Float64, n_forces)
        bond_pass_stats[BOND_PASS_SAMPLE_COUNT_KEY] = [0.0]
        bond_pass_stats[BOND_PASS_COLLECTION_ENABLED_KEY] = [collection_enabled ? 1.0 : 0.0]
        if track_mask === nothing
            if haskey(bond_pass_stats, BOND_PASS_TRACK_MASK_KEY) && length(bond_pass_stats[BOND_PASS_TRACK_MASK_KEY]) == n_forces
                bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = Float64.(bond_pass_stats[BOND_PASS_TRACK_MASK_KEY])
            else
                bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = collection_enabled ? ones(Float64, n_forces) : zeros(Float64, n_forces)
            end
        else
            if length(track_mask) != n_forces
                throw(ArgumentError("track_mask must have length $n_forces."))
            end
            bond_pass_stats[BOND_PASS_TRACK_MASK_KEY] = Float64.(track_mask)
        end
        if !collection_enabled
            delete!(bond_pass_stats, BOND_PASS_SPATIAL_F_AVG_KEY)
            delete!(bond_pass_stats, BOND_PASS_SPATIAL_F2_AVG_KEY)
            delete!(bond_pass_stats, BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY)
        end
        return nothing
    end

    function initialize_spatial_bond_passage_stats!(bond_pass_stats, L::Int)
        bond_passage_collection_enabled(bond_pass_stats) || return nothing
        bond_pass_stats[BOND_PASS_SPATIAL_F_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_F2_AVG_KEY] = zeros(Float64, L)
        bond_pass_stats[BOND_PASS_SPATIAL_SAMPLE_COUNT_KEY] = [0.0]
        return nothing
    end

    function setDummyState(state_to_imitate, ρ_avg, ρ_matrix_avg_cuts, t, bond_pass_stats=nothing;
                           density_int_type=nothing, position_int_type=nothing,
                           keep_directional_densities::Bool=false,
                           simulation_backend=nothing)
        stats_to_use = if isnothing(bond_pass_stats)
            if hasfield(typeof(state_to_imitate), :bond_pass_stats)
                getfield(state_to_imitate, :bond_pass_stats)
            elseif hasfield(typeof(state_to_imitate), :fields)
                try
                    loaded_object_field(state_to_imitate, :bond_pass_stats)
                catch
                    Dict{Symbol,Vector{Float64}}()
                end
            else
                Dict{Symbol,Vector{Float64}}()
            end
        else
            bond_pass_stats
        end
        raw_ρ = loaded_object_field(state_to_imitate, :ρ)
        raw_T = loaded_object_field(state_to_imitate, :T)
        raw_potential = loaded_object_field(state_to_imitate, :potential)
        raw_forcing = loaded_object_field(state_to_imitate, :forcing)
        raw_exp_table = loaded_object_field(state_to_imitate, :exp_table)
        dim_num = ndims(raw_ρ)
        density_type = isnothing(density_int_type) ? eltype(raw_ρ) : density_int_type
        raw_particles = try
            loaded_object_field(state_to_imitate, :particles)
        catch
            nothing
        end
        position_type = isnothing(position_int_type) ?
            particle_position_type_from_particles(raw_particles, Int32) :
            position_int_type
        density_type <: Signed || throw(ArgumentError("density_int_type must be signed. Got $(density_type)."))
        position_type <: Signed || throw(ArgumentError("position_int_type must be signed. Got $(position_type)."))
        ρ = convert_density_array(raw_ρ, density_type)
        ρ₊ = nothing
        ρ₋ = nothing
        if keep_directional_densities
            raw_ρ₊ = try
                loaded_object_field(state_to_imitate, :ρ₊)
            catch
                nothing
            end
            raw_ρ₋ = try
                loaded_object_field(state_to_imitate, :ρ₋)
            catch
                nothing
            end
            if !(isnothing(raw_ρ₊) || isnothing(raw_ρ₋))
                ρ₊ = convert_density_array(raw_ρ₊, density_type)
                ρ₋ = convert_density_array(raw_ρ₋, density_type)
            end
        end
        backend = isnothing(simulation_backend) ? loaded_state_backend(state_to_imitate) : normalize_simulation_backend(simulation_backend)
        if backend == OCCUPANCY_SIMULATION_BACKEND
            return occupancy_state_from_components(
                t,
                ρ,
                ρ₊,
                ρ₋,
                ρ_avg,
                ρ_matrix_avg_cuts,
                stats_to_use,
                raw_T,
                raw_potential,
                raw_forcing,
                raw_exp_table,
            )
        end

        particles = particles_from_density_array(ρ, position_type)
        max_site_occupancy = Int64(maximum(ρ))
        return State(
            t,
            particles,
            ρ,
            ρ₊,
            ρ₋,
            ρ_avg,
            ρ_matrix_avg_cuts,
            stats_to_use,
            max_site_occupancy,
            raw_T,
            raw_potential,
            raw_forcing,
            raw_exp_table
        )
    end

    function populate_densities!(
        ρ::AbstractArray{<:Integer},
        ρ₊,
        ρ₋,
        particles::AbstractVector
    )
        fill!(ρ,  0)
        if !(ρ₊ === nothing)
            fill!(ρ₊, 0)
        end
        if !(ρ₋ === nothing)
            fill!(ρ₋, 0)
        end
    
        for p in particles
            pos = CartesianIndex(p.position...)            # CartesianIndex
            ρ[pos] += one(eltype(ρ))
            # Keep directional arrays neutral/finite for legacy plotting paths.
            if !(ρ₊ === nothing)
                ρ₊[pos] += one(eltype(ρ₊))
            end
            if !(ρ₋ === nothing)
                ρ₋[pos] += one(eltype(ρ₋))
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

    function setState(t, rng, param, T, potential=Potentials.setPotential(zeros(Float64,param.dims)),bond_force=Potentials.setBondForce(([1],[2]),true,0.0);
                      ic ="random", full_corr_tensor=false, int_type::Type{<:Signed}=Int32,
                      position_int_type::Type{<:Signed}=Int32,
                      keep_directional_densities::Bool=false,
                      bond_pass_count_mode::AbstractString="nonzero_magnitude",
                      simulation_backend::AbstractString=PARTICLE_SIMULATION_BACKEND)
        backend = normalize_simulation_backend(simulation_backend)
        if backend == OCCUPANCY_SIMULATION_BACKEND
            return setOccupancyState(
                t,
                rng,
                param,
                T,
                potential,
                bond_force;
                ic=ic,
                full_corr_tensor=full_corr_tensor,
                int_type=int_type,
                keep_directional_densities=keep_directional_densities,
                bond_pass_count_mode=bond_pass_count_mode,
            )
        end
        N = param.N
        dim = length(param.dims)
        if N > typemax(int_type)
            throw(ArgumentError("Requested density int_type $(int_type) cannot represent max site occupancy N=$(N)."))
        end
        if maximum(param.dims) > typemax(position_int_type)
            throw(ArgumentError("Requested position_int_type $(position_int_type) cannot represent lattice coordinates up to $(maximum(param.dims))."))
        end

        # initialize particles
        particles = Vector{Particle{dim,position_int_type}}(undef, N)
        if ic == "flat"
            print("flat ic initialized \n")
            num_sites = prod(param.dims)
            cart_inds = CartesianIndices(param.dims)
            for n in 1:N
                lin_idx = mod(n - 1, num_sites) + 1
                pos_tuple = Tuple(cart_inds[lin_idx])
                particles[n] = Particle{dim,position_int_type}(convert_position_tuple(pos_tuple, position_int_type, dim))
            end
        else
            for n in 1:N
                particles[n] = setParticle(param, rng; ic=ic, position_int_type=position_int_type)
            end
        end
        # Initialize the matrix with dimensions specified by a tuple
        ρ = zeros(int_type, param.dims...)
        ρ₊ = keep_directional_densities ? zeros(int_type, param.dims...) : nothing
        ρ₋ = keep_directional_densities ? zeros(int_type, param.dims...) : nothing
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
        initialize_bond_passage_stats!(bond_pass_stats, length(bond_forces);
                                       track_mask=track_mask,
                                       collection_enabled=String(bond_pass_count_mode) != "none")
        if dim == 1
            initialize_spatial_bond_passage_stats!(bond_pass_stats, param.dims[1])
        end
        max_site_occupancy = Int64(maximum(ρ))
        state = State(t, particles, ρ,ρ₊,ρ₋, ρ_avg, ρ_matrix_avg_cuts, bond_pass_stats, max_site_occupancy, T, potential, bond_forces, exp_table)
        return state
    end

    function reset_statistics!(state)
        dim = ndims(state.ρ)
        state.t = 0
        state.ρ_avg .= state.ρ
        state.max_site_occupancy = Int64(maximum(state.ρ))
        if has_directional_densities(state)
            state.ρ₊ .= state.ρ
            state.ρ₋ .= state.ρ
        end

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
        collection_enabled = bond_passage_collection_enabled(state.bond_pass_stats)
        initialize_bond_passage_stats!(state.bond_pass_stats, length(forcings);
                                       track_mask=existing_mask,
                                       collection_enabled=collection_enabled)
        if dim == 1
            initialize_spatial_bond_passage_stats!(state.bond_pass_stats, size(state.ρ, 1))
        end
        return state
    end

    function configure_bond_passage_tracking!(state, mode::AbstractString="nonzero_magnitude")
        forcings = get_state_forcings!(state)
        track_mask = bond_pass_track_mask_for_forcings(forcings, String(mode))
        initialize_bond_passage_stats!(state.bond_pass_stats, length(forcings);
                                       track_mask=track_mask,
                                       collection_enabled=String(mode) != "none")
        if ndims(state.ρ) == 1
            initialize_spatial_bond_passage_stats!(state.bond_pass_stats, size(state.ρ, 1))
        end
        return state
    end

    @inline function runtime_scratch!(state)
        return get!(STATE_RUNTIME_SCRATCH, state) do
            Dict{Symbol,Any}()
        end
    end

    @inline function cached_int_buffer!(scratch::Dict{Symbol,Any}, key::Symbol, len::Int)
        buffer = get(scratch, key, nothing)
        if !(buffer isa Vector{Int64}) || length(buffer) != len
            buffer = zeros(Int64, len)
            scratch[key] = buffer
        else
            fill!(buffer, 0)
        end
        return buffer
    end

    function latest_bond_passage_counts(state)
        scratch = runtime_scratch!(state)
        n_forces = length(get_state_forcings!(state))
        n_forces == 0 && return Int64[], Int64[]

        if ndims(state.ρ) == 1
            forward_key = :bond_forward_counts_1d
            reverse_key = :bond_reverse_counts_1d
        else
            forward_key = :bond_forward_counts_2d
            reverse_key = :bond_reverse_counts_2d
        end

        forward = get(scratch, forward_key, nothing)
        reverse = get(scratch, reverse_key, nothing)
        if !(forward isa Vector{Int64}) || length(forward) != n_forces ||
           !(reverse isa Vector{Int64}) || length(reverse) != n_forces
            return zeros(Int64, n_forces), zeros(Int64, n_forces)
        end
        return forward, reverse
    end

    @inline function periodic_neighbors_1d!(state, L::Int)
        scratch = runtime_scratch!(state)
        left = get(scratch, :neighbors_1d_left, nothing)
        right = get(scratch, :neighbors_1d_right, nothing)
        if !(left isa Vector{Int}) || !(right isa Vector{Int}) || length(left) != L || length(right) != L
            left, right = precompute_periodic_neighbors_1d(L)
            scratch[:neighbors_1d_left] = left
            scratch[:neighbors_1d_right] = right
        end
        return left, right
    end

    @inline function periodic_neighbors_2d!(state, Lx::Int, Ly::Int)
        scratch = runtime_scratch!(state)
        left_x = get(scratch, :neighbors_2d_left_x, nothing)
        right_x = get(scratch, :neighbors_2d_right_x, nothing)
        down_y = get(scratch, :neighbors_2d_down_y, nothing)
        up_y = get(scratch, :neighbors_2d_up_y, nothing)
        if !(left_x isa Vector{Int}) || !(right_x isa Vector{Int}) || !(down_y isa Vector{Int}) || !(up_y isa Vector{Int}) ||
           length(left_x) != Lx || length(right_x) != Lx || length(down_y) != Ly || length(up_y) != Ly
            left_x, right_x, down_y, up_y = precompute_periodic_neighbors_2d(Lx, Ly)
            scratch[:neighbors_2d_left_x] = left_x
            scratch[:neighbors_2d_right_x] = right_x
            scratch[:neighbors_2d_down_y] = down_y
            scratch[:neighbors_2d_up_y] = up_y
        end
        return left_x, right_x, down_y, up_y
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
        if !haskey(stats, BOND_PASS_COLLECTION_ENABLED_KEY) || length(stats[BOND_PASS_COLLECTION_ENABLED_KEY]) != 1
            stats[BOND_PASS_COLLECTION_ENABLED_KEY] = [1.0]
        end
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            if forcings === nothing
                stats[BOND_PASS_TRACK_MASK_KEY] = bond_passage_collection_enabled(stats) ? ones(Float64, n_forces) : zeros(Float64, n_forces)
            else
                default_mode = bond_passage_collection_enabled(stats) ? "nonzero_magnitude" : "none"
                stats[BOND_PASS_TRACK_MASK_KEY] = bond_pass_track_mask_for_forcings(forcings, default_mode)
            end
        end
        return nothing
    end

    function tracked_force_indices_from_state(state, n_forces::Int)
        stats = state.bond_pass_stats
        bond_passage_collection_enabled(stats) || return Int[]
        if !haskey(stats, BOND_PASS_TRACK_MASK_KEY) || length(stats[BOND_PASS_TRACK_MASK_KEY]) != n_forces
            return collect(1:n_forces)
        end
        mask = stats[BOND_PASS_TRACK_MASK_KEY]
        return [i for i in 1:n_forces if mask[i] > 0.5]
    end

    function ensure_spatial_bond_passage_stats!(state, L::Int)
        stats = state.bond_pass_stats
        bond_passage_collection_enabled(stats) || return nothing
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
                                     from_idx::Integer, to_idx::Integer)
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
                                     from_pos::Tuple{<:Integer,<:Integer}, to_pos::Tuple{<:Integer,<:Integer})
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
        bond_passage_collection_enabled(stats) || return nothing

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
        bond_passage_collection_enabled(stats) || return nothing

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

    @inline function is_static_zero_potential(param)
        return param.potential_type == "zero" && param.fluctuation_type == "no-fluctuation"
    end

    @inline function precompute_periodic_neighbors_1d(L::Int)
        left_neighbors = [i == 1 ? L : i - 1 for i in 1:L]
        right_neighbors = [i == L ? 1 : i + 1 for i in 1:L]
        return left_neighbors, right_neighbors
    end

    @inline function precompute_periodic_neighbors_2d(Lx::Int, Ly::Int)
        left_x = [i == 1 ? Lx : i - 1 for i in 1:Lx]
        right_x = [i == Lx ? 1 : i + 1 for i in 1:Lx]
        down_y = [j == 1 ? Ly : j - 1 for j in 1:Ly]
        up_y = [j == Ly ? 1 : j + 1 for j in 1:Ly]
        return left_x, right_x, down_y, up_y
    end

    @inline endpoint_tuple(indices::AbstractVector{<:Integer}) = Tuple(Int(i) for i in indices)

    function max_forcing_magnitude_per_bond(forcings::Vector{BondForce})
        isempty(forcings) && return 0.0
        bond_totals = Dict{Any,Float64}()
        for force in forcings
            endpoint_1 = endpoint_tuple(force.bond_indices[1])
            endpoint_2 = endpoint_tuple(force.bond_indices[2])
            bond_key = endpoint_1 <= endpoint_2 ? (endpoint_1, endpoint_2) : (endpoint_2, endpoint_1)
            bond_totals[bond_key] = get(bond_totals, bond_key, 0.0) + abs(force.magnitude)
        end
        return maximum(values(bond_totals))
    end

    @inline function hop_rate_normalization(D::Real, max_bond_forcing::Real, scheme::AbstractString)
        if scheme == LEGACY_FORCING_RATE_SCHEME
            return 1.0
        elseif scheme == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
            return max(Float64(D) + Float64(max_bond_forcing), eps(Float64))
        end
        throw(ArgumentError("Unsupported forcing_rate_scheme: $scheme"))
    end

    @inline function bond_rate_prefactor(D::Real, directed_bond_forcing::Real, rate_normalization::Real, scheme::AbstractString)
        base_rate = if scheme == LEGACY_FORCING_RATE_SCHEME
            Float64(D) - max(Float64(directed_bond_forcing), 0.0)
        elseif scheme == SYMMETRIC_NORMALIZED_FORCING_RATE_SCHEME
            Float64(D) + Float64(directed_bond_forcing)
        else
            throw(ArgumentError("Unsupported forcing_rate_scheme: $scheme"))
        end
        return base_rate / rate_normalization
    end

    function directed_bond_forcing_1d(forcings::Vector{BondForce}, from_idx::Integer, to_idx::Integer)
        total_forcing = 0.0
        for force in forcings
            endpoint_1 = force.bond_indices[1][1]
            endpoint_2 = force.bond_indices[2][1]
            active_from = force.direction_flag ? endpoint_1 : endpoint_2
            active_to = force.direction_flag ? endpoint_2 : endpoint_1
            if from_idx == active_from && to_idx == active_to
                total_forcing += force.magnitude
            elseif from_idx == active_to && to_idx == active_from
                total_forcing -= force.magnitude
            end
        end
        return total_forcing
    end

    function directed_bond_forcing_2d(forcings::Vector{BondForce}, from_pos::Tuple{<:Integer,<:Integer}, to_pos::Tuple{<:Integer,<:Integer})
        total_forcing = 0.0
        for force in forcings
            endpoint_1 = (force.bond_indices[1][1], force.bond_indices[1][2])
            endpoint_2 = (force.bond_indices[2][1], force.bond_indices[2][2])
            active_from = force.direction_flag ? endpoint_1 : endpoint_2
            active_to = force.direction_flag ? endpoint_2 : endpoint_1
            if from_pos == active_from && to_pos == active_to
                total_forcing += force.magnitude
            elseif from_pos == active_to && to_pos == active_from
                total_forcing -= force.magnitude
            end
        end
        return total_forcing
    end

    function calculate_jump_probability(D,ΔV,T,exp_table::ExpLookupTable;
                                        directed_bond_forcing=0.0,
                                        rate_normalization=1.0,
                                        forcing_rate_scheme=LEGACY_FORCING_RATE_SCHEME)
        exp_arg = -ΔV / T
        exp_val = lookup_exp(exp_table, exp_arg)
        p = bond_rate_prefactor(D, directed_bond_forcing, rate_normalization, forcing_rate_scheme) * min(1,exp_val)
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

    function update!(param, state, rng; benchmark=false, collect_statistics::Bool=true)
        bench_results = benchmark ? BenchmarkResults() : nothing

        if length(param.dims) == 1
            V = state.potential.V
            T = state.T
            γ = param.γ
            static_zero_potential = is_static_zero_potential(param)
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            scheme = forcing_rate_scheme(param)
            rate_normalization = hop_rate_normalization(param.D, max_forcing_magnitude_per_bond(forcings), scheme)
            L = param.dims[1]
            scratch = runtime_scratch!(state)
            left_neighbors, right_neighbors = periodic_neighbors_1d!(state, L)
            track_bond_passages = bond_passage_collection_enabled(state.bond_pass_stats)
            if track_bond_passages
                ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            end
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = track_bond_passages ? cached_int_buffer!(scratch, :bond_forward_counts_1d, n_forces) : Int64[]
            sweep_reverse_counts = track_bond_passages ? cached_int_buffer!(scratch, :bond_reverse_counts_1d, n_forces) : Int64[]
            if track_bond_passages
                ensure_spatial_bond_passage_stats!(state, L)
            end
            sweep_spatial_forward_counts = track_bond_passages ? cached_int_buffer!(scratch, :spatial_forward_counts_1d, L) : Int64[]
            sweep_spatial_reverse_counts = track_bond_passages ? cached_int_buffer!(scratch, :spatial_reverse_counts_1d, L) : Int64[]
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
                    left_index = left_neighbors[spot_index]
                    right_index = right_neighbors[spot_index]

                    
                    if action_index == 1 # left
                        candidate_spot_index = left_index
                    else # right
                        candidate_spot_index = right_index
                    end

                    # Benchmark jump probability calculation
                    if benchmark
                        t2 = time_ns()
                    end

                    directed_bond_forcing = directed_bond_forcing_1d(forcings, spot_index, candidate_spot_index)
                    if static_zero_potential
                        p_candidate = bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme)
                    else
                        p_candidate = calculate_jump_probability(param.D, V[candidate_spot_index]-V[spot_index], T, state.exp_table;
                                                                 directed_bond_forcing=directed_bond_forcing,
                                                                 rate_normalization=rate_normalization,
                                                                 forcing_rate_scheme=scheme)
                    end

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
                
                p_candidate = clamp(p_candidate, 0.0, 1.0)
                p_stay = 1-p_candidate
                choice = tower_sampling(p_candidate, p_stay, rng)
                
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
                        if track_bond_passages && action_index == 2 && candidate_spot_index == right_index
                            sweep_spatial_forward_counts[spot_index] += 1
                        elseif track_bond_passages && action_index == 1 && candidate_spot_index == left_index
                            # Left jump i->i-1 crosses bond (i-1,i) in reverse orientation.
                            sweep_spatial_reverse_counts[left_index] += 1
                        end

                        state.particles[n] = move_particle(particle, (candidate_spot_index,))
                        new_position = state.particles[n].position[1]
                        state.ρ[spot_index] -= 1
                        state.ρ[new_position] += 1
                        if has_directional_densities(state)
                            state.ρ₊[spot_index] -= 1
                            state.ρ₊[new_position] += 1
                            state.ρ₋[spot_index] -= 1
                            state.ρ₋[new_position] += 1
                        end
                        if state.ρ[new_position] > state.max_site_occupancy
                            state.max_site_occupancy = Int64(state.ρ[new_position])
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
            if collect_statistics && track_bond_passages
                update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
                update_spatial_bond_passage_averages!(state, sweep_spatial_forward_counts, sweep_spatial_reverse_counts)
            end
            
        elseif length(param.dims) == 2
            V = state.potential.V
            T = state.T
            γ = param.γ
            static_zero_potential = is_static_zero_potential(param)
            ffrs = param_ffrs(param)
            forcings = get_state_forcings!(state)
            n_forces = length(forcings)
            scheme = forcing_rate_scheme(param)
            rate_normalization = hop_rate_normalization(param.D, max_forcing_magnitude_per_bond(forcings), scheme)
            Lx, Ly = param.dims
            scratch = runtime_scratch!(state)
            left_x, right_x, down_y, up_y = periodic_neighbors_2d!(state, Lx, Ly)
            track_bond_passages = bond_passage_collection_enabled(state.bond_pass_stats)
            if track_bond_passages
                ensure_bond_passage_stats!(state, n_forces; forcings=forcings)
            end
            tracked_force_indices = tracked_force_indices_from_state(state, n_forces)
            sweep_forward_counts = track_bond_passages ? cached_int_buffer!(scratch, :bond_forward_counts_2d, n_forces) : Int64[]
            sweep_reverse_counts = track_bond_passages ? cached_int_buffer!(scratch, :bond_reverse_counts_2d, n_forces) : Int64[]
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
                    left = (left_x[i], j)
                    right = (right_x[i], j)
                    down = (i, down_y[j])
                    up = (i, up_y[j])

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

                    directed_bond_forcing = directed_bond_forcing_2d(forcings, spot_index, cand)
                    if static_zero_potential
                        p_cand = bond_rate_prefactor(param.D, directed_bond_forcing, rate_normalization, scheme)
                    else
                        p_cand = calculate_jump_probability(param.D, V[cand...]-V[i,j], T, state.exp_table;
                                                            directed_bond_forcing=directed_bond_forcing,
                                                            rate_normalization=rate_normalization,
                                                            forcing_rate_scheme=scheme)
                    end

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
                
                p_cand = clamp(p_cand, 0.0, 1.0)
                p_stay = 1 - p_cand
                choice = tower_sampling(p_cand, p_stay, rng)
                
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
                        state.particles[n] = move_particle(particle, cand)
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
                    new_position = state.particles[n].position
                    state.ρ[new_position...] += 1
                    if has_directional_densities(state)
                        state.ρ₊[i,j] -= 1
                        state.ρ₊[new_position...] += 1
                        state.ρ₋[i,j] -= 1
                        state.ρ₋[new_position...] += 1
                    end
                    if state.ρ[new_position...]<0
                        println("Negative density encountered at position $(new_position)...")
                    elseif state.ρ[new_position...] > state.max_site_occupancy
                        state.max_site_occupancy = Int64(state.ρ[new_position...])
                    end
                end
                
                if benchmark
                    bench_results.density_update_time += (time_ns() - t5) / 1e9
                    bench_results.state_update_time += (time_ns() - t4) / 1e9
                    bench_results.total_samples += 1
                end
                
                t += 1 / param.N
            end
            if collect_statistics && track_bond_passages
                update_bond_passage_averages!(state, sweep_forward_counts, sweep_reverse_counts)
            end
        else
            throw(DomainError("Invalid input - dimension not supported yet"))
        end
        state.t += Δt
        
        if benchmark
            print_benchmark_summary(bench_results)
            return bench_results
        end
    end

    include(joinpath(@__DIR__, "modules_diffusive_no_activity_occupancy.jl"))

    function tower_sampling(weights, w_sum, rng)
        key = w_sum * rand(rng)

        selected = 1
        gathered = weights[selected]
        while gathered < key
            selected += 1
            gathered += weights[selected]
        end

        return selected
    end

    @inline function tower_sampling(w1::Real, w2::Real, rng)
        w_sum = w1 + w2
        key = w_sum * rand(rng)
        return key <= w1 ? 1 : 2
    end
end



function fit_exponential(t, y)
    model(t, p) = p[1] .* exp.(-t ./ p[2]) .+ p[3]  
    p0 = [1.0, 1.0, 0.0]  # Initial guess for [amplitude, decay time, offset]
    fit = Main.LsqFit.curve_fit(model, t, y, p0)
    return fit.param
end

function update_time_averaged_density_field(density_field_history)

end
function calculate_time_averaged_density_field(density_field_history)
    n_dims = length(size(density_field_history))-1
    return mean(density_field_history,dims=n_dims+1)[:,1]

end

function compute_spatial_correlation(ρ)
    F = Main.FFTW.fft(ρ)
    power_spectrum = F .* conj(F)
    corr = real(Main.FFTW.ifft(power_spectrum))
    return Main.FFTW.fftshift(corr) / prod(size(ρ))
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

function averaged_sample_count(state)
    if hasfield(typeof(state), :bond_pass_stats)
        stats = state.bond_pass_stats
        if haskey(stats, FPDiffusive.BOND_PASS_SAMPLE_COUNT_KEY) &&
           !isempty(stats[FPDiffusive.BOND_PASS_SAMPLE_COUNT_KEY])
            return Int(round(stats[FPDiffusive.BOND_PASS_SAMPLE_COUNT_KEY][1]))
        end
    end
    return max(state.t, 0)
end

function update_and_compute_correlations!(state, param,  ρ_history, frame, rng, calc_var_frequency=1, calc_correlations=false; collect_statistics::Bool=true)
    FPDiffusive.update!(param, state, rng; collect_statistics=collect_statistics)
    if collect_statistics && frame%calc_var_frequency==0
        #println("Calculating correlations at frame $frame")
        n_eff = max(averaged_sample_count(state), 1)
        weight_new = min(calc_var_frequency, n_eff)
        weight_prev = n_eff - weight_new

        dim_num= length(param.dims)
        if dim_num==1
            state.ρ_avg = (state.ρ_avg * weight_prev + state.ρ * weight_new) / n_eff
            # @. state.ρ_avg += (state.ρ-state.ρ_avg)*calc_var_frequency/frame
            ρf = float(state.ρ)
            ρ_matrix = ρf * transpose(ρf)
            # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
            state.ρ_matrix_avg_cuts[:full] = (state.ρ_matrix_avg_cuts[:full] * weight_prev + ρ_matrix * weight_new) / n_eff
            # @. state.ρ_matrix_avg_cuts[:full] += (ρ_matrix-state.ρ_matrix_avg_cuts[:full])*calc_var_frequency/frame
            # time_averaged_desnity_field = calculate_time_averaged_density_field(ρ_history[:,1:frame])
        elseif dim_num==2
            # state.ρ_avg = (state.ρ_avg * (frame-calc_var_frequency)+state.ρ*calc_var_frequency)/frame
            @. state.ρ_avg += (state.ρ-state.ρ_avg) * weight_new / n_eff
            
            x_middle = div(param.dims[1],2)
            y_middle = div(param.dims[2],2)
            if haskey(state.ρ_matrix_avg_cuts, :full)
                ρ_matrix = FPDiffusive.outer_density_2D(float(state.ρ)) 
                # state.ρ_matrix_avg = (state.ρ_matrix_avg*(frame-calc_var_frequency)+ρ_matrix*calc_var_frequency)/frame
                @. state.ρ_matrix_avg_cuts[:full] += (ρ_matrix-state.ρ_matrix_avg_cuts[:full]) * weight_new / n_eff
            else
                # @. state.ρ_matrix_avg_cuts[:x_cut] += (state.ρ[:,y_middle] * transpose(state.ρ[:,y_middle])-state.ρ_matrix_avg_cuts[:x_cut])*calc_var_frequency/frame
                # @. state.ρ_matrix_avg_cuts[:y_cut] += (state.ρ[x_middle,:] * transpose(state.ρ[x_middle,:]) - state.ρ_matrix_avg_cuts[:y_cut])*calc_var_frequency/frame
                ρf = float(state.ρ)
                state.ρ_matrix_avg_cuts[:x_cut] .+= (ρf[:, y_middle] * transpose(ρf[:, y_middle]) .- state.ρ_matrix_avg_cuts[:x_cut]) * weight_new / n_eff
                state.ρ_matrix_avg_cuts[:y_cut] .+= (ρf[x_middle, :] * transpose(ρf[x_middle, :]) .- state.ρ_matrix_avg_cuts[:y_cut]) * weight_new / n_eff

                # state.ρ_matrix_avg_cuts[:diag_cut] += (state.ρ[diagind(state.ρ)] * transpose(state.ρ[diagind(state.ρ)]) - state.ρ_matrix_avg_cuts[:diag_cut])*calc_var_frequency/frame
                state.ρ_matrix_avg_cuts[:diag_cut] += (diag(ρf) * transpose(diag(ρf)) - state.ρ_matrix_avg_cuts[:diag_cut]) * weight_new / n_eff
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


function initialize_simulation(state, param, n_frame, calc_correlations; show_progress::Bool=true)
    prg = show_progress ? Progress(n_frame) : nothing
    decay_times = Float64[]
    if calc_correlations
        ρ_history = zeros((param.dims..., n_frame))
        return prg, ρ_history, decay_times
    else
        return prg, nothing, decay_times
    end
end

@inline function normalize_time_lookup(times)
    if isempty(times)
        return nothing
    end
    return Set{Int}(Int.(times))
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

function write_progress_update(progress_file::AbstractString, sweep::Int, t_init::Int, t_end::Int, run_start_epoch::Float64)
    isempty(progress_file) && return
    total = max(t_end - t_init + 1, 1)
    completed = clamp(sweep - t_init + 1, 0, total)
    fraction = completed / total
    elapsed = max(time() - run_start_epoch, 0.0)
    rate = completed > 0 && elapsed > 0.0 ? (completed / elapsed) : 0.0
    remaining = total - completed
    eta = rate > 1e-12 ? (remaining / rate) : -1.0
    eta = isfinite(eta) ? eta : -1.0
    payload = "{\n" *
              "  \"sweep\": $(sweep),\n" *
              "  \"completed\": $(completed),\n" *
              "  \"total\": $(total),\n" *
              "  \"fraction\": $(fraction),\n" *
              "  \"elapsed_seconds\": $(elapsed),\n" *
              "  \"eta_seconds\": $(eta),\n" *
              "  \"updated_at_epoch\": $(time())\n" *
              "}\n"
    try
        mkpath(dirname(progress_file))
        tmp_file = string(progress_file, ".tmp")
        open(tmp_file, "w") do io
            write(io, payload)
        end
        mv(tmp_file, progress_file; force=true)
    catch err
        println("Progress update write failed: $err")
    end
end

function run_simulation!(state, param, n_sweeps, rng; 
                        calc_correlations = false, 
                        show_times = [], 
                        save_times = [],
                        save_dir = "saved_states",
                        plot_flag = false,
                        plotter = nothing,
                        plot_label = "",
                        save_description = nothing,
                        benchmark_frequency = 0,
                        warmup_sweeps::Int = 0,
                        show_progress::Bool = true,
                        progress_file = nothing,
                        progress_interval::Int = 25,
                        snapshot_request_file = nothing,
                        snapshot_tag_prefix = "snapshot",
                        relaxed_ic::Bool=false)
    println("Starting simulation")
    prg, ρ_history, decay_times = initialize_simulation(state, param, n_sweeps, calc_correlations; show_progress=show_progress)

    # Initialize the animation
    t_init = state.t+1
    t_end = t_init + n_sweeps-1
    println("with t_init = $t_init , t_end = $t_end")
    warmup_sweeps = max(warmup_sweeps, 0)
    plot_label = isnothing(plot_label) ? "" : String(plot_label)
    progress_interval = max(progress_interval, 1)
    progress_file_path = isnothing(progress_file) ? "" : strip(String(progress_file))
    snapshot_request_path = isnothing(snapshot_request_file) ? "" : strip(String(snapshot_request_file))
    save_time_lookup = normalize_time_lookup(save_times)
    show_time_lookup = normalize_time_lookup(show_times)
    run_start_epoch = time()
    write_progress_update(progress_file_path, t_init - 1, t_init, t_end, run_start_epoch)
    if warmup_sweeps > 0
        println("Warmup enabled: skipping statistics accumulation for first $warmup_sweeps sweeps of this run.")
    end

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
        
        sweep_since_start = sweep - t_init + 1
        collect_statistics = sweep_since_start > warmup_sweeps
        if warmup_sweeps > 0 && sweep_since_start == warmup_sweeps + 1
            println("Warmup complete at sweep $sweep. Starting statistics accumulation.")
        end

        update_and_compute_correlations!(state, param, ρ_history, sweep, rng; collect_statistics=collect_statistics)

        if !isempty(snapshot_request_path) && isfile(snapshot_request_path)
            try
                rm(snapshot_request_path; force=true)
            catch
                # Best effort: continue even if cleanup fails.
            end
            snapshot_tag = string(snapshot_tag_prefix, "_t", sweep)
            save_state(state, param, save_dir; tag=snapshot_tag, relaxed_ic=relaxed_ic, description=save_description)
            println("Snapshot saved at sweep $sweep")
        end

        # Save state at specified times
        if !isnothing(save_time_lookup) && in(sweep, save_time_lookup)
            save_state(state,param,save_dir; relaxed_ic=relaxed_ic, description=save_description)
            println("State saved at sweep $sweep")
        end

        # Your existing show_times code
        if !isnothing(show_time_lookup) && in(sweep, show_time_lookup) && plot_flag && !isnothing(plotter)
            plotter(sweep, state, param; label=plot_label)
        end
        
        if show_progress && !isnothing(prg)
            next!(prg)
        end

        if sweep_since_start == 1 || sweep_since_start % progress_interval == 0 || sweep == t_end
            write_progress_update(progress_file_path, sweep, t_init, t_end, run_start_epoch)
        end
    end

    println("Simulation complete")
    return state.ρ_avg, state.ρ_matrix_avg_cuts
end
