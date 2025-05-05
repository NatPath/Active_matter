module Potentials
    import Random: AbstractRNG
    using Plots
    export AbstractPotential ,Potential, MultiPotential, IndependentFluctuatingPoints, ProfileSwitchPotential
    export potential_args, choose_potential, potential_update!

    # Abstract type for polymorphism
    abstract type AbstractPotential end

    # Single potential struct
    mutable struct Potential <: AbstractPotential
        V::Array{Float64}
        fluctuation_mask::Array{Float64}
        fluctuation_sign::Int64
    end

    # Multi-potential container
    mutable struct MultiPotential <: AbstractPotential
        potentials::Vector{Potential}
    end

    mutable struct IndependentFluctuatingPoints <: AbstractPotential
        V::Vector{Float64}
        indices::Vector{Int}
        magnitude::Float64
        fluctuation_statistics::String
    end

    mutable struct ProfileSwitchPotential <: AbstractPotential
        potentials::Vector{Potential}
        probabilities::Vector{Float64}
        V::Vector{Float64}
        current::Int
    end
    # Utility: weighted sampling without dependencies
    function weighted_sample(rng::AbstractRNG, weights::Vector{Float64})
        total = sum(weights)
        r = rand(rng) * total
        cum = 0.0
        for (i, w) in enumerate(weights)
            cum += w
            if r <= cum
                return i
            end
        end
        return length(weights)
    end

    function setPotential(V, fluctuation_mask)
        return Potential(V, fluctuation_mask, 1)
    end


    function setMultiPotential(potentials::Vector{Potential})
        return MultiPotential(potentials)
    end

    function setProfileSwitchPotential(pots::Vector{Potential}; probs=nothing, rng::AbstractRNG)
        n = length(pots)
        probabilities = probs === nothing ? fill(1/n, n) : probs
        idx = weighted_sample(rng, probabilities)
        return ProfileSwitchPotential(pots, probabilities, deepcopy(pots[idx].V),idx)
    end

    function potential_update!(p::Potential)
        p.V += p.fluctuation_mask .* p.fluctuation_sign
        p.fluctuation_sign *= -1
    end
    function potential_update!(p::Potential, rng::AbstractRNG)
        potential_update!(p)
    end

    function potential_update!(mp::MultiPotential, rng::AbstractRNG)
        for pot in mp.potentials
            potential_update!(pot, rng)
        end
    end

    function potential_update!(p::IndependentFluctuatingPoints)
        error("IndependentFluctuatingPoints requires an RNG. Use update!(p, rng) instead.")
    end

    function potential_update!(p::IndependentFluctuatingPoints, rng::AbstractRNG)
        for i in p.indices
            if p.fluctuation_statistics == "gaussian"
                p.V[i] = randn(rng,Float32) * p.magnitude
            elseif p.fluctuation_statistics == "uniform"
                p.V[i] = (2*rand(rng)-1) * p.magnitude
            elseif p.fluctuation_statistics == "discrete_wo_0"
                p.V[i] = rand(rng,[-1,1]) * p.magnitude
            elseif p.fluctuation_statistics == "discrete_with_0"
                p.V[i] = rand(rng,[-1,0,1]) * p.magnitude
            end
        end
    end

    function potential_update!(p::ProfileSwitchPotential)
        error("ProfileSwitchPotential requires an RNG. Use potential_update!(p, rng).")
    end

    function potential_update!(p::ProfileSwitchPotential, rng::AbstractRNG)
        # Randomly switch to a new profile
        p.current = weighted_sample(rng, p.probabilities)
        p.V = deepcopy(p.potentials[p.current].V)
    end

    # function potential_update!(p::IndependentFluctuatingPointsGaussian, rng::AbstractRNG)
    #     for i in p.indices
    #         p.V[i] = (2*rand(rng)-1) * p.magnitude
    #     end
    # end

    function choose_potential(v_args,dims; boundary_walls= false, fluctuation_type="plus-minus",rng, plot_flag=false)
        # if get(v_args, "fluctuation_type", "") == "profile_switch"
        #     profiles = get(v_args, "profiles", error("'profiles' key required for profile_switch"))
        #     probs    = get(v_args, "profile_probs", nothing)
        #     pots = [choose_potential(pa, dims; boundary_walls=boundary_walls, fluctuation_type="zero-potential", rng=rng)
        #             for pa in profiles]
        #     println("did you get here?")
        #     return setProfileSwitchPotential(pots; probs=probs, rng=rng)
        # end
        if get(v_args,"multi",false)
            n = get(v_args,"n",2)
            base_args = deepcopy(v_args)
            delete!(base_args,"multi")
            delete!(base_args,"n")
            potentials = [choose_potential(base_args, dims; boundary_walls, fluctuation_type) for _ in 1:n]
            return setMultiPotential(potentials)
        end
        V = zeros(Float64, dims)
        L = dims[1]
        middle = Int(L÷2)
        x = LinRange(0,L,L)
        v_string = v_args["type"]
        magnitude = get(v_args,"magnitude",1.0)

        if v_string == "well"
            for i in 1:L
                #V[i] = exp(-((i - L/2)^2) / (2 * (2)^2))  # Gaussian potential centered in the middle
                width = v_args["width"]
                V[i] = i<= L/2 + width && i> L/2-width  ? magnitude : 0
            end
        elseif v_string == "zero"
            magnitude=1
        elseif v_string == "smudge"
            location = v_args["location"]
            V[location] = magnitude
            V[location-1] = magnitude/2
        elseif v_string == "minus_smudge"
            location = v_args["location"]
            V[location] = -magnitude
            V[location-1] = -magnitude/2
        elseif v_string == "left_smudge"
            location = v_args["location"]
            V[location-1] = magnitude
            V[location] = magnitude/2
        elseif v_string == "left_minus_smudge"
            location = v_args["location"]
            V[location-1] = -magnitude
            V[location] = -magnitude/2
        elseif v_string == "2ratchet"
            location = v_args["location"]
            V[location+2] = magnitude
            V[location+1] = magnitude/2
            V[location-1] = magnitude
            V[location-2] = magnitude/2
        elseif v_string =="modified_smudge"
            location = v_args["location"]
            V[location] = magnitude
            V[location-1] = magnitude/4
            
        elseif v_string == "delta"
            V[middle] = magnitude
        elseif v_string == "linear"
            m = v_args["slope"]
            b = v_args["shift"]
            linear_potential(m,b,x)= m*x.+b
            V = linear_potential(m,b,x)

            magnitude = m*L+b
        elseif v_string == "harmonic"
            k = v_args["k"]
            m_sign = v_args["m_sign"]
            shift = v_args["center"]
            harmonic_oscillator_potential(k,m_sign,shift,x) = k^2*(x.-shift).^2*m_sign
            V = harmonic_oscillator_potential(k,m_sign,shift,x)

            magnitude = k
        elseif v_string == "periodic"
            V0 = v_args["magnitude"]
            T = v_args["period"]
            ϕ = v_args["phase"]
            periodic_potential(V0, a, x,ϕ) = V0 * cos.(2π * (x/T) .+ ϕ )
            V = periodic_potential(V0,T,x,ϕ)

            magnitude=V0
        elseif v_string == "random"
            magnitude= v_args["scale"]
            V = magnitude*rand(rng,dims...)

        else
            error("unsupported potential type : $v_string ")
        end
        if plot_flag
            V_plot=plot(V)
            display(V_plot)
        end
        fluctuating_mask = zeros(Float64,dims)
        if fluctuation_type == "plus-minus"
            fluctuating_mask[middle] = -magnitude/2
            fluctuating_mask[middle-1] = magnitude/2
        elseif fluctuation_type == "zero-potential"
            fluctuating_mask = -V
        elseif fluctuation_type == "reflection"
            fluctuating_mask =  -2*V
        elseif fluctuation_type == "independent-points-discrete"
            points_indices = get(v_args, "points_indices", [L÷2-1,L÷2+1])
            for i in points_indices
                V[i] = rand(rng,[-1,1]) * magnitude
            end
            return IndependentFluctuatingPoints(V, points_indices, magnitude,"discrete")
        elseif fluctuation_type == "independent-points-discrete_wo_0"
            points_indices = get(v_args, "points_indices", [L÷2-1,L÷2+1])
            for i in points_indices
                V[i] = rand(rng,[-1,1]) * magnitude
            end
            return IndependentFluctuatingPoints(V, points_indices, magnitude,"discrete")
        elseif fluctuation_type == "independent-points-discrete_with_0"
            points_indices = get(v_args, "points_indices", [L÷2-1,L÷2+1])
            for i in points_indices
                V[i] = rand(rng,[-1,0,1]) * magnitude
            end
            return IndependentFluctuatingPoints(V, points_indices, magnitude,"discrete_with_0")
        elseif fluctuation_type == "independent-points-uniform"
            points_indices = get(v_args, "points_indices", [L÷2-1,L÷2+1])
            for i in points_indices
                V[i] = (2*rand(rng)-1) * magnitude
            end
            return IndependentFluctuatingPoints(V, points_indices, magnitude,"uniform")
        elseif fluctuation_type == "independent-points-gaussian"
            points_indices = get(v_args, "points_indices", [L÷2-1,L÷2+1])
            for i in points_indices
                V[i] = randn(rng,Float32) * magnitude
            end
            return IndependentFluctuatingPoints(V, points_indices, magnitude,"gaussian")
        elseif fluctuation_type == "profile_switch"
            profiles = get(v_args, "potentials_profiles", "NoProfiles")
            if profiles=="NoProfiles"
                error("'potentials_profiles' key required for profile_switch")
            end
            probs    = get(v_args, "profile_probs", nothing)
            pots = [choose_potential(pa, dims; boundary_walls=boundary_walls, fluctuation_type="zero-potential", rng=rng)
                    for pa in profiles]
            println("did you get here?")
            return setProfileSwitchPotential(pots; probs=probs, rng=rng)
        end
        
        if boundary_walls
            V[1] = 10^5*magnitude
            V[end] = 10^5*magnitude
        end

        return setPotential(V,fluctuating_mask)
    end

    function plot_boltzman_distribution(V,T=1)
        exp_expression= exp.(-V/T)
        
        display(plot(exp_expression/sum(exp_expression)))
    end

    function potential_args(v_string, dims; magnitude = 0.4, simple=false)
        L = dims[1]
        v_well_args = Dict("type"=>"well", "width"=>1, "magnitude"=> magnitude)
        v_zero_args = Dict("type"=>"zero")
        v_smudge_args = Dict("type"=>"smudge","location" => L÷2 , "magnitude" => magnitude)
        v_left_smudge_args = Dict("type"=>"left_smudge","location" => L÷2 , "magnitude" => magnitude)
        v_minus_smudge_args = Dict("type"=>"minus_smudge","location" => L÷2 , "magnitude" => magnitude)
        v_left_minus_smudge_args = Dict("type"=>"left_minus_smudge","location" => L÷2 , "magnitude" => magnitude)
        v_2ratchet_args = Dict("type"=>"2ratchet","location" => L÷2 , "magnitude" => magnitude)
        v_extended_smudge_args = Dict("type"=>"smudge","location" => L÷2 ,"length" => 5, "magnitude" => magnitude)
        v_modified_smudge_args = Dict("type"=>"modified_smudge","location" => L÷2 , "magnitude" => magnitude)
        v_delta_args = Dict("type"=>"delta", "location" => L÷2, "magnitude"=>10^0*magnitude)
        v_linear_args = Dict("type"=> "linear", "slope" => 1, "shift"=>0)
        v_harmonic_args = Dict("type"=>"harmonic", "k" => magnitude, "m_sign"=>1, "center"=> L÷2)
        v_periodic_args = Dict("type"=>"periodic", "period" => L÷4, "magnitude"=>magnitude, "phase"=> L÷2)
        v_random_args = Dict("type"=>"random", "scale"=>magnitude )
        if !simple
            v_ratchet_PmLr = Dict("type"=>"zero","potentials_profiles"=>[potential_args("smudge",dims;magnitude=magnitude,simple=true),potential_args("left_smudge",dims;magnitude=magnitude,simple=true),potential_args("minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true)])
        end

        if v_string== "well"
            return v_well_args
        elseif v_string=="zero"
            return v_zero_args
        elseif v_string == "smudge"
            return v_smudge_args
        elseif v_string == "left_smudge"
            return v_left_smudge_args
        elseif v_string == "minus_smudge"
            return v_minus_smudge_args
        elseif v_string == "left_minus_smudge"
            return v_left_minus_smudge_args
        elseif v_string =="2ratchet"
            return v_2ratchet_args
        elseif v_string =="modified_smudge"
            return v_modified_smudge_args
        elseif v_string == "delta"
            return v_delta_args
        elseif v_string == "linear"
            return v_linear_args
        elseif v_string == "harmonic"
            return v_harmonic_args
        elseif v_string == "periodic"
            return v_periodic_args
        elseif v_string == "random"
            return v_random_args
        elseif v_string == "ratchet_PmLr"
            return v_ratchet_PmLr
        else
            error("unsupported V string")
        end
    end

end