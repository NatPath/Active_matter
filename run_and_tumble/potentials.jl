module Potentials
    import Random: AbstractRNG
    using Plots
    export AbstractPotential ,Potential, MultiPotential, IndependentFluctuatingPoints, ProfileSwitchPotential, BondForce
    export potential_args, choose_potential, potential_update!, bondforce_update!

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
        V::AbstractArray{Float64}
        current::Int
    end

    mutable struct BondForce
        bond_indices::Tuple{Array{Int},Array{Int}}
        direction_flag::Bool # True if the force is directed (1->2), False if it is directed away
        magnitude::Float64
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
        println("if you got here, something is wrong")
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

    function setBondForce(bond_indices, direction_flag, magnitude)
        return BondForce(bond_indices, direction_flag, magnitude)
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

    function bondforce_update!(bf::BondForce)
        bf.direction_flag = !bf.direction_flag
    end

    function choose_bond_force(forcing_type, vertex1, vertex2, magnitude)
        bond_indices = (vertex1, vertex2)
        if forcing_type == "none"
            return setBondForce(bond_indices, true, 0.0)
        elseif forcing_type == "regular"
            return setBondForce(bond_indices, true, magnitude)
        end
    end

    # function potential_update!(p::IndependentFluctuatingPointsGaussian, rng::AbstractRNG)
    #     for i in p.indices
    #         p.V[i] = (2*rand(rng)-1) * p.magnitude
    #     end
    # end
    function check_potential_switch(potential,rng)
        counts= zeros(Int,length(potential.potentials))
        for step in 1:10^9
            potential_update!(potential,rng)
            counts[potential.current]+=1
        end
        println("Profile frequencies: ", counts ./ sum(counts))
    end

    function choose_potential(v_args,dims; boundary_walls = false, fluctuation_type="plus-minus",rng, plot_flag=true)
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
        middle = Int(L÷2)+1
        v_string = v_args["type"]
        magnitude = get(v_args,"magnitude",1.0)
        dim = length(dims)
        fluctuating_mask = zeros(Float64,dims)
        if dim == 1
            x = LinRange(0,L,L)
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
                shift = v_args["shift"]
                b = v_args["b"]
                cut_at = v_args["cut_at"]
                linear_potential(m,b,x)= m*(x.-shift).+b
                # V = collect(linear_potential(m,b,x))
                V[abs.(x.-shift) .> cut_at] .= 0
                V[abs.(x.-shift) .<= cut_at] .= linear_potential(m, b, x[abs.(x.-shift) .<= cut_at])
                magnitude = m*L+b
                print(V)
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
            elseif v_string == "step_potential"
                step = v_args["step"]
                V[1:step] .= magnitude/2
                V[step+1:end] .= -magnitude/2
            elseif v_string == "step_potential_left"
                step = v_args["step"]
                V[1:step] .= magnitude
                V[step+1:end] .= 0
            elseif v_string == "step_potential_left"
                step = v_args["step"]
                V[1:step] .= 0
                V[step+1:end] .= magnitude
            end

            if plot_flag
                V_plot=plot(V)
                display(V_plot)
            end

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
                potential = setProfileSwitchPotential(pots; probs=probs, rng=rng)
                #check_potential_switch(potential,rng)
                return potential
            end
            
            if boundary_walls
                V[1] = 10^5*magnitude
                V[end] = 10^5*magnitude
            end
        elseif dim==2

            if v_string =="zero"
                V.=0
            elseif v_string == "well"
                x = LinRange(0, dims[1], dims[1])
                y = LinRange(0, dims[2], dims[2])
                # Loop through the grid
                for i in axes(V, 1)
                    for j in axes(V, 2)
                        # Compute the condition for the mask
                        if (x[i] - dims[1] / 2)^2 + (y[j] - dims[2] / 2)^2 <= 2
                            V[i, j] = -magnitude
                        end
                    end
                end
            elseif v_string == "2D_x_wall_slide"
                V[middle+1,:] .= magnitude
                V[middle-1,:] .= -magnitude
            elseif v_string == "2D_x_slide"
                V[middle+1,middle] = magnitude
                V[middle-1,middle] = -magnitude
            elseif v_string == "2D_y_slide"
                V[middle,middle+1] = magnitude
                V[middle,middle-1] = -magnitude
            elseif v_string == "xy_slide"
                V[middle+1,middle] = magnitude
                V[middle-1,middle] = -magnitude
                V[middle,middle+1] = magnitude
                V[middle,middle-1] = -magnitude
            elseif v_string == "xy_slide_rotated"
                V[middle+1,middle] = magnitude
                V[middle-1,middle] = -magnitude
                V[middle,middle+1] = -magnitude
                V[middle,middle-1] = magnitude
            elseif v_string == "barrier"
                x = LinRange(0, dims[1], dims[1])
                y = LinRange(0, dims[2], dims[2])
                V[1, :] .= 10^5 * magnitudei
                V[end, :] .= 10^5 * magnitude
                V[:, 1] .= 10^5 * magnitude
                V[:, end] .= 10^5 * magnitude
            else
                error("unsupported potential type : $v_string ")
            end
            
            if plot_flag
                V_plot=heatmap(V, aspect_ratio=:equal, c=:viridis, xlabel="x", ylabel="y", title="2D Potential Map")
                display(V_plot)
            end
            println(fluctuation_type)
            if fluctuation_type == "no-fluctuation"
                fluctuating_mask .= 0
            elseif fluctuation_type == "zero-potential"
                fluctuating_mask = -V
            elseif fluctuation_type == "reflection"
                fluctuating_mask = -2 * V
            elseif fluctuation_type == "profile_switch"
                profiles = get(v_args, "potentials_profiles", "NoProfiles")
                if profiles=="NoProfiles"
                    error("'potentials_profiles' key required for profile_switch")
                end
                probs    = get(v_args, "profile_probs", nothing)
                pots = [choose_potential(pa, dims; boundary_walls=boundary_walls, fluctuation_type="zero-potential", rng=rng)
                        for pa in profiles]
                potential = setProfileSwitchPotential(pots; probs=probs, rng=rng)
                #check_potential_switch(potential,rng)
                return potential
            elseif fluctuation_type == "forcing"
                println("Forcing scenario")
            else
                error("unsupported fluctuation type")
            end


        else
            error("unsupported number of dimensions : $dim")
        end

        return setPotential(V,fluctuating_mask)
    end

    function plot_boltzman_distribution(V,T=1)
        exp_expression= exp.(-V/T)
        
        display(plot(exp_expression/sum(exp_expression)))
    end

    function potential_args(v_string, dims; magnitude = 0.4, simple=false,displacement = 0,cut_at=10^4)
        dim = length(dims)

        if dim == 1
            L = dims[1]
            # 1D cases
            v_well_args = Dict("type"=>"well", "width"=>1, "magnitude"=> magnitude)
            v_zero_args = Dict("type"=>"zero")
            v_smudge_args = Dict("type"=>"smudge","location" => L÷2+displacement , "magnitude" => magnitude)
            v_left_smudge_args = Dict("type"=>"left_smudge","location" => L÷2+displacement , "magnitude" => magnitude)
            v_minus_smudge_args = Dict("type"=>"minus_smudge","location" => L÷2+displacement , "magnitude" => magnitude)
            v_left_minus_smudge_args = Dict("type"=>"left_minus_smudge","location" => L÷2+displacement , "magnitude" => magnitude)
            v_2ratchet_args = Dict("type"=>"2ratchet","location" => L÷2+displacement, "magnitude" => magnitude)
            v_extended_smudge_args = Dict("type"=>"smudge","location" => L÷2+displacement,"length" => 5, "magnitude" => magnitude)
            v_modified_smudge_args = Dict("type"=>"modified_smudge","location" => L÷2+displacement , "magnitude" => magnitude)
            v_delta_args = Dict("type"=>"delta", "location" => L÷2+displacement, "magnitude"=>10^0*magnitude)
            v_linear_args = Dict("type"=> "linear", "slope" => magnitude, "shift"=>L÷2+displacement,"b"=>0, "cut_at"=>cut_at)
            v_harmonic_args = Dict("type"=>"harmonic", "k" => magnitude, "m_sign"=>1, "center"=> L÷2)
            v_periodic_args = Dict("type"=>"periodic", "period" => L÷4, "magnitude"=>magnitude, "phase"=> L÷2)
            v_step_args = Dict("type"=>"step_potential", "step"=>L÷2+displacement, "magnitude"=>magnitude)
            v_random_args = Dict("type"=>"random", "scale"=>magnitude )
            if !simple
                v_ratchet_PmLr = Dict("type"=>"zero","potentials_profiles"=>[potential_args("smudge",dims;magnitude=magnitude,simple=true),potential_args("left_smudge",dims;magnitude=magnitude,simple=true),potential_args("minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true)])
                v_ratchet_PmLr_mini = Dict("type"=>"zero","potentials_profiles"=>[potential_args("smudge",dims;magnitude=magnitude,simple=true),potential_args("left_smudge",dims;magnitude=magnitude,simple=true),potential_args("minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true)])
                v_ratchet_mLmR = Dict("type"=>"zero","potentials_profiles"=>[potential_args("minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true)])
                v_ratchet_pLpR = Dict("type"=>"zero","potentials_profiles"=>[potential_args("smudge",dims;magnitude=magnitude,simple=true),potential_args("left_smudge",dims;magnitude=magnitude,simple=true)])
                v_ratchet_pLpR_gap = Dict("type"=>"zero","potentials_profiles"=>[potential_args("smudge",dims;magnitude=magnitude,simple=true,displacement=1),potential_args("left_smudge",dims;magnitude=magnitude,simple=true,displacement=-1)])
                v_ratchet_mLmR_gap = Dict("type"=>"zero","potentials_profiles"=>[potential_args("minus_smudge",dims;magnitude=magnitude,simple=true,displacement=1),potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true,displacement=-1)])
                v_ratchet_PR_ML = Dict("type"=>"zero","potentials_profiles"=>[potential_args("minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("left_smudge",dims;magnitude=magnitude,simple=true)]) 
                v_linear_slides_cut1 = Dict("type"=>"zero","potentials_profiles"=>[potential_args("linear",dims;magnitude=magnitude,simple=true,cut_at=1),potential_args("linear",dims;magnitude=-magnitude,simple=true,cut_at=1)])
                v_linear_slides_cut2 = Dict("type"=>"zero","potentials_profiles"=>[potential_args("linear",dims;magnitude=magnitude,simple=true,cut_at=2),potential_args("linear",dims;magnitude=-magnitude,simple=true,cut_at=2)])
                v_linear_slides_cut3 = Dict("type"=>"zero","potentials_profiles"=>[potential_args("linear",dims;magnitude=magnitude,simple=true,cut_at=3),potential_args("linear",dims;magnitude=-magnitude,simple=true,cut_at=3)])
                #v_ratchet_PmLr = Dict("type"=>"zero","potentials_profiles"=>[potential_args("left_minus_smudge",dims;magnitude=magnitude,simple=true),potential_args("minus_smudge",dims;magnitude=magnitude,simple=true)])
            end
        elseif dim == 2
            Lx = dims[1]
            Ly = dims[2]
            # 2D cases
            v_zero_args = Dict("type"=>"zero")
            v_well_args = Dict("type"=>"well", "width_x"=>Lx, "width_y"=>Ly, "magnitude"=> magnitude)
            v_harmonic_args = Dict("type"=>"harmonic", "k" => magnitude, "m_sign"=>1, "center_x"=> Lx÷2, "center_y"=> Ly÷2)
            v_periodic_args = Dict("type"=>"periodic", "period_x" => Lx÷4, "period_y" => Ly÷4, "magnitude"=>magnitude, "phase_x"=> dims[1]÷2, "phase_y"=> dims[2]÷2)
            v_random_args = Dict("type"=>"random", "scale"=>magnitude)
            v_2D_x_wall_slide = Dict("type"=>"2D_x_wall_slide","magnitude"=>magnitude)
            v_xy_slide = Dict("type"=>"xy_slide","magnitude"=>magnitude)
            v_xy_slide_rotated = Dict("type"=>"xy_slide_rotated","magnitude"=>magnitude)
            v_x_slide = Dict("type"=>"2D_x_slide","magnitude"=>magnitude)
            v_y_slide = Dict("type"=>"2D_y_slide","magnitude"=>magnitude)
            if !simple
                v_xy_slides =Dict("type"=>"zero","potentials_profiles"=>[potential_args("xy_slide",dims;magnitude=magnitude,simple=true),potential_args("xy_slide",dims;magnitude=-magnitude,simple=true),potential_args("xy_slide_rotated",dims;magnitude=magnitude,simple=true),potential_args("xy_slide_rotated",dims;magnitude=-magnitude,simple=true)])
                v_rotating_1D_slides = Dict("type"=>"zero","potentials_profiles"=>[potential_args("2D_y_slide",dims;magnitude=magnitude,simple=true),potential_args("2D_x_slide",dims;magnitude=magnitude,simple=true),potential_args("2D_x_slide",dims;magnitude=-magnitude,simple=true),potential_args("2D_y_slide",dims;magnitude=-magnitude,simple=true)])
            end
        else
            error("Unsupported number of dimensions: $dim. Only 1D and 2D cases are supported.")
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
        elseif v_string == "step_potential"
            return v_step_args
        elseif v_string == "random"
            return v_random_args
        elseif v_string == "ratchet_PmLr"
            return v_ratchet_PmLr
        elseif v_string == "ratchet_mLmR"
            return v_ratchet_mLmR
        elseif v_string == "ratchet_pLpR"
            return v_ratchet_pLpR
        elseif v_string == "ratchet_pLpR_gap"
            return v_ratchet_pLpR_gap
        elseif v_string == "ratchet_mLmR_gap"
            return v_ratchet_mLmR_gap
        elseif v_string == "linear_slides_cut1"
            return v_linear_slides_cut1
        elseif v_string == "linear_slides_cut2"
            return v_linear_slides_cut2
        elseif v_string == "linear_slides_cut3"
            return v_linear_slides_cut3
        elseif v_string == "ratchet_PR_ML"
            return v_ratchet_PR_ML
        elseif v_string == "2D_x_wall_slide"
            return v_2D_x_wall_slide
        elseif v_string == "2D_x_slide"
            return v_x_slide
        elseif v_string == "2D_y_slide"
            return v_y_slide
        elseif v_string == "rotating_1D_slides"
            return v_rotating_1D_slides
        elseif v_string =="xy_slide"
            return v_xy_slide
        elseif v_string =="xy_slide_rotated"
            return v_xy_slide_rotated
        elseif v_string =="xy_slides"
            return v_xy_slides
        else
            error("unsupported V string")
        end
    end

end
