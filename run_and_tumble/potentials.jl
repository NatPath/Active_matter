module Potentials
    using Plots
    export Potential, potential_args, choose_potential

    mutable struct Potential
        V::Array{Float64}
        fluctuation_mask::Array{Float64}
        fluctuation_sign::Int64
    end

    function setPotential(V, fluctuation_mask)
        return Potential(V, fluctuation_mask, 1)
    end

    function choose_potential(v_args,dims; boundary_walls= false)
        V = zeros(Float64, dims)
        L = dims[1]
        middle = Int(L÷2)
        x = LinRange(0,L,L)
        v_string = v_args["type"]
        magnitude=1
        if v_string == "well"
            for i in 1:L
                #V[i] = exp(-((i - L/2)^2) / (2 * (2)^2))  # Gaussian potential centered in the middle
                width = v_args["width"]
                magnitude = v_args["magnitude"]
                V[i] = i<= L/2 + width && i> L/2-width  ? magnitude : 0
            end
        elseif v_string == "zero"
            magnitude=1
        elseif v_string == "smudge"
            location = v_args["location"]
            magnitude = v_args["magnitude"]
        elseif v_string =="modified_smudge"
            location = v_args["location"]
            magnitude = v_args["magnitude"]
            V[location] = magnitude
            V[location-1] = magnitude/4
            
        elseif v_string == "delta"
            V[middle] = v_args["magnitude"]
            magnitude = v_args["magnitude"]
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
            error("unsupported V string")
        end
        V_plot=plot(V)
        display(V_plot)
        fluctuating_mask = zeros(Float64,dims)
        fluctuating_mask[middle] = -magnitude/2
        fluctuating_mask[middle-1] = magnitude/2
        
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

    function potential_args(v_string, dims; magnitude = 0.4)
        L = dims[1]
        v_well_args = Dict("type"=>"well", "width"=>1, "magnitude"=> magnitude)
        v_zero_args = Dict("type"=>"zero")
        v_smudge_args = Dict("type"=>"smudge","location" => L÷2 , "magnitude" => magnitude)
        v_extended_smudge_args = Dict("type"=>"smudge","location" => L÷2 ,"length" => 5, "magnitude" => magnitude)
        v_modified_smudge_args = Dict("type"=>"modified_smudge","location" => L÷2 , "magnitude" => magnitude)
        v_delta_args = Dict("type"=>"delta", "location" => L÷2, "magnitude"=>10^3*magnitude)
        v_linear_args = Dict("type"=> "linear", "slope" => 1, "shift"=>0)
        v_harmonic_args = Dict("type"=>"harmonic", "k" => magnitude, "m_sign"=>1, "center"=> L÷2)
        v_periodic_args = Dict("type"=>"periodic", "period" => L÷4, "magnitude"=>magnitude, "phase"=> L÷2)
        v_random_args = Dict("type"=>"random", "scale"=>magnitude )
        if v_string== "well"
            return v_well_args
        elseif v_string=="zero"
            return v_zero_args
        elseif v_string == "smudge"
            return v_smudge_args
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
        else
            error("unsupported V string")
        end
    end

end