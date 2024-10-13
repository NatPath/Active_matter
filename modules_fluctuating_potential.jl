#Wrap everything with a module to allow redefinition of type
module FP

    mutable struct Param
        D::Float64  # diffusion constant
        Lx::Int64   # system size along x
        Ly::Int64   # system size along y

        ρ₀::Float64  # density
        N::Int64    # number of particles
    end

    #constructor
    function setParam(D, Lx, Ly, ρ₀)

        N = Int(round( ρ₀*Lx*Ly ))       # number of particles

        param = Param(D, Lx, Ly, ρ₀, N)
        return param
    end


    mutable struct State
        t::Float64              # time
        pos::Array{Int64, 2}    # position vector
        ρ::Array{Int64, 2}      # density field
    end

    function setState(t, param, pos)
        ρ = zeros(Int64, param.Lx, param.Ly)

        for n in 1:param.N
            x, y = pos[n, 1], pos[n,2]
            ρ[x,y] += 1
        end

        state = State(t, pos, ρ)
        return state
    end

    function update!(Δt, param, state, rng)
        D = param.D

        t_end = state.t + Δt
        while state.t ≤ t_end
            n = rand(rng, 1:param.N)

            w = [D, D, D, D]  # right, up, left, down
            move = tower_sampling(w, sum(w), rng)

            if move==1
                state.pos[n,1] = mod1( state.pos[n,1] + 1, param.Lx)
            elseif move==2
                state.pos[n,2] = mod1( state.pos[n,2] + 1, param.Ly)
            elseif move==3
                state.pos[n,1] = mod1( state.pos[n,1] - 1, param.Lx)
            else
                state.pos[n,2] = mod1( state.pos[n,2] - 1, param.Ly)
            end

            state.t += 1/(param.N*sum(w))
        end

        # update the density field
        fill!(state.ρ, 0.0)
        for n in 1:param.N
            x, y = state.pos[n, 1], state.pos[n,2]
            state.ρ[x,y] += 1
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



function make_movie!(state, param, t_gap, n_frame, rng, file_name, in_fps)
    # t_gap = time gap between each frame
    # n_frame = number of frames
    # file_name = name of your simulation file
    # in_fps = number of frames per second

    #initialize
    state.t = 0
    prg = Progress(n_frame)

    #animation macro
    anim = @animate for frame in 1:n_frame
        FP.update!( t_gap, param, state, rng)

        x_range = range(1, param.Lx, length = param.Lx)
        y_range = range(1, param.Ly, length = param.Ly)
        
        heatmap(x_range, y_range, transpose(state.ρ), size=(1000,700), c=cgrad(:inferno), xlims = (0, param.Lx), ylims = (0, param.Ly), clims = (0,3), aspect_ratio = 1, xlabel="x", ylabel="y", cbar_width=5)
        next!(prg)
    end

    println("Simulation complete, producing movie")
    name = @sprintf("%s.gif", file_name)
    gif(anim, name, fps = in_fps)
end



