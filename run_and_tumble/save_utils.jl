module SaveUtils
using Printf
using JLD2
using Dates
using Sockets
export save_aggregation, save_state

function forcing_list(state)
    if !hasfield(typeof(state), :forcing)
        return Any[]
    end
    if state.forcing isa AbstractVector
        return state.forcing
    end
    return [state.forcing]
end

function param_activity(param)
    if hasfield(typeof(param), :ϵ)
        return Float64(getfield(param, :ϵ))
    end
    return 0.0
end

function param_alpha(param)
    if hasfield(typeof(param), :α)
        return Float64(getfield(param, :α))
    end
    return 0.0
end

function two_force_distance_suffix(state, param)
    if !hasfield(typeof(param), :dims) || length(param.dims) != 1
        return ""
    end
    forcings = forcing_list(state)
    if length(forcings) != 2
        return ""
    end

    force1 = forcings[1]
    force2 = forcings[2]
    if length(force1.bond_indices[1]) != 1 || length(force2.bond_indices[1]) != 1
        return ""
    end

    L = param.dims[1]
    s1 = mod1(force1.bond_indices[1][1], L)
    s2 = mod1(force2.bond_indices[1][1], L)
    d_forward = mod(s2 - s1, L)
    d_short = min(d_forward, mod(s1 - s2, L))
    return @sprintf("_fdist-%d_fdistmin-%d", d_forward, d_short)
end

function sanitize_filename_token(value::AbstractString)
    token = replace(strip(value), r"[^A-Za-z0-9._-]+" => "-")
    token = replace(token, r"-{2,}" => "-")
    token = replace(token, r"^-+" => "")
    token = replace(token, r"-+$" => "")
    return token
end

function description_prefix(description)
    if isnothing(description)
        return ""
    end
    desc = strip(String(description))
    isempty(desc) && return ""
    token = sanitize_filename_token(desc)
    isempty(token) && return ""
    return token * "_"
end

function save_aggregation(agg_res,param,total_sweeps,save_dir; description=nothing)
    mkpath(save_dir)
    state = agg_res
    γ = param.γ
    ffr = if hasfield(typeof(param), :ffr)
        param.ffr isa AbstractVector ? (isempty(param.ffr) ? 0.0 : Float64(param.ffr[1])) : Float64(param.ffr)
    elseif hasfield(typeof(param), :ffrs)
        isempty(param.ffrs) ? 0.0 : Float64(param.ffrs[1])
    else
        0.0
    end
    forcing_magnitude = if hasfield(typeof(state), :forcing)
        if state.forcing isa AbstractVector
            isempty(state.forcing) ? 0.0 : sum(force -> force.magnitude, state.forcing)
        else
            state.forcing.magnitude
        end
    else
        0.0
    end
    activity = param_activity(param)
    alpha = param_alpha(param)
    force_distance_suffix = two_force_distance_suffix(state, param)
    description_tag = description_prefix(description)
    dim = length(param.dims)
    filename = @sprintf("%s/%s%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gamma-%.3f_D-%.1f_f_-%.1f_ffr-%.4f%s_t-%d.jld2",
        save_dir,
        description_tag,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        activity,
        param.dims[1],
        param.ρ₀,
        alpha,
        γ,
        param.D,
        forcing_magnitude,
        ffr,
        force_distance_suffix,
        total_sweeps)
    potential = state.potential 
    @save filename state param potential
    return filename
end

# Original save_state function.
function save_state(state, param, save_dir; tag=nothing, ic=nothing, relaxed_ic::Bool=false, description=nothing)
    mkpath(save_dir)
    # Check if param has forcing_fluctuation_rate or ffr field
    if hasfield(typeof(param), :forcing_fluctuation_rate)
        γ = param.γ 
        ffr = param.forcing_fluctuation_rate * param.N
    elseif hasfield(typeof(param), :ffrs)
        γ = param.γ
        ffr = isempty(param.ffrs) ? 0.0 : param.ffrs[1]
    elseif hasfield(typeof(param), :ffr)

        γ = param.γ
        if param.ffr isa AbstractVector
            ffr = isempty(param.ffr) ? 0.0 : Float64(param.ffr[1])
        else
            ffr = Float64(param.ffr)
        end
    else
        error("Parameter object must have either forcing_fluctuation_rate or ffr field")
    end
    forcing_magnitude = if hasfield(typeof(state), :forcing)
        if state.forcing isa AbstractVector
            isempty(state.forcing) ? 0.0 : sum(force -> force.magnitude, state.forcing)
        else
            state.forcing.magnitude
        end
    else
        0.0
    end
    activity = param_activity(param)
    alpha = param_alpha(param)
    force_distance_suffix = two_force_distance_suffix(state, param)
    description_tag = description_prefix(description)
    dim = length(param.dims)
    hostname = Sockets.gethostname()
    host_tag = split(hostname, '.')[1]
    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    run_id = isnothing(tag) ? "$(timestamp)_$(host_tag)" : string(tag)
    ic_tag = isnothing(ic) ? "unspecified" : string(ic)
    if relaxed_ic
        ic_tag = string(ic_tag, "-relaxed_ic")
    end
    filename = @sprintf("%s/%s%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gamma-%.3f_D-%.1f_f_-%.1f_ffr-%.4f%s_ic-%s_t-%d_id-%s.jld2",
        save_dir,
        description_tag,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        activity,
        param.dims[1],
        param.ρ₀,
        alpha,
        γ,
        param.D,
        forcing_magnitude,
        ffr,
        force_distance_suffix,
        ic_tag,
        state.t,
        run_id)
    potential = state.potential 
    @save filename state param potential
    println("Saved a state to $filename")
    return filename
end

end
