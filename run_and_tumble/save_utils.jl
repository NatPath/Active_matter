module SaveUtils
using Printf
using JLD2
using Dates
using Sockets
export save_aggregation, save_state

const MAX_FILENAME_COMPONENT_BYTES = 240

function has_named_property(obj, name::Symbol)
    hasfield(typeof(obj), name) && return true
    try
        hasproperty(obj, name) && return true
    catch
    end
    try
        return name in propertynames(obj)
    catch
        return false
    end
end

function get_named_property(obj, name::Symbol)
    if hasfield(typeof(obj), name)
        return getfield(obj, name)
    end
    return getproperty(obj, name)
end

function forcing_list(state)
    if !has_named_property(state, :forcing)
        return Any[]
    end
    forcing = get_named_property(state, :forcing)
    if forcing isa AbstractVector
        return forcing
    end
    return [forcing]
end

function param_activity(param)
    if has_named_property(param, :ϵ)
        return Float64(get_named_property(param, :ϵ))
    end
    return 0.0
end

function param_alpha(param)
    if has_named_property(param, :α)
        return Float64(get_named_property(param, :α))
    end
    return 0.0
end

function two_force_distance_suffix(state, param)
    if !has_named_property(param, :dims) || length(get_named_property(param, :dims)) != 1
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

    dims = get_named_property(param, :dims)
    L = dims[1]
    s1 = mod1(force1.bond_indices[1][1], L)
    s2 = mod1(force2.bond_indices[1][1], L)
    d_forward = mod(s2 - s1, L)
    d_short = min(d_forward, mod(s1 - s2, L))
    return @sprintf("_fdist-%d_fdistmin-%d", d_forward, d_short)
end

function forcing_fluctuation_rate_value(param)
    if has_named_property(param, :forcing_fluctuation_rate)
        return Float64(get_named_property(param, :forcing_fluctuation_rate)) * Float64(get_named_property(param, :N))
    elseif has_named_property(param, :ffrs)
        ffrs = get_named_property(param, :ffrs)
        return isempty(ffrs) ? 0.0 : Float64(ffrs[1])
    elseif has_named_property(param, :ffr)
        ffr = get_named_property(param, :ffr)
        if ffr isa AbstractVector
            return isempty(ffr) ? 0.0 : Float64(ffr[1])
        end
        return Float64(ffr)
    end
    error("Parameter object must have one of forcing_fluctuation_rate, ffrs, or ffr")
end

function forcing_magnitude_value(state)
    forcings = forcing_list(state)
    isempty(forcings) && return 0.0
    return sum(force -> force.magnitude, forcings)
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

function truncate_token(token::AbstractString, max_chars::Int)
    max_chars <= 0 && return ""
    s = String(token)
    return length(s) <= max_chars ? s : first(s, max_chars)
end

function filename_component_too_long(path::AbstractString)
    return ncodeunits(basename(path)) > MAX_FILENAME_COMPONENT_BYTES
end

function choose_safe_state_filename(
    save_dir::AbstractString;
    description_tag::AbstractString,
    dim::Int,
    potential_type::AbstractString,
    potential_magnitude::Real,
    fluctuation_type::AbstractString,
    activity::Real,
    L::Int,
    rho0::Real,
    alpha::Real,
    gamma::Real,
    D::Real,
    forcing_magnitude::Real,
    ffr::Real,
    force_distance_suffix::AbstractString,
    ic_tag::AbstractString,
    t_val::Integer,
    run_id::AbstractString,
)
    candidates = String[]

    push!(candidates, @sprintf("%s/%s%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gamma-%.3f_D-%.1f_f_-%.1f_ffr-%.4f%s_ic-%s_t-%d_id-%s.jld2",
        save_dir, description_tag, dim, potential_type, potential_magnitude, fluctuation_type, activity,
        L, rho0, alpha, gamma, D, forcing_magnitude, ffr, force_distance_suffix, ic_tag, t_val, run_id))

    desc_short = truncate_token(description_tag, 20)
    ic_short = truncate_token(ic_tag, 24)
    push!(candidates, @sprintf("%s/%s%dD_L%d_rho%.1e_a%.2f_g%.3f_D%.1f_f%.1f_ffr%.4f%s_ic-%s_t%d_id-%s.jld2",
        save_dir, desc_short, dim, L, rho0, alpha, gamma, D, forcing_magnitude, ffr, force_distance_suffix, ic_short, t_val, run_id))

    desc_tiny = truncate_token(description_tag, 12)
    push!(candidates, @sprintf("%s/%s%dD_L%d_rho%.1e_t%d_ic-%s_id-%s.jld2",
        save_dir, desc_tiny, dim, L, rho0, t_val, ic_short, run_id))

    push!(candidates, @sprintf("%s/%dD_L%d_t%d_id-%s.jld2",
        save_dir, dim, L, t_val, run_id))

    for (idx, candidate) in enumerate(candidates)
        if !filename_component_too_long(candidate)
            if idx > 1
                println("WARNING: save_state filename was too long; using compact fallback format #", idx)
            end
            return candidate
        end
    end

    error("Could not build a filename within filesystem limits. Try a shorter save tag (current id='$run_id').")
end

function atomic_jld2_save(filename::AbstractString, state, param, potential; max_attempts::Int=3)
    save_dir = dirname(filename)
    base_name = basename(filename)
    mkpath(save_dir)
    last_err = nothing

    for attempt in 1:max_attempts
        temp_name = joinpath(
            save_dir,
            "." * base_name * ".tmp-" * string(getpid()) * "-" * Dates.format(now(), "yyyymmdd-HHMMSSsss") * "-a" * string(attempt),
        )
        try
            isfile(temp_name) && rm(temp_name; force=true)
            jldopen(temp_name, "w"; iotype=IOStream) do file
                file["state"] = state
                file["param"] = param
                file["potential"] = potential
            end
            if isfile(filename)
                rm(filename; force=true)
            end
            mv(temp_name, filename; force=true)
            return filename
        catch err
            last_err = err
            try
                isfile(temp_name) && rm(temp_name; force=true)
            catch
            end
            if attempt < max_attempts
                println("WARNING: save attempt $(attempt)/$(max_attempts) failed for $(filename): $(sprint(showerror, err))")
                sleep(min(5.0, 0.5 * 2.0^(attempt - 1)))
            end
        end
    end

    if last_err !== nothing
        throw(last_err)
    end
    error("atomic_jld2_save failed unexpectedly for $(filename)")
end

function save_aggregation(agg_res,param,total_sweeps,save_dir; description=nothing)
    mkpath(save_dir)
    state = agg_res
    γ = param.γ
    ffr = forcing_fluctuation_rate_value(param)
    forcing_magnitude = forcing_magnitude_value(state)
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
    atomic_jld2_save(filename, state, param, potential)
    return filename
end

# Original save_state function.
function save_state(state, param, save_dir; tag=nothing, ic=nothing, relaxed_ic::Bool=false, description=nothing)
    mkpath(save_dir)
    γ = param.γ
    ffr = forcing_fluctuation_rate_value(param)
    forcing_magnitude = forcing_magnitude_value(state)
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
    filename = choose_safe_state_filename(
        save_dir;
        description_tag=description_tag,
        dim=dim,
        potential_type=param.potential_type,
        potential_magnitude=param.potential_magnitude,
        fluctuation_type=param.fluctuation_type,
        activity=activity,
        L=param.dims[1],
        rho0=param.ρ₀,
        alpha=alpha,
        gamma=γ,
        D=param.D,
        forcing_magnitude=forcing_magnitude,
        ffr=ffr,
        force_distance_suffix=force_distance_suffix,
        ic_tag=ic_tag,
        t_val=state.t,
        run_id=run_id,
    )
    potential = state.potential 
    atomic_jld2_save(filename, state, param, potential)
    println("Saved a state to $filename")
    return filename
end

end
