module SaveUtils
using Printf
using JLD2
using Dates
using Sockets
export save_aggregation, save_state
function save_aggregation(agg_res,param,total_sweeps,save_dir)
    mkpath(save_dir)
    γ = param.γ
    ffr = param.f
    dim = length(param.dims)
    filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gamma-%.3f_D-%.1f_f_-%.1f_ffr-%.4f_t-%d.jld2",
        save_dir,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        param.ϵ,
        param.dims[1],
        param.ρ₀,
        param.α,
        γ,
        param.D,
        state.forcing.magnitude,
        ffr,
        state.t)
    potential = state.potential 
    @save filename state param potential
    return filename
end

# Original save_state function.
function save_state(state, param, save_dir; tag=nothing, ic=nothing)
    mkpath(save_dir)
    # Check if param has forcing_fluctuation_rate or ffr field
    if hasfield(typeof(param), :forcing_fluctuation_rate)
        γ = param.γ 
        ffr = param.forcing_fluctuation_rate * param.N
    elseif hasfield(typeof(param), :ffr)

        γ = param.γ
        ffr = param.ffr
    else
        error("Parameter object must have either forcing_fluctuation_rate or ffr field")
    end
    dim = length(param.dims)
    hostname = Sockets.gethostname()
    host_tag = split(hostname, '.')[1]
    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    run_id = isnothing(tag) ? "$(timestamp)_$(host_tag)" : string(tag)
    ic_tag = isnothing(ic) ? "unspecified" : string(ic)
    filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gamma-%.3f_D-%.1f_f_-%.1f_ffr-%.4f_ic-%s_t-%d_id-%s.jld2",
        save_dir,
        dim,
        param.potential_type,
        param.potential_magnitude,
        param.fluctuation_type,
        param.ϵ,
        param.dims[1],
        param.ρ₀,
        param.α,
        γ,
        param.D,
        state.forcing.magnitude,
        ffr,
        ic_tag,
        state.t,
        run_id)
    # filename = @sprintf("%s/%dD_potential-%s_Vscale-%.1f_fluctuation-%s_activity-%.2f_L-%d_rho-%.1e_alpha-%.2f_gammap-%.2f_D-%.1f_t-%d.jld2",
    #     save_dir,
    #     dim,
    #     param.potential_type,
    #     param.potential_magnitude,
    #     param.fluctuation_type,
    #     param.ϵ,
    #     param.dims[1],
    #     param.ρ₀,
    #     param.α,
    #     γ′,
    #     param.D,
    #     state.t)
    potential = state.potential 
    @save filename state param potential
    println("Saved a state to $filename")
    return filename
end

end
