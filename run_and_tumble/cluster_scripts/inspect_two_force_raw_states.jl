#!/usr/bin/env julia

using JLD2
using Printf

function usage()
    println("""
Usage:
  julia --startup-file=no cluster_scripts/inspect_two_force_raw_states.jl <file-or-dir> [more files/dirs...]

Description:
  Inspect fetched raw two-force diffusive state files without loading the full simulation module stack.
  Prints saved param fields, forcing bonds, final direction flags, and bond-passage statistics.
  When multiple files are provided, also prints counts of final direction-flag patterns.
""")
end

function collect_files(paths::Vector{String})
    files = String[]
    for path in paths
        abs_path = abspath(path)
        if isfile(abs_path)
            push!(files, abs_path)
        elseif isdir(abs_path)
            for (root, _, names) in walkdir(abs_path)
                for name in names
                    endswith(name, ".jld2") || continue
                    push!(files, joinpath(root, name))
                end
            end
        else
            error("Path does not exist: $path")
        end
    end
    return sort(unique(files))
end

function reconstructed_fields(x)
    return getfield(x, :fields)
end

function param_summary(param)
    vals = reconstructed_fields(param)[1]
    return (
        gamma = vals[1],
        dims = vals[2],
        rho0 = vals[3],
        N = vals[4],
        D = vals[5],
        potential_type = vals[6],
        fluctuation_type = vals[7],
        potential_magnitude = vals[8],
        ffr = vals[9],
    )
end

function forcing_summary(state)
    forcing = reconstructed_fields(state)[12]
    out = NamedTuple[]
    for force in forcing
        vals = reconstructed_fields(force)
        push!(out, (
            bond_indices = vals[1],
            direction_flag = vals[2],
            magnitude = vals[3],
        ))
    end
    return out
end

function bond_pass_summary(state)
    stats = reconstructed_fields(state)[8]
    return (
        forward = get(stats, :bond_pass_forward_avg, missing),
        reverse = get(stats, :bond_pass_reverse_avg, missing),
        total = get(stats, :bond_pass_total_avg, missing),
        total_sq = get(stats, :bond_pass_total_sq_avg, missing),
        sample_count = get(stats, :bond_pass_sample_count, missing),
        track_mask = get(stats, :bond_pass_track_mask, missing),
    )
end

function flag_signature(forces)
    chars = map(forces) do force
        force.direction_flag ? "T" : "F"
    end
    return join(chars, "")
end

function main(args)
    if isempty(args) || any(arg -> arg in ("-h", "--help"), args)
        usage()
        return
    end

    files = collect_files(args)
    if isempty(files)
        println("No .jld2 files found.")
        return
    end

    flag_counts = Dict{String,Int}()

    for file in files
        state = nothing
        param = nothing
        potential = nothing
        JLD2.@load file state param potential

        ps = param_summary(param)
        forces = forcing_summary(state)
        bs = bond_pass_summary(state)
        sig = flag_signature(forces)
        flag_counts[sig] = get(flag_counts, sig, 0) + 1

        println(basename(file))
        @printf("  dims=%s rho0=%s N=%s D=%s potential=%s fluctuation=%s ffr=%s\n",
            repr(ps.dims), repr(ps.rho0), repr(ps.N), repr(ps.D),
            repr(ps.potential_type), repr(ps.fluctuation_type), repr(ps.ffr))
        for (idx, force) in enumerate(forces)
            @printf("  force[%d]: bond=%s direction_flag=%s magnitude=%s\n",
                idx, repr(force.bond_indices), repr(force.direction_flag), repr(force.magnitude))
        end
        println("  final_flag_signature=$(sig)")
        println("  bond_pass_forward_avg=$(repr(bs.forward))")
        println("  bond_pass_reverse_avg=$(repr(bs.reverse))")
        println("  bond_pass_total_avg=$(repr(bs.total))")
        println("  bond_pass_total_sq_avg=$(repr(bs.total_sq))")
        println("  bond_pass_sample_count=$(repr(bs.sample_count))")
        println("  bond_pass_track_mask=$(repr(bs.track_mask))")
        println()
    end

    if length(files) > 1
        println("Final direction-flag signature counts")
        for key in sort(collect(keys(flag_counts)))
            println("  $key => $(flag_counts[key])")
        end
    end
end

main(ARGS)
