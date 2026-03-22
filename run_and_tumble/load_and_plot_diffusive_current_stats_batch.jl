using ArgParse
using JLD2

include("load_and_plot_diffusive_current_stats.jl")

function wildcard_to_regex(pattern::String)
    special = Set(['.', '^', '$', '+', '(', ')', '[', ']', '{', '}', '|', '\\'])
    io = IOBuffer()
    for c in pattern
        if c == '*'
            print(io, ".*")
        elseif c == '?'
            print(io, ".")
        elseif c in special
            print(io, '\\', c)
        else
            print(io, c)
        end
    end
    return Regex("^" * String(take!(io)) * "\$")
end

function collect_matching_files(dir::String, pattern::String; recursive::Bool=false)
    matcher = wildcard_to_regex(pattern)
    files = String[]

    if recursive
        for (root, _, names) in walkdir(dir)
            for name in names
                if occursin(matcher, name)
                    push!(files, joinpath(root, name))
                end
            end
        end
    else
        for name in readdir(dir)
            path = joinpath(dir, name)
            if isfile(path) && occursin(matcher, name)
                push!(files, path)
            end
        end
    end

    sort!(files)
    return files
end

function add_unique!(paths::Vector{String}, seen::Set{String}, candidate::String)
    full_path = abspath(candidate)
    if !(full_path in seen)
        push!(paths, full_path)
        push!(seen, full_path)
    end
end

function collect_input_files(inputs::Vector{String}, default_glob::String; recursive::Bool=false)
    files = String[]
    seen = Set{String}()

    for input in inputs
        if isfile(input)
            if endswith(lowercase(input), ".jld2")
                add_unique!(files, seen, input)
            else
                println("Skipping non-JLD2 file: ", input)
            end
            continue
        end

        if isdir(input)
            for match in collect_matching_files(input, default_glob; recursive=recursive)
                add_unique!(files, seen, match)
            end
            continue
        end

        if occursin('*', input) || occursin('?', input)
            dir = dirname(input)
            dir = isempty(dir) ? "." : dir
            pattern = basename(input)
            if isdir(dir)
                for match in collect_matching_files(dir, pattern; recursive=recursive)
                    add_unique!(files, seen, match)
                end
            else
                println("Skipping glob with missing directory: ", input)
            end
            continue
        end

        println("Skipping missing path: ", input)
    end

    sort!(files)
    return files
end

function parse_batch_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "inputs"
            help = "Input path(s): JLD2 files, directories, or globs"
            required = true
            nargs = '+'
        "--glob"
            help = "Filename wildcard for directory inputs"
            default = "*.jld2"
        "--recursive"
            help = "Recursively scan directory inputs"
            action = :store_true
        "--out_dir"
            help = "Output directory for per-state plot folders"
            default = "results_figures/fitting"
        "--include_abs_mean_in_spatial_f_plot"
            help = "Include |<F>| in spatial F statistics panel"
            action = :store_true
        "--keep_diagonal_in_multiforce_cut"
            help = "Do not smooth the diagonal point in quarter correlation cut"
            action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_batch_commandline()
    recursive = get(args, "recursive", false)
    include_abs_mean = get(args, "include_abs_mean_in_spatial_f_plot", false)
    keep_diag = get(args, "keep_diagonal_in_multiforce_cut", false)
    inputs = String.(args["inputs"])
    default_glob = String(args["glob"])

    files = collect_input_files(inputs, default_glob; recursive=recursive)
    println("Found ", length(files), " matching file(s)")

    if isempty(files)
        return
    end

    for saved_state in files
        try
            println("Processing ", saved_state)
            @load saved_state state param potential
            if length(param.dims) != 1
                println("Skipping non-1D state: ", saved_state)
                continue
            end

            save_diffusive_1d_components(
                saved_state,
                state,
                param,
                potential,
                args["out_dir"];
                include_abs_mean_in_spatial_f_plot=include_abs_mean,
                keep_diagonal_in_multiforce_cut=keep_diag,
            )
        catch e
            println("Failed to process ", saved_state, ": ", e)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
