using JLD2
using ArgParse
using Statistics
using Random
using Dates
include("save_utils.jl")
include("potentials.jl")
include("modules_run_and_tumble.jl")
using .FP
using .SaveUtils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "input"
            help = "Path to either a single parallel results .jld2 file or directory containing multiple files"
            required = true
        "--output_dir"
            help = "Output directory for the recovered dummy states"
            default = "recovered_states"
        "--batch"
            help = "Process all .jld2 files in the input directory"
            action = :store_true
        "--match_mode"
            help = "Match files from parallel_dir with broken_dir based on creation dates"
            action = :store_true
        "--parallel_dir"
            help = "Directory containing parallel simulation results (recover_2D files)"
            default = "saved_states_parallel/recover_2D"
        "--broken_dir"
            help = "Directory containing broken aggregated results with correct parameters"
            default = "dummy_states/passive_case/to_recover"
        "--time_tolerance"
            help = "Maximum time difference in seconds for file matching"
            arg_type = Int
            default = 60
    end
    return parse_args(s)
end

function get_file_creation_time(filepath::String)
    """
    Get file creation time in seconds since epoch
    """
    return mtime(filepath)
end

function get_readable_date(filepath::String)
    """
    Get human readable creation date
    """
    return Dates.unix2datetime(mtime(filepath))
end

function match_files_by_date(parallel_dir::String, broken_dir::String, time_tolerance::Int)
    """
    Match files from two directories based on creation dates
    Returns array of tuples: (broken_file_path, parallel_file_path, time_diff)
    """
    println("Matching files between:")
    println("  Parallel dir: $parallel_dir")
    println("  Broken dir: $broken_dir")
    println("  Time tolerance: $time_tolerance seconds")
    
    # Get all .jld2 files from both directories
    parallel_files = String[]
    broken_files = String[]
    
    if !isdir(parallel_dir)
        error("Parallel directory does not exist: $parallel_dir")
    end
    
    if !isdir(broken_dir)
        error("Broken directory does not exist: $broken_dir")
    end
    
    for file in readdir(parallel_dir)
        if endswith(file, ".jld2")
            push!(parallel_files, joinpath(parallel_dir, file))
        end
    end
    
    for file in readdir(broken_dir)
        if endswith(file, ".jld2")
            push!(broken_files, joinpath(broken_dir, file))
        end
    end
    
    println("Found $(length(parallel_files)) parallel files")
    println("Found $(length(broken_files)) broken files")
    
    # Create date-file pairs
    parallel_dates = [(get_file_creation_time(f), f, get_readable_date(f)) for f in parallel_files]
    broken_dates = [(get_file_creation_time(f), f, get_readable_date(f)) for f in broken_files]
    
    # Sort by date
    sort!(parallel_dates, by=x->x[1])
    sort!(broken_dates, by=x->x[1])
    
    println("\nMatching files by creation date...")
    
    matches = Tuple{String, String, Float64}[]
    unmatched_broken = String[]
    
    for (broken_date, broken_file, broken_readable) in broken_dates
        best_match = ""
        best_match_readable = nothing
        min_diff = Inf
        
        # Find closest parallel file within tolerance
        for (parallel_date, parallel_file, parallel_readable) in parallel_dates
            diff = abs(parallel_date - broken_date)
            if diff < min_diff && diff <= time_tolerance
                min_diff = diff
                best_match = parallel_file
                best_match_readable = parallel_readable
            end
        end
        
        if !isempty(best_match)
            push!(matches, (broken_file, best_match, min_diff))
            println("MATCH: $(basename(broken_file)) ↔ $(basename(best_match)) (diff: $(round(min_diff, digits=1))s)")
            println("  Broken:   $broken_readable")
            println("  Parallel: $best_match_readable")
        else
            push!(unmatched_broken, broken_file)
            println("UNMATCHED: $(basename(broken_file)) - $broken_readable")
        end
    end
    
    println("\nMatching summary:")
    println("  Matched pairs: $(length(matches))")
    println("  Unmatched broken files: $(length(unmatched_broken))")
    
    return matches, unmatched_broken
end

function recover_4D_correlation(normalized_dists, corr_mats, param)
    """
    Attempt to recover proper 4D correlation tensor from parallel results
    """
    println("Original correlation matrix dimensions: ", size(corr_mats[1]))
    println("Number of parallel results: ", length(corr_mats))
    
    # Check what we're dealing with
    corr_dims = ndims(corr_mats[1])
    dist_dims = ndims(normalized_dists[1])
    
    println("Correlation tensor has $corr_dims dimensions")
    println("Density arrays have $dist_dims dimensions") 
    
    # Aggregate the results properly based on dimensions
    if corr_dims == 2 && dist_dims == 1
        # 1D case - this should work fine
        println("Detected 1D case")
        stacked_corr = cat(corr_mats..., dims=3)
        avg_corr = dropdims(mean(stacked_corr, dims=3), dims=3)
        stacked_dists = cat(normalized_dists..., dims=2)
        avg_dists = dropdims(mean(stacked_dists, dims=2), dims=2)
        
    elseif corr_dims == 4 && dist_dims == 2
        # 2D case - this is what we want
        println("Detected proper 2D case with 4D correlation tensor")
        stacked_corr = cat(corr_mats..., dims=5)
        avg_corr = dropdims(mean(stacked_corr, dims=5), dims=5)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
    elseif corr_dims == 3 && dist_dims == 2
        # This is likely the problematic case from cluster
        println("Detected problematic 2D case with 3D correlation tensor")
        println("Correlation tensor size: ", size(corr_mats[1]))
        
        # We can't fully recover the 4D tensor, but we can at least aggregate properly
        # and flag this as problematic
        stacked_corr = cat(corr_mats..., dims=4)
        avg_corr = dropdims(mean(stacked_corr, dims=4), dims=4)
        stacked_dists = cat(normalized_dists..., dims=3)
        avg_dists = dropdims(mean(stacked_dists, dims=3), dims=3)
        
        println("WARNING: Cannot recover full 4D correlation tensor from 3D data!")
        println("The aggregated result will have reduced correlation information.")
        
    else
        error("Unexpected dimensions: corr_dims=$corr_dims, dist_dims=$dist_dims")
    end
    
    println("Final averaged correlation dimensions: ", size(avg_corr))
    println("Final averaged density dimensions: ", size(avg_dists))
    
    return avg_dists, avg_corr
end

function process_single_file(parallel_results_file, output_dir)
    """
    Process a single parallel results file
    """
    println("Processing file: $parallel_results_file")
    
    # Load the parallel results
    @load parallel_results_file normalized_dists corr_mats avg_corr avg_dists
    
    println("Analyzing loaded data...")
    println("Number of runs: ", length(normalized_dists))
    println("Density dimensions: ", size(normalized_dists[1]))
    println("Correlation dimensions: ", size(corr_mats[1]))
    
    # Load parameters from your saved dummy state file
    dummy_state_file = "dummy_states/passive_case/2D_potential-xy_slides_Vscale-16.0_fluctuation-profile_switch_activity-0.00_L-32_rho-1.0e+02_alpha-0.00_gammap-32.00_D-1.0_t-9000000.jld2"
    
    if isfile(dummy_state_file)
        println("Loading parameters from: $dummy_state_file")
        @load dummy_state_file state param potential
        t = state.t
        println("Loaded state time: $t")
        
        dims = param.dims
        println("Loaded param: L=$(param.dims), α=$(param.α), γ′=$(param.γ*param.N)")
        rng = MersenneTwister(123)
        v_args = Potentials.potential_args("xy_slides", dims; magnitude=16.0)
        potential = Potentials.choose_potential(v_args, dims; fluctuation_type="profile_switch", rng=rng)
        reference_state = FP.setState(state.t, rng, param, 1.0, potential)
        state = reference_state
    else
        println("Warning: Could not find dummy state file, using default parameters")
        L = size(normalized_dists[1], 1)
        dims = (L, L)
        
        param = FP.setParam(
            0.0, 1.0, 0.0, dims, 100.0, 1.0,
            "xy_slides", "profile_switch", 16.0
        )
        
        rng = MersenneTwister(123)
        v_args = Potentials.potential_args("xy_slides", dims; magnitude=16.0)
        potential = Potentials.choose_potential(v_args, dims; fluctuation_type="profile_switch", rng=rng)
        reference_state = FP.setState(0, rng, param, 1.0, potential)
        state = reference_state
    end
    
    # Recover the correlation tensor
    recovered_dists, recovered_corr = recover_4D_correlation(normalized_dists, corr_mats, param)
    
    # Create dummy state with recovered data
    nsteps = state.t
    dummy_state = FP.setDummyState(state, recovered_dists, recovered_corr, nsteps)
    
    # Save the recovered dummy state
    filename = save_state(dummy_state, param, output_dir)
    println("Recovered dummy state saved to: $filename")
    
    return filename
end

function process_directory(input_dir, output_dir)
    """
    Process all .jld2 files in a directory
    """
    println("Processing all .jld2 files in directory: $input_dir")
    
    # Find all .jld2 files in the directory
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(input_dir))
    
    if isempty(jld2_files)
        println("No .jld2 files found in $input_dir")
        return
    end
    
    println("Found $(length(jld2_files)) .jld2 files to process")
    
    # Process each file
    processed_files = String[]
    failed_files = String[]
    
    for (i, filename) in enumerate(jld2_files)
        filepath = joinpath(input_dir, filename)
        println("\n=== Processing file $i/$(length(jld2_files)): $filename ===")
        
        try
            output_filename = process_single_file(filepath, output_dir)
            push!(processed_files, output_filename)
            println("✓ Successfully processed: $filename")
        catch e
            println("✗ Failed to process $filename: $e")
            push!(failed_files, filename)
        end
    end
    
    # Summary
    println("\n=== BATCH PROCESSING SUMMARY ===")
    println("Total files: $(length(jld2_files))")
    println("Successfully processed: $(length(processed_files))")
    println("Failed: $(length(failed_files))")
    
    if !isempty(failed_files)
        println("Failed files:")
        for f in failed_files
            println("  - $f")
        end
    end
end

function process_matched_pair(broken_file::String, parallel_file::String, output_dir::String)
    """
    Process a matched pair: use parallel results with parameters from broken file
    """
    println("\nProcessing matched pair:")
    println("  Parallel results: $(basename(parallel_file))")
    println("  Parameters from: $(basename(broken_file))")
    
    # Load parallel results
    @load parallel_file normalized_dists corr_mats
    println("Loaded parallel results with $(length(normalized_dists)) runs")
    
    # Load parameters from broken file
    @load broken_file state param potential
    println("Loaded parameters: L=$(param.dims), α=$(param.α), γ′=$(param.γ*param.N)")
    
    # Recover the correlation tensor using the correct parameters
    recovered_dists, recovered_corr = recover_4D_correlation(normalized_dists, corr_mats, param)
    
    # Create dummy state with recovered data and correct parameters
    nsteps = state.t
    dummy_state = FP.setDummyState(state, recovered_dists, recovered_corr, nsteps)
    
    # Save the recovered dummy state
    filename = save_state(dummy_state, param, output_dir)
    println("Recovered dummy state saved to: $filename")
    
    return filename
end

function process_matched_files(matches, output_dir::String)
    """
    Process all matched file pairs
    """
    println("\n=== PROCESSING MATCHED PAIRS ===")
    
    processed_files = String[]
    failed_files = String[]
    
    for (i, (broken_file, parallel_file, time_diff)) in enumerate(matches)
        println("\n--- Processing pair $i/$(length(matches)) ---")
        
        try
            output_filename = process_matched_pair(broken_file, parallel_file, output_dir)
            push!(processed_files, output_filename)
            println("✓ Successfully processed pair $i")
        catch e
            println("✗ Failed to process pair $i: $e")
            push!(failed_files, (broken_file, parallel_file))
            println(e)
        end
    end
    
    # Summary
    println("\n=== MATCHING PROCESSING SUMMARY ===")
    println("Total matched pairs: $(length(matches))")
    println("Successfully processed: $(length(processed_files))")
    println("Failed: $(length(failed_files))")
    
    if !isempty(failed_files)
        println("Failed pairs:")
        for (broken, parallel) in failed_files
            println("  - $(basename(broken)) ↔ $(basename(parallel))")
        end
    end
    
    return processed_files, failed_files
end

function main()
    args = parse_commandline()
    
    input_path = args["input"]
    output_dir = args["output_dir"]
    batch_mode = args["batch"]
    match_mode = args["match_mode"]
    parallel_dir = args["parallel_dir"]
    broken_dir = args["broken_dir"]
    time_tolerance = args["time_tolerance"]
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    if match_mode
        # Match files by creation date and process pairs
        matches, unmatched = match_files_by_date(parallel_dir, broken_dir, time_tolerance)
        
        if isempty(matches)
            println("No file matches found!")
            return
        end
        
        # Save matching report
        report_file = joinpath(output_dir, "matching_report.txt")
        open(report_file, "w") do f
            println(f, "File Matching Report - $(Dates.now())")
            println(f, "="^50)
            println(f, "Parallel directory: $parallel_dir")
            println(f, "Broken directory: $broken_dir")
            println(f, "Time tolerance: $time_tolerance seconds")
            println(f, "")
            println(f, "MATCHED PAIRS:")
            for (i, (broken, parallel, diff)) in enumerate(matches)
                println(f, "$i. $(basename(broken)) ↔ $(basename(parallel)) ($(round(diff, digits=1))s)")
            end
            println(f, "")
            println(f, "UNMATCHED FILES:")
            for broken in unmatched
                println(f, "- $(basename(broken))")
            end
        end
        println("Matching report saved to: $report_file")
        
        # Process matched pairs
        processed, failed = process_matched_files(matches, output_dir)
        
    elseif batch_mode || isdir(input_path)
        # Process directory
        if !isdir(input_path)
            error("Input path $input_path is not a directory")
        end
        process_directory(input_path, output_dir)
    else
        # Process single file
        if !isfile(input_path)
            error("Input file $input_path does not exist")
        end
        process_single_file(input_path, output_dir)
    end
end

main()
