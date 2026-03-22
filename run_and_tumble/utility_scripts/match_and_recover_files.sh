#!/bin/bash

# Configuration
PARALLEL_DIR="saved_states_parallel/recover_2D_preserved"
BROKEN_DIR="dummy_states/passive_case/to_recover_preserved"
OUTPUT_DIR="recovered_states_matched"
RECOVERY_SCRIPT="recover_4D_from_parallel.jl"
AGGREGATE_SCRIPT="aggregate_results.jl"
LOG_FILE="file_matching.log"
TIME_TOLERANCE=60

# New flags
ALLOW_MULTIPLE=false
AGGREGATE_RESULTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --allow-multiple)
            ALLOW_MULTIPLE=true
            shift
            ;;
        --aggregate)
            AGGREGATE_RESULTS=true
            shift
            ;;
        --parallel-dir)
            PARALLEL_DIR="$2"
            shift 2
            ;;
        --broken-dir)
            BROKEN_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --time-tolerance)
            TIME_TOLERANCE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --allow-multiple     Allow multiple recovered files per parameter set (adds _index suffix)"
            echo "  --aggregate         Aggregate all recovered states using aggregate_results.jl"
            echo "  --parallel-dir DIR  Directory containing parallel simulation results"
            echo "  --broken-dir DIR    Directory containing broken/incomplete states"
            echo "  --output-dir DIR    Output directory for recovered states"
            echo "  --time-tolerance N  Time tolerance for matching (seconds)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "File Matching and Recovery Log - $(date)" > "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "Configuration:" >> "$LOG_FILE"
echo "  Allow multiple results: $ALLOW_MULTIPLE" >> "$LOG_FILE"
echo "  Aggregate results: $AGGREGATE_RESULTS" >> "$LOG_FILE"
echo "  Time tolerance: $TIME_TOLERANCE seconds" >> "$LOG_FILE"

echo "Step 1: Checking directories and recovery script..."
echo "Step 1: Checking directories and recovery script..." >> "$LOG_FILE"

# Check if directories exist
if [[ ! -d "$PARALLEL_DIR" ]]; then
    echo "ERROR: Directory $PARALLEL_DIR does not exist"
    exit 1
fi

if [[ ! -d "$BROKEN_DIR" ]]; then
    echo "ERROR: Directory $BROKEN_DIR does not exist" 
    exit 1
fi

# Check if recovery script exists
if [[ ! -f "$RECOVERY_SCRIPT" ]]; then
    echo "ERROR: Recovery script $RECOVERY_SCRIPT does not exist"
    exit 1
fi

echo "Found directories:"
echo "  Parallel: $PARALLEL_DIR"
echo "  Broken: $BROKEN_DIR"
echo "  Recovery script: $RECOVERY_SCRIPT"

# Check aggregate script if needed
if [[ "$AGGREGATE_RESULTS" == true ]]; then
    if [[ ! -f "$AGGREGATE_SCRIPT" ]]; then
        echo "ERROR: Aggregate script $AGGREGATE_SCRIPT does not exist"
        exit 1
    fi
    echo "  Aggregate script: $AGGREGATE_SCRIPT"
fi

echo "Step 2: Running file matching and recovery using Julia script..."
echo "Step 2: Running file matching and recovery using Julia script..." >> "$LOG_FILE"

# Build Julia command
JULIA_CMD="julia $RECOVERY_SCRIPT dummy --match_mode --parallel_dir \"$PARALLEL_DIR\" --broken_dir \"$BROKEN_DIR\" --output_dir \"$OUTPUT_DIR\" --time_tolerance $TIME_TOLERANCE"

# Add multiple results flag if enabled
if [[ "$ALLOW_MULTIPLE" == true ]]; then
    JULIA_CMD="$JULIA_CMD --allow_multiple"
fi

echo "Executing command:"
echo "$JULIA_CMD"

if eval "$JULIA_CMD" >> "$LOG_FILE" 2>&1; then
    echo "✓ SUCCESS: File matching and recovery completed"
    echo "SUCCESS: File matching and recovery completed at $(date)" >> "$LOG_FILE"
else
    echo "✗ FAILED: File matching and recovery failed"
    echo "FAILED: File matching and recovery failed at $(date)" >> "$LOG_FILE"
    echo "Check $LOG_FILE for detailed error information"
    exit 1
fi

# Step 3: Aggregate results if requested
if [[ "$AGGREGATE_RESULTS" == true ]]; then
    echo ""
    echo "Step 3: Aggregating recovered states..."
    echo "Step 3: Aggregating recovered states..." >> "$LOG_FILE"
    
    # Find all recovered .jld2 files from this run only
    RECOVERED_FILES=($(find "$OUTPUT_DIR" -name "*.jld2" -type f))
    
    if [[ ${#RECOVERED_FILES[@]} -eq 0 ]]; then
        echo "WARNING: No .jld2 files found in $OUTPUT_DIR for aggregation"
        echo "WARNING: No .jld2 files found for aggregation at $(date)" >> "$LOG_FILE"
    else
        echo "Found ${#RECOVERED_FILES[@]} files to aggregate"
        echo "Found ${#RECOVERED_FILES[@]} files to aggregate" >> "$LOG_FILE"
        
        # Prepare aggregation command - pass files directly
        AGG_CMD="julia $AGGREGATE_SCRIPT"
        for file in "${RECOVERED_FILES[@]}"; do
            AGG_CMD="$AGG_CMD \"$file\""
        done
        
        echo "Executing aggregation command:"
        echo "$AGG_CMD"
        
        if eval "$AGG_CMD" >> "$LOG_FILE" 2>&1; then
            echo "✓ SUCCESS: Aggregation completed"
            echo "SUCCESS: Aggregation completed at $(date)" >> "$LOG_FILE"
            
            # Move aggregated result to output directory
            if [[ -d "dummy_states_agg" ]]; then
                AGG_FILES=($(find "dummy_states_agg" -name "*.jld2" -type f))
                if [[ ${#AGG_FILES[@]} -gt 0 ]]; then
                    for agg_file in "${AGG_FILES[@]}"; do
                        filename=$(basename "$agg_file")
                        cp "$agg_file" "$OUTPUT_DIR/aggregated_$filename"
                        echo "  Copied aggregated result: $OUTPUT_DIR/aggregated_$filename"
                    done
                fi
            fi
        else
            echo "✗ WARNING: Aggregation failed"
            echo "WARNING: Aggregation failed at $(date)" >> "$LOG_FILE"
        fi
    fi
fi

echo ""
echo "Results:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Processing log: $LOG_FILE"
echo "  Matching report: $OUTPUT_DIR/matching_report.txt"

# Add detailed file count analysis
echo ""
echo "File Count Analysis:"
echo "===================="
PARALLEL_COUNT=$(find "$PARALLEL_DIR" -name "*.jld2" -type f | wc -l)
BROKEN_COUNT=$(find "$BROKEN_DIR" -name "*.jld2" -type f | wc -l)
RECOVERED_COUNT=$(find "$OUTPUT_DIR" -name "*.jld2" -type f | wc -l)

echo "  Parallel files available: $PARALLEL_COUNT"
echo "  Broken files available: $BROKEN_COUNT"
echo "  Recovered files created: $RECOVERED_COUNT"

# Calculate expected vs actual recovery
if [[ "$ALLOW_MULTIPLE" == true ]]; then
    echo "  Expected recoveries: Up to $PARALLEL_COUNT (multiple parallel files can reuse same parameters)"
else
    echo "  Expected recoveries: Up to $BROKEN_COUNT (limited by broken files, 1:1 matching)"
fi

# Check for specific issues
if [[ $RECOVERED_COUNT -lt $PARALLEL_COUNT ]]; then
    echo ""
    echo "Potential Issues Analysis:"
    echo "========================="
    
    # Check if matching report exists
    if [[ -f "$OUTPUT_DIR/matching_report.txt" ]]; then
        MATCHED_PAIRS=$(grep -c "MATCH:" "$OUTPUT_DIR/matching_report.txt" || echo "0")
        UNMATCHED_FILES=$(grep -c "UNMATCHED:" "$OUTPUT_DIR/matching_report.txt" || echo "0")
        
        echo "  Matched pairs found: $MATCHED_PAIRS"
        echo "  Unmatched files: $UNMATCHED_FILES"
        echo "  Processing failures: $((MATCHED_PAIRS - RECOVERED_COUNT))"
        
        if [[ $UNMATCHED_FILES -gt 0 ]]; then
            echo ""
            echo "  Unmatched parallel files (check time tolerance):"
            grep "UNMATCHED:" "$OUTPUT_DIR/matching_report.txt" | head -5
            if [[ $UNMATCHED_FILES -gt 5 ]]; then
                echo "    ... and $((UNMATCHED_FILES - 5)) more (see full report)"
            fi
        fi
        
        # Check for processing failures in log
        PROCESSING_ERRORS=$(grep -c "Failed to process" "$LOG_FILE" || echo "0")
        if [[ $PROCESSING_ERRORS -gt 0 ]]; then
            echo ""
            echo "  Processing errors found: $PROCESSING_ERRORS"
            echo "  Check $LOG_FILE for detailed error messages"
        fi
    else
        echo "  No matching report found - check if matching step completed"
    fi
fi

# Display summary if available
if [[ -f "$OUTPUT_DIR/matching_report.txt" ]]; then
    echo ""
    echo "Quick Summary:"
    echo "=============="
    grep -E "(MATCHED PAIRS|UNMATCHED FILES)" "$OUTPUT_DIR/matching_report.txt" || true
    
    if [[ "$ALLOW_MULTIPLE" == true ]]; then
        echo ""
        echo "Multiple Results Summary:"
        MULTI_COUNT=$(find "$OUTPUT_DIR" -name "*_[0-9]*.jld2" -type f | wc -l)
        echo "  Files with index suffixes: $MULTI_COUNT"
        
        # Show examples of multiple results
        if [[ $MULTI_COUNT -gt 0 ]]; then
            echo "  Examples:"
            find "$OUTPUT_DIR" -name "*_[0-9]*.jld2" -type f | head -3 | while read file; do
                echo "    - $(basename "$file")"
            done
        fi
    fi
    
    if [[ "$AGGREGATE_RESULTS" == true ]]; then
        echo ""
        echo "Aggregation Summary:"
        AGG_COUNT=$(ls -1 "$OUTPUT_DIR"/aggregated_*.jld2 2>/dev/null | wc -l || echo "0")
        echo "  Aggregated files created: $AGG_COUNT"
    fi
    
    echo ""
    echo "For detailed results, check:"
    echo "  - $OUTPUT_DIR/matching_report.txt (file matching details)"
    echo "  - $LOG_FILE (processing log)"
    
    # Suggest solutions
    echo ""
    echo "Troubleshooting Suggestions:"
    echo "==========================="
    if [[ $UNMATCHED_FILES -gt 0 ]]; then
        echo "  - Try increasing --time-tolerance if many files are unmatched"
        echo "  - Check if parallel and broken files have similar creation times"
    fi
    if [[ $PROCESSING_ERRORS -gt 0 ]]; then
        echo "  - Check $LOG_FILE for specific processing errors"
        echo "  - Verify that parallel files contain valid data structures"
    fi
    if [[ $RECOVERED_COUNT -eq 0 ]]; then
        echo "  - Verify directory paths are correct"
        echo "  - Check if files have .jld2 extension"
        echo "  - Run with --help to see all options"
    fi
fi
