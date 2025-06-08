#!/bin/bash

# Configuration
PARALLEL_DIR="saved_states_parallel/recover_2D_preserved"
BROKEN_DIR="dummy_states/passive_case/to_recover_preserved"
OUTPUT_DIR="recovered_states_matched"
RECOVERY_SCRIPT="recover_4D_from_parallel.jl"
LOG_FILE="file_matching.log"
TIME_TOLERANCE=60

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize log file
echo "File Matching and Recovery Log - $(date)" > "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"

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

echo "Step 2: Running file matching and recovery using Julia script..."
echo "Step 2: Running file matching and recovery using Julia script..." >> "$LOG_FILE"

# Run the Julia script with match mode
echo "Executing command:"
echo "julia $RECOVERY_SCRIPT dummy --match_mode --parallel_dir \"$PARALLEL_DIR\" --broken_dir \"$BROKEN_DIR\" --output_dir \"$OUTPUT_DIR\" --time_tolerance $TIME_TOLERANCE"

if julia "$RECOVERY_SCRIPT" dummy \
    --match_mode \
    --parallel_dir "$PARALLEL_DIR" \
    --broken_dir "$BROKEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --time_tolerance $TIME_TOLERANCE >> "$LOG_FILE" 2>&1; then
    
    echo "✓ SUCCESS: File matching and recovery completed"
    echo "SUCCESS: File matching and recovery completed at $(date)" >> "$LOG_FILE"
else
    echo "✗ FAILED: File matching and recovery failed"
    echo "FAILED: File matching and recovery failed at $(date)" >> "$LOG_FILE"
    echo "Check $LOG_FILE for detailed error information"
    exit 1
fi

echo ""
echo "Results:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Processing log: $LOG_FILE"
echo "  Matching report: $OUTPUT_DIR/matching_report.txt"

# Display summary if available
if [[ -f "$OUTPUT_DIR/matching_report.txt" ]]; then
    echo ""
    echo "Quick Summary:"
    echo "=============="
    grep -E "(MATCHED PAIRS|UNMATCHED FILES)" "$OUTPUT_DIR/matching_report.txt" || true
    echo ""
    echo "For detailed results, check:"
    echo "  - $OUTPUT_DIR/matching_report.txt (file matching details)"
    echo "  - $LOG_FILE (processing log)"
fi
