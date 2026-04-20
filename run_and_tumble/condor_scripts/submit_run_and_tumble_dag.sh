#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_runs>" >&2
  exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]] || [[ "$1" -lt 1 ]]; then
  echo "num_runs must be a positive integer" >&2
  exit 1
fi

NUM_RUNS="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DAG_FILE="$SCRIPT_DIR/run_and_tumble.dag"

mkdir -p "$SCRIPT_DIR/logs"

{
  echo "# Auto-generated DAG for run_and_tumble.jl"
  echo "# Created: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "# num_runs: $NUM_RUNS"
  echo
} > "$DAG_FILE"

run_nodes=()
for i in $(seq 0 $((NUM_RUNS - 1))); do
  {
    echo "JOB RUN${i} run_and_tumble.sub"
    echo "VARS RUN${i} RUN_ID=\"${i}\""
    echo
  } >> "$DAG_FILE"
  run_nodes+=("RUN${i}")
done

{
  echo "JOB AGGREGATE aggregate_results.sub"
  echo
  echo "PARENT ${run_nodes[*]} CHILD AGGREGATE"
} >> "$DAG_FILE"

echo "Generated DAG at $DAG_FILE"
echo "Submitting DAG with $NUM_RUNS run nodes..."
cd "$SCRIPT_DIR"
condor_submit_dag "$DAG_FILE"
