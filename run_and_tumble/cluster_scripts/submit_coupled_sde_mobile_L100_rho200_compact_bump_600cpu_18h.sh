#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_coupled_sde_mobile_600cpu_12h.sh"

if [[ ! -f "${SUBMIT_SCRIPT}" ]]; then
    echo "Missing helper script: ${SUBMIT_SCRIPT}"
    exit 1
fi

export L="${L:-100}"
export RHO0="${RHO0:-200}"
export D0="${D0:-1.0}"
export DT="${DT:-0.005}"
export MU_BATH="${MU_BATH:-1.0}"
export MU_OBJ="${MU_OBJ:-$(awk -v mu="${MU_BATH}" 'BEGIN { printf "%.12g", mu / 1000.0 }')}"
export F0="${F0:-1.5}"
export SIGMA_F="${SIGMA_F:-0.5}"
export PROFILE_TYPE="${PROFILE_TYPE:-compact_bump}"
export HARD_MIN_SEPARATION="${HARD_MIN_SEPARATION:-2.0}"
if [[ -z "${HARD_MIN_SEPARATION}" ]]; then
    export HARD_MIN_SEPARATION_SIGMA="${HARD_MIN_SEPARATION_SIGMA:-4.0}"
else
export HARD_MIN_SEPARATION_SIGMA="${HARD_MIN_SEPARATION_SIGMA:-}"
fi
export SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-20}"
export WARMUP_FRACTION="${WARMUP_FRACTION:-0.0}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export TARGET_UTILIZATION="${TARGET_UTILIZATION:-0.88}"
export PARTICLE_UPDATES_PER_SECOND="${PARTICLE_UPDATES_PER_SECOND:-1170000}"
export N_BINS="${N_BINS:-256}"
export HISTORY_INTERVAL="${HISTORY_INTERVAL:-20000}"
export MAX_HISTORY_RECORDS="${MAX_HISTORY_RECORDS:-2000}"
export SAVE_RAW_HISTORY="${SAVE_RAW_HISTORY:-false}"
export CHECKPOINTS_PER_REPLICA="${CHECKPOINTS_PER_REPLICA:-8}"
export INITIAL_SEPARATION="${INITIAL_SEPARATION:-2}"
export RANDOM_INITIAL_OBJECTS="${RANDOM_INITIAL_OBJECTS:-false}"
export INITIAL_MIN_SEPARATION="${INITIAL_MIN_SEPARATION:-2}"
export INITIAL_MAX_SEPARATION="${INITIAL_MAX_SEPARATION:-2}"
export REQUEST_CPUS="${REQUEST_CPUS:-1}"
export REQUEST_MEMORY="${REQUEST_MEMORY:-6 GB}"
export AGGREGATE_REQUEST_CPUS="${AGGREGATE_REQUEST_CPUS:-1}"

target_hours="${TARGET_HOURS:-18}"
total_cpus="${TOTAL_CPUS:-600}"
if [[ -n "${RUN_ID:-}" ]]; then
    run_id="${RUN_ID}"
else
    run_id="coupled_sde_mobile_L100_rho200_compactbump_width1p0_muobj1e-3_initsep2_hardmin2_bins256_600cpu_18h_$(date +%Y%m%d-%H%M%S)"
fi

exec bash "${SUBMIT_SCRIPT}" \
    --run_id "${run_id}" \
    --target_hours "${target_hours}" \
    --total_cpus "${total_cpus}" \
    "$@"
