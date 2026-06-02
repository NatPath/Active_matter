#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LIVE_SCRIPT="${SCRIPT_DIR}/run_coupled_sde_mobile_live_dashboard.jl"

if [[ ! -f "${LIVE_SCRIPT}" ]]; then
    echo "Missing live dashboard script: ${LIVE_SCRIPT}"
    exit 1
fi

stamp="$(date +%Y%m%d-%H%M%S)"
default_out_dir="${REPO_ROOT}/analysis_outputs/coupled_sde_active_objects/mobile_live/L10_rho200_compactbump_width1_f0_5_dt4p5em4_hardmin2_${stamp}"

JULIA_BIN="${JULIA_BIN:-julia}"
MU_BATH_VALUE="${MU_BATH:-1.0}"
MU_OBJ_VALUE="${MU_OBJ:-$(awk -v mu="${MU_BATH_VALUE}" 'BEGIN { printf "%.12g", mu / 1000.0 }')}"

exec "${JULIA_BIN}" --project="${REPO_ROOT}" --startup-file=no "${LIVE_SCRIPT}" \
    --L "${L:-10}" \
    --rho0 "${RHO0:-200}" \
    --D0 "${D0:-1.0}" \
    --dt "${DT:-0.00045}" \
    --mu_bath "${MU_BATH_VALUE}" \
    --mu_obj "${MU_OBJ_VALUE}" \
    --f0 "${F0:-5.0}" \
    --sigma_f "${SIGMA_F:-0.5}" \
    --profile_type "${PROFILE_TYPE:-compact_bump}" \
    --hard_min_separation "${HARD_MIN_SEPARATION:-2.0}" \
    --separation "${INITIAL_SEPARATION:-2}" \
    --n_steps "${N_STEPS:-5000000}" \
    --warmup_steps "${WARMUP_STEPS:-0}" \
    --sample_interval "${SAMPLE_INTERVAL:-1}" \
    --plot_interval_steps "${PLOT_INTERVAL_STEPS:-20000}" \
    --min_plot_interval_seconds "${MIN_PLOT_INTERVAL_SECONDS:-0.75}" \
    --progress_interval_steps "${PROGRESS_INTERVAL_STEPS:-20000}" \
    --n_bins "${N_BINS:-256}" \
    --trace_points "${TRACE_POINTS:-900}" \
    --out_dir "${OUT_DIR:-${default_out_dir}}" \
    "$@"
