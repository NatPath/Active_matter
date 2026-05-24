#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configuration_files/fluctuating_force_sde}"
mkdir -p "${CONFIG_ROOT}"

DIMS="${DIMS:-1}"
L="${L:-60.0}"
N="${N:-8000}"
D_BATH="${D_BATH:-1.0}"
DT="${DT:-0.05}"
MU_BATH="${MU_BATH:-5.0}"
F0="${F0:-1.0}"
SIGMA_F="${SIGMA_F:-0.5}"
PROFILE_TYPE="${PROFILE_TYPE:-gaussian}"
MOBILE_FORCES="${MOBILE_FORCES:-false}"
FORCE_MOBILITY="${FORCE_MOBILITY:-0.0}"
FORCE_DIFFUSIVITY="${FORCE_DIFFUSIVITY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-10000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1}"
N_BINS="${N_BINS:-80}"
N_RADIAL_BINS="${N_RADIAL_BINS:-20}"
RADIAL_MIN="${RADIAL_MIN:-0.8}"
RADIAL_MAX="${RADIAL_MAX:-}"
EDGE_BINS_FOR_OFFSET="${EDGE_BINS_FOR_OFFSET:-5}"
VARIANCE_FLOOR="${VARIANCE_FLOOR:-1e-6}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-1000}"
MAX_HISTORY_RECORDS="${MAX_HISTORY_RECORDS:-20000}"
SAVE_FORCE_HISTORY="${SAVE_FORCE_HISTORY:-true}"
SAVE_DIR="${SAVE_DIR:-saved_states/fluctuating_force_sde}"
CONFIG_NAME="${CONFIG_NAME:-}"

if [[ "${DIMS}" != "1" && "${DIMS}" != "2" ]]; then
    echo "DIMS must be 1 or 2. Got '${DIMS}'."
    exit 1
fi

if [[ -z "${CONFIG_NAME}" ]]; then
    CONFIG_NAME="${DIMS}d_fixed_origin"
fi

if [[ -z "${FORCE_CENTERS_YAML:-}" ]]; then
    if [[ "${DIMS}" == "1" ]]; then
        FORCE_CENTERS_YAML="[0.0]"
    else
        FORCE_CENTERS_YAML="[[0.0, 0.0]]"
    fi
fi

sanitize_token() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/[.]/p/g; s/-+/-/g; s/^-//; s/-$//'
}

cfg="${CONFIG_ROOT}/$(sanitize_token "${CONFIG_NAME}").yaml"
{
    echo "description: \"fluctuating_force_sde_${CONFIG_NAME}\""
    echo ""
    echo "dims: ${DIMS}"
    echo "L: ${L}"
    echo "N: ${N}"
    echo ""
    echo "D_bath: ${D_BATH}"
    echo "dt: ${DT}"
    echo "mu_bath: ${MU_BATH}"
    echo "f0: ${F0}"
    echo "sigma_f: ${SIGMA_F}"
    echo "profile_type: \"${PROFILE_TYPE}\""
    echo ""
    echo "force_centers: ${FORCE_CENTERS_YAML}"
    echo "mobile_forces: ${MOBILE_FORCES}"
    echo "force_mobility: ${FORCE_MOBILITY}"
    echo "force_diffusivity: ${FORCE_DIFFUSIVITY}"
    echo ""
    echo "warmup_steps: ${WARMUP_STEPS}"
    echo "n_steps: ${PRODUCTION_STEPS}"
    echo "sample_interval: ${SAMPLE_INTERVAL}"
    echo ""
    echo "n_bins: ${N_BINS}"
    echo "n_radial_bins: ${N_RADIAL_BINS}"
    echo "radial_min: ${RADIAL_MIN}"
    if [[ -n "${RADIAL_MAX}" ]]; then
        echo "radial_max: ${RADIAL_MAX}"
    fi
    echo "edge_bins_for_offset: ${EDGE_BINS_FOR_OFFSET}"
    echo "variance_floor: ${VARIANCE_FLOOR}"
    echo ""
    echo "history_interval: ${HISTORY_INTERVAL}"
    echo "max_history_records: ${MAX_HISTORY_RECORDS}"
    echo "save_force_history: ${SAVE_FORCE_HISTORY}"
    echo ""
    echo "seed: 0"
    echo "performance_mode: true"
    echo "cluster_mode: false"
    echo "save_dir: \"${SAVE_DIR}\""
} > "${cfg}"

echo "Generated ${cfg}"
