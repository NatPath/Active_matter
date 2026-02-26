#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

if [[ -f "${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh" ]]; then
    GENERATE_SCRIPT="${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh" ]]; then
    GENERATE_SCRIPT="${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh"
else
    echo "Could not find generate_two_force_d_sweep_configs.sh"
    exit 1
fi

"${GENERATE_SCRIPT}" >/dev/null

D_VALUE="${1:-2}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
JULIA_BIN="${JULIA_BIN:-julia}"

CONFIG_REL="configuration_files/two_force_d_sweep/warmup/d_${D_VALUE}.yaml"
CONFIG_PATH="${REPO_ROOT}/${CONFIG_REL}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}"
    exit 1
fi

cd "${REPO_ROOT}"
echo "Estimating warmup runtime locally for d=${D_VALUE} using ${CONFIG_REL} (sample_size=${SAMPLE_SIZE})"
"${JULIA_BIN}" run_diffusive_no_activity.jl --config "${CONFIG_REL}" --estimate_only --estimate_sample_size "${SAMPLE_SIZE}"
