#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_simulation_meta_local.sh" ]]; then
    LAUNCHER="${SCRIPT_DIR}/run_simulation_meta_local.sh"
elif [[ -f "${SCRIPT_DIR}/../cluster_scripts/run_simulation_meta_local.sh" ]]; then
    LAUNCHER="${SCRIPT_DIR}/../cluster_scripts/run_simulation_meta_local.sh"
else
    echo "Could not find run_simulation_meta_local.sh"
    exit 1
fi

bash "${LAUNCHER}" \
    --simulation two_force_d \
    --run_mode warmup_production \
    --L 128 \
    --rho 10 \
    --warmup_n_sweeps 100000 \
    --n_sweeps 10000000
