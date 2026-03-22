#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_path> [extra run_diffusive_no_activity args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

CONFIG_PATH="$1"
shift || true
EXTRA_ARGS=("$@")

# Match cluster environment setup used by rtp_from_config.sh.
JULIA_SETUP_SCRIPT="${JULIA_SETUP_SCRIPT:-/Local/ph_kafri/julia-1.7.2/bin/setup.sh}"
if [[ -f "${JULIA_SETUP_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${JULIA_SETUP_SCRIPT}"
fi

JULIA_BIN="${JULIA_BIN:-julia}"
if ! command -v "${JULIA_BIN}" >/dev/null 2>&1; then
    echo "Julia executable '${JULIA_BIN}' not found in PATH."
    echo "Tried setup script: ${JULIA_SETUP_SCRIPT}"
    exit 127
fi

julia_procs="${JULIA_NUM_PROCS:-}"
is_aggregate_mode="false"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${arg}" == "--aggregate_state_list" ]]; then
        is_aggregate_mode="true"
        break
    fi
done
if [[ "${is_aggregate_mode}" == "true" ]]; then
    # Aggregation loads many state files and can exceed memory when multiple Julia workers are started.
    # Default to a single process unless explicitly overridden.
    julia_procs="${JULIA_NUM_PROCS_AGGREGATE:-1}"
fi
if [[ -z "${julia_procs}" ]]; then
    for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
        if [[ "${EXTRA_ARGS[$i]}" == "--num_runs" ]]; then
            if (( i + 1 < ${#EXTRA_ARGS[@]} )) && [[ "${EXTRA_ARGS[$((i + 1))]}" =~ ^[0-9]+$ ]]; then
                julia_procs="${EXTRA_ARGS[$((i + 1))]}"
            fi
            break
        fi
    done
fi

julia_cmd=("${JULIA_BIN}")
if [[ -n "${julia_procs}" && "${julia_procs}" =~ ^[0-9]+$ && "${julia_procs}" -gt 1 ]]; then
    julia_cmd=("${JULIA_BIN}" "-p" "${julia_procs}")
fi

cd "${REPO_ROOT}"
"${julia_cmd[@]}" run_diffusive_no_activity.jl --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
