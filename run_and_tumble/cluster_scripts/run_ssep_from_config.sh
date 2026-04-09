#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_path> [extra run_ssep args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_ssep.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_ssep.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

CONFIG_PATH="$1"
shift || true
EXTRA_ARGS=("$@")

JULIA_SETUP_SCRIPT="${JULIA_SETUP_SCRIPT:-${CLUSTER_JULIA_SETUP_SCRIPT:-}}"
if [[ -n "${JULIA_SETUP_SCRIPT}" && -f "${JULIA_SETUP_SCRIPT}" ]]; then
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
"${julia_cmd[@]}" run_ssep.jl --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
