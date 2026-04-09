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

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

CONFIG_PATH="$1"
shift || true
EXTRA_ARGS=("$@")

config_performance_mode="$(awk '
    /^[[:space:]]*#/ {next}
    /^[[:space:]]*performance_mode:[[:space:]]*/ {
        val=$0
        sub(/^[[:space:]]*performance_mode:[[:space:]]*/, "", val)
        gsub(/[[:space:]]+$/, "", val)
        gsub(/^"/, "", val); gsub(/"$/, "", val)
        print tolower(val)
        exit
    }
    /^[[:space:]]*cluster_mode:[[:space:]]*/ {
        val=$0
        sub(/^[[:space:]]*cluster_mode:[[:space:]]*/, "", val)
        gsub(/[[:space:]]+$/, "", val)
        gsub(/^"/, "", val); gsub(/"$/, "", val)
        print tolower(val)
        exit
    }' "${CONFIG_PATH}" || true)"
if [[ "${config_performance_mode}" =~ ^(true|1|yes|on)$ ]]; then
    export RUN_AND_TUMBLE_HEADLESS=1
fi

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
