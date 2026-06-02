#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_path> [extra run_coupled_sde_active_objects args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_coupled_sde_active_objects.jl" ]]; then
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
if [[ -z "${JULIA_SETUP_SCRIPT}" && -f "/Local/ph_kafri/julia-1.7.2/bin/setup.sh" ]]; then
    JULIA_SETUP_SCRIPT="/Local/ph_kafri/julia-1.7.2/bin/setup.sh"
fi
if [[ -n "${JULIA_SETUP_SCRIPT}" && -f "${JULIA_SETUP_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${JULIA_SETUP_SCRIPT}"
fi

JULIA_BIN="${JULIA_BIN:-julia}"
if ! command -v "${JULIA_BIN}" >/dev/null 2>&1; then
    echo "Julia executable '${JULIA_BIN}' not found in PATH." >&2
    echo "Tried setup script: ${JULIA_SETUP_SCRIPT}" >&2
    echo "Set JULIA_BIN or CLUSTER_JULIA_SETUP_SCRIPT in ${cluster_env_path}." >&2
    exit 127
fi

cd "${REPO_ROOT}"
"${JULIA_BIN}" --startup-file=no run_coupled_sde_active_objects.jl --config "${CONFIG_PATH}" "${EXTRA_ARGS[@]}"
