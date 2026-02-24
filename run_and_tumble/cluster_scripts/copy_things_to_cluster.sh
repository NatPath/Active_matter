#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash copy_things_to_cluster.sh [options]

Options:
  --with-d-sweep-configs   Also generate/copy configuration_files/two_force_d_sweep/*/d_*.yaml
  --remote-user <user>     Remote SSH user (default: nativmr)
  --remote-host <host>     Remote SSH host (default: tech-ui02.hep.technion.ac.il)
  --remote-dir <path>      Remote target directory (default: /storage/ph_kafri/nativmr/run_and_tumble)
  -h, --help               Show this help

Notes:
  - By default this script copies source/code + top-level YAML files only.
  - d-sweep YAML files are excluded by default to avoid pushing large generated sets every time.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

remote_user="${REMOTE_USER:-nativmr}"
remote_host="${REMOTE_HOST:-tech-ui02.hep.technion.ac.il}"
remote_dir="${REMOTE_DIR:-/storage/ph_kafri/nativmr/run_and_tumble}"
include_d_sweep_configs="${INCLUDE_D_SWEEP_CONFIGS:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-d-sweep-configs)
            include_d_sweep_configs="true"
            shift 1
            ;;
        --remote-user)
            remote_user="${2:-}"
            shift 2
            ;;
        --remote-host)
            remote_host="${2:-}"
            shift 2
            ;;
        --remote-dir)
            remote_dir="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

to_bool() {
    local raw
    raw="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    [[ "${raw}" == "1" || "${raw}" == "true" || "${raw}" == "yes" || "${raw}" == "on" ]]
}

copy_items=(
    "${REPO_ROOT}"/*.jl
    "${REPO_ROOT}"/configuration_files/*.yaml
    "${REPO_ROOT}"/cluster_scripts/*.sh
)

if to_bool "${include_d_sweep_configs}"; then
    if [[ -f "${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh" ]]; then
        GENERATE_SCRIPT="${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh"
    elif [[ -f "${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh" ]]; then
        GENERATE_SCRIPT="${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh"
    else
        echo "Could not find generate_two_force_d_sweep_configs.sh"
        exit 1
    fi
    "${GENERATE_SCRIPT}" >/dev/null
    copy_items+=(
        "${REPO_ROOT}"/configuration_files/two_force_d_sweep/warmup/*.yaml
        "${REPO_ROOT}"/configuration_files/two_force_d_sweep/production/*.yaml
    )
fi

target="${remote_user}@${remote_host}:${remote_dir}/"
echo "Copying files to ${target}"
echo "Include d-sweep configs: ${include_d_sweep_configs}"
scp "${copy_items[@]}" "${target}"
# scp match_and_recover_files.sh nativmr@tech-ui02.hep.technion.ac.il:/storage/ph_kafri/nativmr/run_and_tumble/

# scp -rp dummy_states/passive_case/to_recover/* nativmr@tech-ui02.hep.technion.ac.il:/storage/ph_kafri/nativmr/run_and_tumble/
