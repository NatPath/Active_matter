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
  - By default this script copies source/code, utility scripts, and top-level YAML files.
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

ssh_control_dir=""
ssh_control_path=""

cleanup_ssh_master() {
    if [[ -n "${ssh_control_path}" ]]; then
        ssh -o ControlPath="${ssh_control_path}" -O exit "${remote_user}@${remote_host}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${ssh_control_dir}" && -d "${ssh_control_dir}" ]]; then
        rm -rf "${ssh_control_dir}"
    fi
}

open_ssh_master() {
    ssh_control_dir="$(mktemp -d /tmp/copy_to_cluster_XXXXXX)"
    ssh_control_path="${ssh_control_dir}/control.sock"
    ssh -MNf \
        -o ControlMaster=yes \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "${remote_user}@${remote_host}"
}

ssh_with_master() {
    ssh \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "$@"
}

scp_with_master() {
    scp \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "$@"
}

copy_group() {
    local destination="$1"
    shift
    local -a files=("$@")
    if (( ${#files[@]} == 0 )); then
        return 0
    fi
    scp_with_master "${files[@]}" "${remote_user}@${remote_host}:${destination}/"
}

shopt -s nullglob
trap cleanup_ssh_master EXIT

root_jl_files=(
    "${REPO_ROOT}"/*.jl
)
top_level_yaml_files=(
    "${REPO_ROOT}"/configuration_files/*.yaml
)
cluster_script_files=(
    "${REPO_ROOT}"/cluster_scripts/*.sh
)
utility_script_files=(
    "${REPO_ROOT}"/utility_scripts/*.jl
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
    d_sweep_warmup_files=(
        "${REPO_ROOT}"/configuration_files/two_force_d_sweep/warmup/*.yaml
    )
    d_sweep_production_files=(
        "${REPO_ROOT}"/configuration_files/two_force_d_sweep/production/*.yaml
    )
fi

echo "Copying files to ${remote_user}@${remote_host}:${remote_dir}/"
echo "Include d-sweep configs: ${include_d_sweep_configs}"
open_ssh_master
ssh_with_master "${remote_user}@${remote_host}" \
    "mkdir -p '${remote_dir}' '${remote_dir}/configuration_files' '${remote_dir}/cluster_scripts' '${remote_dir}/utility_scripts'"

copy_group "${remote_dir}" "${root_jl_files[@]}"
copy_group "${remote_dir}/configuration_files" "${top_level_yaml_files[@]}"
copy_group "${remote_dir}/cluster_scripts" "${cluster_script_files[@]}"
copy_group "${remote_dir}/utility_scripts" "${utility_script_files[@]}"

if to_bool "${include_d_sweep_configs}"; then
    ssh_with_master "${remote_user}@${remote_host}" \
        "mkdir -p '${remote_dir}/configuration_files/two_force_d_sweep/warmup' '${remote_dir}/configuration_files/two_force_d_sweep/production'"
    copy_group "${remote_dir}/configuration_files/two_force_d_sweep/warmup" "${d_sweep_warmup_files[@]}"
    copy_group "${remote_dir}/configuration_files/two_force_d_sweep/production" "${d_sweep_production_files[@]}"
fi
# scp match_and_recover_files.sh nativmr@tech-ui02.hep.technion.ac.il:/storage/ph_kafri/nativmr/run_and_tumble/

# scp -rp dummy_states/passive_case/to_recover/* nativmr@tech-ui02.hep.technion.ac.il:/storage/ph_kafri/nativmr/run_and_tumble/
