#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_replicas_from_tags.sh \
      --config <path> \
      --state_dir <path> \
      --num_replicas <int> \
      --replica_tag_prefix <prefix> \
      --save_tag <tag>

Behavior:
  - resolves one saved state per replica using expected IDs:
      *_id-${replica_tag_prefix}<replica_index>.jld2
  - writes a temporary state list file
  - runs run_diffusive_no_activity.jl in aggregate-only mode
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
elif [[ -f "${SCRIPT_DIR}/../cluster_scripts/run_diffusive_no_activity_from_config.sh" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/../cluster_scripts/run_diffusive_no_activity_from_config.sh"
else
    echo "Could not find run_diffusive_no_activity_from_config.sh"
    exit 1
fi

config_path=""
state_dir=""
num_replicas=""
replica_tag_prefix=""
save_tag=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config_path="${2:-}"
            shift 2
            ;;
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --replica_tag_prefix)
            replica_tag_prefix="${2:-}"
            shift 2
            ;;
        --save_tag)
            save_tag="${2:-}"
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

if [[ -z "${config_path}" || -z "${state_dir}" || -z "${num_replicas}" || -z "${replica_tag_prefix}" || -z "${save_tag}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
    exit 1
fi
if [[ ! -d "${state_dir}" ]]; then
    echo "State directory not found: ${state_dir}"
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi

state_list_file="$(mktemp)"
trap 'rm -f "${state_list_file}"' EXIT

for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_tag="${replica_tag_prefix}${replica_idx}"
    matched_state="$(ls -1t "${state_dir}"/*"_id-${replica_tag}.jld2" 2>/dev/null | head -n 1 || true)"
    if [[ -z "${matched_state}" ]]; then
        echo "Missing replica state for replica ${replica_idx} (tag=${replica_tag}) in ${state_dir}"
        exit 1
    fi
    echo "${matched_state}" >> "${state_list_file}"
done

echo "Aggregating ${num_replicas} replica states from ${state_dir}"
echo "Save tag: ${save_tag}"
bash "${RUNNER_SCRIPT}" "${config_path}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"

