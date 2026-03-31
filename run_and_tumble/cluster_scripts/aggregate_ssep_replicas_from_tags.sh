#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_ssep_replicas_from_tags.sh \
      --config <path> \
      --state_dir <path> \
      --aggregated_dir <path> \
      --num_replicas <int> \
      --replica_tag_prefix <prefix> \
      --save_tag <tag>

Behavior:
  - resolves one saved SSEP state per replica using expected IDs:
      *_id-${replica_tag_prefix}<replica_index>.jld2
  - writes a temporary state list file
  - runs run_ssep.jl in aggregate-only mode
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_ssep_from_config.sh" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/run_ssep_from_config.sh"
elif [[ -f "${SCRIPT_DIR}/../cluster_scripts/run_ssep_from_config.sh" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/../cluster_scripts/run_ssep_from_config.sh"
else
    echo "Could not find run_ssep_from_config.sh"
    exit 1
fi

config_path=""
state_dir=""
aggregated_dir=""
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
        --aggregated_dir)
            aggregated_dir="${2:-}"
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

if [[ -z "${aggregated_dir}" ]]; then
    aggregated_dir="$(cd "$(dirname "${state_dir}")" && pwd)/aggregated"
fi
mkdir -p "${aggregated_dir}"

latest_state_for_id_tag() {
    local root_dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime

    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${root_dir}" -type f -name "*_id-${id_tag}.jld2" -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

sidecar_path_for_aggregate() {
    local aggregate_path="$1"
    printf "%s.inputs.txt" "${aggregate_path%.jld2}"
}

state_list_file="$(mktemp)"
aggregate_config="$(mktemp "${TMPDIR:-/tmp}/ssep_aggregate_config.XXXXXX.yaml")"
trap 'rm -f "${state_list_file}" "${aggregate_config}"' EXIT

for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_tag="${replica_tag_prefix}${replica_idx}"
    matched_state="$(latest_state_for_id_tag "${state_dir}" "${replica_tag}")"
    if [[ -z "${matched_state}" ]]; then
        echo "Missing replica state for replica ${replica_idx} (tag=${replica_tag}) in ${state_dir}"
        exit 1
    fi
    echo "${matched_state}" >> "${state_list_file}"
done

echo "Aggregating ${num_replicas} replica states from ${state_dir}"
echo "Aggregate output dir: ${aggregated_dir}"
echo "Save tag: ${save_tag}"
awk -v save_dir_line="save_dir: \"${aggregated_dir}\"" '
BEGIN {seen_save=0}
{
    if ($0 ~ /^save_dir:[[:space:]]*/) {
        print save_dir_line
        seen_save=1
        next
    }
    print
}
END {
    if (!seen_save) print save_dir_line
}' "${config_path}" > "${aggregate_config}"

bash "${RUNNER_SCRIPT}" "${aggregate_config}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"

new_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "${save_tag}")"
if [[ -z "${new_aggregate}" ]]; then
    echo "Aggregation finished but no output aggregate with save_tag='${save_tag}' was found under ${aggregated_dir}"
    exit 1
fi

aggregate_sidecar="$(sidecar_path_for_aggregate "${new_aggregate}")"
awk 'NF && !seen[$0]++' "${state_list_file}" > "${aggregate_sidecar}"

echo "Aggregate completed."
echo "  new_aggregate=${new_aggregate}"
echo "  aggregate_inputs=${aggregate_sidecar}"
