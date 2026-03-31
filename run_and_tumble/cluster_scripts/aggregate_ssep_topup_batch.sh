#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_ssep_topup_batch.sh \
      --config <path> \
      --target_run_info <path> \
      --raw_state_dir <path> \
      --num_replicas <int> \
      --replica_tag_prefix <prefix> \
      --save_tag <aggregate_tag> \
      --archive_dir <path> \
      [--job_info <path>]

Behavior:
  - resolves the current aggregate from the target run's aggregated/ directory
  - resolves one raw top-up state per replica via *_id-${replica_tag_prefix}<idx>.jld2
  - aggregates [current aggregate + new raw states] using run_ssep.jl aggregate mode
  - archives the superseded aggregate file(s) matching the same save tag
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

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

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
    done < <(find "${root_dir}" -maxdepth 1 \( -type f -o -type l \) -name "*_id-${id_tag}.jld2" -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

sidecar_path_for_aggregate() {
    local aggregate_path="$1"
    printf "%s.inputs.txt" "${aggregate_path%.jld2}"
}

config_path=""
target_run_info=""
raw_state_dir=""
num_replicas=""
replica_tag_prefix=""
save_tag=""
archive_dir=""
job_info=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config_path="${2:-}"
            shift 2
            ;;
        --target_run_info)
            target_run_info="${2:-}"
            shift 2
            ;;
        --raw_state_dir)
            raw_state_dir="${2:-}"
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
        --archive_dir)
            archive_dir="${2:-}"
            shift 2
            ;;
        --job_info)
            job_info="${2:-}"
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

if [[ -z "${config_path}" || -z "${target_run_info}" || -z "${raw_state_dir}" || -z "${num_replicas}" || -z "${replica_tag_prefix}" || -z "${save_tag}" || -z "${archive_dir}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
    exit 1
fi
if [[ ! -f "${target_run_info}" ]]; then
    echo "Target run_info not found: ${target_run_info}"
    exit 1
fi
if [[ ! -d "${raw_state_dir}" ]]; then
    echo "Raw state directory not found: ${raw_state_dir}"
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi

aggregated_dir="$(read_run_info_value "${target_run_info}" "aggregated_dir")"
if [[ -z "${aggregated_dir}" ]]; then
    run_root="$(read_run_info_value "${target_run_info}" "run_root")"
    [[ -n "${run_root}" ]] || {
        echo "Could not resolve aggregated_dir or run_root from ${target_run_info}"
        exit 1
    }
    aggregated_dir="${run_root}/aggregated"
fi
mkdir -p "${aggregated_dir}" "${archive_dir}"

current_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "${save_tag}")"
if [[ -z "${current_aggregate}" ]]; then
    echo "Could not find current aggregate with save_tag='${save_tag}' under ${aggregated_dir}"
    exit 1
fi

state_list_file="$(mktemp)"
new_raw_state_list_file="$(mktemp)"
base_input_list_file="$(mktemp)"
aggregate_config="$(mktemp "${TMPDIR:-/tmp}/ssep_topup_aggregate_config.XXXXXX.yaml")"
trap 'rm -f "${state_list_file}" "${new_raw_state_list_file}" "${base_input_list_file}" "${aggregate_config}"' EXIT

current_aggregate_sidecar="$(sidecar_path_for_aggregate "${current_aggregate}")"
if [[ -f "${current_aggregate_sidecar}" ]]; then
    cp "${current_aggregate_sidecar}" "${base_input_list_file}"
else
    : > "${base_input_list_file}"
fi

echo "${current_aggregate}" > "${state_list_file}"
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_tag="${replica_tag_prefix}${replica_idx}"
    matched_state="$(latest_state_for_id_tag "${raw_state_dir}" "${replica_tag}")"
    if [[ -z "${matched_state}" ]]; then
        echo "Missing top-up raw state for replica ${replica_idx} (tag=${replica_tag}) in ${raw_state_dir}"
        exit 1
    fi
    echo "${matched_state}" >> "${state_list_file}"
    echo "${matched_state}" >> "${new_raw_state_list_file}"
done

echo "Aggregating existing aggregate plus ${num_replicas} new raw states"
echo "  current_aggregate=${current_aggregate}"
echo "  raw_state_dir=${raw_state_dir}"
echo "  aggregated_dir=${aggregated_dir}"
echo "  save_tag=${save_tag}"

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

new_aggregate_sidecar="$(sidecar_path_for_aggregate "${new_aggregate}")"
if [[ -s "${base_input_list_file}" ]]; then
    awk 'NF && !seen[$0]++' "${base_input_list_file}" "${new_raw_state_list_file}" > "${new_aggregate_sidecar}"
else
    echo "WARNING: base aggregate input sidecar not found for ${current_aggregate}; not writing lineage metadata for the new aggregate."
    new_aggregate_sidecar=""
fi

while IFS= read -r -d '' old_aggregate; do
    [[ "${old_aggregate}" == "${new_aggregate}" ]] && continue
    mv "${old_aggregate}" "${archive_dir}/"
    old_sidecar="$(sidecar_path_for_aggregate "${old_aggregate}")"
    if [[ -f "${old_sidecar}" ]]; then
        mv "${old_sidecar}" "${archive_dir}/"
    fi
done < <(find "${aggregated_dir}" -maxdepth 1 \( -type f -o -type l \) -name "*_id-${save_tag}.jld2" -print0 2>/dev/null)

if [[ -n "${job_info}" ]]; then
    cat > "${job_info}" <<EOF
timestamp=$(date +%Y%m%d-%H%M%S)
target_run_info=${target_run_info}
raw_state_dir=${raw_state_dir}
current_aggregate=${current_aggregate}
new_aggregate=${new_aggregate}
archive_dir=${archive_dir}
num_replicas=${num_replicas}
save_tag=${save_tag}
EOF
fi

echo "Top-up aggregation completed."
echo "  new_aggregate=${new_aggregate}"
if [[ -n "${new_aggregate_sidecar}" ]]; then
    echo "  aggregate_inputs=${new_aggregate_sidecar}"
else
    echo "  aggregate_inputs=<unavailable>"
fi
echo "  archive_dir=${archive_dir}"
