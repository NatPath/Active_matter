#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_diffusive_origin_batch.sh \
      --config <production_config> \
      --raw_state_dir <dir> \
      --num_replicas <int> \
      --raw_tag_prefix <prefix> \
      --aggregate_root <dir> \
      --batch_tag <tag> \
      --cumulative_tag <tag>

Options:
  --archive_subdir <name>   archive directory under aggregate_root (default: archive)
  --archive_stamp <token>   archive stamp for previous cumulative aggregate (default: batch_tag)

Behavior:
  - resolves one raw production state per replica from raw_state_dir using:
      *_id-${raw_tag_prefix}<replica_index>.jld2
  - aggregates the current batch into:
      <aggregate_root>/batches/
  - accumulates the batch aggregate onto the latest cumulative aggregate in:
      <aggregate_root>/current/
    and archives the previous cumulative aggregate under:
      <aggregate_root>/<archive_subdir>/<archive_stamp>/
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing helper script: ${RUNNER_SCRIPT}"
    exit 1
fi

config_path=""
raw_state_dir=""
num_replicas=""
raw_tag_prefix=""
aggregate_root=""
batch_tag=""
cumulative_tag=""
archive_subdir="archive"
archive_stamp=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config_path="${2:-}"
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
        --raw_tag_prefix)
            raw_tag_prefix="${2:-}"
            shift 2
            ;;
        --aggregate_root)
            aggregate_root="${2:-}"
            shift 2
            ;;
        --batch_tag)
            batch_tag="${2:-}"
            shift 2
            ;;
        --cumulative_tag)
            cumulative_tag="${2:-}"
            shift 2
            ;;
        --archive_subdir)
            archive_subdir="${2:-}"
            shift 2
            ;;
        --archive_stamp)
            archive_stamp="${2:-}"
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

if [[ -z "${config_path}" || -z "${raw_state_dir}" || -z "${num_replicas}" ||
      -z "${raw_tag_prefix}" || -z "${aggregate_root}" || -z "${batch_tag}" ||
      -z "${cumulative_tag}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
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
for token_name in batch_tag cumulative_tag archive_subdir; do
    token_value="${!token_name}"
    if [[ -z "${token_value}" || ! "${token_value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "--${token_name} must match [A-Za-z0-9._-]+. Got '${token_value}'."
        exit 1
    fi
done
if [[ -z "${archive_stamp}" ]]; then
    archive_stamp="${batch_tag}"
fi
if ! [[ "${archive_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_stamp must match [A-Za-z0-9._-]+. Got '${archive_stamp}'."
    exit 1
fi

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
    done < <(find "${root_dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

latest_cumulative_state() {
    local current_dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime
    [[ -d "${current_dir}" ]] || { printf ""; return 0; }

    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${current_dir}" -maxdepth 1 -type f -name "*_id-aggregated_${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

rewrite_save_dir_config() {
    local source_config="$1"
    local target_config="$2"
    local save_dir="$3"
    local save_dir_line="save_dir: \"${save_dir}\""

    awk -v save_dir_line="${save_dir_line}" '
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
    }' "${source_config}" > "${target_config}"
}

aggregate_state_list() {
    local cfg="$1"
    local list_file="$2"
    local save_tag="$3"
    local output_dir="$4"
    local expected_id="aggregated_${save_tag}"
    local output_state

    bash "${RUNNER_SCRIPT}" "${cfg}" --aggregate_state_list "${list_file}" --save_tag "${save_tag}" >&2
    output_state="$(latest_state_for_id_tag "${output_dir}" "${expected_id}")"
    if [[ -z "${output_state}" ]]; then
        echo "Aggregation finished but no output with id tag '${expected_id}' was found in ${output_dir}" >&2
        exit 1
    fi
    printf "%s" "${output_state}"
}

batch_dir="${aggregate_root}/batches"
current_dir="${aggregate_root}/current"
archive_dir="${aggregate_root}/${archive_subdir}/${archive_stamp}"
work_dir="${aggregate_root}/work/${batch_tag}"
lineage_file="${aggregate_root}/lineage.csv"
mkdir -p "${batch_dir}" "${current_dir}" "${archive_dir}" "${work_dir}"

batch_state_list="${work_dir}/batch_states.txt"
: > "${batch_state_list}"
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    raw_tag="${raw_tag_prefix}${replica_idx}"
    raw_state="$(latest_state_for_id_tag "${raw_state_dir}" "${raw_tag}")"
    if [[ -z "${raw_state}" ]]; then
        echo "Missing raw state for replica ${replica_idx}: tag=${raw_tag} dir=${raw_state_dir}"
        exit 1
    fi
    echo "${raw_state}" >> "${batch_state_list}"
done

batch_config="${work_dir}/batch_aggregate_config.yaml"
cumulative_config="${work_dir}/cumulative_aggregate_config.yaml"
rewrite_save_dir_config "${config_path}" "${batch_config}" "${batch_dir}"
rewrite_save_dir_config "${config_path}" "${cumulative_config}" "${current_dir}"

echo "Aggregating current production batch:"
echo "  raw_state_dir=${raw_state_dir}"
echo "  num_replicas=${num_replicas}"
echo "  batch_dir=${batch_dir}"
batch_aggregate_state="$(aggregate_state_list "${batch_config}" "${batch_state_list}" "${batch_tag}" "${batch_dir}")"
echo "Batch aggregate: ${batch_aggregate_state}"

previous_cumulative_state="$(latest_cumulative_state "${current_dir}" "${cumulative_tag}")"
archived_previous_state=""
restore_previous="false"
if [[ -n "${previous_cumulative_state}" ]]; then
    archived_previous_state="${archive_dir}/$(basename "${previous_cumulative_state}")"
    mv -f "${previous_cumulative_state}" "${archived_previous_state}"
    restore_previous="true"
    echo "Archived previous cumulative aggregate: ${archived_previous_state}"
fi

restore_on_failure() {
    if [[ "${restore_previous}" == "true" && -n "${archived_previous_state}" && -f "${archived_previous_state}" ]]; then
        mkdir -p "${current_dir}"
        mv -f "${archived_previous_state}" "${current_dir}/$(basename "${archived_previous_state}")"
        echo "Restored previous cumulative aggregate after failure."
    fi
}
trap restore_on_failure ERR

cumulative_state_list="${work_dir}/cumulative_states.txt"
: > "${cumulative_state_list}"
if [[ -n "${archived_previous_state}" ]]; then
    echo "${archived_previous_state}" >> "${cumulative_state_list}"
fi
echo "${batch_aggregate_state}" >> "${cumulative_state_list}"

echo "Accumulating batch aggregate into cumulative aggregate:"
echo "  current_dir=${current_dir}"
cumulative_state="$(aggregate_state_list "${cumulative_config}" "${cumulative_state_list}" "${cumulative_tag}" "${current_dir}")"
restore_previous="false"
trap - ERR
echo "Cumulative aggregate: ${cumulative_state}"

if [[ ! -f "${lineage_file}" ]]; then
    echo "timestamp,batch_tag,cumulative_tag,raw_state_dir,batch_aggregate_state,previous_cumulative_state,archived_previous_state,cumulative_state,num_replicas" > "${lineage_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$(date +%Y%m%d-%H%M%S)" \
    "${batch_tag}" \
    "${cumulative_tag}" \
    "${raw_state_dir}" \
    "${batch_aggregate_state}" \
    "${previous_cumulative_state}" \
    "${archived_previous_state}" \
    "${cumulative_state}" \
    "${num_replicas}" >> "${lineage_file}"

echo "Aggregation and cumulative accumulation completed."
echo "Lineage: ${lineage_file}"
