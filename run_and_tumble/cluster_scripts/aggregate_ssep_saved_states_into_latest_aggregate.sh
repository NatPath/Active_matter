#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_ssep_saved_states_into_latest_aggregate.sh \
      --run_id <id> \
      [--mode auto|production]

Behavior:
  - resolves an existing SSEP production run from --run_id
  - finds the successfully saved production replica states for that run from manifest/save_tag data
  - if a latest aggregate with lineage metadata exists, adds only the raw states not yet represented there
  - if no aggregate exists yet, aggregates all currently saved raw replica states
  - if a latest aggregate exists but predates lineage metadata, rebuilds it from the
    currently saved raw production states instead of incrementing blindly
  - archives the superseded aggregate and its sidecar after the new aggregate is written
  - for cluster use, submit this helper via:
      cluster_scripts/submit_ssep_saved_states_into_latest_aggregate.sh

Safety:
  - does not increment an aggregate without lineage metadata; it rebuilds from current raw states
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_ssep.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_ssep.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

RUNNER_SCRIPT="${SCRIPT_DIR}/run_ssep_from_config.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/ssep/single_center_bond/run_registry.csv"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing runner script: ${RUNNER_SCRIPT}"
    exit 1
fi

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

find_run_info_by_run_id() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local candidate=""

    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/ssep/single_center_bond/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    if [[ -f "${REGISTRY_FILE}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${REGISTRY_FILE}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _warmup _numrep _cpus _mem reg_run_root _submit_dir _log_dir _state_dir _config_path _aggregate_run_id <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

resolve_target_production_run_info() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local resolved_info

    resolved_info="$(find_run_info_by_run_id "${lookup_run_id}" "${mode_hint}" || true)"
    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve SSEP production run_info for run_id='${lookup_run_id}' (mode=${mode_hint})." >&2
        return 1
    fi
    if [[ "$(read_run_info_value "${resolved_info}" "mode")" != "production" ]]; then
        echo "run_id='${lookup_run_id}' does not resolve to an SSEP production run." >&2
        return 1
    fi
    echo "${resolved_info}"
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

run_id=""
mode="auto"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
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

if [[ -z "${run_id}" ]]; then
    echo "--run_id is required."
    usage
    exit 1
fi

target_run_info="$(resolve_target_production_run_info "${run_id}" "${mode}")"
target_run_root="$(read_run_info_value "${target_run_info}" "run_root")"
state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
aggregated_dir="$(read_run_info_value "${target_run_info}" "aggregated_dir")"
config_path="$(read_run_info_value "${target_run_info}" "runtime_config")"
manifest_path="$(read_run_info_value "${target_run_info}" "manifest")"
aggregate_run_id="$(read_run_info_value "${target_run_info}" "aggregate_run_id")"

if [[ -z "${config_path}" ]]; then
    config_path="$(read_run_info_value "${target_run_info}" "config_path")"
fi
if [[ -z "${aggregated_dir}" ]]; then
    aggregated_dir="${target_run_root}/aggregated"
fi
if [[ -z "${aggregate_run_id}" ]]; then
    aggregate_run_id="aggregated_${run_id}"
fi

[[ -n "${target_run_root}" && -d "${target_run_root}" ]] || {
    echo "Could not resolve target run_root from ${target_run_info}"
    exit 1
}
[[ -n "${state_dir}" && -d "${state_dir}" ]] || {
    echo "Could not resolve source state_dir from ${target_run_info}"
    exit 1
}
[[ -n "${aggregated_dir}" && -d "${aggregated_dir}" ]] || {
    echo "Could not resolve aggregated_dir from ${target_run_info}"
    exit 1
}
[[ -n "${config_path}" && -f "${config_path}" ]] || {
    echo "Could not resolve config_path/runtime_config from ${target_run_info}"
    exit 1
}
[[ -n "${manifest_path}" && -f "${manifest_path}" ]] || {
    echo "Could not resolve manifest from ${target_run_info}"
    exit 1
}

job_token="manual_saved_states_$(date +%Y%m%d-%H%M%S)"
job_root="${target_run_root}/manual_aggregate_jobs/${job_token}"
archive_dir="${aggregated_dir}/archive/${job_token}"
mkdir -p "${job_root}" "${archive_dir}"

state_list_file="${job_root}/aggregate_state_list.txt"
raw_state_list_file="${job_root}/new_raw_state_list.txt"
base_input_list_file="${job_root}/base_input_state_list.txt"
aggregate_config="${job_root}/aggregate_config.yaml"
job_info="${job_root}/run_info.txt"

current_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "${aggregate_run_id}")"
current_aggregate_sidecar=""
increment_from_existing_aggregate="false"
if [[ -n "${current_aggregate}" ]]; then
    current_aggregate_sidecar="$(sidecar_path_for_aggregate "${current_aggregate}")"
    if [[ -f "${current_aggregate_sidecar}" ]]; then
        cp "${current_aggregate_sidecar}" "${base_input_list_file}"
        increment_from_existing_aggregate="true"
    else
        echo "Latest aggregate exists but has no input sidecar: ${current_aggregate}"
        echo "Falling back to rebuilding from the currently saved raw production states."
        : > "${base_input_list_file}"
    fi
else
    : > "${base_input_list_file}"
fi

staged_old_dir="${job_root}/staged_previous_aggregate"
staged_old_aggregate=""
staged_old_sidecar=""
aggregation_succeeded="false"
restore_staged_previous_aggregate() {
    if [[ "${aggregation_succeeded}" == "true" ]]; then
        return
    fi
    if [[ -n "${staged_old_aggregate}" && -f "${staged_old_aggregate}" ]]; then
        mv "${staged_old_aggregate}" "${aggregated_dir}/"
    fi
    if [[ -n "${staged_old_sidecar}" && -f "${staged_old_sidecar}" ]]; then
        mv "${staged_old_sidecar}" "${aggregated_dir}/"
    fi
}
trap restore_staged_previous_aggregate EXIT

: > "${raw_state_list_file}"
resolved_count=0
missing_count=0
while IFS=',' read -r row_type job_name submit_file output_file error_file log_file save_tag extra; do
    [[ "${row_type}" == "replica" ]] || continue
    [[ -n "${save_tag}" ]] || continue
    matched_state="$(latest_state_for_id_tag "${state_dir}" "${save_tag}")"
    if [[ -z "${matched_state}" ]]; then
        missing_count=$((missing_count + 1))
        continue
    fi
    resolved_count=$((resolved_count + 1))
    if [[ -s "${base_input_list_file}" ]] && grep -Fqx "${matched_state}" "${base_input_list_file}"; then
        continue
    fi
    echo "${matched_state}" >> "${raw_state_list_file}"
done < "${manifest_path}"

if [[ ! -s "${raw_state_list_file}" ]]; then
    if [[ -n "${current_aggregate}" ]]; then
        echo "No new saved raw states were found beyond the latest aggregate."
        echo "  current_aggregate=${current_aggregate}"
        exit 0
    fi
    echo "No saved replica states were found to aggregate under ${state_dir}."
    exit 1
fi

: > "${state_list_file}"
if [[ "${increment_from_existing_aggregate}" == "true" ]]; then
    echo "${current_aggregate}" >> "${state_list_file}"
elif [[ -n "${current_aggregate}" ]]; then
    mkdir -p "${staged_old_dir}"
    staged_old_aggregate="${staged_old_dir}/$(basename "${current_aggregate}")"
    mv "${current_aggregate}" "${staged_old_aggregate}"
    if [[ -f "${current_aggregate_sidecar}" ]]; then
        staged_old_sidecar="${staged_old_dir}/$(basename "${current_aggregate_sidecar}")"
        mv "${current_aggregate_sidecar}" "${staged_old_sidecar}"
    fi
fi
cat "${raw_state_list_file}" >> "${state_list_file}"

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

echo "Aggregating saved states for run_id=${run_id}"
echo "  current_aggregate=${current_aggregate:-<none>}"
echo "  new_raw_states=$(wc -l < "${raw_state_list_file}")"
echo "  resolved_manifest_states=${resolved_count}"
echo "  missing_manifest_states=${missing_count}"

bash "${RUNNER_SCRIPT}" "${aggregate_config}" --aggregate_state_list "${state_list_file}" --save_tag "${aggregate_run_id}"

new_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "${aggregate_run_id}")"
if [[ -z "${new_aggregate}" ]]; then
    echo "Aggregation finished but no output aggregate with save_tag='${aggregate_run_id}' was found under ${aggregated_dir}"
    exit 1
fi

new_aggregate_sidecar="$(sidecar_path_for_aggregate "${new_aggregate}")"
if [[ -s "${base_input_list_file}" ]]; then
    awk 'NF && !seen[$0]++' "${base_input_list_file}" "${raw_state_list_file}" > "${new_aggregate_sidecar}"
else
    awk 'NF && !seen[$0]++' "${raw_state_list_file}" > "${new_aggregate_sidecar}"
fi

aggregation_succeeded="true"

while IFS= read -r -d '' old_aggregate; do
    [[ "${old_aggregate}" == "${new_aggregate}" ]] && continue
    mv "${old_aggregate}" "${archive_dir}/"
    old_sidecar="$(sidecar_path_for_aggregate "${old_aggregate}")"
    if [[ -f "${old_sidecar}" ]]; then
        mv "${old_sidecar}" "${archive_dir}/"
    fi
done < <(find "${aggregated_dir}" -maxdepth 1 \( -type f -o -type l \) -name "*_id-${aggregate_run_id}.jld2" -print0 2>/dev/null)

if [[ -n "${staged_old_aggregate}" && -f "${staged_old_aggregate}" ]]; then
    mv "${staged_old_aggregate}" "${archive_dir}/"
fi
if [[ -n "${staged_old_sidecar}" && -f "${staged_old_sidecar}" ]]; then
    mv "${staged_old_sidecar}" "${archive_dir}/"
fi

cat > "${job_info}" <<EOF
timestamp=$(date +%Y%m%d-%H%M%S)
target_run_id=${run_id}
target_run_info=${target_run_info}
config_path=${config_path}
manifest_path=${manifest_path}
state_dir=${state_dir}
aggregated_dir=${aggregated_dir}
current_aggregate=${current_aggregate}
new_aggregate=${new_aggregate}
aggregate_run_id=${aggregate_run_id}
resolved_manifest_states=${resolved_count}
missing_manifest_states=${missing_count}
new_raw_state_count=$(wc -l < "${raw_state_list_file}")
archive_dir=${archive_dir}
state_list_file=${state_list_file}
raw_state_list_file=${raw_state_list_file}
aggregate_inputs=${new_aggregate_sidecar}
EOF

echo "Saved-state aggregation completed."
echo "  new_aggregate=${new_aggregate}"
echo "  aggregate_inputs=${new_aggregate_sidecar}"
echo "  archive_dir=${archive_dir}"
echo "  job_info=${job_info}"
