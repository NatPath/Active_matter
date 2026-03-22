#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash tail_two_force_d_add_repeats_logs.sh --run_id <id> [options]

Required:
  --run_id <id>                  Existing two_force_d production run_id or chain run_id

Options:
  --repo_root <path>             Repo checkout to inspect (default: repo containing this script)
  --mode <auto|production|warmup_production>
                                 How to resolve --run_id (default: auto)
  --job_label <label>            Restrict to add-repeats batches whose token starts with this label
  --job_token <token>            Restrict to an exact/partial add-repeats batch token
  --batch_path <path>            Inspect this add-repeats batch directory directly
  --d_values <csv>               Restrict to a subset of d values
  --only <problems|failed|done|all>
                                 Which nodes to show (default: problems)
  --tail_lines <int>             Lines per log tail (default: 25)
  --max_nodes <int>              Stop after showing this many nodes (default: 20, 0 = unlimited)
  -h, --help                     Show help

Behavior:
  - Resolves the add-repeats batch the same way as check_two_force_d_add_repeats_status.sh.
  - Prints tails from the per-node .err/.out/.log files for matching replica and aggregate jobs.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT_DEFAULT}/run_diffusive_no_activity.jl" ]]; then
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/two_force_d_add_repeats_utils.sh"

run_id=""
repo_root="${REPO_ROOT_DEFAULT}"
mode="auto"
job_label=""
job_token=""
batch_path=""
d_values_csv=""
only_mode="problems"
tail_lines="25"
max_nodes="20"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --repo_root)
            repo_root="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --job_label)
            job_label="${2:-}"
            shift 2
            ;;
        --job_token)
            job_token="${2:-}"
            shift 2
            ;;
        --batch_path)
            batch_path="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --only)
            only_mode="${2:-}"
            shift 2
            ;;
        --tail_lines)
            tail_lines="${2:-}"
            shift 2
            ;;
        --max_nodes)
            max_nodes="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${run_id}" ]]; then
    echo "--run_id is required." >&2
    usage
    exit 1
fi
if [[ ! "${tail_lines}" =~ ^[0-9]+$ ]] || (( tail_lines <= 0 )); then
    echo "--tail_lines must be a positive integer. Got '${tail_lines}'." >&2
    exit 1
fi
if ! [[ "${max_nodes}" =~ ^[0-9]+$ ]] || (( max_nodes < 0 )); then
    echo "--max_nodes must be a non-negative integer. Got '${max_nodes}'." >&2
    exit 1
fi
case "${only_mode}" in
    problems|failed|done|all)
        ;;
    *)
        echo "--only must be one of: problems, failed, done, all." >&2
        exit 1
        ;;
esac

requested_d_values=()
if [[ -n "${d_values_csv}" ]]; then
    two_force_parse_csv_ints "${d_values_csv}" requested_d_values
fi

target_run_info="$(two_force_resolve_target_production_run_info "${repo_root}" "${run_id}" "${mode}")"
target_run_id="$(two_force_read_key_value "${target_run_info}" "run_id")"
target_state_dir="$(two_force_read_key_value "${target_run_info}" "state_dir")"
target_L="$(two_force_read_key_value "${target_run_info}" "L")"
target_rho="$(two_force_read_key_value "${target_run_info}" "rho0")"
run_hash="$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
rho_tag="$(two_force_sanitize_token "${target_rho}")"

if [[ -n "${batch_path}" ]]; then
    job_root="${batch_path}"
else
    job_root="$(two_force_pick_add_repeats_job_root "${repo_root}" "${target_run_id}" "${job_label}" "${job_token}" || true)"
    if [[ -z "${job_root}" ]]; then
        echo "No add-repeats batch found for target_run_id='${target_run_id}' under ${repo_root}/runs/two_force_d/add_repeats_jobs" >&2
        exit 1
    fi
fi

job_info="${job_root}/job_info.txt"
manifest="${job_root}/manifest.csv"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"

if [[ ! -f "${job_info}" || ! -f "${manifest}" ]]; then
    echo "Batch is missing job_info.txt or manifest.csv: ${job_root}" >&2
    exit 1
fi

aggregated_subdir="$(two_force_read_key_value "${job_info}" "aggregated_subdir")"
raw_subdir="$(two_force_read_key_value "${job_info}" "raw_subdir")"
raw_state_dir_default="$(two_force_read_key_value "${job_info}" "raw_state_dir")"
job_timestamp="$(two_force_read_key_value "${job_info}" "timestamp")"
job_start_epoch="$(two_force_timestamp_to_epoch "${job_timestamp}")"

aggregated_dir="${target_state_dir}"
if [[ -n "${aggregated_subdir}" ]]; then
    aggregated_dir="${target_state_dir}/${aggregated_subdir}"
fi

echo "Log tails for add-repeats batch"
echo "  target_run_id=${target_run_id}"
echo "  job_root=${job_root}"
echo "  only=${only_mode}"
echo "  tail_lines=${tail_lines}"
echo "  max_nodes=${max_nodes}"

shown_count=0

while IFS=',' read -r d_val new_repeats _existing_raw_count _warmup_state _config_path manifest_raw_state_dir; do
    [[ "${d_val}" == "d" ]] && continue
    if (( ${#requested_d_values[@]} > 0 )); then
        keep="false"
        for requested_d in "${requested_d_values[@]}"; do
            if [[ "${requested_d}" == "${d_val}" ]]; then
                keep="true"
                break
            fi
        done
        [[ "${keep}" == "true" ]] || continue
    fi

    effective_raw_state_dir="${manifest_raw_state_dir:-${raw_state_dir_default}}"
    if [[ -z "${effective_raw_state_dir}" ]]; then
        effective_raw_state_dir="${target_state_dir}/${raw_subdir}"
    fi

    expected_repeats="${new_repeats}"
    if ! [[ "${expected_repeats}" =~ ^[0-9]+$ ]]; then
        expected_repeats="0"
    fi

    for ((replica_idx = 1; replica_idx <= expected_repeats; replica_idx++)); do
        two_force_resolve_replica_node_metadata \
            "${submit_dir}" \
            "${log_dir}" \
            "${target_run_id}" \
            "${job_timestamp}" \
            "${d_val}" \
            "${replica_idx}" \
            save_tag \
            replica_out \
            replica_err \
            replica_log \
            replica_submit
        saved_path="$(two_force_latest_state_for_id_tag_top_level "${effective_raw_state_dir}" "${save_tag}")"
        zero_path="$(two_force_any_state_for_id_tag_top_level "${effective_raw_state_dir}" "${save_tag}")"
        condor_state="$(two_force_condor_log_state "${replica_log}")"
        node_state="$(two_force_classify_node_state "${saved_path}" "${zero_path}" "${condor_state}")"
        two_force_should_show_problem_state "${only_mode}" "${node_state}" || continue

        echo
        echo "Replica d=${d_val} state=${node_state} tag=${save_tag}"
        if [[ -n "${saved_path}" ]]; then
            echo "  saved=${saved_path}"
        elif [[ -n "${zero_path}" ]]; then
            echo "  zero_size=${zero_path}"
        fi
        two_force_print_file_tail "err" "${replica_err}" "${tail_lines}"
        two_force_print_file_tail "out" "${replica_out}" "${tail_lines}"
        two_force_print_file_tail "log" "${replica_log}" "${tail_lines}"
        shown_count=$((shown_count + 1))
        if (( max_nodes > 0 && shown_count >= max_nodes )); then
            exit 0
        fi
    done

    while IFS= read -r segment_submit; do
        [[ -n "${segment_submit}" ]] || continue
        save_tag="$(two_force_extract_save_tag_from_submit "${segment_submit}")"
        segment_out="$(two_force_extract_submit_path_value "${segment_submit}" "output")"
        segment_err="$(two_force_extract_submit_path_value "${segment_submit}" "error")"
        segment_log="$(two_force_extract_submit_path_value "${segment_submit}" "log")"
        condor_state="$(two_force_condor_log_state "${segment_log}")"
        [[ "${condor_state}" == "missing" ]] && continue
        node_state="$(two_force_classify_node_state "" "" "${condor_state}")"
        two_force_should_show_problem_state "${only_mode}" "${node_state}" || continue

        echo
        echo "Segment d=${d_val} state=${node_state} submit=$(basename "${segment_submit}") tag=${save_tag}"
        two_force_print_file_tail "err" "${segment_err}" "${tail_lines}"
        two_force_print_file_tail "out" "${segment_out}" "${tail_lines}"
        two_force_print_file_tail "log" "${segment_log}" "${tail_lines}"
        shown_count=$((shown_count + 1))
        if (( max_nodes > 0 && shown_count >= max_nodes )); then
            exit 0
        fi
    done < <(find "${submit_dir}" -maxdepth 1 -type f -name "seg_d_${d_val}_r_*_s_*.sub" | sort -V)

    aggregate_submit="${submit_dir}/d_${d_val}_aggregate.sub"
    if [[ -f "${aggregate_submit}" ]]; then
        aggregate_out="$(two_force_extract_submit_path_value "${aggregate_submit}" "output")"
        aggregate_err="$(two_force_extract_submit_path_value "${aggregate_submit}" "error")"
        aggregate_log="$(two_force_extract_submit_path_value "${aggregate_submit}" "log")"
        aggregate_tag="aggregated_saved_r${run_hash}_d${d_val}"
        aggregate_legacy_tag="aggregated_saved_L${target_L}_rho${rho_tag}_rid${run_hash}_d${d_val}"
        aggregate_saved="$(two_force_latest_state_for_id_tag_top_level "${aggregated_dir}" "${aggregate_tag}")"
        if [[ -z "${aggregate_saved}" ]]; then
            aggregate_saved="$(two_force_latest_state_for_id_tag_top_level "${target_state_dir}" "${aggregate_tag}")"
        fi
        if [[ -z "${aggregate_saved}" ]]; then
            aggregate_saved="$(two_force_latest_state_for_id_tag_top_level "${aggregated_dir}" "${aggregate_legacy_tag}")"
        fi
        if [[ -z "${aggregate_saved}" ]]; then
            aggregate_saved="$(two_force_latest_state_for_id_tag_top_level "${target_state_dir}" "${aggregate_legacy_tag}")"
        fi
        aggregate_zero="$(two_force_any_state_for_id_tag_top_level "${aggregated_dir}" "${aggregate_tag}")"
        if [[ -z "${aggregate_zero}" ]]; then
            aggregate_zero="$(two_force_any_state_for_id_tag_top_level "${target_state_dir}" "${aggregate_tag}")"
        fi
        if [[ -z "${aggregate_zero}" ]]; then
            aggregate_zero="$(two_force_any_state_for_id_tag_top_level "${aggregated_dir}" "${aggregate_legacy_tag}")"
        fi
        if [[ -z "${aggregate_zero}" ]]; then
            aggregate_zero="$(two_force_any_state_for_id_tag_top_level "${target_state_dir}" "${aggregate_legacy_tag}")"
        fi
        aggregate_condor_state="$(two_force_condor_log_state "${aggregate_log}")"
        aggregate_state="$(two_force_classify_node_state "${aggregate_saved}" "${aggregate_zero}" "${aggregate_condor_state}" "${job_start_epoch}")"
        two_force_should_show_problem_state "${only_mode}" "${aggregate_state}" || continue

        echo
        echo "Aggregate d=${d_val} state=${aggregate_state} tag=${aggregate_tag}"
        if [[ -n "${aggregate_saved}" ]]; then
            echo "  saved=${aggregate_saved}"
        elif [[ -n "${aggregate_zero}" ]]; then
            echo "  zero_size=${aggregate_zero}"
        fi
        two_force_print_file_tail "err" "${aggregate_err}" "${tail_lines}"
        two_force_print_file_tail "out" "${aggregate_out}" "${tail_lines}"
        two_force_print_file_tail "log" "${aggregate_log}" "${tail_lines}"
        shown_count=$((shown_count + 1))
        if (( max_nodes > 0 && shown_count >= max_nodes )); then
            exit 0
        fi
    fi
done < "${manifest}"

if (( shown_count == 0 )); then
    echo
    echo "No nodes matched --only=${only_mode}."
fi
