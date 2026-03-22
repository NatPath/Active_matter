#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash check_two_force_d_add_repeats_status.sh --run_id <id> [options]

Required:
  --run_id <id>                  Existing two_force_d production run_id or chain run_id

Options:
  --repo_root <path>             Repo checkout to inspect (default: repo containing this script)
  --mode <auto|production|warmup_production>
                                 How to resolve --run_id (default: auto)
  --job_label <label>            Restrict to add-repeats batches whose token starts with this label
  --job_token <token>            Restrict to an exact/partial add-repeats batch token
  --batch_path <path>            Inspect this add-repeats batch directory directly
  --d_values <csv>               Restrict report to a subset of d values
  --list_batches                 List matching add-repeats batch directories and exit
  -h, --help                     Show help

Behavior:
  - Resolves the production run for --run_id.
  - Finds the newest matching add-repeats batch unless --batch_path is given.
  - Summarizes replica and aggregate status using:
      submit/*.sub
      logs/*.out|*.err|*.log
      raw repeat-batch .jld2 outputs
      aggregated .jld2 outputs
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
list_batches="false"

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
        --list_batches)
            list_batches="true"
            shift
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
if [[ ! -d "${repo_root}" ]]; then
    echo "--repo_root does not exist: ${repo_root}" >&2
    exit 1
fi
if [[ -n "${batch_path}" && ! -d "${batch_path}" ]]; then
    echo "--batch_path does not exist: ${batch_path}" >&2
    exit 1
fi

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

if [[ "${list_batches}" == "true" ]]; then
    two_force_list_add_repeats_job_roots "${repo_root}" "${target_run_id}" "${job_label}" "${job_token}"
    exit 0
fi

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

if [[ ! -f "${job_info}" ]]; then
    echo "Missing job_info.txt: ${job_info}" >&2
    exit 1
fi
if [[ ! -f "${manifest}" ]]; then
    echo "Missing manifest.csv: ${manifest}" >&2
    exit 1
fi

job_timestamp="$(two_force_read_key_value "${job_info}" "timestamp")"
aggregated_subdir="$(two_force_read_key_value "${job_info}" "aggregated_subdir")"
archive_subdir="$(two_force_read_key_value "${job_info}" "archive_subdir")"
raw_subdir="$(two_force_read_key_value "${job_info}" "raw_subdir")"
raw_state_dir="$(two_force_read_key_value "${job_info}" "raw_state_dir")"
expected_repeats_default="$(two_force_read_key_value "${job_info}" "num_repeats")"
selected_d_values_csv="$(two_force_read_key_value "${job_info}" "selected_d_values")"
job_start_epoch="$(two_force_timestamp_to_epoch "${job_timestamp}")"

aggregated_dir="${target_state_dir}"
if [[ -n "${aggregated_subdir}" ]]; then
    aggregated_dir="${target_state_dir}/${aggregated_subdir}"
fi
archive_root=""

echo "Add-repeats batch summary"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${target_run_info}"
echo "  job_root=${job_root}"
echo "  timestamp=${job_timestamp}"
echo "  raw_state_dir=${raw_state_dir}"
echo "  aggregated_dir=${aggregated_dir}"
echo "  manifest=${manifest}"
echo "  selected_d_values=${selected_d_values_csv}"

total_expected=0
total_done=0
total_running=0
total_submitted=0
total_failed=0
total_missing=0
total_zero_size=0
total_unknown=0
aggregate_done=0
aggregate_preexisting=0
aggregate_failed=0
aggregate_pending=0

printf "\n%-6s %-22s %-16s %-16s %-8s %-8s %-8s\n" "d" "replicas(done/exp)" "replica_state_counts" "aggregate" "agg.err" "r.err" "saved"

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

    expected_repeats="${new_repeats}"
    [[ -n "${expected_repeats}" ]] || expected_repeats="${expected_repeats_default:-0}"
    if ! [[ "${expected_repeats}" =~ ^[0-9]+$ ]]; then
        expected_repeats="0"
    fi

    effective_raw_state_dir="${manifest_raw_state_dir:-${raw_state_dir}}"
    if [[ -z "${effective_raw_state_dir}" ]]; then
        effective_raw_state_dir="${target_state_dir}/${raw_subdir}"
    fi

    d_done=0
    d_running=0
    d_submitted=0
    d_failed=0
    d_missing=0
    d_zero_size=0
    d_unknown=0
    d_err_count=0
    d_saved_count=0

    replica_indices=()
    if (( expected_repeats > 0 )); then
        for ((replica_idx = 1; replica_idx <= expected_repeats; replica_idx++)); do
            replica_indices+=("${replica_idx}")
        done
    fi

    for replica_idx in "${replica_indices[@]}"; do
        save_tag=""
        replica_submit=""
        replica_log=""
        replica_err=""
        replica_out=""
        saved_path=""
        zero_path=""
        condor_state=""
        node_state=""

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
        if [[ -n "${saved_path}" ]]; then
            d_saved_count=$((d_saved_count + 1))
        fi
        condor_state="$(two_force_condor_log_state "${replica_log}")"
        node_state="$(two_force_classify_node_state "${saved_path}" "${zero_path}" "${condor_state}")"
        if [[ "${node_state}" == "missing" ]]; then
            segment_best_state="missing"
            while IFS= read -r segment_submit; do
                [[ -n "${segment_submit}" ]] || continue
                segment_log="$(two_force_extract_submit_path_value "${segment_submit}" "log")"
                segment_err="$(two_force_extract_submit_path_value "${segment_submit}" "error")"
                segment_condor_state="$(two_force_condor_log_state "${segment_log}")"
                [[ "${segment_condor_state}" == "missing" ]] && continue
                segment_node_state="$(two_force_classify_node_state "" "" "${segment_condor_state}")"
                segment_best_state="$(two_force_pick_better_state "${segment_best_state}" "${segment_node_state}")"
                if [[ -n "${segment_err}" && -s "${segment_err}" ]]; then
                    d_err_count=$((d_err_count + 1))
                fi
            done < <(find "${submit_dir}" -maxdepth 1 -type f -name "seg_d_${d_val}_r_${replica_idx}_s_*.sub" | sort -V)
            node_state="${segment_best_state}"
        fi
        case "${node_state}" in
            done)
                d_done=$((d_done + 1))
                ;;
            running)
                d_running=$((d_running + 1))
                ;;
            submitted)
                d_submitted=$((d_submitted + 1))
                ;;
            failed|missing_output)
                d_failed=$((d_failed + 1))
                ;;
            zero_size)
                d_zero_size=$((d_zero_size + 1))
                ;;
            missing)
                d_missing=$((d_missing + 1))
                ;;
            *)
                d_unknown=$((d_unknown + 1))
                ;;
        esac
        if [[ -n "${replica_err}" && -s "${replica_err}" ]]; then
            d_err_count=$((d_err_count + 1))
        fi
    done

    aggregate_submit="${submit_dir}/d_${d_val}_aggregate.sub"
    aggregate_log="$(two_force_extract_submit_path_value "${aggregate_submit}" "log" 2>/dev/null || true)"
    aggregate_err="$(two_force_extract_submit_path_value "${aggregate_submit}" "error" 2>/dev/null || true)"
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

    case "${aggregate_state}" in
        done)
            aggregate_done=$((aggregate_done + 1))
            ;;
        preexisting)
            aggregate_preexisting=$((aggregate_preexisting + 1))
            aggregate_pending=$((aggregate_pending + 1))
            ;;
        failed|missing_output|zero_size)
            aggregate_failed=$((aggregate_failed + 1))
            ;;
        *)
            aggregate_pending=$((aggregate_pending + 1))
            ;;
    esac

    total_expected=$((total_expected + expected_repeats))
    total_done=$((total_done + d_done))
    total_running=$((total_running + d_running))
    total_submitted=$((total_submitted + d_submitted))
    total_failed=$((total_failed + d_failed))
    total_missing=$((total_missing + d_missing))
    total_zero_size=$((total_zero_size + d_zero_size))
    total_unknown=$((total_unknown + d_unknown))

    replica_counts="ok=${d_done} run=${d_running} sub=${d_submitted} fail=${d_failed} miss=${d_missing}"
    if (( d_zero_size > 0 || d_unknown > 0 )); then
        replica_counts="${replica_counts} z=${d_zero_size} u=${d_unknown}"
    fi
    aggregate_err_flag="0"
    if [[ -n "${aggregate_err}" && -s "${aggregate_err}" ]]; then
        aggregate_err_flag="1"
    fi

    printf "%-6s %-22s %-16s %-16s %-8s %-8s %-8s\n" \
        "${d_val}" \
        "${d_done}/${#replica_indices[@]}" \
        "${replica_counts}" \
        "${aggregate_state}" \
        "${aggregate_err_flag}" \
        "${d_err_count}" \
        "${d_saved_count}"
done < "${manifest}"

echo
echo "Replica totals"
echo "  expected=${total_expected}"
echo "  done=${total_done}"
echo "  running=${total_running}"
echo "  submitted=${total_submitted}"
echo "  failed=${total_failed}"
echo "  missing=${total_missing}"
echo "  zero_size=${total_zero_size}"
echo "  unknown=${total_unknown}"

echo "Aggregate totals"
echo "  done=${aggregate_done}"
echo "  preexisting=${aggregate_preexisting}"
echo "  failed=${aggregate_failed}"
echo "  pending=${aggregate_pending}"

echo "Paths"
echo "  submit_dir=${submit_dir}"
echo "  log_dir=${log_dir}"
