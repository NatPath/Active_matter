#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash summarize_two_force_d_add_repeats_failures.sh --run_id <id> [options]

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
  --top <int>                    Show top N signatures per d (default: 10)
  -h, --help                     Show help

Behavior:
  - Scans failed/missing-output/zero-size replica jobs in the selected add-repeats batch.
  - Groups them by a simple signature extracted from stderr/stdout/condor log tail.
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
top_n="10"

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
        --top)
            top_n="${2:-}"
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
if ! [[ "${top_n}" =~ ^[0-9]+$ ]] || (( top_n <= 0 )); then
    echo "--top must be a positive integer. Got '${top_n}'." >&2
    exit 1
fi

requested_d_values=()
if [[ -n "${d_values_csv}" ]]; then
    two_force_parse_csv_ints "${d_values_csv}" requested_d_values
fi

target_run_info="$(two_force_resolve_target_production_run_info "${repo_root}" "${run_id}" "${mode}")"
target_run_id="$(two_force_read_key_value "${target_run_info}" "run_id")"
target_state_dir="$(two_force_read_key_value "${target_run_info}" "state_dir")"

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

raw_subdir="$(two_force_read_key_value "${job_info}" "raw_subdir")"
raw_state_dir_default="$(two_force_read_key_value "${job_info}" "raw_state_dir")"
job_timestamp="$(two_force_read_key_value "${job_info}" "timestamp")"

echo "Failure summary for add-repeats batch"
echo "  target_run_id=${target_run_id}"
echo "  job_root=${job_root}"
echo "  top=${top_n}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

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

    sig_file="${tmp_dir}/d_${d_val}.txt"
    : > "${sig_file}"
    fail_count=0

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
        if [[ "${node_state}" != "failed" && "${node_state}" != "missing_output" && "${node_state}" != "zero_size" ]]; then
            continue
        fi

        signature="$(two_force_failure_signature "${replica_err}" "${replica_out}" "${replica_log}")"
        printf "%s\t%s\t%s\n" "${signature}" "${node_state}" "$(basename "${replica_submit}")" >> "${sig_file}"
        fail_count=$((fail_count + 1))
    done

    while IFS= read -r segment_submit; do
        [[ -n "${segment_submit}" ]] || continue
        segment_out="$(two_force_extract_submit_path_value "${segment_submit}" "output")"
        segment_err="$(two_force_extract_submit_path_value "${segment_submit}" "error")"
        segment_log="$(two_force_extract_submit_path_value "${segment_submit}" "log")"
        condor_state="$(two_force_condor_log_state "${segment_log}")"
        node_state="$(two_force_classify_node_state "" "" "${condor_state}")"
        if [[ "${node_state}" != "failed" && "${node_state}" != "missing_output" && "${node_state}" != "zero_size" ]]; then
            continue
        fi

        signature="$(two_force_failure_signature "${segment_err}" "${segment_out}" "${segment_log}")"
        printf "%s\t%s\t%s\n" "${signature}" "${node_state}" "$(basename "${segment_submit}")" >> "${sig_file}"
        fail_count=$((fail_count + 1))
    done < <(find "${submit_dir}" -maxdepth 1 -type f -name "seg_d_${d_val}_r_*_s_*.sub" | sort -V)

    echo
    echo "d=${d_val} failed_nodes=${fail_count}"
    if (( fail_count == 0 )); then
        echo "  none"
        continue
    fi

    awk -F'\t' '{count[$1]++} END {for (sig in count) printf "%07d\t%s\n", count[sig], sig}' "${sig_file}" \
        | sort -rn \
        | head -n "${top_n}" \
        | while IFS=$'\t' read -r count signature; do
            count="$((10#${count}))"
            sample="$(awk -F'\t' -v sig="${signature}" '$1 == sig {print $2 " " $3; exit}' "${sig_file}")"
            printf "  %s  %s\n" "${count}" "${signature}"
            printf "      sample=%s\n" "${sample}"
        done
done < "${manifest}"
