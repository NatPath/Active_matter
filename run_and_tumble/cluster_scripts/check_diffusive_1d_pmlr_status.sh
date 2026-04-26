#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash check_diffusive_1d_pmlr_status.sh [--run_id <id> | --batch_path <path> | --latest] [options]

Options:
  --run_id <id>          Existing diffusive_1d_pmlr run id
  --batch_path <path>    Inspect this run directory directly
  --latest               Inspect the newest run under runs/diffusive_1d_pmlr/{warmup,production}
  --mode <auto|warmup|production>
                         How to resolve --run_id (default: auto)
  --max_problems <int>   Maximum problem nodes to print (default: 12)
  --snippet_lines <int>  Lines of error snippet per problem node (default: 3)
  -h, --help             Show help

Behavior:
  - Reads run_info.txt, manifest.csv, config, logs, and saved states from an existing run root.
  - Summarizes config, saved-state counts, Condor node states, and first error snippets.
  - Focuses on the direct batch artifacts only; it does not reach out to the cluster.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ ! -f "${REPO_ROOT}/run_diffusive_no_activity.jl" ]]; then
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}" >&2
    exit 1
fi

read_run_info_value() {
    local file_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${file_path}" | tail -n 1
}

read_config_value() {
    local file_path="$1"
    local key="$2"
    [[ -f "${file_path}" ]] || return 0
    awk -F: -v k="${key}" '
        $1 ~ "^[[:space:]]*" k "[[:space:]]*$" {
            sub(/^[^:]*:[[:space:]]*/, "", $0)
            sub(/[[:space:]]+#.*$/, "", $0)
            gsub(/^"/, "", $0)
            gsub(/"$/, "", $0)
            gsub(/^'\''/, "", $0)
            gsub(/'\''$/, "", $0)
            print $0
            exit
        }
    ' "${file_path}"
}

latest_state_for_id_tag() {
    local root_dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime

    [[ -d "${root_dir}" ]] || {
        printf ""
        return 0
    }

    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${root_dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

condor_log_state() {
    local log_path="$1"
    [[ -f "${log_path}" ]] || {
        printf "missing"
        return 0
    }
    if rg -q "Job was held\\." "${log_path}"; then
        printf "held"
    elif rg -q "Job terminated\\." "${log_path}"; then
        printf "terminated"
    elif rg -q "executing on host" "${log_path}"; then
        printf "running"
    elif rg -q "Job submitted from host" "${log_path}"; then
        printf "submitted"
    else
        printf "unknown"
    fi
}

first_error_snippet() {
    local path="$1"
    local snippet_lines="$2"
    [[ -f "${path}" ]] || return 0
    rg -n -m "${snippet_lines}" \
        "ERROR:|MethodError|ArgumentError|LoadError|Stacktrace|Exception|Traceback|FATAL|No such file|command not found|syntax error|invalid" \
        "${path}" 2>/dev/null || true
}

resolve_run_root() {
    local run_id="$1"
    local batch_path="$2"
    local mode="$3"
    local latest_flag="$4"
    local candidate roots_found registry_file registry_row reg_run_root
    local -a matches=()

    if [[ -n "${batch_path}" ]]; then
        if [[ "${batch_path}" != /* ]]; then
            batch_path="${REPO_ROOT}/${batch_path}"
        fi
        [[ -d "${batch_path}" ]] || {
            echo "batch_path not found: ${batch_path}" >&2
            return 1
        }
        printf "%s" "${batch_path}"
        return 0
    fi

    if [[ "${latest_flag}" == "true" ]]; then
        candidate="$(
            find "${REPO_ROOT}/runs/diffusive_1d_pmlr" -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' 2>/dev/null \
                | sort -nr | awk 'NR==1 { $1=""; sub(/^ /, ""); print }'
        )"
        [[ -n "${candidate}" ]] || {
            echo "No diffusive_1d_pmlr runs found under ${REPO_ROOT}/runs/diffusive_1d_pmlr" >&2
            return 1
        }
        printf "%s" "${candidate}"
        return 0
    fi

    [[ -n "${run_id}" ]] || {
        echo "Provide --run_id, --batch_path, or --latest." >&2
        return 1
    }

    if [[ "${mode}" == "auto" || "${mode}" == "warmup" ]]; then
        candidate="${REPO_ROOT}/runs/diffusive_1d_pmlr/warmup/${run_id}"
        [[ -d "${candidate}" ]] && matches+=("${candidate}")
    fi
    if [[ "${mode}" == "auto" || "${mode}" == "production" ]]; then
        candidate="${REPO_ROOT}/runs/diffusive_1d_pmlr/production/${run_id}"
        [[ -d "${candidate}" ]] && matches+=("${candidate}")
    fi

    if (( ${#matches[@]} == 1 )); then
        printf "%s" "${matches[0]}"
        return 0
    elif (( ${#matches[@]} > 1 )); then
        echo "run_id '${run_id}' exists in multiple modes. Add --mode warmup or --mode production." >&2
        return 1
    fi

    registry_file="${REPO_ROOT}/runs/diffusive_1d_pmlr/run_registry.csv"
    if [[ -f "${registry_file}" ]]; then
        registry_row="$(awk -F, -v rid="${run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${registry_file}")"
        if [[ -n "${registry_row}" ]]; then
            if [[ "$(awk -F, '{print NF}' <<< "${registry_row}")" -ge 12 ]]; then
                IFS=',' read -r _ts _rid _mode _L _rho _gamma _potential _ns _nr _cpus _mem reg_run_root _rest <<< "${registry_row}"
                if [[ -n "${reg_run_root}" && -d "${reg_run_root}" ]]; then
                    printf "%s" "${reg_run_root}"
                    return 0
                fi
            fi
        fi
    fi

    echo "Could not resolve run root for run_id='${run_id}'." >&2
    return 1
}

run_id=""
batch_path=""
mode="auto"
latest_flag="false"
max_problems="12"
snippet_lines="3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --batch_path)
            batch_path="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --latest)
            latest_flag="true"
            shift 1
            ;;
        --max_problems)
            max_problems="${2:-}"
            shift 2
            ;;
        --snippet_lines)
            snippet_lines="${2:-}"
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

case "${mode}" in
    auto|warmup|production)
        ;;
    *)
        echo "--mode must be auto, warmup, or production. Got '${mode}'." >&2
        exit 1
        ;;
esac
if ! [[ "${max_problems}" =~ ^[0-9]+$ ]] || (( max_problems < 0 )); then
    echo "--max_problems must be a non-negative integer. Got '${max_problems}'." >&2
    exit 1
fi
if ! [[ "${snippet_lines}" =~ ^[0-9]+$ ]] || (( snippet_lines <= 0 )); then
    echo "--snippet_lines must be a positive integer. Got '${snippet_lines}'." >&2
    exit 1
fi

run_root="$(resolve_run_root "${run_id}" "${batch_path}" "${mode}" "${latest_flag}")"
run_info="${run_root}/run_info.txt"
manifest="${run_root}/manifest.csv"

[[ -f "${run_info}" ]] || {
    echo "run_info.txt not found: ${run_info}" >&2
    exit 1
}
[[ -f "${manifest}" ]] || {
    echo "manifest.csv not found: ${manifest}" >&2
    exit 1
}

run_id_resolved="$(read_run_info_value "${run_info}" "run_id")"
mode_resolved="$(read_run_info_value "${run_info}" "mode")"
simulation="$(read_run_info_value "${run_info}" "simulation")"
state_dir="$(read_run_info_value "${run_info}" "state_dir")"
[[ -n "${state_dir}" ]] || state_dir="$(read_run_info_value "${run_info}" "raw_state_dir")"
log_dir="$(read_run_info_value "${run_info}" "log_dir")"
config_path="$(read_run_info_value "${run_info}" "runtime_config")"
[[ -n "${config_path}" ]] || config_path="$(read_run_info_value "${run_info}" "config_path")"

forcing_rate_scheme="$(read_config_value "${config_path}" "forcing_rate_scheme")"
bond_pass_count_mode="$(read_config_value "${config_path}" "bond_pass_count_mode")"

replica_rows="$(awk -F, 'NR > 1 && $1 == "replica" {count++} END {print count+0}' "${manifest}")"
aggregate_rows="$(awk -F, 'NR > 1 && $1 == "aggregate" {count++} END {print count+0}' "${manifest}")"
saved_state_count="$(find "${state_dir}" -maxdepth 1 -type f -name '*.jld2' ! -size 0 2>/dev/null | wc -l | tr -d ' ')"
zero_state_count="$(find "${state_dir}" -maxdepth 1 -type f -name '*.jld2' -size 0 2>/dev/null | wc -l | tr -d ' ')"

declare -A status_counts=()
problem_rows_file="$(mktemp)"
trap 'rm -f "${problem_rows_file}"' EXIT

while IFS=',' read -r job_type job_name submit_file output_file error_file log_file save_tag initial_state; do
    [[ "${job_type}" == "job_type" ]] && continue

    condor_state="$(condor_log_state "${log_file}")"
    saved_path=""
    if [[ -n "${save_tag}" && "${job_type}" == "replica" ]]; then
        saved_path="$(latest_state_for_id_tag "${state_dir}" "${save_tag}")"
    fi
    err_snippet="$(first_error_snippet "${error_file}" "${snippet_lines}")"
    out_snippet="$(first_error_snippet "${output_file}" "${snippet_lines}")"

    status="unknown"
    if [[ -n "${saved_path}" ]]; then
        status="saved"
    elif [[ -n "${err_snippet}" || -n "${out_snippet}" ]]; then
        status="error"
    else
        case "${condor_state}" in
            held)
                status="held"
                ;;
            terminated)
                status="terminated_no_state"
                ;;
            running)
                status="running"
                ;;
            submitted)
                status="submitted"
                ;;
            missing)
                status="missing_log"
                ;;
            *)
                status="unknown"
                ;;
        esac
    fi

    status_counts["${status}"]=$(( ${status_counts["${status}"]:-0} + 1 ))

    case "${status}" in
        saved|running|submitted)
            ;;
        *)
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "${job_type}" "${job_name}" "${status}" "${save_tag}" "${error_file}" "${output_file}" "${log_file}" "${initial_state}" "${saved_path}" \
                >> "${problem_rows_file}"
            ;;
    esac
done < "${manifest}"

echo "Run"
echo "  run_id=${run_id_resolved}"
echo "  mode=${mode_resolved}"
echo "  simulation=${simulation}"
echo "  run_root=${run_root}"
echo "  run_info=${run_info}"
echo "  manifest=${manifest}"

echo "Config"
echo "  config=${config_path}"
echo "  L=$(read_run_info_value "${run_info}" "L")"
echo "  rho0=$(read_run_info_value "${run_info}" "rho0")"
echo "  gamma=$(read_run_info_value "${run_info}" "gamma")"
echo "  potential_type=$(read_run_info_value "${run_info}" "potential_type")"
echo "  fluctuation_type=$(read_run_info_value "${run_info}" "fluctuation_type")"
echo "  potential_strength=$(read_run_info_value "${run_info}" "potential_strength")"
echo "  n_sweeps=$(read_run_info_value "${run_info}" "n_sweeps")"
echo "  num_replicas=$(read_run_info_value "${run_info}" "num_replicas")"
echo "  forcing_rate_scheme=${forcing_rate_scheme:-unknown}"
echo "  bond_pass_count_mode=${bond_pass_count_mode:-unknown}"

echo "Artifacts"
echo "  state_dir=${state_dir}"
echo "  log_dir=${log_dir}"
echo "  replica_rows=${replica_rows}"
echo "  aggregate_rows=${aggregate_rows}"
echo "  saved_states=${saved_state_count}"
echo "  zero_size_states=${zero_state_count}"

echo "Status"
for key in saved error held terminated_no_state running submitted missing_log unknown; do
    if [[ -n "${status_counts[${key}]:-}" ]]; then
        echo "  ${key}=${status_counts[${key}]}"
    fi
done

problem_count="$(wc -l < "${problem_rows_file}" | tr -d ' ')"
echo "Problems"
echo "  problem_nodes=${problem_count}"

if (( problem_count > 0 && max_problems > 0 )); then
    shown=0
    while IFS=',' read -r job_type job_name status save_tag error_file output_file log_file initial_state saved_path; do
        echo
        echo "Node ${job_name}"
        echo "  type=${job_type}"
        echo "  status=${status}"
        [[ -n "${save_tag}" ]] && echo "  save_tag=${save_tag}"
        [[ -n "${initial_state}" ]] && echo "  initial_state=${initial_state}"
        [[ -n "${saved_path}" ]] && echo "  saved_path=${saved_path}"
        [[ -f "${error_file}" ]] && echo "  err=${error_file}"
        [[ -f "${output_file}" ]] && echo "  out=${output_file}"
        [[ -f "${log_file}" ]] && echo "  log=${log_file}"

        err_snippet="$(first_error_snippet "${error_file}" "${snippet_lines}")"
        out_snippet="$(first_error_snippet "${output_file}" "${snippet_lines}")"
        if [[ -n "${err_snippet}" ]]; then
            echo "  err_snippet:"
            while IFS= read -r line; do
                echo "    ${line}"
            done <<< "${err_snippet}"
        elif [[ -n "${out_snippet}" ]]; then
            echo "  out_snippet:"
            while IFS= read -r line; do
                echo "    ${line}"
            done <<< "${out_snippet}"
        elif [[ -f "${error_file}" && -s "${error_file}" ]]; then
            echo "  err_tail:"
            tail -n "${snippet_lines}" "${error_file}" | sed 's/^/    /'
        elif [[ -f "${output_file}" && -s "${output_file}" ]]; then
            echo "  out_tail:"
            tail -n "${snippet_lines}" "${output_file}" | sed 's/^/    /'
        fi

        shown=$((shown + 1))
        if (( shown >= max_problems )); then
            break
        fi
    done < "${problem_rows_file}"
fi
