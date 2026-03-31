#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_two_force_d_topup_repeats.sh \
      --run_id <id> \
      [options]

Required:
  --run_id <id>                     Existing two_force_d run_id to top up

Options:
  --mode <auto|production|warmup_production>
                                    How to resolve --run_id (default: production)
  --d_values <csv>                  d values to top up (default: 96,128)
  --num_repeats <int>               New repeats to add per selected d (default: 800)
  --n_sweeps <int>                  Sweeps for each new repeat
                                    (default: infer from target run_info n_sweeps)
  --include_raw_dir <path>          Existing raw repeat-batch root to include in the new aggregation
                                    (repeatable; forwarded as --extra_raw_dir)
  --request_cpus <int>              Replica request_cpus
                                    (default: infer from target run_info or 1)
  --request_memory <value>          Replica/aggregate request_memory
                                    (default: infer from target run_info or "2 GB")
  --aggregate_request_cpus <int>    Aggregate request_cpus (default: 1)
  --julia_num_procs_aggregate <int> Aggregate JULIA_NUM_PROCS_AGGREGATE (default: 1)
  --replica_retries <int>           DAG retry count for replica nodes (default: 2)
  --estimate_runtime                Enable runtime estimation prints inside replica jobs
  --estimate_sample_size <int>      Sample sweeps used when --estimate_runtime is enabled
                                    (default: 100)
  --dag_maxjobs <int>               Forwarded to condor_submit_dag -maxjobs
                                    (default: 0, no DAGMan submitted-node throttle)
  --dag_maxidle <int>               Forwarded to condor_submit_dag -maxidle
                                    (default: 0, no DAGMan idle-proc throttle)
  --job_label <label>               Optional batch label
                                    (default: topup_d<d-values>_nr<num_repeats>)
  --batch_name <name>               Optional Condor batch_name
  --no_submit                       Generate files only; do not submit the DAG
  -h, --help                        Show help

Behavior:
  - Uses submit_two_force_d_add_repeats.sh under the hood.
  - Always aggregates incrementally into <run>/states/aggregated.
  - Existing aggregate files for each selected d are used as the base input when present.
  - New raw states from this top-up batch are added on top.
  - Any --include_raw_dir paths are also folded into the aggregate, which is how you carry
    forward finished states from older partial add-repeats batches.
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

ADD_REPEATS_SCRIPT="${SCRIPT_DIR}/submit_two_force_d_add_repeats.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/two_force_d/run_registry.csv"

if [[ ! -f "${ADD_REPEATS_SCRIPT}" ]]; then
    echo "Missing helper script: ${ADD_REPEATS_SCRIPT}"
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
        candidate="${REPO_ROOT}/runs/two_force_d/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "warmup_production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup_production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "warmup" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    if [[ -f "${REGISTRY_FILE}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${REGISTRY_FILE}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _dmin _dmax _dstep _cpus _mem reg_run_root _log_dir _state_dir _warmup_state_dir <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

run_id=""
mode="production"
d_values_csv="96,128"
num_repeats="800"
n_sweeps=""
include_raw_dirs=()
request_cpus=""
request_memory=""
aggregate_request_cpus="1"
julia_num_procs_aggregate="1"
replica_retries="2"
estimate_runtime="false"
estimate_sample_size="100"
dag_maxjobs="0"
dag_maxidle="0"
job_label=""
batch_name=""
no_submit="false"

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
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --num_repeats)
            num_repeats="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --include_raw_dir)
            include_raw_dirs+=("${2:-}")
            shift 2
            ;;
        --request_cpus)
            request_cpus="${2:-}"
            shift 2
            ;;
        --request_memory)
            request_memory="${2:-}"
            shift 2
            ;;
        --aggregate_request_cpus)
            aggregate_request_cpus="${2:-}"
            shift 2
            ;;
        --julia_num_procs_aggregate)
            julia_num_procs_aggregate="${2:-}"
            shift 2
            ;;
        --replica_retries)
            replica_retries="${2:-}"
            shift 2
            ;;
        --estimate_runtime)
            estimate_runtime="true"
            shift
            ;;
        --estimate_sample_size)
            estimate_sample_size="${2:-}"
            shift 2
            ;;
        --dag_maxjobs)
            dag_maxjobs="${2:-}"
            shift 2
            ;;
        --dag_maxidle)
            dag_maxidle="${2:-}"
            shift 2
            ;;
        --job_label)
            job_label="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
            shift 2
            ;;
        --no_submit)
            no_submit="true"
            shift
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

case "${mode}" in
    auto|production|warmup_production)
        ;;
    *)
        echo "--mode must be one of: auto, production, warmup_production."
        exit 1
        ;;
esac

for numeric_name in num_repeats n_sweeps request_cpus aggregate_request_cpus julia_num_procs_aggregate replica_retries estimate_sample_size dag_maxjobs dag_maxidle; do
    numeric_value="${!numeric_name}"
    if [[ -n "${numeric_value}" ]] && ! [[ "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "--${numeric_name} must be a non-negative integer. Got '${numeric_value}'."
        exit 1
    fi
done
if ! [[ "${num_repeats}" =~ ^[0-9]+$ ]] || (( num_repeats <= 0 )); then
    echo "--num_repeats must be a positive integer. Got '${num_repeats}'."
    exit 1
fi
if [[ -n "${n_sweeps}" ]] && (( n_sweeps <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps}'."
    exit 1
fi
if [[ -n "${request_cpus}" ]] && (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if (( aggregate_request_cpus <= 0 )); then
    echo "--aggregate_request_cpus must be a positive integer. Got '${aggregate_request_cpus}'."
    exit 1
fi
if (( julia_num_procs_aggregate <= 0 )); then
    echo "--julia_num_procs_aggregate must be a positive integer. Got '${julia_num_procs_aggregate}'."
    exit 1
fi
if (( replica_retries < 0 )); then
    echo "--replica_retries must be a non-negative integer. Got '${replica_retries}'."
    exit 1
fi
for include_raw_dir in "${include_raw_dirs[@]}"; do
    if [[ -z "${include_raw_dir}" || ! -d "${include_raw_dir}" ]]; then
        echo "--include_raw_dir is invalid: ${include_raw_dir}"
        exit 1
    fi
done

run_info_path="$(find_run_info_by_run_id "${run_id}" "${mode}")" || {
    echo "Could not resolve run_info.txt for run_id='${run_id}' mode='${mode}'."
    exit 1
}

if [[ -z "${n_sweeps}" ]]; then
    n_sweeps="$(read_run_info_value "${run_info_path}" "n_sweeps")"
    if ! [[ "${n_sweeps}" =~ ^[0-9]+$ ]] || (( n_sweeps <= 0 )); then
        echo "Could not infer a valid n_sweeps from ${run_info_path}; pass --n_sweeps explicitly."
        exit 1
    fi
fi

if [[ -z "${request_cpus}" ]]; then
    request_cpus="$(read_run_info_value "${run_info_path}" "request_cpus")"
    if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
        request_cpus="1"
    fi
fi

if [[ -z "${request_memory}" ]]; then
    request_memory="$(read_run_info_value "${run_info_path}" "request_memory")"
    if [[ -z "${request_memory}" ]]; then
        request_memory="2 GB"
    fi
fi

if [[ -z "${job_label}" ]]; then
    d_values_slug="$(printf "%s" "${d_values_csv}" | sed -E 's/[[:space:]]+//g; s/,/_/g')"
    job_label="topup_d${d_values_slug}_nr${num_repeats}"
    job_label="$(sanitize_token "${job_label}")"
fi

echo "Submitting top-up add-repeats batch:"
echo "  run_info=${run_info_path}"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  d_values=${d_values_csv}"
echo "  num_repeats=${num_repeats}"
echo "  n_sweeps=${n_sweeps}"
echo "  request_cpus=${request_cpus}"
echo "  request_memory=${request_memory}"
echo "  aggregate_request_cpus=${aggregate_request_cpus}"
echo "  julia_num_procs_aggregate=${julia_num_procs_aggregate}"
echo "  replica_retries=${replica_retries}"
echo "  performance_mode=true"
echo "  estimate_runtime=${estimate_runtime}"
echo "  estimate_sample_size=${estimate_sample_size}"
echo "  dag_maxjobs=${dag_maxjobs}"
echo "  dag_maxidle=${dag_maxidle}"
echo "  aggregated_subdir=aggregated"
echo "  job_label=${job_label}"
if [[ -n "${batch_name}" ]]; then
    echo "  batch_name=${batch_name}"
fi
if (( ${#include_raw_dirs[@]} > 0 )); then
    echo "  include_raw_dirs=$(IFS=:; echo "${include_raw_dirs[*]}")"
fi
if [[ "${no_submit}" == "true" ]]; then
    echo "  no_submit=true"
fi

submit_cmd=(
    bash "${ADD_REPEATS_SCRIPT}"
    --run_id "${run_id}"
    --mode "${mode}"
    --d_values "${d_values_csv}"
    --n_sweeps "${n_sweeps}"
    --num_repeats "${num_repeats}"
    --request_cpus "${request_cpus}"
    --request_memory "${request_memory}"
    --aggregate_request_cpus "${aggregate_request_cpus}"
    --julia_num_procs_aggregate "${julia_num_procs_aggregate}"
    --replica_retries "${replica_retries}"
    --estimate_sample_size "${estimate_sample_size}"
    --aggregated_subdir "aggregated"
    --dag_maxjobs "${dag_maxjobs}"
    --dag_maxidle "${dag_maxidle}"
    --job_label "${job_label}"
)
if [[ "${estimate_runtime}" == "true" ]]; then
    submit_cmd+=(--estimate_runtime)
fi
if [[ -n "${batch_name}" ]]; then
    submit_cmd+=(--batch_name "${batch_name}")
fi
for include_raw_dir in "${include_raw_dirs[@]}"; do
    submit_cmd+=(--extra_raw_dir "${include_raw_dir}")
done
if [[ "${no_submit}" == "true" ]]; then
    submit_cmd+=(--no_submit)
fi

"${submit_cmd[@]}"
