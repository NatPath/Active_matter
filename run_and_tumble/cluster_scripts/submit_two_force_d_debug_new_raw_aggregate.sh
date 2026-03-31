#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_two_force_d_debug_new_raw_aggregate.sh --run_id <run_id> --raw_dir <path> [options]

Purpose:
  Submit a clean debug aggregation job that aggregates only explicitly provided raw repeat-batch
  directories, without touching the main states/aggregated outputs.

Required:
  --run_id <id>                    two_force_d run_id
  --raw_dir <path>                 repeat-batch directory containing raw saved states (repeatable)

Options:
  --mode <auto|warmup|production|warmup_production>
                                   default: auto
  --d_values <csv>                 default: 96,128
  --label <token>                  label used in the debug output subdir (default: new_raw_only)
  --aggregated_subdir <name>       explicit output subdir under states/ (default: auto)
  --request_cpus <int>             default: 1
  --request_memory <value>         default: 4 GB
  --julia_num_procs_aggregate <n>  default: 1
  --batch_name <name>              optional explicit Condor batch name
  --keep_going                     continue on per-d failures
  --no_submit                      prepare only; do not submit
  -h, --help                       show this help

Example:
  bash cluster_scripts/submit_two_force_d_debug_new_raw_aggregate.sh \
    --run_id <run_id> \
    --raw_dir /path/to/repeat_batch_A \
    --raw_dir /path/to/repeat_batch_B \
    --d_values 96,128 \
    --label post_t8e9_debug
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_AGG_SCRIPT="${SCRIPT_DIR}/submit_aggregate_two_force_d_saved_files.sh"
if [[ ! -f "${SUBMIT_AGG_SCRIPT}" ]]; then
    echo "Missing submit helper: ${SUBMIT_AGG_SCRIPT}"
    exit 1
fi

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

run_id=""
mode="auto"
d_values_csv="96,128"
label="new_raw_only"
aggregated_subdir=""
request_cpus="1"
request_memory="4 GB"
julia_num_procs_aggregate="1"
batch_name=""
keep_going="false"
no_submit="false"
raw_dirs=()

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
        --label)
            label="${2:-}"
            shift 2
            ;;
        --aggregated_subdir)
            aggregated_subdir="${2:-}"
            shift 2
            ;;
        --raw_dir|--extra_raw_dir)
            raw_dirs+=("${2:-}")
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
        --julia_num_procs_aggregate)
            julia_num_procs_aggregate="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
            shift 2
            ;;
        --keep_going)
            keep_going="true"
            shift
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
if (( ${#raw_dirs[@]} == 0 )); then
    echo "At least one --raw_dir is required."
    usage
    exit 1
fi
for raw_dir in "${raw_dirs[@]}"; do
    if [[ -z "${raw_dir}" || ! -d "${raw_dir}" ]]; then
        echo "Invalid --raw_dir: ${raw_dir}"
        exit 1
    fi
done
if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if ! [[ "${julia_num_procs_aggregate}" =~ ^[0-9]+$ ]] || (( julia_num_procs_aggregate <= 0 )); then
    echo "--julia_num_procs_aggregate must be a positive integer. Got '${julia_num_procs_aggregate}'."
    exit 1
fi
if [[ -n "${aggregated_subdir}" ]] && ! [[ "${aggregated_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--aggregated_subdir must match [A-Za-z0-9._-]+. Got '${aggregated_subdir}'."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_label="$(sanitize_token "${label}")"
if [[ -z "${aggregated_subdir}" ]]; then
    aggregated_subdir="debug_new_raw_${safe_label}_${timestamp}"
fi
if [[ -z "${batch_name}" ]]; then
    batch_name="two_force_d_debug_new_raw_${safe_label}"
fi

submit_cmd=(
    bash "${SUBMIT_AGG_SCRIPT}"
    --run_id "${run_id}"
    --mode "${mode}"
    --d_values "${d_values_csv}"
    --num_files 0
    --aggregated_subdir "${aggregated_subdir}"
    --exclude_aggregated_inputs
    --only_extra_raw_inputs
    --request_cpus "${request_cpus}"
    --request_memory "${request_memory}"
    --julia_num_procs_aggregate "${julia_num_procs_aggregate}"
    --batch_name "${batch_name}"
)
for raw_dir in "${raw_dirs[@]}"; do
    submit_cmd+=(--extra_raw_dir "${raw_dir}")
done
if [[ "${keep_going}" == "true" ]]; then
    submit_cmd+=(--keep_going)
fi
if [[ "${no_submit}" == "true" ]]; then
    submit_cmd+=(--no_submit)
fi

echo "Submitting debug raw-only aggregation:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  d_values=${d_values_csv}"
echo "  aggregated_subdir=${aggregated_subdir}"
echo "  request_cpus=${request_cpus}"
echo "  request_memory=${request_memory}"
echo "  JULIA_NUM_PROCS_AGGREGATE=${julia_num_procs_aggregate}"
echo "  batch_name=${batch_name}"
echo "  raw_dirs=$(IFS=:; echo "${raw_dirs[*]}")"
if [[ "${no_submit}" == "true" ]]; then
    echo "  no_submit=true"
fi
echo "Outputs will be written under:"
echo "  <run state_dir>/${aggregated_subdir}/"

"${submit_cmd[@]}"
