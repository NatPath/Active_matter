#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_aggregate_two_force_d_saved_files.sh --run_id <run_id> [options]

Required:
  --run_id <id>                    two_force_d run_id (production or warmup_production chain id)

Aggregation options (forwarded to aggregate_two_force_d_saved_files.sh):
  --mode <auto|warmup|production|warmup_production>   default: auto
  --num_files <int>                files per d (0 means all; default: 0)
  --state_dir <path>               override state_dir
  --config_dir <path>              override config_dir
  --extra_raw_dir <path>           also include raw states from this top-level directory
                                   (repeatable)
  --only_extra_raw_inputs          do not scan top-level raw files in state_dir; aggregate only
                                   files found under --extra_raw_dir paths
  --aggregated_subdir <name>       save aggregated outputs under <state_dir>/<name> (default: aggregated)
  --exclude_aggregated_inputs      do not use existing aggregated files as discovery inputs
  --incremental_from_existing_aggregate
                                   use the current aggregate for each d as the base input and add
                                   only raw files from --extra_raw_dir paths when available
  --archive_existing_aggregates    archive existing non-partial aggregates before replacing them
  --archive_subdir <name>          archive subdirectory under aggregated_subdir (default: archive)
  --archive_stamp <token>          archive stamp token (default: current timestamp in aggregation script)
  --d_min <int>                    override d_min
  --d_max <int>                    override d_max
  --d_step <int>                   override d_step
  --d_values <csv>                 explicit comma-separated d list (for example: 96,128)
  --force                          re-aggregate existing outputs
  --dry_run                        dry run only
  --keep_going                     continue on per-d failures

Submit options:
  --request_cpus <int>             Condor request_cpus (default: 1)
  --request_memory <value>         Condor request_memory (default: "5 GB")
  --julia_num_procs_aggregate <n>  sets JULIA_NUM_PROCS_AGGREGATE in job env (default: 1)
  --batch_name <name>              Condor batch_name (default: auto)
  --no_submit                      generate files only; do not call condor_submit
  -h, --help                       show this help
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

AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_two_force_d_saved_files.sh"
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregation script: ${AGGREGATE_SCRIPT}"
    exit 1
fi

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

run_id=""
mode="auto"
num_files="0"
state_dir=""
config_dir=""
extra_raw_dirs=()
only_extra_raw_inputs="false"
aggregated_subdir="aggregated"
exclude_aggregated_inputs="false"
incremental_from_existing_aggregate="false"
archive_existing_aggregates="false"
archive_subdir="archive"
archive_stamp=""
d_min=""
d_max=""
d_step=""
d_values_csv=""
force_reaggregate="false"
dry_run="false"
keep_going="false"

request_cpus="1"
request_memory="5 GB"
julia_num_procs_aggregate="1"
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
        --num_files)
            num_files="${2:-}"
            shift 2
            ;;
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --config_dir)
            config_dir="${2:-}"
            shift 2
            ;;
        --extra_raw_dir)
            extra_raw_dirs+=("${2:-}")
            shift 2
            ;;
        --only_extra_raw_inputs)
            only_extra_raw_inputs="true"
            shift
            ;;
        --aggregated_subdir)
            aggregated_subdir="${2:-}"
            shift 2
            ;;
        --exclude_aggregated_inputs)
            exclude_aggregated_inputs="true"
            shift
            ;;
        --incremental_from_existing_aggregate)
            incremental_from_existing_aggregate="true"
            shift
            ;;
        --archive_existing_aggregates)
            archive_existing_aggregates="true"
            shift
            ;;
        --archive_subdir)
            archive_subdir="${2:-}"
            shift 2
            ;;
        --archive_stamp)
            archive_stamp="${2:-}"
            shift 2
            ;;
        --d_min)
            d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            d_step="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --force)
            force_reaggregate="true"
            shift
            ;;
        --dry_run)
            dry_run="true"
            shift
            ;;
        --keep_going)
            keep_going="true"
            shift
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
    auto|warmup|production|warmup_production)
        ;;
    *)
        echo "--mode must be one of: auto, warmup, production, warmup_production."
        exit 1
        ;;
esac

if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if ! [[ "${julia_num_procs_aggregate}" =~ ^[0-9]+$ ]] || (( julia_num_procs_aggregate <= 0 )); then
    echo "--julia_num_procs_aggregate must be a positive integer. Got '${julia_num_procs_aggregate}'."
    exit 1
fi
if [[ -z "${aggregated_subdir}" ]] || ! [[ "${aggregated_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--aggregated_subdir must match [A-Za-z0-9._-]+. Got '${aggregated_subdir}'."
    exit 1
fi
if [[ "${archive_existing_aggregates}" == "true" && -z "${aggregated_subdir}" ]]; then
    echo "--archive_existing_aggregates requires a non-empty --aggregated_subdir."
    exit 1
fi
if [[ -n "${archive_subdir}" ]] && ! [[ "${archive_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_subdir must match [A-Za-z0-9._-]+ when provided. Got '${archive_subdir}'."
    exit 1
fi
if [[ -n "${archive_stamp}" ]] && ! [[ "${archive_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_stamp must match [A-Za-z0-9._-]+ when provided. Got '${archive_stamp}'."
    exit 1
fi
if [[ -n "${num_files}" ]] && ! [[ "${num_files}" =~ ^[0-9]+$ ]]; then
    echo "--num_files must be a non-negative integer. Got '${num_files}'."
    exit 1
fi
for extra_dir in "${extra_raw_dirs[@]}"; do
    if [[ -z "${extra_dir}" || ! -d "${extra_dir}" ]]; then
        echo "--extra_raw_dir is invalid: ${extra_dir}"
        exit 1
    fi
done
if [[ "${only_extra_raw_inputs}" == "true" && ${#extra_raw_dirs[@]} -eq 0 ]]; then
    echo "--only_extra_raw_inputs requires at least one --extra_raw_dir."
    exit 1
fi
for numeric_name in d_min d_max d_step; do
    numeric_value="${!numeric_name}"
    if [[ -n "${numeric_value}" ]] && ! [[ "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "--${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done
if [[ -n "${d_values_csv}" ]] && ! [[ "${d_values_csv}" =~ ^[0-9,]+$ ]]; then
    echo "--d_values must be a comma-separated list of integers. Got '${d_values_csv}'."
    exit 1
fi
if [[ -n "${d_values_csv}" && ( -n "${d_min}" || -n "${d_max}" || -n "${d_step}" ) ]]; then
    echo "--d_values cannot be combined with --d_min/--d_max/--d_step."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_run_id="$(sanitize_token "${run_id}")"
job_root="${REPO_ROOT}/runs/two_force_d/aggregation_jobs/${safe_run_id}_${timestamp}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
mkdir -p "${submit_dir}" "${log_dir}"

if [[ -z "${batch_name}" ]]; then
    batch_name="two_force_d_aggregate_saved_${safe_run_id}"
fi

launcher_script="${submit_dir}/run_aggregate_saved_files.sh"
submit_file="${submit_dir}/aggregate_saved_files.sub"
output_file="${log_dir}/aggregate_saved_files.out"
error_file="${log_dir}/aggregate_saved_files.err"
log_file="${log_dir}/aggregate_saved_files.log"
meta_file="${job_root}/job_info.txt"

agg_args=("--run_id" "${run_id}" "--mode" "${mode}")
agg_args+=("--aggregated_subdir" "${aggregated_subdir}")
if [[ "${archive_existing_aggregates}" == "true" ]]; then
    agg_args+=("--archive_existing_aggregates")
fi
if [[ -n "${archive_subdir}" ]]; then
    agg_args+=("--archive_subdir" "${archive_subdir}")
fi
if [[ -n "${archive_stamp}" ]]; then
    agg_args+=("--archive_stamp" "${archive_stamp}")
fi
if [[ -n "${num_files}" ]]; then
    agg_args+=("--num_files" "${num_files}")
fi
if [[ -n "${state_dir}" ]]; then
    agg_args+=("--state_dir" "${state_dir}")
fi
if [[ -n "${config_dir}" ]]; then
    agg_args+=("--config_dir" "${config_dir}")
fi
for extra_raw_dir in "${extra_raw_dirs[@]}"; do
    agg_args+=("--extra_raw_dir" "${extra_raw_dir}")
done
if [[ "${only_extra_raw_inputs}" == "true" ]]; then
    agg_args+=("--only_extra_raw_inputs")
fi
if [[ -n "${d_min}" ]]; then
    agg_args+=("--d_min" "${d_min}")
fi
if [[ -n "${d_max}" ]]; then
    agg_args+=("--d_max" "${d_max}")
fi
if [[ -n "${d_step}" ]]; then
    agg_args+=("--d_step" "${d_step}")
fi
if [[ -n "${d_values_csv}" ]]; then
    agg_args+=("--d_values" "${d_values_csv}")
fi
if [[ "${force_reaggregate}" == "true" ]]; then
    agg_args+=("--force")
fi
if [[ "${exclude_aggregated_inputs}" == "true" ]]; then
    agg_args+=("--exclude_aggregated_inputs")
fi
if [[ "${incremental_from_existing_aggregate}" == "true" ]]; then
    agg_args+=("--incremental_from_existing_aggregate")
fi
if [[ "${dry_run}" == "true" ]]; then
    agg_args+=("--dry_run")
fi
if [[ "${keep_going}" == "true" ]]; then
    agg_args+=("--keep_going")
fi

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf '%q' "${REPO_ROOT}")"
    echo "export JULIA_NUM_PROCS_AGGREGATE=$(printf '%q' "${julia_num_procs_aggregate}")"
    printf "bash %q" "${AGGREGATE_SCRIPT}"
    for arg in "${agg_args[@]}"; do
        printf " %q" "${arg}"
    done
    echo
} > "${launcher_script}"
chmod +x "${launcher_script}"

cat > "${submit_file}" <<EOF
Universe   = vanilla
Executable = ${launcher_script}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${output_file}
error      = ${error_file}
log        = ${log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${batch_name}
queue
EOF

{
    echo "timestamp=${timestamp}"
    echo "run_id=${run_id}"
    echo "job_root=${job_root}"
    echo "submit_file=${submit_file}"
    echo "launcher_script=${launcher_script}"
    echo "output_file=${output_file}"
    echo "error_file=${error_file}"
    echo "log_file=${log_file}"
    echo "request_cpus=${request_cpus}"
    echo "request_memory=${request_memory}"
    echo "julia_num_procs_aggregate=${julia_num_procs_aggregate}"
    echo "mode=${mode}"
    echo "num_files=${num_files:-}"
    echo "state_dir=${state_dir:-}"
    echo "config_dir=${config_dir:-}"
    if (( ${#extra_raw_dirs[@]} > 0 )); then
        echo "extra_raw_dirs=$(IFS=:; echo "${extra_raw_dirs[*]}")"
    else
        echo "extra_raw_dirs="
    fi
    echo "only_extra_raw_inputs=${only_extra_raw_inputs}"
    echo "aggregated_subdir=${aggregated_subdir}"
    echo "exclude_aggregated_inputs=${exclude_aggregated_inputs}"
    echo "incremental_from_existing_aggregate=${incremental_from_existing_aggregate}"
    echo "archive_existing_aggregates=${archive_existing_aggregates}"
    echo "archive_subdir=${archive_subdir}"
    echo "archive_stamp=${archive_stamp:-}"
    echo "d_min=${d_min:-}"
    echo "d_max=${d_max:-}"
    echo "d_step=${d_step:-}"
    echo "d_values=${d_values_csv:-}"
    echo "force=${force_reaggregate}"
    echo "dry_run=${dry_run}"
    echo "keep_going=${keep_going}"
} > "${meta_file}"

echo "Prepared aggregation submit artifacts:"
echo "  job_root=${job_root}"
echo "  submit_file=${submit_file}"
echo "  launcher_script=${launcher_script}"
echo "  output_file=${output_file}"
echo "  error_file=${error_file}"
echo "  log_file=${log_file}"
echo "  request_cpus=${request_cpus}"
echo "  request_memory=${request_memory}"
echo "  JULIA_NUM_PROCS_AGGREGATE=${julia_num_procs_aggregate}"
echo "  aggregated_subdir=${aggregated_subdir}"
if (( ${#extra_raw_dirs[@]} > 0 )); then
    echo "  extra_raw_dirs=$(IFS=:; echo "${extra_raw_dirs[*]}")"
fi
echo "  only_extra_raw_inputs=${only_extra_raw_inputs}"
echo "  exclude_aggregated_inputs=${exclude_aggregated_inputs}"
echo "  incremental_from_existing_aggregate=${incremental_from_existing_aggregate}"
echo "  archive_existing_aggregates=${archive_existing_aggregates}"
echo "  archive_subdir=${archive_subdir}"
if [[ -n "${archive_stamp}" ]]; then
    echo "  archive_stamp=${archive_stamp}"
fi

if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated submit file but did not submit."
    echo "Submit manually with:"
    echo "  condor_submit ${submit_file}"
    exit 0
fi

submit_output="$(condor_submit "${submit_file}")"
echo "${submit_output}"
cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
cluster_id="${cluster_id:-NA}"
echo "Submitted aggregation job:"
echo "  cluster_id=${cluster_id}"
echo "  submit_file=${submit_file}"
echo "  logs=${log_dir}"
