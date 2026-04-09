#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_active_objects_histogram_dag.sh \
      --config <path> \
      --num_replicas <int> \
      --n_sweeps <int> \
      --tr <int> \
      [--run_id <token>] \
      [--request_cpus <int>] \
      [--request_memory <value>] \
      [--aggregate_request_cpus <int>] \
      [--max_sweep <int>] \
      [--no_submit]

Behavior:
  - Creates a run folder under runs/active_objects/steady_state_histograms/<run_id>/
  - Copies the config into a cluster/runtime variant that writes states into run_root/states/
  - Launches one active-object replica per DAG node with save tags:
      replica_<run_id>_r<idx>
  - Launches one final aggregation node that exports:
      P_object1_ss(x), P_object2_ss(x), and P_distance_ss(d)
    into run_root/histograms/ using whatever saved states finished successfully
  - If you want to add repeats to an existing run_id, use:
      cluster_scripts/submit_active_objects_add_states_to_histograms.sh
    The --run_id flag here only names a fresh run; it is not a top-up mechanism.
  - Plotting is always disabled on the cluster path. Generate plots locally from
    the saved histogram artifacts if needed.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_active_objects_from_config.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_active_object_histograms_from_tags.sh"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing active-object runner wrapper: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing histogram aggregate helper: ${AGGREGATE_SCRIPT}"
    exit 1
fi
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

config_path=""
num_replicas=""
n_sweeps_override=""
run_id=""
request_cpus="1"
request_memory="5 GB"
aggregate_request_cpus="1"
tr_sweeps=""
max_sweep=""
no_plot="true"
no_submit="false"
registry_file="${REPO_ROOT}/runs/active_objects/steady_state_histograms/run_registry.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config_path="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps_override="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
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
        --tr|--min_sweep)
            tr_sweeps="${2:-}"
            shift 2
            ;;
        --max_sweep)
            max_sweep="${2:-}"
            shift 2
            ;;
        --plot_per_run)
            echo "--plot_per_run is not supported in the cluster DAG. Plot locally from saved histogram artifacts."
            exit 1
            ;;
        --no_plot)
            no_plot="true"
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

if [[ -z "${config_path}" || -z "${num_replicas}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi
if [[ -z "${n_sweeps_override}" ]]; then
    echo "--n_sweeps is required."
    usage
    exit 1
fi
if [[ -z "${tr_sweeps}" ]]; then
    echo "--tr is required."
    usage
    exit 1
fi

config_path="$(realpath "${config_path}")"
if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi
if [[ -n "${n_sweeps_override}" ]] && { ! [[ "${n_sweeps_override}" =~ ^[0-9]+$ ]] || (( n_sweeps_override <= 0 )); }; then
    echo "--n_sweeps must be a positive integer when provided. Got '${n_sweeps_override}'."
    exit 1
fi
for numeric_name in request_cpus aggregate_request_cpus; do
    numeric_value="${!numeric_name}"
    if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
        echo "--${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done
if [[ -n "${tr_sweeps}" ]] && ! [[ "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    echo "--tr must be an integer. Got '${tr_sweeps}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer. Got '${max_sweep}'."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
config_stem="$(basename "${config_path}")"
config_stem="${config_stem%.*}"

read_config_value() {
    local file_path="$1"
    local key="$2"
    awk -F: -v wanted="${key}" '
        /^[[:space:]]*#/ {next}
        $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
            value = substr($0, index($0, ":") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"/, "", value); gsub(/"$/, "", value)
            print value
            exit
        }' "${file_path}" || true
}

motion_scheme="$(read_config_value "${config_path}" "object_motion_scheme")"
refresh_sweeps_cfg="$(read_config_value "${config_path}" "object_refresh_sweeps")"
memory_sweeps_cfg="$(read_config_value "${config_path}" "object_memory_sweeps")"
hop_probability_cfg="$(read_config_value "${config_path}" "object_kappa")"
motion_token=""
case "${motion_scheme}" in
    hard_refresh)
        if [[ -n "${refresh_sweeps_cfg}" ]]; then
            refresh_sweeps_token="${refresh_sweeps_cfg%.*}"
            motion_token="_oref${refresh_sweeps_token}"
        fi
        ;;
    exponential_memory)
        if [[ -n "${memory_sweeps_cfg}" ]]; then
            memory_sweeps_token="${memory_sweeps_cfg%.*}"
            motion_token="_omem${memory_sweeps_token}"
        fi
        ;;
    per_hop_probability|per_hop|hop_probability|hop_triggered|hop)
        if [[ -n "${hop_probability_cfg}" ]]; then
            hop_probability_token="${hop_probability_cfg//+/}"
            motion_token="_ohop${hop_probability_token}"
        fi
        ;;
esac

if [[ -z "${run_id}" ]]; then
    run_id="${config_stem}${motion_token}_nr${num_replicas}_hist_${timestamp}"
fi
if ! [[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run_id must match [A-Za-z0-9._-]+. Got '${run_id}'."
    exit 1
fi

run_root="${REPO_ROOT}/runs/active_objects/steady_state_histograms/${run_id}"
if [[ -e "${run_root}" ]]; then
    echo "Run root already exists: ${run_root}"
    echo "submit_active_objects_histogram_dag.sh creates fresh runs only."
    echo "To add repeats on top of an existing run, use:"
    echo "  bash cluster_scripts/submit_active_objects_add_states_to_histograms.sh --run_id ${run_id} --nr <count> --ns <sweeps>"
    exit 1
fi
effective_n_sweeps="${n_sweeps_override}"
if [[ -n "${effective_n_sweeps}" && "${effective_n_sweeps}" =~ ^[0-9]+$ && "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    if (( tr_sweeps >= effective_n_sweeps )); then
        echo "--tr (${tr_sweeps}) must be smaller than --n_sweeps (${effective_n_sweeps})."
        exit 1
    fi
fi
config_dir="${run_root}/configs"
submit_dir="${run_root}/submit"
log_dir="${run_root}/logs"
state_dir="${run_root}/states"
hist_dir="${run_root}/histograms"
runtime_config="${config_dir}/$(basename "${config_path}")"
run_info="${run_root}/run_info.txt"
manifest="${run_root}/manifest.csv"
dag_file="${submit_dir}/active_objects_histograms.dag"
aggregate_submit_file="${submit_dir}/active_objects_histograms_aggregate.sub"
aggregate_output_file="${log_dir}/active_objects_histograms_aggregate.out"
aggregate_error_file="${log_dir}/active_objects_histograms_aggregate.err"
aggregate_log_file="${log_dir}/active_objects_histograms_aggregate.log"
aggregate_save_tag="aggregated_${run_id}"
job_batch_name="${run_id}"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${state_dir}" "${hist_dir}"

prepare_runtime_config() {
    local src="$1"
    local dest="$2"
    local forced_save_dir="$3"
    local forced_n_sweeps="${4:-}"

    local save_dir_line n_sweeps_line performance_mode_line cluster_mode_line plot_final_line
    local save_final_plot_line live_plot_line save_live_plot_line
    save_dir_line="save_dir: \"${forced_save_dir}\""
    n_sweeps_line=""
    if [[ -n "${forced_n_sweeps}" ]]; then
        n_sweeps_line="n_sweeps: ${forced_n_sweeps}"
    fi
    performance_mode_line="performance_mode: true"
    cluster_mode_line="cluster_mode: true"
    plot_final_line="plot_final: false"
    save_final_plot_line="save_final_plot: false"
    live_plot_line="live_plot: false"
    save_live_plot_line="save_live_plot: false"

    awk \
        -v save_dir_line="${save_dir_line}" \
        -v n_sweeps_line="${n_sweeps_line}" \
        -v performance_mode_line="${performance_mode_line}" \
        -v cluster_mode_line="${cluster_mode_line}" \
        -v plot_final_line="${plot_final_line}" \
        -v save_final_plot_line="${save_final_plot_line}" \
        -v live_plot_line="${live_plot_line}" \
        -v save_live_plot_line="${save_live_plot_line}" '
        BEGIN {
            seen_save = seen_perf = seen_cluster = 0
            seen_n_sweeps = 0
            seen_plot_final = seen_save_plot = seen_live_plot = seen_save_live_plot = 0
        }
        {
            if ($0 ~ /^[[:space:]]*save_dir:[[:space:]]*/) {
                print save_dir_line
                seen_save = 1
                next
            }
            if (n_sweeps_line != "" && $0 ~ /^[[:space:]]*n_sweeps:[[:space:]]*/) {
                print n_sweeps_line
                seen_n_sweeps = 1
                next
            }
            if ($0 ~ /^[[:space:]]*performance_mode:[[:space:]]*/) {
                print performance_mode_line
                seen_perf = 1
                next
            }
            if ($0 ~ /^[[:space:]]*cluster_mode:[[:space:]]*/) {
                print cluster_mode_line
                seen_cluster = 1
                next
            }
            if ($0 ~ /^[[:space:]]*plot_final:[[:space:]]*/) {
                print plot_final_line
                seen_plot_final = 1
                next
            }
            if ($0 ~ /^[[:space:]]*save_final_plot:[[:space:]]*/) {
                print save_final_plot_line
                seen_save_plot = 1
                next
            }
            if ($0 ~ /^[[:space:]]*live_plot:[[:space:]]*/) {
                print live_plot_line
                seen_live_plot = 1
                next
            }
            if ($0 ~ /^[[:space:]]*save_live_plot:[[:space:]]*/) {
                print save_live_plot_line
                seen_save_live_plot = 1
                next
            }
            print
        }
        END {
            if (!seen_save) print save_dir_line
            if (n_sweeps_line != "" && !seen_n_sweeps) print n_sweeps_line
            if (!seen_perf) print performance_mode_line
            if (!seen_cluster) print cluster_mode_line
            if (!seen_plot_final) print plot_final_line
            if (!seen_save_plot) print save_final_plot_line
            if (!seen_live_plot) print live_plot_line
            if (!seen_save_live_plot) print save_live_plot_line
        }' "${src}" > "${dest}"
}

prepare_runtime_config "${config_path}" "${runtime_config}" "${state_dir}" "${n_sweeps_override}"

: > "${dag_file}"
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag" > "${manifest}"

replica_job_ids=()
replica_tag_prefix="replica_${run_id}_r"
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_submit_file="${submit_dir}/active_objects_hist_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/active_objects_hist_replica_${replica_idx}.out"
    replica_error_file="${log_dir}/active_objects_hist_replica_${replica_idx}.err"
    replica_log_file="${log_dir}/active_objects_hist_replica_${replica_idx}.log"
    replica_tag="${replica_tag_prefix}${replica_idx}"

    cat > "${replica_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${RUNNER_SCRIPT} ${runtime_config} --save_tag ${replica_tag} --performance_mode
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${replica_output_file}
error      = ${replica_error_file}
log        = ${replica_log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${job_batch_name}
queue
EOF

    job_id="R${replica_idx}"
    replica_job_ids+=("${job_id}")
    printf "JOB %s %s\n" "${job_id}" "${replica_submit_file}" >> "${dag_file}"
    printf "replica,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${replica_submit_file}" "${replica_output_file}" "${replica_error_file}" "${replica_log_file}" "${replica_tag}" \
        >> "${manifest}"
done

aggregate_arguments="${AGGREGATE_SCRIPT} --state_dir ${state_dir} --output_dir ${hist_dir} --save_tag aggregated_${run_id} --all_states_recursive --min_sweep ${tr_sweeps}"
if [[ -n "${max_sweep}" ]]; then
    aggregate_arguments="${aggregate_arguments} --max_sweep ${max_sweep}"
fi
aggregate_arguments="${aggregate_arguments} --no_plot"

cat > "${aggregate_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${aggregate_arguments}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${aggregate_output_file}
error      = ${aggregate_error_file}
log        = ${aggregate_log_file}
request_cpus = ${aggregate_request_cpus}
request_memory = ${request_memory}
batch_name = ${job_batch_name}
queue
EOF

printf "FINAL AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s\n" \
    "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${aggregate_save_tag}" \
    >> "${manifest}"

dag_append_post_notification_script "${dag_file}" "AGG" "${submit_dir}" "${log_dir}" "${run_root}" "${run_id}" "active_objects_histograms" "${REPO_ROOT}"

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=production
config_path=${config_path}
runtime_config=${runtime_config}
run_root=${run_root}
state_dir=${state_dir}
histogram_dir=${hist_dir}
submit_dir=${submit_dir}
log_dir=${log_dir}
manifest=${manifest}
num_replicas=${num_replicas}
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
effective_n_sweeps=${effective_n_sweeps}
tr_sweeps=${tr_sweeps}
max_sweep=${max_sweep}
no_plot=${no_plot}
dag_file=${dag_file}
aggregate_save_tag=${aggregate_save_tag}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
EOF

cluster_id=""
submit_output=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated DAG but not submitting: ${dag_file}"
    cluster_id="NO_SUBMIT"
else
    submit_output="$(condor_submit_dag "${dag_file}")"
    cluster_id="$(printf '%s\n' "${submit_output}" | sed -nE 's/.*submitted to cluster ([0-9]+).*/\1/p' | tail -n 1)"
    cluster_id="${cluster_id:-NA}"
fi

if [[ -n "${cluster_id}" ]]; then
    {
        echo "cluster_id=${cluster_id}"
        echo "submit_time=$(date +%Y-%m-%dT%H:%M:%S)"
    } >> "${run_info}"
fi

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,tr_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,histogram_dir,config_path,aggregate_save_tag" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" \
    "${run_id}" \
    "production" \
    "$(read_config_value "${config_path}" "L")" \
    "$(read_config_value "${config_path}" "ρ₀")" \
    "${effective_n_sweeps}" \
    "${tr_sweeps}" \
    "${num_replicas}" \
    "${request_cpus}" \
    "${request_memory}" \
    "${run_root}" \
    "${submit_dir}" \
    "${log_dir}" \
    "${state_dir}" \
    "${hist_dir}" \
    "${runtime_config}" \
    "${aggregate_save_tag}" \
    >> "${registry_file}"

echo "Prepared active-object steady-state histogram DAG:"
echo "  run_id=${run_id}"
echo "  run_root=${run_root}"
echo "  runtime_config=${runtime_config}"
echo "  state_dir=${state_dir}"
echo "  histogram_dir=${hist_dir}"
echo "  dag_file=${dag_file}"
echo "  aggregate_save_tag=${aggregate_save_tag}"
echo "  cluster_id=${cluster_id}"
if [[ -n "${submit_output}" ]]; then
    echo "${submit_output}"
fi
