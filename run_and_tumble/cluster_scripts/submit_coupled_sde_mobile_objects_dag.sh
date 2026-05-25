#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_coupled_sde_mobile_objects_dag.sh \
      --config <path> \
      --num_replicas <int> \
      [--run_id <token>] \
      [--request_cpus <int>] \
      [--request_memory <value>] \
      [--aggregate_request_cpus <int>] \
      [--checkpoint_interval_steps <int>] \
      [--no_submit]

Behavior:
  - Creates runs/coupled_sde_active_objects/mobile_objects/<run_id>/.
  - Runs one Condor node per mobile-object replica.
  - Final DAG node aggregates binned conditional diffusivity and P_ss diagnostics.
  - If checkpoint_interval_steps > 0, each replica atomically updates one latest checkpoint file.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_coupled_sde_active_objects_from_config.sh"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_coupled_sde_mobile_objects.sh"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"

for required in "${RUNNER_SCRIPT}" "${ANALYZE_SCRIPT}" "${DAG_NOTIFY_UTILS}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing helper script: ${required}"
        exit 1
    fi
done
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

config_path=""
num_replicas=""
run_id=""
request_cpus="1"
request_memory="5 GB"
aggregate_request_cpus="1"
checkpoint_interval_steps="${CHECKPOINT_INTERVAL_STEPS:-0}"
no_submit="false"

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
        --checkpoint_interval_steps)
            checkpoint_interval_steps="${2:-}"
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

if [[ -z "${config_path}" || -z "${num_replicas}" ]]; then
    echo "--config and --num_replicas are required."
    usage
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi
for numeric_name in request_cpus aggregate_request_cpus; do
    numeric_value="${!numeric_name}"
    if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
        echo "--${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done
if ! [[ "${checkpoint_interval_steps}" =~ ^[0-9]+$ ]]; then
    echo "--checkpoint_interval_steps must be a nonnegative integer. Got '${checkpoint_interval_steps}'."
    exit 1
fi

config_path="$(realpath "${config_path}")"
if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
config_stem="$(basename "${config_path}")"
config_stem="${config_stem%.*}"
if [[ -z "${run_id}" ]]; then
    run_id="${config_stem}_nr${num_replicas}_${timestamp}"
fi
if ! [[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run_id must match [A-Za-z0-9._-]+. Got '${run_id}'."
    exit 1
fi

run_root="${REPO_ROOT}/runs/coupled_sde_active_objects/mobile_objects/${run_id}"
if [[ -e "${run_root}" ]]; then
    echo "Run root already exists: ${run_root}"
    exit 1
fi

runtime_config_dir="${run_root}/configs"
submit_dir="${run_root}/submit"
log_dir="${run_root}/logs"
state_dir="${run_root}/states"
checkpoint_dir="${run_root}/checkpoints"
analysis_dir="${run_root}/analysis"
manifest="${run_root}/manifest.csv"
run_info="${run_root}/run_info.txt"
dag_file="${submit_dir}/coupled_sde_mobile_objects.dag"
runtime_config="${runtime_config_dir}/$(basename "${config_path}")"
aggregate_submit_file="${submit_dir}/coupled_sde_mobile_aggregate.sub"
aggregate_output_file="${log_dir}/coupled_sde_mobile_aggregate.out"
aggregate_error_file="${log_dir}/coupled_sde_mobile_aggregate.err"
aggregate_log_file="${log_dir}/coupled_sde_mobile_aggregate.log"
aggregate_save_tag="aggregated_${run_id}"

mkdir -p "${runtime_config_dir}" "${submit_dir}" "${log_dir}" "${state_dir}" "${checkpoint_dir}" "${analysis_dir}"

awk \
    -v save_dir_line="save_dir: \"${state_dir}\"" \
    -v performance_mode_line="performance_mode: true" \
    -v cluster_mode_line="cluster_mode: true" '
    BEGIN { seen_save = seen_perf = seen_cluster = 0 }
    /^[[:space:]]*save_dir:[[:space:]]*/ {
        print save_dir_line
        seen_save = 1
        next
    }
    /^[[:space:]]*performance_mode:[[:space:]]*/ {
        print performance_mode_line
        seen_perf = 1
        next
    }
    /^[[:space:]]*cluster_mode:[[:space:]]*/ {
        print cluster_mode_line
        seen_cluster = 1
        next
    }
    { print }
    END {
        if (!seen_save) print save_dir_line
        if (!seen_perf) print performance_mode_line
        if (!seen_cluster) print cluster_mode_line
    }' "${config_path}" > "${runtime_config}"

: > "${dag_file}"
echo "job_type,job_name,config_path,submit_file,output_file,error_file,log_file,save_tag,checkpoint_file" > "${manifest}"

for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    job_id="R${replica_idx}"
    replica_tag="replica_${run_id}_r${replica_idx}"
    checkpoint_file="${checkpoint_dir}/${replica_tag}.checkpoint.jld2"
    submit_file="${submit_dir}/mobile_r${replica_idx}.sub"
    output_file="${log_dir}/mobile_r${replica_idx}.out"
    error_file="${log_dir}/mobile_r${replica_idx}.err"
    log_file="${log_dir}/mobile_r${replica_idx}.log"
    runner_arguments="${RUNNER_SCRIPT} ${runtime_config} --save_tag ${replica_tag} --performance_mode"
    if (( checkpoint_interval_steps > 0 )); then
        runner_arguments="${runner_arguments} --checkpoint_path ${checkpoint_file} --checkpoint_interval_steps ${checkpoint_interval_steps}"
    fi

    cat > "${submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${runner_arguments}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${output_file}
error      = ${error_file}
log        = ${log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${run_id}
queue
EOF

    printf "JOB %s %s\n" "${job_id}" "${submit_file}" >> "${dag_file}"
    printf "replica,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${runtime_config}" "${submit_file}" "${output_file}" "${error_file}" "${log_file}" "${replica_tag}" "${checkpoint_file}" >> "${manifest}"
done

cat > "${aggregate_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${ANALYZE_SCRIPT} --state_dir ${state_dir} --output_dir ${analysis_dir} --save_tag ${aggregate_save_tag} --no_plot
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${aggregate_output_file}
error      = ${aggregate_error_file}
log        = ${aggregate_log_file}
request_cpus = ${aggregate_request_cpus}
request_memory = ${request_memory}
batch_name = ${run_id}
queue
EOF

printf "FINAL AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s,%s,%s\n" \
    "${runtime_config}" "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${aggregate_save_tag}" "" >> "${manifest}"

dag_append_post_notification_script "${dag_file}" "AGG" "${submit_dir}" "${log_dir}" "${run_root}" "${run_id}" "coupled_sde_mobile_objects" "${REPO_ROOT}"

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=mobile_objects
config_path=${config_path}
runtime_config=${runtime_config}
run_root=${run_root}
state_dir=${state_dir}
checkpoint_dir=${checkpoint_dir}
analysis_dir=${analysis_dir}
submit_dir=${submit_dir}
log_dir=${log_dir}
manifest=${manifest}
num_replicas=${num_replicas}
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
checkpoint_interval_steps=${checkpoint_interval_steps}
dag_file=${dag_file}
aggregate_save_tag=${aggregate_save_tag}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
EOF

cat > "${run_root}/cluster_followup_commands.txt" <<EOF
# Inspect progress on the cluster filesystem:
condor_q -nobatch
find "${state_dir}" -maxdepth 1 -type f -name '*.jld2' | wc -l
find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.checkpoint.jld2' | wc -l

# Aggregate completed replicas at any time:
bash cluster_scripts/analyze_coupled_sde_mobile_objects.sh --state_dir "${state_dir}" --output_dir "${analysis_dir}" --save_tag "${aggregate_save_tag}" --no_plot
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

{
    echo "cluster_id=${cluster_id}"
    echo "submit_time=$(date +%Y-%m-%dT%H:%M:%S)"
} >> "${run_info}"

registry_file="${REPO_ROOT}/runs/coupled_sde_active_objects/run_registry.csv"
mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,num_configs,num_replicas,request_cpus,request_memory,run_root,state_dir,analysis_dir,config_dir,aggregate_save_tag" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "mobile_objects" "1" "${num_replicas}" \
    "${request_cpus}" "${request_memory}" "${run_root}" "${state_dir}" "${analysis_dir}" "$(dirname "${config_path}")" "${aggregate_save_tag}" >> "${registry_file}"

echo "Prepared coupled-SDE mobile-object DAG:"
echo "  run_id=${run_id}"
echo "  run_root=${run_root}"
echo "  state_dir=${state_dir}"
echo "  analysis_dir=${analysis_dir}"
echo "  dag_file=${dag_file}"
echo "  cluster_id=${cluster_id}"
if [[ -n "${submit_output}" ]]; then
    echo "${submit_output}"
fi
