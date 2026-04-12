#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_ssep_single_center_bond_L256_dag.sh --num_replicas <int> [options]

Required:
  --num_replicas <int>         Number of independent single-core replica jobs

Optional:
  --rho <float>               Density rho0 for the run batch (default: 0.5)
  --run_id <id>               Explicit run id for the batch
  --request_memory <value>    Replica and aggregate request_memory (default: "5 GB")
  --request_cpus <int>        Replica request_cpus (default: 1)
  --aggregate_request_cpus    Aggregate request_cpus (default: 1)
  --no_submit                 Generate files only; do not call condor_submit_dag
  -h, --help                  Show this help

Behavior:
  - Uses the fixed production setup:
      ctmc_1d, L=256, D=1, f=1, ffr=1,
      selected cuts [8, 16, 32, 64, 128] with automatic mirrored counterparts,
      performance mode, n_sweeps=5e8, warmup_sweeps=1e5
  - Generates rho-specific runtime config, description, and run labels
  - Generates one Condor submit file per replica, each requesting a single core
  - Generates one aggregation child job after all replicas complete
  - Creates a run folder under runs/ssep/single_center_bond/production/<run_id>
    with configs, submit files, logs, states, aggregated, manifest, and run_info
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
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_ssep_replicas_from_tags.sh"
SOURCE_CONFIG_PATH="${REPO_ROOT}/configuration_files/ssep_ctmc_single_center_bond_L256_collapse_template.yaml"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing runner script: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate helper: ${AGGREGATE_SCRIPT}"
    exit 1
fi
if [[ ! -f "${SOURCE_CONFIG_PATH}" ]]; then
    echo "Missing source config: ${SOURCE_CONFIG_PATH}"
    exit 1
fi
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

ensure_cluster_shared_dir_permissions() {
    local path="$1"
    local mode="$2"
    chmod "${mode}" "${path}" 2>/dev/null || true
}

num_replicas=""
rho="0.5"
run_id=""
request_memory="5 GB"
request_cpus="1"
aggregate_request_cpus="1"
no_submit="${NO_SUBMIT:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --rho)
            rho="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --request_memory)
            request_memory="${2:-}"
            shift 2
            ;;
        --request_cpus)
            request_cpus="${2:-}"
            shift 2
            ;;
        --aggregate_request_cpus)
            aggregate_request_cpus="${2:-}"
            shift 2
            ;;
        --no_submit)
            no_submit="true"
            shift 1
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

for numeric_name in num_replicas request_cpus aggregate_request_cpus; do
    value="${!numeric_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "--${numeric_name} must be a positive integer. Got '${value}'."
        exit 1
    fi
done

if ! [[ "${rho}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]; then
    echo "--rho must be a positive decimal number less than 1. Got '${rho}'."
    exit 1
fi
if ! awk -v rho="${rho}" 'BEGIN { exit !(rho > 0.0 && rho < 1.0) }'; then
    echo "--rho must satisfy 0 < rho < 1. Got '${rho}'."
    exit 1
fi

rho_display="$(awk -v rho="${rho}" 'BEGIN { printf "%.6g", rho }')"
rho_slug="$(printf "%s" "${rho_display}" | sed -E 's/[^0-9]+//g')"
if [[ -z "${rho_slug}" ]]; then
    echo "Failed to derive rho slug from '${rho_display}'."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${run_id}" ]]; then
    run_label="ssep_ctmc_single_center_bond_L256_rho${rho_slug}_ns500000000_nr${num_replicas}_dag"
    run_id="$(slugify "${run_label}")_${timestamp}"
fi

description_name="ssep_ctmc_1d_single_center_bond_L256_rho${rho_slug}_collapse"

run_root="${REPO_ROOT}/runs/ssep/single_center_bond/production/${run_id}"
config_dir="${run_root}/configs"
submit_dir="${run_root}/submit"
log_dir="${run_root}/logs"
state_dir="${run_root}/states"
aggregated_dir="${run_root}/aggregated"
run_info="${run_root}/run_info.txt"
manifest="${run_root}/manifest.csv"
registry_file="${REPO_ROOT}/runs/ssep/single_center_bond/run_registry.csv"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${state_dir}" "${aggregated_dir}"
ensure_cluster_shared_dir_permissions "${run_root}" 755
ensure_cluster_shared_dir_permissions "${config_dir}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777
ensure_cluster_shared_dir_permissions "${state_dir}" 1777
ensure_cluster_shared_dir_permissions "${aggregated_dir}" 1777

runtime_config="${config_dir}/${description_name}.yaml"
save_dir_line="save_dir: \"${state_dir}\""
rho_line="ρ₀: ${rho_display}"
description_line="description: \"${description_name}\""
awk -v save_dir_line="${save_dir_line}" -v rho_line="${rho_line}" -v description_line="${description_line}" '
BEGIN {
    seen_save=0
    seen_rho=0
    seen_description=0
}
{
    if ($0 ~ /^save_dir:[[:space:]]*/) {
        print save_dir_line
        seen_save=1
        next
    }
    if ($0 ~ /^ρ₀:[[:space:]]*/) {
        print rho_line
        seen_rho=1
        next
    }
    if ($0 ~ /^description:[[:space:]]*/) {
        print description_line
        seen_description=1
        next
    }
    print
}
END {
    if (!seen_rho) print rho_line
    if (!seen_description) print description_line
    if (!seen_save) print save_dir_line
}' "${SOURCE_CONFIG_PATH}" > "${runtime_config}"

save_tag_base="${run_id}"
replica_tag_prefix="replica_${save_tag_base}_r"
aggregate_run_id="aggregated_${save_tag_base}"
job_batch_name="${JOB_BATCH_NAME:-${run_id}}"

dag_file="${submit_dir}/ssep_single_center_bond_production.dag"
aggregate_submit_file="${submit_dir}/ssep_single_center_bond_aggregate.sub"
aggregate_output_file="${log_dir}/ssep_single_center_bond_aggregate.out"
aggregate_error_file="${log_dir}/ssep_single_center_bond_aggregate.err"
aggregate_log_file="${log_dir}/ssep_single_center_bond_aggregate.log"

: > "${dag_file}"
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag" > "${manifest}"

replica_job_ids=()
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_submit_file="${submit_dir}/ssep_single_center_bond_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/ssep_single_center_bond_r${replica_idx}.out"
    replica_error_file="${log_dir}/ssep_single_center_bond_r${replica_idx}.err"
    replica_log_file="${log_dir}/ssep_single_center_bond_r${replica_idx}.log"
    replica_tag="${replica_tag_prefix}${replica_idx}"
    replica_runner_arguments="${RUNNER_SCRIPT} ${runtime_config} --save_tag ${replica_tag}"

    cat > "${replica_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${replica_runner_arguments}
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

aggregate_arguments="${AGGREGATE_SCRIPT} --config ${runtime_config} --state_dir ${state_dir} --aggregated_dir ${aggregated_dir} --num_replicas ${num_replicas} --replica_tag_prefix ${replica_tag_prefix} --save_tag ${aggregate_run_id}"
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

printf "JOB AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
printf "PARENT %s CHILD AGG\n" "${replica_job_ids[*]}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s\n" \
    "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${aggregate_run_id}" \
    >> "${manifest}"
dag_append_final_notification_node "${dag_file}" "${submit_dir}" "${log_dir}" "${run_root}" "${run_id}" "ssep_single_center_bond_production" "${REPO_ROOT}"

cluster_id=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated DAG but not submitting: ${dag_file}"
    cluster_id="NO_SUBMIT"
else
    submit_output="$(condor_submit_dag "${dag_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=production
simulation=ssep_single_center_bond_ctmc_1d
L=256
rho0=${rho_display}
D=1.0
force_strength=1.0
ffr=1.0
n_sweeps=500000000
warmup_sweeps=100000
num_replicas=${num_replicas}
replica_strategy=dag
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
job_batch_name=${job_batch_name}
run_root=${run_root}
config_dir=${config_dir}
submit_dir=${submit_dir}
log_dir=${log_dir}
state_dir=${state_dir}
aggregated_dir=${aggregated_dir}
source_config=${SOURCE_CONFIG_PATH}
runtime_config=${runtime_config}
dag_file=${dag_file}
manifest=${manifest}
replica_tag_prefix=${replica_tag_prefix}
aggregate_run_id=${aggregate_run_id}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
cluster_id=${cluster_id}
correlation_cut_offsets=8,16,32,64,128
EOF

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,warmup_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,config_path,aggregate_run_id" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "production" "256" "${rho_display}" "500000000" "100000" "${num_replicas}" \
    "${request_cpus}" "${request_memory}" "${run_root}" "${submit_dir}" "${log_dir}" "${state_dir}" "${runtime_config}" "${aggregate_run_id}" \
    >> "${registry_file}"

echo "SSEP DAG submission prepared."
echo "  run_id=${run_id}"
echo "  rho0=${rho_display}"
echo "  aggregate_run_id=${aggregate_run_id}"
echo "  dag_file=${dag_file}"
echo "  run_info=${run_info}"
echo "  manifest=${manifest}"
echo "  state_dir=${state_dir}"
echo "  aggregated_dir=${aggregated_dir}"
echo "  cluster_id=${cluster_id}"
