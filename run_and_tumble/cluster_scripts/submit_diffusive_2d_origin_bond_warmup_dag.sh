#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_diffusive_2d_origin_bond_warmup_dag.sh \
      --L <int> --rho <float> --n_sweeps <int> --nr <int> [options]

Defaults:
  L=64, rho=1000, force=1, ffr=1, n_sweeps=20000, num_replicas=600

Options:
  --L <int>                    Even 2D lattice size (default: 64)
  --rho <float>                Density rho0 (default: 1000)
  --force_strength <float>     Bond force magnitude (default: 1.0)
  --ffr <float>                Fluctuating forcing rate (default: 1.0)
  --n_sweeps <int>             Warmup sweeps per replica (default: 20000)
  --nr <int>                   Number of independent warmup replicas (default: 600)
  --num_replicas <int>         Alias for --nr
  --run_id <id>                Explicit run id
  --request_cpus <int>         Replica request_cpus (default: 1)
  --request_memory <value>     Replica request_memory (default: "6 GB")
  --replica_retries <int>      DAG retry count for replica nodes (default: 2)
  --dag_maxjobs <int>          Optional DAGMan replica throttle, 0 disables (default: 0)
  --batch_name <name>          Condor batch_name (default: run id)
  --no_submit                  Generate files only; do not call condor_submit_dag
  -h, --help                   Show this help

Behavior:
  - Generates one Condor DAG node per independent warmup replica.
  - Saves terminal states under runs/diffusive_2d_origin_bond/warmup/<run_id>/states.
  - Uses compact restart-state storage explicitly:
      int_type=auto, position_int_type=auto, keep_directional_densities=false
  - Writes run_info.txt as a record, but the next production batch can use
    --source_run_id <this run_id> instead of reading run_info.
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

RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing runner script: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

slugify() {
    printf "%s" "$1" | sed -E 's/[+]/p/g; s/-/m/g; s/[.]/p/g; s/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//'
}

format_float() {
    awk -v x="$1" 'BEGIN { printf "%.12g", x }'
}

ensure_cluster_shared_dir_permissions() {
    local path="$1"
    local mode="$2"
    chmod "${mode}" "${path}" 2>/dev/null || true
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "--${name} must be a positive integer. Got '${value}'."
        exit 1
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "--${name} must be a non-negative integer. Got '${value}'."
        exit 1
    fi
}

require_positive_float() {
    local name="$1"
    local value="$2"
    if ! awk -v x="${value}" 'BEGIN { exit !(x > 0.0) }'; then
        echo "--${name} must be a positive number. Got '${value}'."
        exit 1
    fi
}

L="64"
rho="1000"
force_strength="1.0"
ffr="1.0"
n_sweeps="20000"
num_replicas="600"
run_id=""
request_cpus="1"
request_memory="6 GB"
replica_retries="2"
dag_maxjobs="0"
batch_name=""
no_submit="${NO_SUBMIT:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --L)
            L="${2:-}"
            shift 2
            ;;
        --rho)
            rho="${2:-}"
            shift 2
            ;;
        --force_strength)
            force_strength="${2:-}"
            shift 2
            ;;
        --ffr)
            ffr="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --nr|--num_replicas)
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
        --replica_retries)
            replica_retries="${2:-}"
            shift 2
            ;;
        --dag_maxjobs)
            dag_maxjobs="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
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

require_positive_int "L" "${L}"
require_positive_int "n_sweeps" "${n_sweeps}"
require_positive_int "num_replicas" "${num_replicas}"
require_positive_int "request_cpus" "${request_cpus}"
require_nonnegative_int "replica_retries" "${replica_retries}"
require_nonnegative_int "dag_maxjobs" "${dag_maxjobs}"
require_positive_float "rho" "${rho}"
require_positive_float "force_strength" "${force_strength}"
require_positive_float "ffr" "${ffr}"
if (( L % 2 != 0 )); then
    echo "--L must be even so the origin bond maps to [L/2,L/2] -> [L/2+1,L/2]. Got L=${L}."
    exit 1
fi

rho_display="$(format_float "${rho}")"
force_display="$(format_float "${force_strength}")"
ffr_display="$(format_float "${ffr}")"
rho_slug="$(slugify "${rho_display}")"
force_slug="$(slugify "${force_display}")"
ffr_slug="$(slugify "${ffr_display}")"

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${run_id}" ]]; then
    run_id="diffusive_2d_origin_bond_L${L}_rho${rho_slug}_f${force_slug}_ffr${ffr_slug}_warmup_ns${n_sweeps}_nr${num_replicas}_${timestamp}"
else
    run_id="$(slugify "${run_id}")"
fi
job_batch_name="${batch_name:-${run_id}}"

run_root="${REPO_ROOT}/runs/diffusive_2d_origin_bond/warmup/${run_id}"
config_dir="${run_root}/configs"
submit_dir="${run_root}/submit"
log_dir="${run_root}/logs"
state_dir="${run_root}/states"
run_info="${run_root}/run_info.txt"
manifest="${run_root}/manifest.csv"
registry_file="${REPO_ROOT}/runs/diffusive_2d_origin_bond/run_registry.csv"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${state_dir}"
ensure_cluster_shared_dir_permissions "${run_root}" 755
ensure_cluster_shared_dir_permissions "${config_dir}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777
ensure_cluster_shared_dir_permissions "${state_dir}" 1777

center=$(( L / 2 ))
next_x=$(( center + 1 ))
description_name="diffusive_2d_origin_xbond_L${L}_rho${rho_slug}_f${force_slug}_ffr${ffr_slug}_warmup"
runtime_config="${config_dir}/diffusive_2d_origin_bond_warmup.yaml"
cat > "${runtime_config}" <<EOF
# Generated by cluster_scripts/submit_diffusive_2d_origin_bond_warmup_dag.sh
dim_num: 2
L: ${L}
ρ₀: ${rho_display}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${n_sweeps}
warmup_sweeps: 0
performance_mode: true
description: "${description_name}"
int_type: "auto"
position_int_type: "auto"
keep_directional_densities: false

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [[${center}, ${center}], [${next_x}, ${center}]]
forcing_magnitudes: [${force_display}]
ffrs: [${ffr_display}]
forcing_direction_flags: [true]
forcing_fluctuation_type: "alternating_direction"
forcing_rate_scheme: "symmetric_normalized"
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "${state_dir}"
progress_interval: 1000
EOF

save_tag_prefix="warmup_${run_id}_r"
dag_file="${submit_dir}/diffusive_2d_origin_bond_warmup.dag"

: > "${dag_file}"
if (( dag_maxjobs > 0 )); then
    printf "MAXJOBS REPLICAS %s\n" "${dag_maxjobs}" >> "${dag_file}"
fi
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag,initial_state" > "${manifest}"

replica_job_ids=()
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_submit_file="${submit_dir}/warmup_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/warmup_r${replica_idx}.out"
    replica_error_file="${log_dir}/warmup_r${replica_idx}.err"
    replica_log_file="${log_dir}/warmup_r${replica_idx}.log"
    replica_tag="${save_tag_prefix}${replica_idx}"
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
    if (( replica_retries > 0 )); then
        printf "RETRY %s %s\n" "${job_id}" "${replica_retries}" >> "${dag_file}"
    fi
    if (( dag_maxjobs > 0 )); then
        printf "CATEGORY %s REPLICAS\n" "${job_id}" >> "${dag_file}"
    fi
    printf "replica,%s,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${replica_submit_file}" "${replica_output_file}" "${replica_error_file}" "${replica_log_file}" "${replica_tag}" "" \
        >> "${manifest}"
done

dag_append_final_notification_node "${dag_file}" "${submit_dir}" "${log_dir}" "${run_root}" "${run_id}" "diffusive_2d_origin_bond_warmup" "${REPO_ROOT}"

cluster_id=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated warmup DAG but not submitting: ${dag_file}"
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
mode=warmup
simulation=diffusive_2d_origin_bond
dim_num=2
L=${L}
rho0=${rho_display}
D=1.0
force_strength=${force_display}
ffr=${ffr_display}
forcing_rate_scheme=symmetric_normalized
n_sweeps=${n_sweeps}
num_replicas=${num_replicas}
request_cpus=${request_cpus}
request_memory=${request_memory}
replica_retries=${replica_retries}
dag_maxjobs=${dag_maxjobs}
job_batch_name=${job_batch_name}
run_root=${run_root}
config_dir=${config_dir}
submit_dir=${submit_dir}
log_dir=${log_dir}
state_dir=${state_dir}
runtime_config=${runtime_config}
dag_file=${dag_file}
manifest=${manifest}
save_tag_prefix=${save_tag_prefix}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG:-}
cluster_id=${cluster_id}
next_step_hint=bash cluster_scripts/submit_diffusive_2d_origin_bond_production_batch.sh --source_run_id ${run_id} --L ${L} --rho ${rho_display} --n_sweeps <production_sweeps> --nr ${num_replicas}
EOF

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,config_path,save_tag_prefix" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "warmup" "${L}" "${rho_display}" "${n_sweeps}" "${num_replicas}" \
    "${request_cpus}" "${request_memory}" "${run_root}" "${submit_dir}" "${log_dir}" "${state_dir}" "${runtime_config}" "${save_tag_prefix}" \
    >> "${registry_file}"

echo "Prepared diffusive 2D origin-bond warmup DAG."
echo "  run_id=${run_id}"
echo "  dag_file=${dag_file}"
echo "  run_info=${run_info}"
echo "  states=${state_dir}"
echo "  save_tag_prefix=${save_tag_prefix}"
