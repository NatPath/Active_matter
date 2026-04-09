#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

if [[ $# -gt 0 ]]; then
    META_SCRIPT="${SCRIPT_DIR}/submit_single_origin_bond_meta.sh"
    if [[ ! -f "${META_SCRIPT}" && -f "${REPO_ROOT}/cluster_scripts/submit_single_origin_bond_meta.sh" ]]; then
        META_SCRIPT="${REPO_ROOT}/cluster_scripts/submit_single_origin_bond_meta.sh"
    fi
    if [[ ! -f "${META_SCRIPT}" ]]; then
        echo "Could not find submit_single_origin_bond_meta.sh"
        exit 1
    fi

    forwarded_args=("$@")
    mode_arg=""
    for ((i = 0; i < ${#forwarded_args[@]}; i++)); do
        if [[ "${forwarded_args[$i]}" == "--mode" ]]; then
            if (( i + 1 >= ${#forwarded_args[@]} )); then
                echo "Missing value for --mode."
                exit 1
            fi
            mode_arg="${forwarded_args[$((i + 1))]}"
            break
        fi
    done
    if [[ -z "${mode_arg}" ]]; then
        forwarded_args=(--mode warmup "${forwarded_args[@]}")
    elif [[ "${mode_arg}" != "warmup" ]]; then
        echo "submit_single_origin_bond_warmup.sh only supports --mode warmup when invoked with CLI arguments."
        exit 1
    fi

    exec bash "${META_SCRIPT}" "${forwarded_args[@]}"
fi

if [[ -f "${SCRIPT_DIR}/generate_single_origin_bond_configs.sh" ]]; then
    GENERATE_SCRIPT="${SCRIPT_DIR}/generate_single_origin_bond_configs.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/generate_single_origin_bond_configs.sh" ]]; then
    GENERATE_SCRIPT="${REPO_ROOT}/cluster_scripts/generate_single_origin_bond_configs.sh"
else
    echo "Could not find generate_single_origin_bond_configs.sh"
    exit 1
fi

if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh" ]]; then
    RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_config.sh" ]]; then
    RUNNER_SCRIPT="${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_config.sh"
else
    echo "Could not find run_diffusive_no_activity_from_config.sh"
    exit 1
fi

if [[ -f "${SCRIPT_DIR}/aggregate_replicas_from_tags.sh" ]]; then
    AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_replicas_from_tags.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/aggregate_replicas_from_tags.sh" ]]; then
    AGGREGATE_SCRIPT="${REPO_ROOT}/cluster_scripts/aggregate_replicas_from_tags.sh"
else
    echo "Could not find aggregate_replicas_from_tags.sh"
    exit 1
fi
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

"${GENERATE_SCRIPT}"

SOURCE_CONFIG_PATH="${REPO_ROOT}/configuration_files/single_origin_bond/warmup/params.yaml"
DEFAULT_SUBMIT_DIR="${SCRIPT_DIR}/generated_submit/single_origin_bond/warmup"
DEFAULT_LOG_DIR="${REPO_ROOT}/condor_logs/single_origin_bond/warmup"
DEFAULT_STATE_DIR="${REPO_ROOT}/saved_states/single_origin_bond/warmup"
CONFIG_DIR="${RUN_CONFIG_DIR:-$(dirname "${SOURCE_CONFIG_PATH}")}"
SUBMIT_DIR="${RUN_SUBMIT_DIR:-${DEFAULT_SUBMIT_DIR}}"
LOG_DIR="${RUN_LOG_DIR:-${DEFAULT_LOG_DIR}}"
STATE_DIR="${RUN_STATE_DIR:-${DEFAULT_STATE_DIR}}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
JOB_BATCH_NAME="${JOB_BATCH_NAME:-single_origin_bond_warmup}"

REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-5 GB}"
NUM_REPLICAS="${NUM_REPLICAS:-1}"
REPLICA_STRATEGY="${REPLICA_STRATEGY:-mp}"
NO_SUBMIT="${NO_SUBMIT:-false}"

if ! [[ "${NUM_REPLICAS}" =~ ^[0-9]+$ ]] || (( NUM_REPLICAS <= 0 )); then
    echo "NUM_REPLICAS must be a positive integer. Got '${NUM_REPLICAS}'."
    exit 1
fi
if [[ "${REPLICA_STRATEGY}" != "mp" && "${REPLICA_STRATEGY}" != "dag" ]]; then
    echo "REPLICA_STRATEGY must be 'mp' or 'dag'. Got '${REPLICA_STRATEGY}'."
    exit 1
fi

mkdir -p "${SUBMIT_DIR}" "${LOG_DIR}" "${STATE_DIR}" "${CONFIG_DIR}"

runtime_config="${SOURCE_CONFIG_PATH}"
if [[ "${CONFIG_DIR}" != "$(dirname "${SOURCE_CONFIG_PATH}")" || "${STATE_DIR}" != "${DEFAULT_STATE_DIR}" ]]; then
    runtime_config="${CONFIG_DIR}/single_origin_bond_warmup.yaml"
    save_dir_line="save_dir: \"${STATE_DIR}\""
    awk -v save_dir_line="${save_dir_line}" '
    BEGIN {seen_save=0}
    {
        if ($0 ~ /^save_dir:[[:space:]]*/) {
            print save_dir_line
            seen_save=1
            next
        }
        print
    }
    END {
        if (!seen_save) print save_dir_line
    }' "${SOURCE_CONFIG_PATH}" > "${runtime_config}"
fi

save_tag_base="${RUN_ID:-single_origin_bond_warmup}"
submit_file="${SUBMIT_DIR}/single_origin_bond_warmup.sub"
output_file="${LOG_DIR}/single_origin_bond.out"
error_file="${LOG_DIR}/single_origin_bond.err"
log_file="${LOG_DIR}/single_origin_bond.log"

if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
    dag_file="${SUBMIT_DIR}/single_origin_bond_warmup.dag"
    aggregate_submit_file="${SUBMIT_DIR}/single_origin_bond_warmup_aggregate.sub"
    aggregate_output_file="${LOG_DIR}/single_origin_bond_aggregate.out"
    aggregate_error_file="${LOG_DIR}/single_origin_bond_aggregate.err"
    aggregate_log_file="${LOG_DIR}/single_origin_bond_aggregate.log"
    aggregate_request_cpus="${AGGREGATE_REQUEST_CPUS:-1}"

    : > "${dag_file}"
    job_ids=()
    for ((replica_idx = 1; replica_idx <= NUM_REPLICAS; replica_idx++)); do
        replica_submit_file="${SUBMIT_DIR}/single_origin_bond_warmup_replica_${replica_idx}.sub"
        replica_output_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.out"
        replica_error_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.err"
        replica_log_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.log"
        replica_tag="replica_${save_tag_base}_r${replica_idx}"
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
request_cpus = ${REQUEST_CPUS}
request_memory = ${REQUEST_MEMORY}
batch_name = ${JOB_BATCH_NAME}
queue
EOF
        job_id="R${replica_idx}"
        job_ids+=("${job_id}")
        printf "JOB %s %s\n" "${job_id}" "${replica_submit_file}" >> "${dag_file}"
    done

    aggregate_arguments="${AGGREGATE_SCRIPT} --config ${runtime_config} --state_dir ${STATE_DIR} --num_replicas ${NUM_REPLICAS} --replica_tag_prefix replica_${save_tag_base}_r --save_tag aggregated_${save_tag_base}"
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
request_memory = ${REQUEST_MEMORY}
batch_name = ${JOB_BATCH_NAME}
queue
EOF
    printf "JOB AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
    printf "PARENT %s CHILD AGG\n" "${job_ids[*]}" >> "${dag_file}"
    dag_append_final_notification_node "${dag_file}" "${SUBMIT_DIR}" "${LOG_DIR}" "$(dirname "${SUBMIT_DIR}")" "${JOB_BATCH_NAME}" "single_origin_bond_warmup" "${REPO_ROOT}"

    echo "Submitting single-origin-bond warmup DAG"
    if [[ "${NO_SUBMIT}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated warmup DAG but not submitting: ${dag_file}"
        cluster_id="NO_SUBMIT"
    else
        submit_output="$(condor_submit_dag "${dag_file}")"
        echo "${submit_output}"
        cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
        cluster_id="${cluster_id:-NA}"
    fi
    submit_file="${dag_file}"
    output_file="${aggregate_output_file}"
    error_file="${aggregate_error_file}"
    log_file="${aggregate_log_file}"
else
    runner_arguments="${RUNNER_SCRIPT} ${runtime_config}"
    if (( NUM_REPLICAS > 1 )); then
        runner_arguments="${runner_arguments} --num_runs ${NUM_REPLICAS} --save_tag aggregated_${save_tag_base}"
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
request_cpus = ${REQUEST_CPUS}
request_memory = ${REQUEST_MEMORY}
batch_name = ${JOB_BATCH_NAME}
queue
EOF

    echo "Submitting single-origin-bond warmup job"
    if [[ "${NO_SUBMIT}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated warmup submit file but not submitting: ${submit_file}"
        cluster_id="NO_SUBMIT"
    else
        submit_output="$(condor_submit "${submit_file}")"
        echo "${submit_output}"
        cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
        cluster_id="${cluster_id:-NA}"
    fi
fi

if [[ -n "${MANIFEST_PATH}" ]]; then
    mkdir -p "$(dirname "${MANIFEST_PATH}")"
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
        echo "mode,cluster_id,config_path,submit_file,output_file,error_file,log_file,save_dir,initial_state" > "${MANIFEST_PATH}"
    fi
    manifest_mode="warmup"
    if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
        manifest_mode="warmup_dag"
    fi
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${manifest_mode}" "${cluster_id}" "${runtime_config}" "${submit_file}" \
        "${output_file}" "${error_file}" "${log_file}" "${STATE_DIR}" "" >> "${MANIFEST_PATH}"
fi

echo "Warmup submission completed."
echo "Submit file: ${submit_file}"
echo "Logs: ${LOG_DIR}"
echo "States: ${STATE_DIR}"
