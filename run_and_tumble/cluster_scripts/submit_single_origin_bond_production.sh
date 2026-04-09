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
        forwarded_args=(--mode production "${forwarded_args[@]}")
    elif [[ "${mode_arg}" != "production" ]]; then
        echo "submit_single_origin_bond_production.sh only supports --mode production when invoked with CLI arguments."
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

if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity_from_latest_state.sh" ]]; then
    LATEST_STATE_RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_latest_state.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_latest_state.sh" ]]; then
    LATEST_STATE_RUNNER_SCRIPT="${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_latest_state.sh"
else
    echo "Could not find run_diffusive_no_activity_from_latest_state.sh"
    exit 1
fi

"${GENERATE_SCRIPT}"

SOURCE_CONFIG_PATH="${REPO_ROOT}/configuration_files/single_origin_bond/production/params.yaml"
DEFAULT_WARMUP_STATE_DIR="${REPO_ROOT}/saved_states/single_origin_bond/warmup"
DEFAULT_SUBMIT_DIR="${SCRIPT_DIR}/generated_submit/single_origin_bond/production"
DEFAULT_LOG_DIR="${REPO_ROOT}/condor_logs/single_origin_bond/production"
DEFAULT_STATE_DIR="${REPO_ROOT}/saved_states/single_origin_bond/production"
CONFIG_DIR="${RUN_CONFIG_DIR:-$(dirname "${SOURCE_CONFIG_PATH}")}"
WARMUP_STATE_DIR="${WARMUP_STATE_DIR:-${DEFAULT_WARMUP_STATE_DIR}}"
SUBMIT_DIR="${RUN_SUBMIT_DIR:-${DEFAULT_SUBMIT_DIR}}"
LOG_DIR="${RUN_LOG_DIR:-${DEFAULT_LOG_DIR}}"
STATE_DIR="${RUN_STATE_DIR:-${DEFAULT_STATE_DIR}}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
JOB_BATCH_NAME="${JOB_BATCH_NAME:-single_origin_bond_production}"

REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-5 GB}"
REQUIRE_INITIAL_STATE="${REQUIRE_INITIAL_STATE:-false}"
RUN_REGISTRY_PATH="${RUN_REGISTRY_PATH:-${REPO_ROOT}/runs/single_origin_bond/run_registry.csv}"
CONTINUE_RUN_ID="${CONTINUE_RUN_ID:-}"
CONTINUE_STATE_DIR="${CONTINUE_STATE_DIR:-}"
REQUIRE_CONTINUE_STATE="${REQUIRE_CONTINUE_STATE:-true}"
NUM_REPLICAS="${NUM_REPLICAS:-1}"
REPLICA_STRATEGY="${REPLICA_STRATEGY:-mp}"
DEFER_INITIAL_STATE_LOOKUP="${DEFER_INITIAL_STATE_LOOKUP:-false}"
NO_SUBMIT="${NO_SUBMIT:-false}"

if ! [[ "${NUM_REPLICAS}" =~ ^[0-9]+$ ]] || (( NUM_REPLICAS <= 0 )); then
    echo "NUM_REPLICAS must be a positive integer. Got '${NUM_REPLICAS}'."
    exit 1
fi
if [[ "${REPLICA_STRATEGY}" != "mp" && "${REPLICA_STRATEGY}" != "dag" ]]; then
    echo "REPLICA_STRATEGY must be 'mp' or 'dag'. Got '${REPLICA_STRATEGY}'."
    exit 1
fi

lookup_registry_row_by_run_id() {
    local lookup_run_id="$1"
    local registry_path="$2"
    awk -F, -v rid="${lookup_run_id}" '
        NR == 1 {next}
        $2 == rid {row = $0}
        END {print row}
    ' "${registry_path}"
}

latest_matching_state() {
    local base_dir="$1"
    shift
    local pattern=""
    local candidate=""
    for pattern in "$@"; do
        candidate="$(ls -1t "${base_dir}"/${pattern} 2>/dev/null | head -n 1 || true)"
        if [[ -n "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

if [[ -n "${CONTINUE_RUN_ID}" && -n "${CONTINUE_STATE_DIR}" ]]; then
    echo "Set only one of CONTINUE_RUN_ID or CONTINUE_STATE_DIR."
    exit 1
fi

if [[ -n "${CONTINUE_RUN_ID}" ]]; then
    if [[ ! -f "${RUN_REGISTRY_PATH}" ]]; then
        echo "Cannot resolve CONTINUE_RUN_ID='${CONTINUE_RUN_ID}': registry not found: ${RUN_REGISTRY_PATH}"
        exit 1
    fi
    continue_registry_row="$(lookup_registry_row_by_run_id "${CONTINUE_RUN_ID}" "${RUN_REGISTRY_PATH}")"
    if [[ -z "${continue_registry_row}" ]]; then
        echo "Cannot resolve CONTINUE_RUN_ID='${CONTINUE_RUN_ID}': not found in ${RUN_REGISTRY_PATH}"
        exit 1
    fi
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_cpus reg_mem reg_run_root reg_log_dir reg_state_dir reg_warmup_state_dir reg_ffr reg_force <<< "${continue_registry_row}"
    if [[ "${reg_mode}" != "production" ]]; then
        echo "CONTINUE_RUN_ID='${CONTINUE_RUN_ID}' is mode='${reg_mode}', expected mode='production'."
        exit 1
    fi
    CONTINUE_STATE_DIR="${reg_state_dir}"
fi

if [[ -n "${CONTINUE_STATE_DIR}" && ! -d "${CONTINUE_STATE_DIR}" ]]; then
    echo "CONTINUE_STATE_DIR does not exist: ${CONTINUE_STATE_DIR}"
    exit 1
fi

mkdir -p "${SUBMIT_DIR}" "${LOG_DIR}" "${STATE_DIR}" "${CONFIG_DIR}"

runtime_config="${SOURCE_CONFIG_PATH}"
if [[ "${CONFIG_DIR}" != "$(dirname "${SOURCE_CONFIG_PATH}")" || "${STATE_DIR}" != "${DEFAULT_STATE_DIR}" ]]; then
    runtime_config="${CONFIG_DIR}/single_origin_bond_production.yaml"
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

latest_state=""
runner_arguments="${RUNNER_SCRIPT} ${runtime_config}"
if [[ -n "${CONTINUE_STATE_DIR}" ]]; then
    latest_state="$(latest_matching_state "${CONTINUE_STATE_DIR}" \
        "single_origin_bond_prod_*.jld2" \
        "single_origin_bond_production_*.jld2" \
        "single_origin_bond_*.jld2" \
        "*.jld2" || true)"
    if [[ -z "${latest_state}" ]]; then
        if [[ "${REQUIRE_CONTINUE_STATE}" == "true" ]]; then
            echo "ERROR: no continuation production state found under ${CONTINUE_STATE_DIR}"
            exit 1
        else
            echo "Skipping production submission: no continuation production state found under ${CONTINUE_STATE_DIR}"
            exit 0
        fi
    fi
    runner_arguments="${runner_arguments} --continue ${latest_state}"
else
    if [[ "${DEFER_INITIAL_STATE_LOOKUP}" == "true" ]]; then
        latest_state="DEFERRED:${WARMUP_STATE_DIR}"
        runner_arguments="${LATEST_STATE_RUNNER_SCRIPT} --runner_script ${RUNNER_SCRIPT} --config ${runtime_config} --state_dir ${WARMUP_STATE_DIR} --pattern single_origin_bond_warmup_*.jld2 --pattern *.jld2"
    else
        latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/single_origin_bond_warmup_*.jld2 2>/dev/null | head -n 1 || true)"
        if [[ -z "${latest_state}" ]]; then
            latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/*.jld2 2>/dev/null | head -n 1 || true)"
        fi
        if [[ -z "${latest_state}" ]]; then
            if [[ "${REQUIRE_INITIAL_STATE}" == "true" ]]; then
                echo "ERROR: no warmup state found under ${WARMUP_STATE_DIR}"
                exit 1
            else
                echo "Skipping production submission: no warmup state found under ${WARMUP_STATE_DIR}"
                exit 0
            fi
        fi
        runner_arguments="${runner_arguments} --initial_state ${latest_state}"
    fi
fi

submit_file="${SUBMIT_DIR}/single_origin_bond_production.sub"
output_file="${LOG_DIR}/single_origin_bond.out"
error_file="${LOG_DIR}/single_origin_bond.err"
log_file="${LOG_DIR}/single_origin_bond.log"
save_tag_base="${RUN_ID:-single_origin_bond_production}"

if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
    dag_file="${SUBMIT_DIR}/single_origin_bond_production.dag"
    aggregate_submit_file="${SUBMIT_DIR}/single_origin_bond_production_aggregate.sub"
    aggregate_output_file="${LOG_DIR}/single_origin_bond_aggregate.out"
    aggregate_error_file="${LOG_DIR}/single_origin_bond_aggregate.err"
    aggregate_log_file="${LOG_DIR}/single_origin_bond_aggregate.log"
    aggregate_request_cpus="${AGGREGATE_REQUEST_CPUS:-1}"

    : > "${dag_file}"
    job_ids=()
    for ((replica_idx = 1; replica_idx <= NUM_REPLICAS; replica_idx++)); do
        replica_submit_file="${SUBMIT_DIR}/single_origin_bond_production_replica_${replica_idx}.sub"
        replica_output_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.out"
        replica_error_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.err"
        replica_log_file="${LOG_DIR}/single_origin_bond_r${replica_idx}.log"
        replica_tag="replica_${save_tag_base}_r${replica_idx}"
        replica_runner_arguments="${runner_arguments} --save_tag ${replica_tag}"
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
    dag_append_final_notification_node "${dag_file}" "${SUBMIT_DIR}" "${LOG_DIR}" "$(dirname "${SUBMIT_DIR}")" "${JOB_BATCH_NAME}" "single_origin_bond_production" "${REPO_ROOT}"

    if [[ -n "${CONTINUE_STATE_DIR}" ]]; then
        echo "Submitting single-origin-bond production continuation DAG from ${latest_state}"
    else
        echo "Submitting single-origin-bond production DAG with initial_state=${latest_state}"
    fi
    if [[ "${NO_SUBMIT}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated production DAG but not submitting: ${dag_file}"
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

    if [[ -n "${CONTINUE_STATE_DIR}" ]]; then
        echo "Submitting single-origin-bond production continuation job from ${latest_state}"
    else
        echo "Submitting single-origin-bond production job with initial_state=${latest_state}"
    fi
    if [[ "${NO_SUBMIT}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated production submit file but not submitting: ${submit_file}"
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
    manifest_mode="production"
    if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
        manifest_mode="production_dag"
    fi
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${manifest_mode}" "${cluster_id}" "${runtime_config}" "${submit_file}" \
        "${output_file}" "${error_file}" "${log_file}" "${STATE_DIR}" "${latest_state}" >> "${MANIFEST_PATH}"
fi

echo "Production submission completed."
echo "Submit file: ${submit_file}"
echo "Logs: ${LOG_DIR}"
echo "States: ${STATE_DIR}"
