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

if [[ -f "${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh" ]]; then
    GENERATE_SCRIPT="${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh" ]]; then
    GENERATE_SCRIPT="${REPO_ROOT}/cluster_scripts/generate_two_force_d_sweep_configs.sh"
else
    echo "Could not find generate_two_force_d_sweep_configs.sh"
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

if [[ -f "${SCRIPT_DIR}/aggregate_two_force_d_saved_files.sh" ]]; then
    AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_two_force_d_saved_files.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/aggregate_two_force_d_saved_files.sh" ]]; then
    AGGREGATE_SCRIPT="${REPO_ROOT}/cluster_scripts/aggregate_two_force_d_saved_files.sh"
else
    echo "Could not find aggregate_two_force_d_saved_files.sh"
    exit 1
fi

if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity_from_latest_state.sh" ]]; then
    LATEST_STATE_RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_latest_state.sh"
elif [[ -f "${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_latest_state.sh" ]]; then
    LATEST_STATE_RUNNER_SCRIPT="${REPO_ROOT}/cluster_scripts/run_diffusive_no_activity_from_latest_state.sh"
else
    echo "Could not find run_diffusive_no_activity_from_latest_state.sh"
    exit 1
fi
SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Could not find spacing utils script: ${SPACING_UTILS}"
    exit 1
fi
source "${SPACING_UTILS}"

"${GENERATE_SCRIPT}"

D_MIN="${D_MIN:-2}"
D_MAX="${D_MAX:-128}"
D_STEP="${D_STEP:-2}"
D_SPACING="${D_SPACING:-linear}"
D_SPACING="$(two_force_d_normalize_spacing_mode "${D_SPACING}")" || {
    echo "Invalid D_SPACING='${D_SPACING}'."
    exit 1
}
if [[ -n "${D_VALUES_CSV:-}" ]]; then
    D_VALUES=()
    if ! two_force_d_csv_to_array "${D_VALUES_CSV}" D_VALUES; then
        echo "Invalid D_VALUES_CSV='${D_VALUES_CSV}'."
        exit 1
    fi
else
    mapfile -t D_VALUES < <(two_force_d_generate_d_values "${D_SPACING}" "${D_MIN}" "${D_MAX}" "${D_STEP}")
fi
if (( ${#D_VALUES[@]} == 0 )); then
    echo "No d values to submit (spacing='${D_SPACING}', range=${D_MIN}:${D_STEP}:${D_MAX})."
    exit 1
fi
SOURCE_CONFIG_DIR="${REPO_ROOT}/configuration_files/two_force_d_sweep/production"
DEFAULT_WARMUP_STATE_DIR="${REPO_ROOT}/saved_states/two_force_d_sweep/warmup"
DEFAULT_SUBMIT_DIR="${SCRIPT_DIR}/generated_submit/two_force_d_sweep/production"
DEFAULT_LOG_DIR="${REPO_ROOT}/condor_logs/two_force_d_sweep/production"
DEFAULT_STATE_DIR="${REPO_ROOT}/saved_states/two_force_d_sweep/production"
CONFIG_DIR="${RUN_CONFIG_DIR:-${SOURCE_CONFIG_DIR}}"
WARMUP_STATE_DIR="${WARMUP_STATE_DIR:-${DEFAULT_WARMUP_STATE_DIR}}"
SUBMIT_DIR="${RUN_SUBMIT_DIR:-${DEFAULT_SUBMIT_DIR}}"
LOG_DIR="${RUN_LOG_DIR:-${DEFAULT_LOG_DIR}}"
STATE_DIR="${RUN_STATE_DIR:-${DEFAULT_STATE_DIR}}"
MANIFEST_PATH="${MANIFEST_PATH:-}"
JOB_BATCH_NAME="${JOB_BATCH_NAME:-two_force_d_production}"

REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-2 GB}"
REQUIRE_INITIAL_STATE="${REQUIRE_INITIAL_STATE:-false}"
RUN_REGISTRY_PATH="${RUN_REGISTRY_PATH:-${REPO_ROOT}/runs/two_force_d/run_registry.csv}"
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
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_dmin reg_dmax reg_dstep reg_cpus reg_mem reg_run_root reg_log_dir reg_state_dir reg_warmup_state_dir <<< "${continue_registry_row}"
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

mkdir -p "${SUBMIT_DIR}" "${LOG_DIR}" "${STATE_DIR}"
if [[ "${CONFIG_DIR}" != "${SOURCE_CONFIG_DIR}" ]]; then
    mkdir -p "${CONFIG_DIR}"
fi

if [[ -n "${MANIFEST_PATH}" ]]; then
    mkdir -p "$(dirname "${MANIFEST_PATH}")"
    if [[ ! -f "${MANIFEST_PATH}" ]]; then
        echo "d,mode,cluster_id,config_path,submit_file,output_file,error_file,log_file,save_dir,initial_state" > "${MANIFEST_PATH}"
    fi
fi

save_tag_base="${RUN_ID:-two_force_d_production}"
aggregate_run_id="${RUN_ID:-}"
if [[ -z "${aggregate_run_id}" ]]; then
    config_dir_abs="$(cd "${CONFIG_DIR}" 2>/dev/null && pwd || true)"
    if [[ -n "${config_dir_abs}" && "${config_dir_abs}" =~ /runs/two_force_d/(warmup|production)/([^/]+)/configs$ ]]; then
        aggregate_run_id="${BASH_REMATCH[2]}"
    fi
fi
dag_file="${SUBMIT_DIR}/two_force_d_production.dag"
if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
    if [[ -z "${aggregate_run_id}" ]]; then
        echo "DAG aggregation requires a resolvable run_id (set RUN_ID or use run config dir under runs/two_force_d/.../configs)."
        exit 1
    fi
    : > "${dag_file}"
fi

for d in "${D_VALUES[@]}"; do
    config_path="${SOURCE_CONFIG_DIR}/d_${d}.yaml"
    if [[ ! -f "${config_path}" ]]; then
        config_path="${REPO_ROOT}/two_force_d_sweep/production/d_${d}.yaml"
    fi
    if [[ ! -f "${config_path}" ]]; then
        config_path="${REPO_ROOT}/d_${d}.yaml"
    fi
    if [[ ! -f "${config_path}" ]]; then
        echo "Skipping d=${d}: could not find production config file"
        continue
    fi
    runtime_config="${config_path}"
    if [[ "${CONFIG_DIR}" != "${SOURCE_CONFIG_DIR}" || "${STATE_DIR}" != "${DEFAULT_STATE_DIR}" ]]; then
        runtime_config="${CONFIG_DIR}/d_${d}.yaml"
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
        }' "${config_path}" > "${runtime_config}"
    fi

    latest_state=""
    runner_arguments="${RUNNER_SCRIPT} ${runtime_config}"
    if [[ -n "${CONTINUE_STATE_DIR}" ]]; then
        latest_state="$(latest_matching_state "${CONTINUE_STATE_DIR}" \
            "aggregated/two_force_d${d}_prod_*.jld2" \
            "aggregated/two_force_d${d}_production_*.jld2" \
            "aggregated/two_force_d${d}_*.jld2" \
            "two_force_d${d}_prod_*.jld2" \
            "two_force_d${d}_production_*.jld2" \
            "two_force_d${d}_*.jld2" || true)"
        if [[ -z "${latest_state}" ]]; then
            if [[ "${REQUIRE_CONTINUE_STATE}" == "true" ]]; then
                echo "ERROR: no continuation production state found for d=${d} under ${CONTINUE_STATE_DIR}"
                exit 1
            else
                echo "Skipping d=${d}: no continuation production state found under ${CONTINUE_STATE_DIR}"
                continue
            fi
        fi
        runner_arguments="${runner_arguments} --continue ${latest_state}"
    else
        if [[ "${DEFER_INITIAL_STATE_LOOKUP}" == "true" ]]; then
            latest_state="DEFERRED:${WARMUP_STATE_DIR}/two_force_d${d}_warmup_*"
            runner_arguments="${LATEST_STATE_RUNNER_SCRIPT} --runner_script ${RUNNER_SCRIPT} --config ${runtime_config} --state_dir ${WARMUP_STATE_DIR} --pattern aggregated/two_force_d${d}_warmup_*.jld2 --pattern aggregated/two_force_d${d}_*.jld2 --pattern two_force_d${d}_warmup_*.jld2 --pattern two_force_d${d}_*.jld2"
        else
            latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/two_force_d${d}_warmup_* 2>/dev/null | head -n 1 || true)"
            if [[ -z "${latest_state}" ]]; then
                latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/aggregated/two_force_d${d}_warmup_* 2>/dev/null | head -n 1 || true)"
            fi
            if [[ -z "${latest_state}" ]]; then
                latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/aggregated/two_force_d${d}_* 2>/dev/null | head -n 1 || true)"
            fi
            if [[ -z "${latest_state}" ]]; then
                if [[ "${REQUIRE_INITIAL_STATE}" == "true" ]]; then
                    echo "ERROR: no warmup state matching ${WARMUP_STATE_DIR}/two_force_d${d}_warmup_* for d=${d}"
                    exit 1
                else
                    echo "Skipping d=${d}: no warmup state matching ${WARMUP_STATE_DIR}/two_force_d${d}_warmup_*"
                    continue
                fi
            fi
            runner_arguments="${runner_arguments} --initial_state ${latest_state}"
        fi
    fi
    if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
        replica_job_ids=()
        for ((replica_idx = 1; replica_idx <= NUM_REPLICAS; replica_idx++)); do
            replica_tag="replica_${save_tag_base}_d${d}_r${replica_idx}"
            replica_runner_arguments="${runner_arguments} --save_tag ${replica_tag}"
            replica_submit_file="${SUBMIT_DIR}/production_d_${d}_replica_${replica_idx}.sub"
            replica_output_file="${LOG_DIR}/d_${d}_r${replica_idx}.out"
            replica_error_file="${LOG_DIR}/d_${d}_r${replica_idx}.err"
            replica_log_file="${LOG_DIR}/d_${d}_r${replica_idx}.log"
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
            job_id="D${d}R${replica_idx}"
            replica_job_ids+=("${job_id}")
            printf "JOB %s %s\n" "${job_id}" "${replica_submit_file}" >> "${dag_file}"
        done

        aggregate_submit_file="${SUBMIT_DIR}/production_d_${d}_aggregate.sub"
        aggregate_output_file="${LOG_DIR}/d_${d}_aggregate.out"
        aggregate_error_file="${LOG_DIR}/d_${d}_aggregate.err"
        aggregate_log_file="${LOG_DIR}/d_${d}_aggregate.log"
        aggregate_request_cpus="${AGGREGATE_REQUEST_CPUS:-1}"
        aggregate_arguments="${AGGREGATE_SCRIPT} --run_id ${aggregate_run_id} --mode production --state_dir ${STATE_DIR} --config_dir ${CONFIG_DIR} --d_min ${d} --d_max ${d} --d_step 1 --num_files ${NUM_REPLICAS} --aggregated_subdir aggregated"
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
        aggregate_job_id="D${d}A"
        printf "JOB %s %s\n" "${aggregate_job_id}" "${aggregate_submit_file}" >> "${dag_file}"
        printf "PARENT %s CHILD %s\n" "${replica_job_ids[*]}" "${aggregate_job_id}" >> "${dag_file}"

        if [[ -n "${MANIFEST_PATH}" ]]; then
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "${d}" "production_dag" "DAG" "${runtime_config}" "${aggregate_submit_file}" \
                "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${STATE_DIR}" "${latest_state}" >> "${MANIFEST_PATH}"
        fi
    else
        if (( NUM_REPLICAS > 1 )); then
            runner_arguments="${runner_arguments} --num_runs ${NUM_REPLICAS} --save_tag aggregated_${save_tag_base}_d${d}"
        fi
        submit_file="${SUBMIT_DIR}/production_d_${d}.sub"
        output_file="${LOG_DIR}/d_${d}.out"
        error_file="${LOG_DIR}/d_${d}.err"
        log_file="${LOG_DIR}/d_${d}.log"
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
            echo "Submitting production continuation for d=${d} from ${latest_state}"
        else
            echo "Submitting production job for d=${d} with initial_state=${latest_state}"
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
        if [[ -n "${MANIFEST_PATH}" ]]; then
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "${d}" "production" "${cluster_id}" "${runtime_config}" "${submit_file}" \
                "${output_file}" "${error_file}" "${log_file}" "${STATE_DIR}" "${latest_state}" >> "${MANIFEST_PATH}"
        fi
    fi
done

if [[ "${REPLICA_STRATEGY}" == "dag" && "${NUM_REPLICAS}" -gt 1 ]]; then
    if ! grep -q '^JOB ' "${dag_file}" 2>/dev/null; then
        echo "No DAG nodes were generated for production submission."
        exit 0
    fi
    echo "Submitting production DAG: ${dag_file}"
    if [[ "${NO_SUBMIT}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated production DAG but not submitting: ${dag_file}"
    else
        submit_output="$(condor_submit_dag "${dag_file}")"
        echo "${submit_output}"
    fi
fi

echo "Production submission pass completed."
echo "Submit files: ${SUBMIT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "States: ${STATE_DIR}"
if [[ -n "${CONTINUE_STATE_DIR}" ]]; then
    echo "Continued from states: ${CONTINUE_STATE_DIR}"
fi
