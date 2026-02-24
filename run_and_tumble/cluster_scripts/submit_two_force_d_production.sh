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

"${GENERATE_SCRIPT}"

D_MIN="${D_MIN:-2}"
D_MAX="${D_MAX:-128}"
D_STEP="${D_STEP:-2}"
D_VALUES=($(seq "${D_MIN}" "${D_STEP}" "${D_MAX}"))
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

    latest_state="$(ls -1t "${WARMUP_STATE_DIR}"/two_force_d${d}_warmup_* 2>/dev/null | head -n 1 || true)"

    if [[ -z "${latest_state}" ]]; then
        if [[ "${REQUIRE_INITIAL_STATE}" == "true" ]]; then
            echo "ERROR: no warmup state matching ${WARMUP_STATE_DIR}/two_force_d${d}_warmup_* for d=${d}"
            exit 1
        else
            echo "Skipping d=${d}: no warmup state matching ${WARMUP_STATE_DIR}/two_force_d${d}_warmup_*"
            continue
        fi
    fi

    submit_file="${SUBMIT_DIR}/production_d_${d}.sub"
    output_file="${LOG_DIR}/d_${d}.out"
    error_file="${LOG_DIR}/d_${d}.err"
    log_file="${LOG_DIR}/d_${d}.log"
    cat > "${submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${RUNNER_SCRIPT} ${runtime_config} --initial_state ${latest_state}
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

    echo "Submitting production job for d=${d} with initial_state=${latest_state}"
    submit_output="$(condor_submit "${submit_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
    if [[ -n "${MANIFEST_PATH}" ]]; then
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${d}" "production" "${cluster_id}" "${runtime_config}" "${submit_file}" \
            "${output_file}" "${error_file}" "${log_file}" "${STATE_DIR}" "${latest_state}" >> "${MANIFEST_PATH}"
    fi
done

echo "Production submission pass completed."
echo "Submit files: ${SUBMIT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "States: ${STATE_DIR}"
