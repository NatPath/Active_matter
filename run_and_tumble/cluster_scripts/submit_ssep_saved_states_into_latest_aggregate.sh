#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_ssep_saved_states_into_latest_aggregate.sh --run_id <id> [options]

Required:
  --run_id <id>                  Existing SSEP production run_id

Options:
  --mode <auto|production>       How to resolve --run_id (default: auto)
  --request_cpus <int>           Condor request_cpus (default: 1)
  --request_memory <value>       Condor request_memory (default: "2 GB")
  --batch_name <name>            Condor batch_name (default: auto)
  --job_label <label>            Optional label in the submit token
  --no_submit                    Generate files only; do not call condor_submit
  -h, --help                     Show help

Behavior:
  - Resolves the target SSEP production run from --run_id
  - Submits one aggregation job that folds currently saved raw production states
    into the latest aggregate without double counting
  - The heavy aggregation work is done by:
      cluster_scripts/aggregate_ssep_saved_states_into_latest_aggregate.sh
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

AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_ssep_saved_states_into_latest_aggregate.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/ssep/single_center_bond/run_registry.csv"

if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate helper: ${AGGREGATE_SCRIPT}"
    exit 1
fi

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

find_run_info_by_run_id() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local candidate=""

    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/ssep/single_center_bond/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    if [[ -f "${REGISTRY_FILE}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${REGISTRY_FILE}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _warmup _numrep _cpus _mem reg_run_root _submit_dir _log_dir _state_dir _config_path _aggregate_run_id <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

resolve_target_production_run_info() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local resolved_info

    resolved_info="$(find_run_info_by_run_id "${lookup_run_id}" "${mode_hint}" || true)"
    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve SSEP production run_info for run_id='${lookup_run_id}' (mode=${mode_hint})." >&2
        return 1
    fi
    if [[ "$(read_run_info_value "${resolved_info}" "mode")" != "production" ]]; then
        echo "run_id='${lookup_run_id}' does not resolve to an SSEP production run." >&2
        return 1
    fi
    echo "${resolved_info}"
}

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

ensure_cluster_shared_dir_permissions() {
    local path="$1"
    local mode="$2"
    chmod "${mode}" "${path}" 2>/dev/null || true
}

run_id=""
mode="auto"
request_cpus="1"
request_memory="2 GB"
batch_name=""
job_label=""
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
        --request_cpus)
            request_cpus="${2:-}"
            shift 2
            ;;
        --request_memory)
            request_memory="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
            shift 2
            ;;
        --job_label)
            job_label="${2:-}"
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
    auto|production)
        ;;
    *)
        echo "--mode must be one of: auto, production."
        exit 1
        ;;
esac

if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi

target_run_info="$(resolve_target_production_run_info "${run_id}" "${mode}")"
target_run_root="$(read_run_info_value "${target_run_info}" "run_root")"
[[ -n "${target_run_root}" && -d "${target_run_root}" ]] || {
    echo "Could not resolve target run_root from ${target_run_info}"
    exit 1
}

timestamp="$(date +%Y%m%d-%H%M%S)"
job_token_base="saved_states_aggregate_${timestamp}"
if [[ -n "${job_label}" ]]; then
    job_token_base="${job_label}_${job_token_base}"
fi
job_token="$(sanitize_token "${job_token_base}")"
job_root="${target_run_root}/manual_aggregate_jobs/${job_token}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
mkdir -p "${submit_dir}" "${log_dir}"
ensure_cluster_shared_dir_permissions "${target_run_root}" 755
ensure_cluster_shared_dir_permissions "${job_root}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777

if [[ -z "${batch_name}" ]]; then
    batch_name="${run_id}_saved_state_aggregate"
fi

launcher_script="${submit_dir}/run_saved_states_aggregate.sh"
submit_file="${submit_dir}/saved_states_aggregate.sub"
output_file="${log_dir}/saved_states_aggregate.out"
error_file="${log_dir}/saved_states_aggregate.err"
log_file="${log_dir}/saved_states_aggregate.log"
meta_file="${job_root}/submit_info.txt"

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf '%q' "${REPO_ROOT}")"
    printf "bash %q --run_id %q --mode %q\n" "${AGGREGATE_SCRIPT}" "${run_id}" "${mode}"
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
    echo "mode=${mode}"
    echo "target_run_info=${target_run_info}"
    echo "target_run_root=${target_run_root}"
    echo "job_token=${job_token}"
    echo "job_root=${job_root}"
    echo "submit_file=${submit_file}"
    echo "launcher_script=${launcher_script}"
    echo "output_file=${output_file}"
    echo "error_file=${error_file}"
    echo "log_file=${log_file}"
    echo "request_cpus=${request_cpus}"
    echo "request_memory=${request_memory}"
    echo "batch_name=${batch_name}"
} > "${meta_file}"

echo "Prepared SSEP saved-state aggregate submit artifacts:"
echo "  job_root=${job_root}"
echo "  submit_file=${submit_file}"
echo "  launcher_script=${launcher_script}"
echo "  output_file=${output_file}"
echo "  error_file=${error_file}"
echo "  log_file=${log_file}"

cluster_id=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated submit file but did not call condor_submit."
    cluster_id="NO_SUBMIT"
else
    submit_output="$(condor_submit "${submit_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

echo "cluster_id=${cluster_id}" >> "${meta_file}"
echo "  cluster_id=${cluster_id}"
