#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_active_objects_saved_states_into_histograms.sh --run_id <id> [options]

Required:
  --run_id <id>                  Existing active-object histogram run_id

Options:
  --tr <int>                     Override histogram thermalization cutoff
  --max_sweep <int>              Override histogram max_sweep
  --request_cpus <int>           Condor request_cpus (default: 1)
  --request_memory <value>       Condor request_memory (default: "2 GB")
  --batch_name <name>            Condor batch_name (default: auto)
  --job_label <label>            Optional label in the submit token
  --no_submit                    Generate files only; do not call condor_submit
  -h, --help                     Show help

Behavior:
  - Resolves the target active-object histogram run from --run_id
  - Submits one aggregation job that rebuilds histogram outputs from the currently
    saved active-object states under that run
  - Uses recursive state discovery so top-up batches are included automatically
  - Uses the run's saved tr_sweeps by default when present
  - The heavy aggregation work is done by:
      cluster_scripts/aggregate_active_object_histograms_from_tags.sh
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_active_objects.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_active_objects.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_active_object_histograms_from_tags.sh"

if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate helper: ${AGGREGATE_SCRIPT}"
    exit 1
fi

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
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
tr_sweeps=""
max_sweep=""
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
        --tr|--min_sweep)
            tr_sweeps="${2:-}"
            shift 2
            ;;
        --max_sweep)
            max_sweep="${2:-}"
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

if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if [[ -n "${tr_sweeps}" ]] && ! [[ "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    echo "--tr must be an integer. Got '${tr_sweeps}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer. Got '${max_sweep}'."
    exit 1
fi

target_run_root="${REPO_ROOT}/runs/active_objects/steady_state_histograms/${run_id}"
target_run_info="${target_run_root}/run_info.txt"
if [[ ! -f "${target_run_info}" ]]; then
    echo "Could not resolve active-object run_info for run_id='${run_id}'."
    exit 1
fi

target_state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
target_hist_dir="$(read_run_info_value "${target_run_info}" "histogram_dir")"
target_effective_n_sweeps="$(read_run_info_value "${target_run_info}" "effective_n_sweeps")"
target_tr_sweeps="$(read_run_info_value "${target_run_info}" "tr_sweeps")"
if [[ -z "${target_tr_sweeps}" ]]; then
    target_tr_sweeps="$(read_run_info_value "${target_run_info}" "min_sweep")"
fi
target_max_sweep="$(read_run_info_value "${target_run_info}" "max_sweep")"

[[ -n "${target_state_dir}" && -d "${target_state_dir}" ]] || {
    echo "Could not resolve source state_dir from ${target_run_info}"
    exit 1
}
[[ -n "${target_hist_dir}" ]] || {
    echo "Could not resolve histogram_dir from ${target_run_info}"
    exit 1
}
mkdir -p "${target_hist_dir}"

if [[ -z "${tr_sweeps}" ]]; then
    tr_sweeps="${target_tr_sweeps:-0}"
fi
if [[ -z "${max_sweep}" ]]; then
    max_sweep="${target_max_sweep:-}"
fi

if [[ -n "${target_effective_n_sweeps}" && "${target_effective_n_sweeps}" =~ ^[0-9]+$ && "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    if (( tr_sweeps >= target_effective_n_sweeps )); then
        echo "Resolved histogram tr=${tr_sweeps} is not valid for effective_n_sweeps=${target_effective_n_sweeps}."
        echo "Pass --tr explicitly with a value smaller than the saved run length."
        exit 1
    fi
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
job_token_base="saved_states_histograms_${timestamp}"
if [[ -n "${job_label}" ]]; then
    job_token_base="${job_label}_${job_token_base}"
fi
job_token="$(sanitize_token "${job_token_base}")"
job_root="${target_run_root}/manual_histogram_jobs/${job_token}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
mkdir -p "${submit_dir}" "${log_dir}"
ensure_cluster_shared_dir_permissions "${target_run_root}" 755
ensure_cluster_shared_dir_permissions "${job_root}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777

if [[ -z "${batch_name}" ]]; then
    batch_name="${run_id}_saved_states_histograms"
fi

launcher_script="${submit_dir}/run_saved_states_histograms.sh"
submit_file="${submit_dir}/saved_states_histograms.sub"
output_file="${log_dir}/saved_states_histograms.out"
error_file="${log_dir}/saved_states_histograms.err"
log_file="${log_dir}/saved_states_histograms.log"
meta_file="${job_root}/submit_info.txt"
aggregate_save_tag="aggregated_${run_id}"

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf '%q' "${REPO_ROOT}")"
    printf "bash %q --state_dir %q --output_dir %q --save_tag %q --all_states_recursive --min_sweep %q --no_plot" \
        "${AGGREGATE_SCRIPT}" "${target_state_dir}" "${target_hist_dir}" "${aggregate_save_tag}" "${tr_sweeps}"
    if [[ -n "${max_sweep}" ]]; then
        printf " --max_sweep %q" "${max_sweep}"
    fi
    printf "\n"
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

cat > "${meta_file}" <<EOF
target_run_id=${run_id}
target_run_root=${target_run_root}
target_state_dir=${target_state_dir}
target_hist_dir=${target_hist_dir}
aggregate_save_tag=${aggregate_save_tag}
effective_n_sweeps=${target_effective_n_sweeps}
tr_sweeps=${tr_sweeps}
max_sweep=${max_sweep}
submit_file=${submit_file}
output_file=${output_file}
error_file=${error_file}
log_file=${log_file}
launcher_script=${launcher_script}
EOF

echo "Prepared active-object saved-state histogram rebuild job"
echo "  target_run_root: ${target_run_root}"
echo "  state_dir: ${target_state_dir}"
echo "  histogram_dir: ${target_hist_dir}"
echo "  tr_sweeps: ${tr_sweeps}"
if [[ -n "${max_sweep}" ]]; then
    echo "  max_sweep: ${max_sweep}"
fi
echo "  submit_file: ${submit_file}"
echo "  launcher_script: ${launcher_script}"

if [[ "${no_submit}" == "true" ]]; then
    echo "No-submit mode enabled; skipping condor_submit."
    exit 0
fi

submit_output="$(condor_submit "${submit_file}")"
echo "${submit_output}"
cluster_id="$(printf '%s\n' "${submit_output}" | sed -nE 's/.*cluster[[:space:]]+([0-9]+).*/\1/p' | tail -n 1)"
if [[ -n "${cluster_id}" ]]; then
    echo "cluster_id=${cluster_id}" >> "${meta_file}"
fi
