#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_active_objects_state_dir_into_histograms.sh \
      --state_dir <path> \
      --output_dir <path> \
      --save_tag <tag> [options]

Required:
  --state_dir <path>             Directory containing saved active-object states
  --output_dir <path>            Histogram output directory
  --save_tag <tag>               Save tag for the standalone aggregate

Options:
  --tr <int>                     Histogram thermalization cutoff (default: 0)
  --max_sweep <int>              Optional histogram max_sweep
  --request_cpus <int>           Condor request_cpus (default: 1)
  --request_memory <value>       Condor request_memory (default: "5 GB")
  --batch_name <name>            Condor batch_name (default: auto)
  --job_label <label>            Optional label in the submit token
  --no_submit                    Generate files only; do not call condor_submit
  -h, --help                     Show help

Behavior:
  - Submits one aggregation job against exactly the provided state_dir
  - Uses recursive state discovery under state_dir
  - Does not use any base aggregate
  - Runs:
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

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

ensure_cluster_shared_dir_permissions() {
    local path="$1"
    local mode="$2"
    chmod "${mode}" "${path}" 2>/dev/null || true
}

state_dir=""
output_dir=""
save_tag=""
tr_sweeps="0"
max_sweep=""
request_cpus="1"
request_memory="5 GB"
batch_name=""
job_label=""
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --output_dir)
            output_dir="${2:-}"
            shift 2
            ;;
        --save_tag)
            save_tag="${2:-}"
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

if [[ -z "${state_dir}" || -z "${output_dir}" || -z "${save_tag}" ]]; then
    echo "--state_dir, --output_dir, and --save_tag are required."
    usage
    exit 1
fi

if [[ ! -d "${state_dir}" ]]; then
    echo "State directory not found: ${state_dir}"
    exit 1
fi
if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if ! [[ "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    echo "--tr must be an integer. Got '${tr_sweeps}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer. Got '${max_sweep}'."
    exit 1
fi

state_dir="$(cd "${state_dir}" && pwd)"
mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"

timestamp="$(date +%Y%m%d-%H%M%S)"
job_token_base="state_dir_histograms_${timestamp}"
if [[ -n "${job_label}" ]]; then
    job_token_base="${job_label}_${job_token_base}"
fi
job_token="$(sanitize_token "${job_token_base}")"
job_parent="$(dirname "${output_dir}")"
job_root="${job_parent}/manual_histogram_jobs/${job_token}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
mkdir -p "${submit_dir}" "${log_dir}"
ensure_cluster_shared_dir_permissions "${job_parent}" 755
ensure_cluster_shared_dir_permissions "${job_root}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777

if [[ -z "${batch_name}" ]]; then
    batch_name="$(sanitize_token "$(basename "${state_dir}")")_standalone_histograms"
fi

launcher_script="${submit_dir}/run_state_dir_histograms.sh"
submit_file="${submit_dir}/state_dir_histograms.sub"
output_file="${log_dir}/state_dir_histograms.out"
error_file="${log_dir}/state_dir_histograms.err"
log_file="${log_dir}/state_dir_histograms.log"
meta_file="${job_root}/submit_info.txt"

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd $(printf '%q' "${REPO_ROOT}")"
    printf "bash %q --state_dir %q --output_dir %q --save_tag %q --all_states_recursive --min_sweep %q --no_plot" \
        "${AGGREGATE_SCRIPT}" "${state_dir}" "${output_dir}" "${save_tag}" "${tr_sweeps}"
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
state_dir=${state_dir}
output_dir=${output_dir}
save_tag=${save_tag}
tr_sweeps=${tr_sweeps}
max_sweep=${max_sweep}
submit_file=${submit_file}
output_file=${output_file}
error_file=${error_file}
log_file=${log_file}
launcher_script=${launcher_script}
EOF

echo "Prepared active-object standalone state-dir histogram job"
echo "  state_dir: ${state_dir}"
echo "  output_dir: ${output_dir}"
echo "  save_tag: ${save_tag}"
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
