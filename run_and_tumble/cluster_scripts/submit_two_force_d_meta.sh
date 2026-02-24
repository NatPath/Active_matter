#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_two_force_d_meta.sh --mode <warmup|production> --L <int> --rho <value> --n_sweeps <int> [options]

Required:
  --mode              warmup or production
  --L                 system size (even integer)
  --rho               density value for ρ₀
  --n_sweeps          number of sweeps for selected mode

Optional:
  --request_memory    Condor request_memory value (e.g. "4 GB")
  --request_cpus      Condor request_cpus value
  --warmup_state_dir  warmup state directory for production mode
  --d_min             minimum d (default: 2)
  --d_max             maximum d (default: L/4)
  --d_step            d step (default: 2)
  --run_label         optional custom run label prefix
  -h, --help          show this help

Behavior:
  - warmup: submits without initial_state
  - production: requires initial_state for every d and errors if missing
  - creates a run folder under runs/two_force_d/<mode>/<run_id> with
    per-run configs, submit files, logs, states, manifest, and run_info
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

WARMUP_SCRIPT="${SCRIPT_DIR}/submit_two_force_d_warmup.sh"
PRODUCTION_SCRIPT="${SCRIPT_DIR}/submit_two_force_d_production.sh"
if [[ ! -f "${WARMUP_SCRIPT}" || ! -f "${PRODUCTION_SCRIPT}" ]]; then
    echo "Could not find submit scripts in ${SCRIPT_DIR}"
    exit 1
fi

mode=""
L_val=""
rho_val=""
n_sweeps_val=""
request_memory=""
request_cpus=""
warmup_state_dir=""
d_min="2"
d_max=""
d_step="2"
run_label=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --L)
            L_val="${2:-}"
            shift 2
            ;;
        --rho)
            rho_val="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps_val="${2:-}"
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
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
            shift 2
            ;;
        --d_min)
            d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            d_step="${2:-}"
            shift 2
            ;;
        --run_label)
            run_label="${2:-}"
            shift 2
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

if [[ -z "${mode}" || -z "${L_val}" || -z "${rho_val}" || -z "${n_sweeps_val}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ "${mode}" != "warmup" && "${mode}" != "production" ]]; then
    echo "--mode must be 'warmup' or 'production'."
    exit 1
fi

if ! [[ "${L_val}" =~ ^[0-9]+$ ]] || (( L_val <= 0 )) || (( L_val % 2 != 0 )); then
    echo "--L must be a positive even integer. Got '${L_val}'."
    exit 1
fi

if ! [[ "${n_sweeps_val}" =~ ^[0-9]+$ ]] || (( n_sweeps_val <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps_val}'."
    exit 1
fi

if ! [[ "${d_min}" =~ ^[0-9]+$ ]] || ! [[ "${d_step}" =~ ^[0-9]+$ ]]; then
    echo "--d_min and --d_step must be positive integers."
    exit 1
fi

if [[ -z "${d_max}" ]]; then
    d_max="$((L_val / 4))"
fi
if ! [[ "${d_max}" =~ ^[0-9]+$ ]]; then
    echo "--d_max must be an integer. Got '${d_max}'."
    exit 1
fi
if (( d_step <= 0 || d_max < d_min )); then
    echo "Invalid d range: d_min=${d_min}, d_max=${d_max}, d_step=${d_step}."
    exit 1
fi

export L="${L_val}"
export RHO0="${rho_val}"
export D_MIN="${d_min}"
export D_MAX="${d_max}"
export D_STEP="${d_step}"

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

timestamp="$(date +%Y%m%d-%H%M%S)"
rho_tag="$(slugify "${rho_val}")"
if [[ -z "${run_label}" ]]; then
    run_label="${mode}_L${L_val}_rho${rho_tag}_ns${n_sweeps_val}_d${d_min}-${d_max}-s${d_step}"
fi
run_label="$(slugify "${run_label}")"
run_id="${run_label}_${timestamp}"
run_root="${REPO_ROOT}/runs/two_force_d/${mode}/${run_id}"
run_config_dir="${run_root}/configs"
run_submit_dir="${run_root}/submit"
run_log_dir="${run_root}/logs"
run_state_dir="${run_root}/states"
run_report_dir="${run_root}/reports"
run_manifest="${run_root}/manifest.csv"
run_info="${run_root}/run_info.txt"
registry_file="${REPO_ROOT}/runs/two_force_d/run_registry.csv"

mkdir -p "${run_config_dir}" "${run_submit_dir}" "${run_log_dir}" "${run_state_dir}" "${run_report_dir}"

export RUN_ID="${run_id}"
export JOB_BATCH_NAME="${run_id}"
export RUN_CONFIG_DIR="${run_config_dir}"
export RUN_SUBMIT_DIR="${run_submit_dir}"
export RUN_LOG_DIR="${run_log_dir}"
export RUN_STATE_DIR="${run_state_dir}"
export MANIFEST_PATH="${run_manifest}"

if [[ -n "${request_memory}" ]]; then
    export REQUEST_MEMORY="${request_memory}"
fi
if [[ -n "${request_cpus}" ]]; then
    export REQUEST_CPUS="${request_cpus}"
fi

request_cpus_effective="${REQUEST_CPUS:-1}"
request_memory_effective="${REQUEST_MEMORY:-2 GB}"

echo "Preparing two-force d sweep:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  L=${L}"
echo "  rho0=${RHO0}"
echo "  n_sweeps=${n_sweeps_val}"
echo "  d range: ${D_MIN}:${D_STEP}:${D_MAX}"
echo "  request_cpus=${request_cpus_effective}"
echo "  request_memory=${request_memory_effective}"
echo "  run_root=${run_root}"
echo "  run_logs=${run_log_dir}"
echo "  run_states=${run_state_dir}"

if [[ "${mode}" == "production" && -z "${warmup_state_dir}" ]]; then
    if [[ -f "${registry_file}" ]]; then
        warmup_state_dir="$(
            awk -F, -v L="${L}" -v rho="${RHO0}" -v dmin="${D_MIN}" -v dmax="${D_MAX}" -v dstep="${D_STEP}" '
                $3=="warmup" && $4==L && $5==rho && $7==dmin && $8==dmax && $9==dstep {
                    state_dir=$14
                }
                END {
                    if (state_dir != "") print state_dir
                }' "${registry_file}"
        )"
    fi
    if [[ -z "${warmup_state_dir}" ]]; then
        warmup_state_dir="${REPO_ROOT}/saved_states/two_force_d_sweep/warmup"
    fi
elif [[ "${mode}" == "warmup" && -z "${warmup_state_dir}" ]]; then
    warmup_state_dir="${run_state_dir}"
fi

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=${mode}
L=${L}
rho0=${RHO0}
n_sweeps=${n_sweeps_val}
d_min=${D_MIN}
d_max=${D_MAX}
d_step=${D_STEP}
request_cpus=${request_cpus_effective}
request_memory=${request_memory_effective}
run_root=${run_root}
config_dir=${run_config_dir}
submit_dir=${run_submit_dir}
log_dir=${run_log_dir}
state_dir=${run_state_dir}
manifest=${run_manifest}
warmup_state_dir=${warmup_state_dir}
EOF

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,d_min,d_max,d_step,request_cpus,request_memory,run_root,log_dir,state_dir,warmup_state_dir" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "${mode}" "${L}" "${RHO0}" "${n_sweeps_val}" \
    "${D_MIN}" "${D_MAX}" "${D_STEP}" "${request_cpus_effective}" "${request_memory_effective}" \
    "${run_root}" "${run_log_dir}" "${run_state_dir}" "${warmup_state_dir}" >> "${registry_file}"

if [[ "${mode}" == "warmup" ]]; then
    export WARMUP_SWEEPS="${n_sweeps_val}"
    bash "${WARMUP_SCRIPT}"
else
    export PRODUCTION_SWEEPS="${n_sweeps_val}"
    export REQUIRE_INITIAL_STATE="true"
    export WARMUP_STATE_DIR="${warmup_state_dir}"
    bash "${PRODUCTION_SCRIPT}"
fi

echo "Run manifest: ${run_manifest}"
echo "Run info: ${run_info}"
