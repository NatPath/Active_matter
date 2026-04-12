#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_active_objects_L64_rho100_d16_hard_refresh_hist.sh \
      --num_replicas <int> \
      --n_sweeps <int> \
      --kappa <value> \
      --ots <int> \
      [--run_id <token>] \
      [--request_cpus <int>] \
      [--request_memory <value>] \
      [--aggregate_request_cpus <int>] \
      [--max_sweep <int>] \
      [--no_submit]

Behavior:
  - Fixed physical setup:
      L = 64, rho = 100, d = 16, hard_refresh
  - Builds a derived config with:
      object_kappa = --kappa
      object_refresh_sweeps = --ots
      object_memory_sweeps = --ots
      object_history_interval = --ots
  - Sets histogram tr = ots automatically
  - Delegates to cluster_scripts/submit_active_objects_histogram_dag.sh

Example:
  bash cluster_scripts/submit_active_objects_L64_rho100_d16_hard_refresh_hist.sh \
      --num_replicas 600 \
      --n_sweeps 10000000 \
      --kappa 1e-6 \
      --ots 10000
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configuration_files/active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k1e-6.yaml"
WRAPPER="${SCRIPT_DIR}/submit_active_objects_histogram_dag.sh"

num_replicas=""
n_sweeps=""
kappa=""
ots=""
run_id=""
request_cpus=""
request_memory=""
aggregate_request_cpus=""
max_sweep=""
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --kappa)
            kappa="${2:-}"
            shift 2
            ;;
        --ots)
            ots="${2:-}"
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
        --aggregate_request_cpus)
            aggregate_request_cpus="${2:-}"
            shift 2
            ;;
        --max_sweep)
            max_sweep="${2:-}"
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

if [[ -z "${num_replicas}" || -z "${n_sweeps}" || -z "${kappa}" || -z "${ots}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Base config not found: ${BASE_CONFIG}"
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi
if ! [[ "${n_sweeps}" =~ ^[0-9]+$ ]] || (( n_sweeps <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps}'."
    exit 1
fi
if ! [[ "${ots}" =~ ^[0-9]+$ ]] || (( ots <= 0 )); then
    echo "--ots must be a positive integer. Got '${ots}'."
    exit 1
fi
if ! [[ "${kappa}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "--kappa must be numeric. Got '${kappa}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer. Got '${max_sweep}'."
    exit 1
fi
if (( ots >= n_sweeps )); then
    echo "--ots (${ots}) must be smaller than --n_sweeps (${n_sweeps})."
    exit 1
fi

sanitize_token() {
    local raw="$1"
    raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
    raw="${raw//+}"
    raw="${raw//[^a-z0-9._-]/-}"
    raw="${raw//--/-}"
    printf '%s' "${raw}"
}

kappa_token="k$(sanitize_token "${kappa}")"
ots_token="oref${ots}"
timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${run_id}" ]]; then
    run_id="active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_${kappa_token}_${ots_token}_nr${num_replicas}_hist_${timestamp}"
fi

tmp_config="$(mktemp "${TMPDIR:-/tmp}/active_objects_L64_rho100_d16_hr_${kappa_token}_${ots_token}.XXXXXX.yaml")"
trap 'rm -f "${tmp_config}"' EXIT

description="active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_${kappa_token}"

awk \
    -v description_line="description: \"${description}\"" \
    -v refresh_line="object_refresh_sweeps: ${ots}" \
    -v memory_line="object_memory_sweeps: ${ots}.0" \
    -v kappa_line="object_kappa: ${kappa}" \
    -v history_line="object_history_interval: ${ots}" '
    BEGIN {
        seen_desc = seen_refresh = seen_memory = seen_kappa = seen_history = 0
    }
    {
        if ($0 ~ /^[[:space:]]*description:[[:space:]]*/) {
            print description_line
            seen_desc = 1
            next
        }
        if ($0 ~ /^[[:space:]]*object_refresh_sweeps:[[:space:]]*/) {
            print refresh_line
            seen_refresh = 1
            next
        }
        if ($0 ~ /^[[:space:]]*object_memory_sweeps:[[:space:]]*/) {
            print memory_line
            seen_memory = 1
            next
        }
        if ($0 ~ /^[[:space:]]*object_kappa:[[:space:]]*/) {
            print kappa_line
            seen_kappa = 1
            next
        }
        if ($0 ~ /^[[:space:]]*object_history_interval:[[:space:]]*/) {
            print history_line
            seen_history = 1
            next
        }
        print
    }
    END {
        if (!seen_desc) print description_line
        if (!seen_refresh) print refresh_line
        if (!seen_memory) print memory_line
        if (!seen_kappa) print kappa_line
        if (!seen_history) print history_line
    }' "${BASE_CONFIG}" > "${tmp_config}"

cmd=(bash "${WRAPPER}"
    --config "${tmp_config}"
    --num_replicas "${num_replicas}"
    --n_sweeps "${n_sweeps}"
    --tr "${ots}"
    --run_id "${run_id}")

if [[ -n "${request_cpus}" ]]; then
    cmd+=(--request_cpus "${request_cpus}")
fi
if [[ -n "${request_memory}" ]]; then
    cmd+=(--request_memory "${request_memory}")
fi
if [[ -n "${aggregate_request_cpus}" ]]; then
    cmd+=(--aggregate_request_cpus "${aggregate_request_cpus}")
fi
if [[ -n "${max_sweep}" ]]; then
    cmd+=(--max_sweep "${max_sweep}")
fi
if [[ "${no_submit}" == "true" ]]; then
    cmd+=(--no_submit)
fi

"${cmd[@]}"
