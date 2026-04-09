#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_active_objects_two_objects_hard_refresh_hist.sh \
      --L <int> \
      --rho <value> \
      --d <int> \
      --kappa <value> \
      --ots <int> \
      --num_replicas <int> \
      --n_sweeps <int> \
      [--param_config <path>] \
      [--base_config <path>] \
      [--run_id <token>] \
      [--request_cpus <int>] \
      [--request_memory <value>] \
      [--aggregate_request_cpus <int>] \
      [--max_sweep <int>] \
      [--no_submit]

Behavior:
  - Fixed model family:
      active_objects, 1D, two objects, hard_refresh
  - Accepts either direct flags or a simple YAML parameter file via --param_config
  - Direct flags override values from --param_config
  - Builds a derived runtime config with:
      L = --L
      ρ₀ = --rho
      forcing_distance_d = --d
      object_kappa = --kappa
      object_refresh_sweeps = --ots
      object_memory_sweeps = --ots
      object_history_interval = --ots
  - Sets histogram tr = ots automatically
  - Delegates to cluster_scripts/submit_active_objects_histogram_dag.sh

Simple config keys:
  L, rho (or ρ₀), d (or forcing_distance_d), kappa (or object_kappa),
  ots (or object_refresh_sweeps), num_replicas, n_sweeps,
  request_cpus, request_memory, aggregate_request_cpus, max_sweep, run_id

Example:
  bash cluster_scripts/submit_active_objects_two_objects_hard_refresh_hist.sh \
      --L 32 --rho 100 --d 16 --kappa 5e-6 --ots 10000 \
      --num_replicas 600 --n_sweeps 10000000

  bash cluster_scripts/submit_active_objects_two_objects_hard_refresh_hist.sh \
      --param_config configuration_files/active_objects_1d_two_objects_hard_refresh_submit_example.yaml
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_BASE_CONFIG="${REPO_ROOT}/configuration_files/active_objects_1d_two_objects_hard_refresh_template.yaml"
WRAPPER="${SCRIPT_DIR}/submit_active_objects_histogram_dag.sh"

L=""
rho=""
d=""
kappa=""
ots=""
num_replicas=""
n_sweeps=""
param_config=""
base_config="${DEFAULT_BASE_CONFIG}"
run_id=""
request_cpus=""
request_memory=""
aggregate_request_cpus=""
max_sweep=""
no_submit="false"

read_yaml_value() {
    local file_path="$1"
    local key="$2"
    awk -F: -v wanted="${key}" '
        /^[[:space:]]*#/ {next}
        $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
            value = substr($0, index($0, ":") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"/, "", value)
            gsub(/"$/, "", value)
            print value
            exit
        }' "${file_path}" || true
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --L)
            L="${2:-}"
            shift 2
            ;;
        --rho|--rho0)
            rho="${2:-}"
            shift 2
            ;;
        --d)
            d="${2:-}"
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
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --param_config)
            param_config="${2:-}"
            shift 2
            ;;
        --base_config)
            base_config="${2:-}"
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

if [[ -n "${param_config}" ]]; then
    param_config="$(realpath "${param_config}")"
    [[ -f "${param_config}" ]] || {
        echo "Parameter config not found: ${param_config}"
        exit 1
    }
    [[ -n "${L}" ]] || L="$(read_yaml_value "${param_config}" "L")"
    if [[ -z "${rho}" ]]; then
        rho="$(read_yaml_value "${param_config}" "rho")"
        [[ -n "${rho}" ]] || rho="$(read_yaml_value "${param_config}" "ρ₀")"
    fi
    if [[ -z "${d}" ]]; then
        d="$(read_yaml_value "${param_config}" "d")"
        [[ -n "${d}" ]] || d="$(read_yaml_value "${param_config}" "forcing_distance_d")"
    fi
    if [[ -z "${kappa}" ]]; then
        kappa="$(read_yaml_value "${param_config}" "kappa")"
        [[ -n "${kappa}" ]] || kappa="$(read_yaml_value "${param_config}" "object_kappa")"
    fi
    if [[ -z "${ots}" ]]; then
        ots="$(read_yaml_value "${param_config}" "ots")"
        [[ -n "${ots}" ]] || ots="$(read_yaml_value "${param_config}" "object_refresh_sweeps")"
    fi
    [[ -n "${num_replicas}" ]] || num_replicas="$(read_yaml_value "${param_config}" "num_replicas")"
    [[ -n "${n_sweeps}" ]] || n_sweeps="$(read_yaml_value "${param_config}" "n_sweeps")"
    [[ -n "${run_id}" ]] || run_id="$(read_yaml_value "${param_config}" "run_id")"
    [[ -n "${request_cpus}" ]] || request_cpus="$(read_yaml_value "${param_config}" "request_cpus")"
    [[ -n "${request_memory}" ]] || request_memory="$(read_yaml_value "${param_config}" "request_memory")"
    [[ -n "${aggregate_request_cpus}" ]] || aggregate_request_cpus="$(read_yaml_value "${param_config}" "aggregate_request_cpus")"
    [[ -n "${max_sweep}" ]] || max_sweep="$(read_yaml_value "${param_config}" "max_sweep")"
fi

base_config="$(realpath "${base_config}")"
if [[ ! -f "${base_config}" ]]; then
    echo "Base config not found: ${base_config}"
    exit 1
fi
if [[ -z "${L}" || -z "${rho}" || -z "${d}" || -z "${kappa}" || -z "${ots}" || -z "${num_replicas}" || -z "${n_sweeps}" ]]; then
    echo "Missing required parameters."
    usage
    exit 1
fi

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )); then
    echo "--L must be a positive integer. Got '${L}'."
    exit 1
fi
if ! [[ "${d}" =~ ^[0-9]+$ ]] || (( d < 0 )); then
    echo "--d must be a nonnegative integer. Got '${d}'."
    exit 1
fi
if (( d > L - 2 )); then
    echo "--d (${d}) must satisfy 0 <= d <= L - 2 for two inferred bonds on a ring of size L=${L}."
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
if ! [[ "${rho}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "--rho must be numeric. Got '${rho}'."
    exit 1
fi
if ! [[ "${kappa}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "--kappa must be numeric. Got '${kappa}'."
    exit 1
fi
if [[ -n "${request_cpus}" ]] && { ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); }; then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if [[ -n "${aggregate_request_cpus}" ]] && { ! [[ "${aggregate_request_cpus}" =~ ^[0-9]+$ ]] || (( aggregate_request_cpus <= 0 )); }; then
    echo "--aggregate_request_cpus must be a positive integer. Got '${aggregate_request_cpus}'."
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

rho_token="rho$(sanitize_token "${rho}")"
kappa_token="k$(sanitize_token "${kappa}")"
ots_token="oref${ots}"
timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${run_id}" ]]; then
    run_id="active_objects_1d_two_objects_L${L}_${rho_token}_d${d}_hard_refresh_${kappa_token}_${ots_token}_nr${num_replicas}_hist_${timestamp}"
fi

tmp_config="$(mktemp "${TMPDIR:-/tmp}/active_objects_L${L}_${rho_token}_d${d}_hr_${kappa_token}_${ots_token}.XXXXXX.yaml")"
trap 'rm -f "${tmp_config}"' EXIT

description="active_objects_1d_two_objects_L${L}_${rho_token}_d${d}_hard_refresh_${kappa_token}"

awk \
    -v description_line="description: \"${description}\"" \
    -v L_line="L: ${L}" \
    -v rho_line="ρ₀: ${rho}" \
    -v d_line="forcing_distance_d: ${d}" \
    -v refresh_line="object_refresh_sweeps: ${ots}" \
    -v memory_line="object_memory_sweeps: ${ots}.0" \
    -v kappa_line="object_kappa: ${kappa}" \
    -v history_line="object_history_interval: ${ots}" '
    BEGIN {
        seen_desc = seen_L = seen_rho = seen_d = 0
        seen_refresh = seen_memory = seen_kappa = seen_history = 0
    }
    {
        if ($0 ~ /^[[:space:]]*description:[[:space:]]*/) {
            print description_line
            seen_desc = 1
            next
        }
        if ($0 ~ /^[[:space:]]*L:[[:space:]]*/) {
            print L_line
            seen_L = 1
            next
        }
        if ($0 ~ /^[[:space:]]*ρ₀:[[:space:]]*/) {
            print rho_line
            seen_rho = 1
            next
        }
        if ($0 ~ /^[[:space:]]*forcing_distance_d:[[:space:]]*/) {
            print d_line
            seen_d = 1
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
        if (!seen_L) print L_line
        if (!seen_rho) print rho_line
        if (!seen_d) print d_line
        if (!seen_refresh) print refresh_line
        if (!seen_memory) print memory_line
        if (!seen_kappa) print kappa_line
        if (!seen_history) print history_line
    }' "${base_config}" > "${tmp_config}"

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
