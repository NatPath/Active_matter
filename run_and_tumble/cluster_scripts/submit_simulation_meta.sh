#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_simulation_meta.sh \
      --simulation <two_force_d|single_origin_force|single_origin_bond> \
      --run_mode <warmup|production|warm_production|warmup_production> \
      --n_sweeps <int> \
      [options]

Required:
  --simulation         Simulation family
  --run_mode           Run mode
  --n_sweeps           Sweeps for selected mode (production sweeps for warm_production)

General options:
  --warmup_n_sweeps    Warmup sweeps for warm_production
  --num_replicas       Number of replicas for production stage (default: 1)
  --replica_strategy   mp or dag (default: mp)
  --request_cpus       Condor request_cpus override
  --request_memory     Condor request_memory override
  --run_label          Optional run label prefix
  --continue_run_id    Continue from production run_id (production mode only)
  --warmup_run_id      Initialize production from warmup run_id (production mode only)
  --warmup_state_dir   Explicit warmup state dir (production mode only)

Simulation parameters:
  --L <int>            System size
  --rho <value>        Density

two_force_d options:
  --d_spacing <linear|log_midpoints>
  --d_min <int>
  --d_max <int>
  --d_step <int>

single_origin_force options:
  --ffr <value>
  --force_strength <value>

Defaults policy (strict):
  - Core params are grouped and defaulted only when none were provided:
      two_force_d core: L,rho defaults to 128,100
      single_origin_force core: L,rho defaults to 256,10000
  - If any core param is provided, all core params must be provided.
  - two_force_d d-range is all-or-none:
      if none provided => defaults d_min=2,d_max=L/4,d_step=2
      if any provided  => require all three
  - two_force_d spacing:
      default          => linear
      --d_spacing log_midpoints => d in {2,4,8,...} ∪ {6,12,24,...} up to d_max (odd d skipped)
  - single_origin_force forcing params are all-or-none:
      if none provided => defaults ffr=1.0,force_strength=1.0
      if any provided  => require both

Examples:
  bash submit_simulation_meta.sh --simulation two_force_d --run_mode warmup --n_sweeps 100000
  bash submit_simulation_meta.sh --simulation single_origin_force --run_mode production --n_sweeps 1000000 --L 256 --rho 10000 --ffr 1.0 --force_strength 1.0
  bash submit_simulation_meta.sh --simulation two_force_d --run_mode warm_production --warmup_n_sweeps 100000 --n_sweeps 1000000 --num_replicas 8 --replica_strategy dag
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

TWO_FORCE_META="${SCRIPT_DIR}/submit_two_force_d_meta.sh"
SINGLE_META="${SCRIPT_DIR}/submit_single_origin_bond_meta.sh"
COPY_SCRIPT="${SCRIPT_DIR}/copy_data_from_cluster.sh"
if [[ ! -f "${TWO_FORCE_META}" || ! -f "${SINGLE_META}" ]]; then
    echo "Missing family meta scripts under ${SCRIPT_DIR}"
    exit 1
fi

simulation=""
run_mode=""
n_sweeps=""
warmup_n_sweeps=""

L_val=""
rho_val=""
d_min=""
d_max=""
d_step=""
d_spacing="linear"
ffr_val=""
force_strength_val=""

num_replicas="1"
replica_strategy="mp"
request_cpus=""
request_memory=""
run_label=""
continue_run_id=""
warmup_run_id=""
warmup_state_dir=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --simulation)
            simulation="${2:-}"
            shift 2
            ;;
        --run_mode)
            run_mode="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --warmup_n_sweeps)
            warmup_n_sweeps="${2:-}"
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
        --d_min)
            d_min="${2:-}"
            shift 2
            ;;
        --d_spacing)
            d_spacing="${2:-}"
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
        --ffr)
            ffr_val="${2:-}"
            shift 2
            ;;
        --force_strength)
            force_strength_val="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --replica_strategy)
            replica_strategy="${2:-}"
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
        --run_label)
            run_label="${2:-}"
            shift 2
            ;;
        --continue_run_id)
            continue_run_id="${2:-}"
            shift 2
            ;;
        --warmup_run_id)
            warmup_run_id="${2:-}"
            shift 2
            ;;
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
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

is_number() {
    local value="$1"
    [[ "${value}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]
}

if [[ -z "${simulation}" || -z "${run_mode}" || -z "${n_sweeps}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

case "${simulation}" in
    single_origin_force|single_origin_bond)
        simulation="single_origin_bond"
        ;;
    two_force_d)
        ;;
    *)
        echo "--simulation must be two_force_d, single_origin_force, or single_origin_bond. Got '${simulation}'."
        exit 1
        ;;
esac

case "${run_mode}" in
    warmup)
        mode_mapped="warmup"
        ;;
    production)
        mode_mapped="production"
        ;;
    warm_production|warmup_production)
        mode_mapped="warmup_production"
        ;;
    *)
        echo "--run_mode must be warmup, production, warm_production, or warmup_production. Got '${run_mode}'."
        exit 1
        ;;
esac

if ! [[ "${n_sweeps}" =~ ^[0-9]+$ ]] || (( n_sweeps <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps}'."
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi
if [[ "${replica_strategy}" != "mp" && "${replica_strategy}" != "dag" ]]; then
    echo "--replica_strategy must be mp or dag. Got '${replica_strategy}'."
    exit 1
fi

if [[ "${mode_mapped}" == "warmup_production" ]]; then
    if [[ -z "${warmup_n_sweeps}" ]]; then
        echo "--warmup_n_sweeps is required for run_mode '${run_mode}'."
        exit 1
    fi
    if ! [[ "${warmup_n_sweeps}" =~ ^[0-9]+$ ]] || (( warmup_n_sweeps <= 0 )); then
        echo "--warmup_n_sweeps must be a positive integer. Got '${warmup_n_sweeps}'."
        exit 1
    fi
    if [[ -n "${continue_run_id}" || -n "${warmup_run_id}" || -n "${warmup_state_dir}" ]]; then
        echo "run_mode '${run_mode}' does not accept --continue_run_id, --warmup_run_id, or --warmup_state_dir."
        exit 1
    fi
else
    if [[ -n "${warmup_n_sweeps}" ]]; then
        echo "--warmup_n_sweeps is only valid with run_mode warm_production/warmup_production."
        exit 1
    fi
fi

if [[ "${mode_mapped}" == "warmup" ]]; then
    if [[ -n "${continue_run_id}" || -n "${warmup_run_id}" || -n "${warmup_state_dir}" ]]; then
        echo "run_mode warmup does not accept --continue_run_id, --warmup_run_id, or --warmup_state_dir."
        exit 1
    fi
fi

defaults_used="false"
if [[ -z "${L_val}" && -z "${rho_val}" ]]; then
    defaults_used="true"
    if [[ "${simulation}" == "two_force_d" ]]; then
        L_val="128"
        rho_val="100"
    else
        L_val="256"
        rho_val="10000"
    fi
elif [[ -n "${L_val}" && -n "${rho_val}" ]]; then
    :
else
    echo "Core parameters are all-or-none: provide both --L and --rho, or neither to use defaults."
    exit 1
fi

if ! [[ "${L_val}" =~ ^[0-9]+$ ]] || (( L_val <= 0 )) || (( L_val % 2 != 0 )); then
    echo "--L must be a positive even integer. Got '${L_val}'."
    exit 1
fi

if [[ "${simulation}" == "two_force_d" ]]; then
    case "${d_spacing}" in
        linear|log_midpoints)
            ;;
        *)
            echo "--d_spacing must be linear or log_midpoints. Got '${d_spacing}'."
            exit 1
            ;;
    esac
    if [[ -z "${d_min}" && -z "${d_max}" && -z "${d_step}" ]]; then
        d_min="2"
        d_max="$((L_val / 4))"
        d_step="2"
    else
        if [[ -z "${d_min}" || -z "${d_max}" || -z "${d_step}" ]]; then
            echo "two_force_d d-range is all-or-none: provide --d_min --d_max --d_step together, or none."
            exit 1
        fi
    fi

    if ! [[ "${d_min}" =~ ^[0-9]+$ ]] || ! [[ "${d_max}" =~ ^[0-9]+$ ]] || ! [[ "${d_step}" =~ ^[0-9]+$ ]]; then
        echo "--d_min, --d_max, and --d_step must be positive integers."
        exit 1
    fi
    if (( d_step <= 0 || d_max < d_min )); then
        echo "Invalid d-range: d_min=${d_min}, d_max=${d_max}, d_step=${d_step}."
        exit 1
    fi
else
    if [[ "${d_spacing}" != "linear" ]]; then
        echo "--d_spacing is only supported for simulation=two_force_d."
        exit 1
    fi
    if [[ -z "${ffr_val}" && -z "${force_strength_val}" ]]; then
        ffr_val="1.0"
        force_strength_val="1.0"
    else
        if [[ -z "${ffr_val}" || -z "${force_strength_val}" ]]; then
            echo "single_origin_force forcing params are all-or-none: provide both --ffr and --force_strength, or none."
            exit 1
        fi
    fi
    if ! is_number "${ffr_val}" || ! is_number "${force_strength_val}"; then
        echo "--ffr and --force_strength must be numeric."
        exit 1
    fi
fi

declare -a cmd
if [[ "${simulation}" == "two_force_d" ]]; then
    cmd=(
        bash "${TWO_FORCE_META}"
        --mode "${mode_mapped}"
        --L "${L_val}"
        --rho "${rho_val}"
        --n_sweeps "${n_sweeps}"
        --d_spacing "${d_spacing}"
        --d_min "${d_min}"
        --d_max "${d_max}"
        --d_step "${d_step}"
        --num_replicas "${num_replicas}"
        --replica_strategy "${replica_strategy}"
    )
else
    cmd=(
        bash "${SINGLE_META}"
        --mode "${mode_mapped}"
        --L "${L_val}"
        --rho "${rho_val}"
        --n_sweeps "${n_sweeps}"
        --ffr "${ffr_val}"
        --force_strength "${force_strength_val}"
        --num_replicas "${num_replicas}"
        --replica_strategy "${replica_strategy}"
    )
fi

if [[ -n "${warmup_n_sweeps}" ]]; then
    cmd+=(--warmup_n_sweeps "${warmup_n_sweeps}")
fi
if [[ -n "${request_cpus}" ]]; then
    cmd+=(--request_cpus "${request_cpus}")
fi
if [[ -n "${request_memory}" ]]; then
    cmd+=(--request_memory "${request_memory}")
fi
if [[ -n "${run_label}" ]]; then
    cmd+=(--run_label "${run_label}")
fi
if [[ -n "${continue_run_id}" ]]; then
    cmd+=(--continue_run_id "${continue_run_id}")
fi
if [[ -n "${warmup_run_id}" ]]; then
    cmd+=(--warmup_run_id "${warmup_run_id}")
fi
if [[ -n "${warmup_state_dir}" ]]; then
    cmd+=(--warmup_state_dir "${warmup_state_dir}")
fi

echo "Submitting normalized run:"
echo "  simulation=${simulation}"
echo "  run_mode=${mode_mapped}"
echo "  defaults_used=${defaults_used}"
echo "  L=${L_val}"
echo "  rho=${rho_val}"
if [[ "${simulation}" == "two_force_d" ]]; then
    echo "  d_spacing=${d_spacing}"
    echo "  d_range=${d_min}:${d_step}:${d_max}"
else
    echo "  ffr=${ffr_val}"
    echo "  force_strength=${force_strength_val}"
fi
echo "  n_sweeps=${n_sweeps}"
if [[ -n "${warmup_n_sweeps}" ]]; then
    echo "  warmup_n_sweeps=${warmup_n_sweeps}"
fi

submit_log="$(mktemp)"
if ! "${cmd[@]}" 2>&1 | tee "${submit_log}"; then
    echo "Submission failed."
    rm -f "${submit_log}"
    exit 1
fi

chain_run_info_path="$(awk -F= '/chain_run_info=/{v=$2} END{print v}' "${submit_log}")"
run_info_path="$(awk -F': ' '/^Run info: /{v=$2} END{print v}' "${submit_log}")"
if [[ -n "${chain_run_info_path}" ]]; then
    run_info_path="${chain_run_info_path}"
fi
rm -f "${submit_log}"

if [[ -z "${run_info_path}" || ! -f "${run_info_path}" ]]; then
    echo "Could not resolve run_info path from submit output."
    exit 0
fi

run_info_val() {
    local key="$1"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

timestamp_now="$(date +%Y%m%d-%H%M%S)"
meta_registry="${REPO_ROOT}/runs/simulation_meta/run_registry.csv"
mkdir -p "$(dirname "${meta_registry}")"
if [[ ! -f "${meta_registry}" ]]; then
    echo "timestamp,simulation,run_mode,run_info,chain_run_id,run_id,warmup_run_id,production_run_id,L,rho,d_min,d_max,d_step,ffr,force_strength,n_sweeps,warmup_n_sweeps,num_replicas,replica_strategy,defaults_used" > "${meta_registry}"
fi

chain_run_id="$(run_info_val chain_run_id)"
run_id="$(run_info_val run_id)"
warmup_run_id_out="$(run_info_val warmup_run_id)"
production_run_id_out="$(run_info_val production_run_id)"
if [[ -z "${run_id}" ]]; then
    run_id="${chain_run_id}"
fi

printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp_now}" "${simulation}" "${mode_mapped}" "${run_info_path}" "${chain_run_id}" "${run_id}" \
    "${warmup_run_id_out}" "${production_run_id_out}" "${L_val}" "${rho_val}" "${d_min}" "${d_max}" "${d_step}" \
    "${ffr_val}" "${force_strength_val}" "${n_sweeps}" "${warmup_n_sweeps}" "${num_replicas}" "${replica_strategy}" "${defaults_used}" >> "${meta_registry}"

echo "Submission summary:"
echo "  run_info=${run_info_path}"
if [[ -n "${chain_run_id}" ]]; then
    echo "  chain_run_id=${chain_run_id}"
fi
if [[ -n "${warmup_run_id_out}" ]]; then
    echo "  warmup_run_id=${warmup_run_id_out}"
fi
if [[ -n "${production_run_id_out}" ]]; then
    echo "  production_run_id=${production_run_id_out}"
fi
if [[ -n "${run_id}" && -z "${production_run_id_out}" ]]; then
    echo "  run_id=${run_id}"
fi
echo "  meta_registry=${meta_registry}"

if [[ -n "${production_run_id_out}" ]]; then
    echo "Fetch production results:"
    echo "  bash ${COPY_SCRIPT} --run_id ${production_run_id_out} --plot"
elif [[ -n "${run_id}" ]]; then
    echo "Fetch results:"
    echo "  bash ${COPY_SCRIPT} --run_id ${run_id} --plot"
fi
