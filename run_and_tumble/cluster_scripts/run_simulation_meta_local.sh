#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_simulation_meta_local.sh \
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
  --status_interval    Progress snapshot interval in seconds (default: 20)
  --plot_sweeps        Enable per-sweep plotting in local runs
  --run_label          Optional run label prefix
  --warmup_state_dir   Production-only: explicit warmup state directory

Simulation parameters:
  --L <int>            System size
  --rho <value>        Density

two_force_d options:
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
  - single_origin_force forcing params are all-or-none:
      if none provided => defaults ffr=1.0,force_strength=1.0
      if any provided  => require both

Examples:
  bash run_simulation_meta_local.sh --simulation two_force_d --run_mode warmup --n_sweeps 50000
  bash run_simulation_meta_local.sh --simulation single_origin_force --run_mode production --n_sweeps 200000 --L 256 --rho 10000 --ffr 1.0 --force_strength 1.0
  bash run_simulation_meta_local.sh --simulation two_force_d --run_mode warm_production --warmup_n_sweeps 50000 --n_sweeps 200000 --num_replicas 4
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

TWO_FORCE_LOCAL="${SCRIPT_DIR}/run_two_force_d_meta_local.sh"
SINGLE_LOCAL="${SCRIPT_DIR}/run_single_origin_bond_meta_local.sh"
if [[ ! -f "${TWO_FORCE_LOCAL}" || ! -f "${SINGLE_LOCAL}" ]]; then
    echo "Missing local family meta scripts under ${SCRIPT_DIR}"
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
ffr_val=""
force_strength_val=""

num_replicas="1"
status_interval="20"
plot_sweeps="false"
run_label=""
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
        --status_interval)
            status_interval="${2:-}"
            shift 2
            ;;
        --plot_sweeps)
            plot_sweeps="true"
            shift 1
            ;;
        --run_label)
            run_label="${2:-}"
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

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

run_and_capture_run_info() {
    local -a cmd=("$@")
    local temp_log
    temp_log="$(mktemp)"
    if ! "${cmd[@]}" 2>&1 | tee "${temp_log}"; then
        rm -f "${temp_log}"
        return 1
    fi
    local info
    info="$(awk -F': ' '/^Run info: /{v=$2} END{print v}' "${temp_log}")"
    rm -f "${temp_log}"
    printf "%s" "${info}"
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
if ! [[ "${status_interval}" =~ ^[0-9]+$ ]] || (( status_interval <= 0 )); then
    echo "--status_interval must be a positive integer. Got '${status_interval}'."
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
    if [[ -n "${warmup_state_dir}" ]]; then
        echo "run_mode '${run_mode}' does not accept --warmup_state_dir (it uses the warmup stage output)."
        exit 1
    fi
else
    if [[ -n "${warmup_n_sweeps}" ]]; then
        echo "--warmup_n_sweeps is only valid with run_mode warm_production/warmup_production."
        exit 1
    fi
fi

if [[ "${mode_mapped}" == "warmup" && -n "${warmup_state_dir}" ]]; then
    echo "--warmup_state_dir is only valid for production mode."
    exit 1
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

echo "Running normalized local simulation:"
echo "  simulation=${simulation}"
echo "  run_mode=${mode_mapped}"
echo "  defaults_used=${defaults_used}"
echo "  L=${L_val}"
echo "  rho=${rho_val}"
if [[ "${simulation}" == "two_force_d" ]]; then
    echo "  d_range=${d_min}:${d_step}:${d_max}"
else
    echo "  ffr=${ffr_val}"
    echo "  force_strength=${force_strength_val}"
fi
echo "  n_sweeps=${n_sweeps}"
if [[ -n "${warmup_n_sweeps}" ]]; then
    echo "  warmup_n_sweeps=${warmup_n_sweeps}"
fi
echo "  num_replicas=${num_replicas}"
echo "  status_interval=${status_interval}"
echo "  plot_sweeps=${plot_sweeps}"

assemble_local_cmd() {
    local -n cmd_ref="$1"
    local family_mode="$2"
    local family_num_replicas="$3"
    local local_warmup_state_dir="$4"
    local local_run_label="$5"
    local local_n_sweeps="$6"

    cmd_ref=()
    if [[ "${simulation}" == "two_force_d" ]]; then
        cmd_ref=(
            bash "${TWO_FORCE_LOCAL}"
            --mode "${family_mode}"
            --L "${L_val}"
            --rho "${rho_val}"
            --n_sweeps "${local_n_sweeps}"
            --d_min "${d_min}"
            --d_max "${d_max}"
            --d_step "${d_step}"
            --num_replicas "${family_num_replicas}"
            --status_interval "${status_interval}"
        )
    else
        cmd_ref=(
            bash "${SINGLE_LOCAL}"
            --mode "${family_mode}"
            --L "${L_val}"
            --rho "${rho_val}"
            --n_sweeps "${local_n_sweeps}"
            --ffr "${ffr_val}"
            --force_strength "${force_strength_val}"
            --num_replicas "${family_num_replicas}"
            --status_interval "${status_interval}"
        )
    fi

    if [[ "${plot_sweeps}" == "true" ]]; then
        cmd_ref+=(--plot_sweeps)
    fi
    if [[ -n "${local_run_label}" ]]; then
        cmd_ref+=(--run_label "${local_run_label}")
    fi
    if [[ -n "${local_warmup_state_dir}" ]]; then
        cmd_ref+=(--warmup_state_dir "${local_warmup_state_dir}")
    fi
}

final_run_info=""
chain_run_id=""
warmup_run_id=""
production_run_id=""

if [[ "${mode_mapped}" == "warmup_production" ]]; then
    chain_timestamp="$(date +%Y%m%d-%H%M%S)"
    rho_tag="$(slugify "${rho_val}")"
    if [[ -n "${run_label}" ]]; then
        chain_base="$(slugify "${run_label}")"
    else
        if [[ "${simulation}" == "two_force_d" ]]; then
            chain_base="local_two_force_warmup_production_L${L_val}_rho${rho_tag}_wns${warmup_n_sweeps}_pns${n_sweeps}_d${d_min}-${d_max}-s${d_step}"
        else
            chain_base="local_single_warmup_production_L${L_val}_rho${rho_tag}_wns${warmup_n_sweeps}_pns${n_sweeps}_f${force_strength_val}_ffr${ffr_val}"
        fi
        if (( num_replicas > 1 )); then
            chain_base="${chain_base}_nr${num_replicas}"
        fi
    fi

    warmup_label="${chain_base}_warmup"
    production_label="${chain_base}_production"
    chain_run_id="${chain_base}_${chain_timestamp}"

    assemble_local_cmd warmup_cmd "warmup" "1" "" "${warmup_label}" "${warmup_n_sweeps}"
    warmup_info="$(run_and_capture_run_info "${warmup_cmd[@]}")"
    if [[ -z "${warmup_info}" || ! -f "${warmup_info}" ]]; then
        echo "Failed to resolve warmup run_info."
        exit 1
    fi
    warmup_state_dir_resolved="$(run_info_value "${warmup_info}" "state_dir")"
    warmup_run_id="$(run_info_value "${warmup_info}" "run_id")"
    if [[ -z "${warmup_state_dir_resolved}" || ! -d "${warmup_state_dir_resolved}" ]]; then
        echo "Warmup state_dir is invalid: ${warmup_state_dir_resolved}"
        exit 1
    fi

    assemble_local_cmd production_cmd "production" "${num_replicas}" "${warmup_state_dir_resolved}" "${production_label}" "${n_sweeps}"
    production_info="$(run_and_capture_run_info "${production_cmd[@]}")"
    if [[ -z "${production_info}" || ! -f "${production_info}" ]]; then
        echo "Failed to resolve production run_info."
        exit 1
    fi
    production_run_id="$(run_info_value "${production_info}" "run_id")"

    chain_root="${REPO_ROOT}/runs/simulation_meta/local/warmup_production/${chain_run_id}"
    mkdir -p "${chain_root}"
    final_run_info="${chain_root}/run_info.txt"
    cat > "${final_run_info}" <<EOF
chain_run_id=${chain_run_id}
timestamp=${chain_timestamp}
simulation=${simulation}
run_mode=warmup_production
L=${L_val}
rho0=${rho_val}
warmup_n_sweeps=${warmup_n_sweeps}
production_n_sweeps=${n_sweeps}
num_replicas=${num_replicas}
d_min=${d_min}
d_max=${d_max}
d_step=${d_step}
ffr=${ffr_val}
force_strength=${force_strength_val}
warmup_run_id=${warmup_run_id}
warmup_run_info=${warmup_info}
warmup_state_dir=${warmup_state_dir_resolved}
production_run_id=${production_run_id}
production_run_info=${production_info}
EOF
else
    assemble_local_cmd one_cmd "${mode_mapped}" "${num_replicas}" "${warmup_state_dir}" "${run_label}" "${n_sweeps}"
    final_run_info="$(run_and_capture_run_info "${one_cmd[@]}")"
    if [[ -z "${final_run_info}" || ! -f "${final_run_info}" ]]; then
        echo "Failed to resolve run_info."
        exit 1
    fi
fi

meta_registry="${REPO_ROOT}/runs/simulation_meta/local_run_registry.csv"
mkdir -p "$(dirname "${meta_registry}")"
if [[ ! -f "${meta_registry}" ]]; then
    echo "timestamp,simulation,run_mode,run_info,chain_run_id,run_id,warmup_run_id,production_run_id,L,rho,d_min,d_max,d_step,ffr,force_strength,n_sweeps,warmup_n_sweeps,num_replicas,status_interval,plot_sweeps,defaults_used" > "${meta_registry}"
fi

resolved_run_id="$(run_info_value "${final_run_info}" "run_id")"
if [[ -z "${resolved_run_id}" ]]; then
    resolved_run_id="$(run_info_value "${final_run_info}" "chain_run_id")"
fi
if [[ -z "${warmup_run_id}" ]]; then
    warmup_run_id="$(run_info_value "${final_run_info}" "warmup_run_id")"
fi
if [[ -z "${production_run_id}" ]]; then
    production_run_id="$(run_info_value "${final_run_info}" "production_run_id")"
fi
if [[ -z "${chain_run_id}" ]]; then
    chain_run_id="$(run_info_value "${final_run_info}" "chain_run_id")"
fi

printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$(date +%Y%m%d-%H%M%S)" "${simulation}" "${mode_mapped}" "${final_run_info}" "${chain_run_id}" "${resolved_run_id}" \
    "${warmup_run_id}" "${production_run_id}" "${L_val}" "${rho_val}" "${d_min}" "${d_max}" "${d_step}" \
    "${ffr_val}" "${force_strength_val}" "${n_sweeps}" "${warmup_n_sweeps}" "${num_replicas}" "${status_interval}" "${plot_sweeps}" "${defaults_used}" >> "${meta_registry}"

echo "Local unified run summary:"
echo "  run_info=${final_run_info}"
if [[ -n "${chain_run_id}" ]]; then
    echo "  chain_run_id=${chain_run_id}"
fi
if [[ -n "${warmup_run_id}" ]]; then
    echo "  warmup_run_id=${warmup_run_id}"
fi
if [[ -n "${production_run_id}" ]]; then
    echo "  production_run_id=${production_run_id}"
fi
if [[ -n "${resolved_run_id}" && -z "${production_run_id}" ]]; then
    echo "  run_id=${resolved_run_id}"
fi
echo "  registry=${meta_registry}"
