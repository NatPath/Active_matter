#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_two_force_d_meta.sh --n_sweeps <int> [--mode <warmup|production|warmup_production>] [--L <int>] [--rho <value>] [options]

Required:
  --n_sweeps          number of sweeps for selected mode (production sweeps for warmup_production)
  --mode/--L/--rho    required unless --continue_run_id or --warmup_run_id is provided (warmup_production always requires them)

Optional:
  --request_memory    Condor request_memory value (e.g. "5 GB")
  --request_cpus      Condor request_cpus value
  --num_replicas      number of independent replicas to run and aggregate per job (default: 1)
  --replica_strategy  replica execution strategy on cluster: mp or dag (default: mp)
  --warmup_n_sweeps   warmup sweeps for --mode warmup_production
  --warmup_run_id     specific warmup run_id to initialize production from
  --warmup_state_dir  warmup state directory for production mode
  --continue_run_id   continue accumulation from an existing run_id (same mode)
  --d_spacing         d spacing mode: linear (default) or log_midpoints
  --d_min             minimum d (default: 2)
  --d_max             maximum d (default: L/4)
  --d_step            d step (default: 2)
  --run_label         optional custom run label prefix
  -h, --help          show this help

Behavior:
  - warmup: submits without initial_state
  - production initialization: starts from warmup state (via --warmup_run_id, --warmup_state_dir, or auto-registry lookup)
  - production continuation: starts from latest state under --continue_run_id
  - warmup_production: submits one chained DAG
      1) warmup single-process sweep per d (no replicas)
      2) production run(s) initialized from those warmup states (num_replicas + replica_strategy)
  - with --replica_strategy mp and --num_replicas > 1: each d uses one Condor job with Julia multi-process workers, saving one aggregated state per d
    default request_cpus is num_replicas + 1 (main process + workers)
  - with --replica_strategy dag and --num_replicas > 1: each d uses DAG nodes (replica jobs -> aggregation child)
    default request_cpus is 1 per replica node (unless --request_cpus is provided)
  - with --continue_run_id: mode/L/rho (and default d-range) are inferred from registry
  - with --continue_run_id: submits with --continue per d and accumulates averages
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
SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
if [[ ! -f "${WARMUP_SCRIPT}" || ! -f "${PRODUCTION_SCRIPT}" ]]; then
    echo "Could not find submit scripts in ${SCRIPT_DIR}"
    exit 1
fi
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Could not find spacing utils script: ${SPACING_UTILS}"
    exit 1
fi
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"
source "${SPACING_UTILS}"

registry_file="${REPO_ROOT}/runs/two_force_d/run_registry.csv"

lookup_registry_row_by_run_id() {
    local lookup_run_id="$1"
    local registry_path="$2"
    awk -F, -v rid="${lookup_run_id}" '
        NR == 1 {next}
        $2 == rid {row = $0}
        END {print row}
    ' "${registry_path}"
}

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

infer_d_spacing_from_run_id() {
    local rid="$1"
    if [[ "${rid}" =~ -lm(_|$) ]]; then
        echo "log_midpoints"
    else
        echo "linear"
    fi
}

submit_chained_warmup_production() {
    local self_script="${SCRIPT_DIR}/submit_two_force_d_meta.sh"
    if [[ ! -f "${self_script}" ]]; then
        self_script="${REPO_ROOT}/cluster_scripts/submit_two_force_d_meta.sh"
    fi
    if [[ ! -f "${self_script}" ]]; then
        echo "Could not locate submit_two_force_d_meta.sh for chained submission."
        exit 1
    fi

    local_slugify() {
        printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
    }

    local chain_no_submit="${NO_SUBMIT:-false}"
    local chain_timestamp
    chain_timestamp="$(date +%Y%m%d-%H%M%S)"
    local rho_tag
    rho_tag="$(local_slugify "${rho_val}")"
    local d_max_for_label="${d_max:-auto}"
    local spacing_tag
    spacing_tag="$(two_force_d_spacing_tag "${d_spacing}" "${d_step}")"
    local chain_base
    if [[ -n "${run_label}" ]]; then
        chain_base="$(local_slugify "${run_label}")"
        if [[ "${chain_base}" != *"-${spacing_tag}"* && "${chain_base}" != *"_${spacing_tag}"* ]]; then
            chain_base="${chain_base}_${spacing_tag}"
        fi
    else
        chain_base="two_force_warmup_production_L${L_val}_rho${rho_tag}_wns${warmup_n_sweeps}_pns${n_sweeps_val}_d${d_min}-${d_max_for_label}-${spacing_tag}"
        if (( num_replicas > 1 )); then
            chain_base="${chain_base}_nr${num_replicas}_${replica_strategy}"
        fi
    fi

    local warmup_label="${chain_base}_warmup"
    local production_label="${chain_base}_production"

    local -a warmup_cmd
    warmup_cmd=(
        bash "${self_script}"
        --mode warmup
        --L "${L_val}"
        --rho "${rho_val}"
        --n_sweeps "${warmup_n_sweeps}"
        --num_replicas 1
        --replica_strategy mp
        --request_cpus 1
        --d_spacing "${d_spacing}"
        --d_min "${d_min}"
        --d_step "${d_step}"
        --run_label "${warmup_label}"
    )
    if [[ -n "${d_max}" ]]; then
        warmup_cmd+=(--d_max "${d_max}")
    fi
    if [[ -n "${request_memory}" ]]; then
        warmup_cmd+=(--request_memory "${request_memory}")
    fi

    echo "Preparing chained warmup stage (single process per d, NO_SUBMIT)..."
    local warmup_output
    warmup_output="$(
        NO_SUBMIT=true NO_DAG_NOTIFICATION=true "${warmup_cmd[@]}"
    )"
    printf "%s\n" "${warmup_output}"

    local warmup_run_info
    warmup_run_info="$(printf "%s\n" "${warmup_output}" | awk -F': ' '/^Run info: /{print $2}' | tail -n 1)"
    if [[ -z "${warmup_run_info}" || ! -f "${warmup_run_info}" ]]; then
        echo "Failed to resolve warmup run_info from chained warmup stage."
        exit 1
    fi
    local warmup_run_id_local warmup_state_dir_local warmup_manifest_local
    local warmup_d_min_local warmup_d_max_local warmup_d_step_local
    warmup_run_id_local="$(read_run_info_value "${warmup_run_info}" "run_id")"
    warmup_state_dir_local="$(read_run_info_value "${warmup_run_info}" "state_dir")"
    warmup_manifest_local="$(read_run_info_value "${warmup_run_info}" "manifest")"
    warmup_d_min_local="$(read_run_info_value "${warmup_run_info}" "d_min")"
    warmup_d_max_local="$(read_run_info_value "${warmup_run_info}" "d_max")"
    warmup_d_step_local="$(read_run_info_value "${warmup_run_info}" "d_step")"
    if [[ -z "${warmup_manifest_local}" || ! -f "${warmup_manifest_local}" ]]; then
        echo "Warmup manifest is missing: ${warmup_manifest_local}"
        exit 1
    fi

    local -a production_cmd
    production_cmd=(
        bash "${self_script}"
        --mode production
        --L "${L_val}"
        --rho "${rho_val}"
        --n_sweeps "${n_sweeps_val}"
        --num_replicas "${num_replicas}"
        --replica_strategy "${replica_strategy}"
        --warmup_state_dir "${warmup_state_dir_local}"
        --d_spacing "${d_spacing}"
        --d_min "${warmup_d_min_local}"
        --d_max "${warmup_d_max_local}"
        --d_step "${warmup_d_step_local}"
        --run_label "${production_label}"
    )
    if [[ -n "${request_memory}" ]]; then
        production_cmd+=(--request_memory "${request_memory}")
    fi
    if [[ -n "${request_cpus}" ]]; then
        production_cmd+=(--request_cpus "${request_cpus}")
    fi

    echo "Preparing chained production stage (NO_SUBMIT, deferred warmup state lookup)..."
    local production_output
    production_output="$(
        NO_SUBMIT=true NO_DAG_NOTIFICATION=true DEFER_INITIAL_STATE_LOOKUP=true "${production_cmd[@]}"
    )"
    printf "%s\n" "${production_output}"

    local production_run_info
    production_run_info="$(printf "%s\n" "${production_output}" | awk -F': ' '/^Run info: /{print $2}' | tail -n 1)"
    if [[ -z "${production_run_info}" || ! -f "${production_run_info}" ]]; then
        echo "Failed to resolve production run_info from chained production stage."
        exit 1
    fi

    local production_run_id_local production_manifest_local production_submit_dir
    production_run_id_local="$(read_run_info_value "${production_run_info}" "run_id")"
    production_manifest_local="$(read_run_info_value "${production_run_info}" "manifest")"
    production_submit_dir="$(read_run_info_value "${production_run_info}" "submit_dir")"
    if [[ -z "${production_manifest_local}" || ! -f "${production_manifest_local}" ]]; then
        echo "Production manifest is missing: ${production_manifest_local}"
        exit 1
    fi

    local chain_run_id="${chain_base}_${chain_timestamp}"
    local chain_root="${REPO_ROOT}/runs/two_force_d/warmup_production/${chain_run_id}"
    local chain_submit_dir="${chain_root}/submit"
    local chain_log_dir="${chain_root}/logs"
    local chain_run_info_file="${chain_root}/run_info.txt"
    local chain_dag_file="${chain_submit_dir}/two_force_d_warmup_production.dag"
    mkdir -p "${chain_submit_dir}" "${chain_log_dir}"
    : > "${chain_dag_file}"

    local -A warmup_job_by_d=()
    local -a warmup_job_ids=()
    local job_id
    while IFS=, read -r mf_d mf_mode mf_cluster mf_cfg mf_submit mf_out mf_err mf_log mf_save mf_init; do
        if [[ "${mf_d}" == "d" ]]; then
            continue
        fi
        if [[ -z "${mf_d}" || -z "${mf_submit}" ]]; then
            continue
        fi
        if [[ ! -f "${mf_submit}" ]]; then
            echo "Warmup submit file missing for d=${mf_d}: ${mf_submit}"
            exit 1
        fi
        job_id="W${mf_d}"
        warmup_job_by_d["${mf_d}"]="${job_id}"
        warmup_job_ids+=("${job_id}")
        printf "JOB %s %s\n" "${job_id}" "${mf_submit}" >> "${chain_dag_file}"
    done < "${warmup_manifest_local}"

    if (( ${#warmup_job_ids[@]} == 0 )); then
        echo "No warmup jobs were generated for chained run."
        exit 1
    fi

    if [[ "${replica_strategy}" == "dag" && "${num_replicas}" -gt 1 ]]; then
        local production_dag_file="${production_submit_dir}/two_force_d_production.dag"
        if [[ ! -f "${production_dag_file}" ]]; then
            echo "Production DAG file missing: ${production_dag_file}"
            exit 1
        fi
        printf "SUBDAG EXTERNAL PRODUCTION %s\n" "${production_dag_file}" >> "${chain_dag_file}"
        printf "PARENT %s CHILD PRODUCTION\n" "${warmup_job_ids[*]}" >> "${chain_dag_file}"
    else
        while IFS=, read -r mf_d mf_mode mf_cluster mf_cfg mf_submit mf_out mf_err mf_log mf_save mf_init; do
            if [[ "${mf_d}" == "d" ]]; then
                continue
            fi
            if [[ -z "${mf_d}" || -z "${mf_submit}" ]]; then
                continue
            fi
            if [[ ! -f "${mf_submit}" ]]; then
                echo "Production submit file missing for d=${mf_d}: ${mf_submit}"
                exit 1
            fi
            if [[ -z "${warmup_job_by_d[${mf_d}]:-}" ]]; then
                echo "No matching warmup job found for production d=${mf_d}."
                exit 1
            fi
            job_id="P${mf_d}"
            printf "JOB %s %s\n" "${job_id}" "${mf_submit}" >> "${chain_dag_file}"
            printf "PARENT %s CHILD %s\n" "${warmup_job_by_d[${mf_d}]}" "${job_id}" >> "${chain_dag_file}"
        done < "${production_manifest_local}"
    fi

    dag_append_final_notification_node "${chain_dag_file}" "${chain_submit_dir}" "${chain_log_dir}" "${chain_root}" "${chain_run_id}" "two_force_d_warmup_production" "${REPO_ROOT}"

    local chain_cluster_id
    if [[ "${chain_no_submit}" == "true" ]]; then
        echo "NO_SUBMIT=true; generated chained DAG but not submitting: ${chain_dag_file}"
        chain_cluster_id="NO_SUBMIT"
    else
        local submit_output
        submit_output="$(condor_submit_dag "${chain_dag_file}")"
        echo "${submit_output}"
        chain_cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
        chain_cluster_id="${chain_cluster_id:-NA}"
    fi

    cat > "${chain_run_info_file}" <<EOF
chain_run_id=${chain_run_id}
timestamp=${chain_timestamp}
mode=warmup_production
L=${L_val}
rho0=${rho_val}
warmup_n_sweeps=${warmup_n_sweeps}
production_n_sweeps=${n_sweeps_val}
num_replicas=${num_replicas}
replica_strategy=${replica_strategy}
d_min=${warmup_d_min_local}
d_max=${warmup_d_max_local}
d_step=${warmup_d_step_local}
d_spacing=${d_spacing}
d_values=${d_values_csv}
warmup_run_id=${warmup_run_id_local}
warmup_run_info=${warmup_run_info}
warmup_manifest=${warmup_manifest_local}
warmup_state_dir=${warmup_state_dir_local}
production_run_id=${production_run_id_local}
production_run_info=${production_run_info}
production_manifest=${production_manifest_local}
chain_dag=${chain_dag_file}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
chain_cluster_id=${chain_cluster_id}
EOF

    echo "Chained submission prepared:"
    echo "  chain_run_id=${chain_run_id}"
    echo "  warmup_run_id=${warmup_run_id_local}"
    echo "  production_run_id=${production_run_id_local}"
    echo "  chain_dag=${chain_dag_file}"
    echo "  chain_cluster_id=${chain_cluster_id}"
    echo "  chain_run_info=${chain_run_info_file}"
}

mode=""
L_val=""
rho_val=""
n_sweeps_val=""
warmup_n_sweeps=""
request_memory=""
request_cpus=""
num_replicas="1"
replica_strategy="mp"
warmup_state_dir=""
warmup_run_id=""
d_min="2"
d_max=""
d_step="2"
d_spacing="linear"
d_spacing_set="false"
d_min_set="false"
d_max_set="false"
d_step_set="false"
run_label=""
continue_run_id=""

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
        --warmup_n_sweeps)
            warmup_n_sweeps="${2:-}"
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
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --replica_strategy)
            replica_strategy="${2:-}"
            shift 2
            ;;
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
            shift 2
            ;;
        --warmup_run_id)
            warmup_run_id="${2:-}"
            shift 2
            ;;
        --continue_run_id)
            continue_run_id="${2:-}"
            shift 2
            ;;
        --d_min)
            d_min="${2:-}"
            d_min_set="true"
            shift 2
            ;;
        --d_spacing)
            d_spacing="${2:-}"
            d_spacing_set="true"
            shift 2
            ;;
        --d_max)
            d_max="${2:-}"
            d_max_set="true"
            shift 2
            ;;
        --d_step)
            d_step="${2:-}"
            d_step_set="true"
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

d_spacing="$(two_force_d_normalize_spacing_mode "${d_spacing}")" || {
    echo "--d_spacing must be linear or log_midpoints. Got '${d_spacing}'."
    exit 1
}

if [[ -z "${n_sweeps_val}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

continue_state_dir=""
if [[ -n "${continue_run_id}" ]]; then
    if [[ ! -f "${registry_file}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': registry file not found: ${registry_file}"
        exit 1
    fi

    continue_registry_row="$(lookup_registry_row_by_run_id "${continue_run_id}" "${registry_file}")"
    if [[ -z "${continue_registry_row}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': not found in ${registry_file}"
        exit 1
    fi

    IFS=',' read -r cont_ts cont_run_id cont_mode cont_L cont_rho cont_ns cont_dmin cont_dmax cont_dstep cont_cpus cont_mem cont_run_root cont_log_dir cont_state_dir cont_warmup_state_dir <<< "${continue_registry_row}"

    if [[ -z "${mode}" ]]; then
        mode="${cont_mode}"
    elif [[ "${mode}" != "${cont_mode}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': source mode='${cont_mode}', requested mode='${mode}'."
        exit 1
    fi

    if [[ -z "${L_val}" ]]; then
        L_val="${cont_L}"
    elif [[ "${L_val}" != "${cont_L}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': source L='${cont_L}', requested L='${L_val}'."
        exit 1
    fi

    if [[ -z "${rho_val}" ]]; then
        rho_val="${cont_rho}"
    elif [[ "${rho_val}" != "${cont_rho}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': source rho='${cont_rho}', requested rho='${rho_val}'."
        exit 1
    fi

    if [[ "${d_min_set}" != "true" && -n "${cont_dmin}" ]]; then
        d_min="${cont_dmin}"
    fi
    if [[ "${d_max_set}" != "true" && -n "${cont_dmax}" ]]; then
        d_max="${cont_dmax}"
    fi
    if [[ "${d_step_set}" != "true" && -n "${cont_dstep}" ]]; then
        d_step="${cont_dstep}"
    fi

    cont_spacing="$(infer_d_spacing_from_run_id "${cont_run_id}")"
    if [[ "${d_spacing_set}" != "true" ]]; then
        d_spacing="${cont_spacing}"
    elif [[ "${d_spacing}" != "${cont_spacing}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': source spacing='${cont_spacing}', requested spacing='${d_spacing}'."
        exit 1
    fi

    continue_state_dir="${cont_state_dir}"
    if [[ -z "${continue_state_dir}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': empty state_dir in registry."
        exit 1
    fi
    if [[ ! -d "${continue_state_dir}" ]]; then
        echo "Cannot continue from run_id='${continue_run_id}': state_dir does not exist: ${continue_state_dir}"
        exit 1
    fi

    if [[ "${d_min}" != "${cont_dmin}" || "${d_max}" != "${cont_dmax}" || "${d_step}" != "${cont_dstep}" ]]; then
        echo "WARNING: continuation source d-range is ${cont_dmin}:${cont_dstep}:${cont_dmax}; requested d-range is ${d_min}:${d_step}:${d_max}."
    fi
fi
if [[ -n "${continue_run_id}" && -n "${warmup_run_id}" ]]; then
    echo "Set only one of --continue_run_id or --warmup_run_id."
    exit 1
fi

if [[ -n "${warmup_run_id}" ]]; then
    if [[ ! -f "${registry_file}" ]]; then
        echo "Cannot initialize from warmup_run_id='${warmup_run_id}': registry file not found: ${registry_file}"
        exit 1
    fi
    warmup_registry_row="$(lookup_registry_row_by_run_id "${warmup_run_id}" "${registry_file}")"
    if [[ -z "${warmup_registry_row}" ]]; then
        echo "Cannot initialize from warmup_run_id='${warmup_run_id}': not found in ${registry_file}"
        exit 1
    fi
    IFS=',' read -r warm_ts warm_run_id warm_mode warm_L warm_rho warm_ns warm_dmin warm_dmax warm_dstep warm_cpus warm_mem warm_run_root warm_log_dir warm_state_dir warm_warmup_state_dir <<< "${warmup_registry_row}"
    if [[ "${warm_mode}" != "warmup" ]]; then
        echo "--warmup_run_id expects a warmup run. Got mode='${warm_mode}'."
        exit 1
    fi
    if [[ -z "${mode}" ]]; then
        mode="production"
    elif [[ "${mode}" != "production" ]]; then
        echo "--warmup_run_id can only be used with --mode production."
        exit 1
    fi
    if [[ -z "${L_val}" ]]; then
        L_val="${warm_L}"
    elif [[ "${L_val}" != "${warm_L}" ]]; then
        echo "warmup_run_id mismatch: source L='${warm_L}', requested L='${L_val}'."
        exit 1
    fi
    if [[ -z "${rho_val}" ]]; then
        rho_val="${warm_rho}"
    elif [[ "${rho_val}" != "${warm_rho}" ]]; then
        echo "warmup_run_id mismatch: source rho='${warm_rho}', requested rho='${rho_val}'."
        exit 1
    fi
    if [[ "${d_min_set}" != "true" && -n "${warm_dmin}" ]]; then
        d_min="${warm_dmin}"
    fi
    if [[ "${d_max_set}" != "true" && -n "${warm_dmax}" ]]; then
        d_max="${warm_dmax}"
    fi
    if [[ "${d_step_set}" != "true" && -n "${warm_dstep}" ]]; then
        d_step="${warm_dstep}"
    fi
    warm_spacing="$(infer_d_spacing_from_run_id "${warm_run_id}")"
    if [[ "${d_spacing_set}" != "true" ]]; then
        d_spacing="${warm_spacing}"
    elif [[ "${d_spacing}" != "${warm_spacing}" ]]; then
        echo "warmup_run_id mismatch: source spacing='${warm_spacing}', requested spacing='${d_spacing}'."
        exit 1
    fi
    warmup_state_dir="${warm_state_dir}"
    if [[ -z "${warmup_state_dir}" || ! -d "${warmup_state_dir}" ]]; then
        echo "Resolved warmup state_dir is invalid: ${warmup_state_dir}"
        exit 1
    fi
fi

if [[ -z "${mode}" || -z "${L_val}" || -z "${rho_val}" ]]; then
    echo "Missing required arguments. Provide --mode/--L/--rho or use --continue_run_id / --warmup_run_id."
    usage
    exit 1
fi

if [[ "${mode}" != "warmup" && "${mode}" != "production" && "${mode}" != "warmup_production" ]]; then
    echo "--mode must be 'warmup', 'production', or 'warmup_production'."
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
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi
if [[ "${replica_strategy}" != "mp" && "${replica_strategy}" != "dag" ]]; then
    echo "--replica_strategy must be 'mp' or 'dag'. Got '${replica_strategy}'."
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

mapfile -t D_VALUES < <(two_force_d_generate_d_values "${d_spacing}" "${d_min}" "${d_max}" "${d_step}")
if (( ${#D_VALUES[@]} == 0 )); then
    echo "No d values produced for spacing='${d_spacing}' and range ${d_min}:${d_step}:${d_max}."
    exit 1
fi
d_values_csv="$(IFS=,; echo "${D_VALUES[*]}")"
d_spacing_tag="$(two_force_d_spacing_tag "${d_spacing}" "${d_step}")"

if [[ "${mode}" == "warmup_production" ]]; then
    if [[ -n "${continue_run_id}" || -n "${warmup_run_id}" || -n "${warmup_state_dir}" ]]; then
        echo "--mode warmup_production does not accept --continue_run_id, --warmup_run_id, or --warmup_state_dir."
        exit 1
    fi
    if [[ -z "${warmup_n_sweeps}" ]]; then
        echo "--warmup_n_sweeps is required for --mode warmup_production."
        exit 1
    fi
    if ! [[ "${warmup_n_sweeps}" =~ ^[0-9]+$ ]] || (( warmup_n_sweeps <= 0 )); then
        echo "--warmup_n_sweeps must be a positive integer. Got '${warmup_n_sweeps}'."
        exit 1
    fi
    submit_chained_warmup_production
    exit 0
fi

export L="${L_val}"
export RHO0="${rho_val}"
export D_MIN="${d_min}"
export D_MAX="${d_max}"
export D_STEP="${d_step}"
export D_SPACING="${d_spacing}"
export D_VALUES_CSV="${d_values_csv}"
export NUM_REPLICAS="${num_replicas}"
export REPLICA_STRATEGY="${replica_strategy}"

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

timestamp="$(date +%Y%m%d-%H%M%S)"
rho_tag="$(slugify "${rho_val}")"
if [[ -z "${run_label}" ]]; then
    run_label="two_force_${mode}_L${L_val}_rho${rho_tag}_ns${n_sweeps_val}_d${d_min}-${d_max}-${d_spacing_tag}"
    if (( num_replicas > 1 )); then
        run_label="${run_label}_nr${num_replicas}_${replica_strategy}"
    fi
elif [[ "${run_label}" != *"-${d_spacing_tag}"* && "${run_label}" != *"_${d_spacing_tag}"* ]]; then
    run_label="${run_label}_${d_spacing_tag}"
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
elif (( num_replicas > 1 )); then
    if [[ "${replica_strategy}" == "mp" ]]; then
        export REQUEST_CPUS="$((num_replicas + 1))"
    else
        export REQUEST_CPUS="1"
    fi
fi

request_cpus_effective="${REQUEST_CPUS:-1}"
request_memory_effective="${REQUEST_MEMORY:-5 GB}"

if [[ -n "${continue_run_id}" ]]; then
    export CONTINUE_STATE_DIR="${continue_state_dir}"
    export REQUIRE_CONTINUE_STATE="true"
fi

echo "Preparing two-force d sweep:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  L=${L}"
echo "  rho0=${RHO0}"
echo "  n_sweeps=${n_sweeps_val}"
echo "  num_replicas=${NUM_REPLICAS}"
echo "  replica_strategy=${REPLICA_STRATEGY}"
echo "  d_spacing=${D_SPACING}"
echo "  d range: ${D_MIN}:${D_STEP}:${D_MAX}"
echo "  d values: ${D_VALUES_CSV}"
echo "  request_cpus=${request_cpus_effective}"
echo "  request_memory=${request_memory_effective}"
echo "  run_root=${run_root}"
echo "  run_logs=${run_log_dir}"
echo "  run_states=${run_state_dir}"
if [[ -n "${continue_run_id}" ]]; then
    echo "  continue_from_run_id=${continue_run_id}"
    echo "  continue_state_dir=${continue_state_dir}"
fi
if [[ -n "${warmup_run_id}" ]]; then
    echo "  warmup_run_id=${warmup_run_id}"
fi

if [[ "${mode}" == "production" && -z "${continue_run_id}" && -z "${warmup_state_dir}" ]]; then
    if [[ -f "${registry_file}" ]]; then
        warmup_state_dir="$(
            awk -F, -v L="${L}" -v rho="${RHO0}" -v dmin="${D_MIN}" -v dmax="${D_MAX}" -v dstep="${D_STEP}" -v spacing="${D_SPACING}" '
                $3=="warmup" && $4==L && $5==rho && $7==dmin && $8==dmax && $9==dstep {
                    if (spacing == "log_midpoints" && $2 !~ /-lm(_|$)/) next
                    if (spacing != "log_midpoints" && $2 ~ /-lm(_|$)/) next
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
num_replicas=${NUM_REPLICAS}
replica_strategy=${REPLICA_STRATEGY}
d_min=${D_MIN}
d_max=${D_MAX}
d_step=${D_STEP}
d_spacing=${D_SPACING}
d_values=${D_VALUES_CSV}
request_cpus=${request_cpus_effective}
request_memory=${request_memory_effective}
run_root=${run_root}
config_dir=${run_config_dir}
submit_dir=${run_submit_dir}
log_dir=${run_log_dir}
state_dir=${run_state_dir}
manifest=${run_manifest}
warmup_state_dir=${warmup_state_dir}
continue_run_id=${continue_run_id}
continue_state_dir=${continue_state_dir}
warmup_run_id=${warmup_run_id}
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
    if [[ -n "${continue_run_id}" ]]; then
        export REQUIRE_CONTINUE_STATE="true"
    else
        export REQUIRE_INITIAL_STATE="true"
        export WARMUP_STATE_DIR="${warmup_state_dir}"
    fi
    bash "${PRODUCTION_SCRIPT}"
fi

echo "Run manifest: ${run_manifest}"
echo "Run info: ${run_info}"
