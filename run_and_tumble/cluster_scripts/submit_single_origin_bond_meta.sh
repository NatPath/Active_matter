#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_single_origin_bond_meta.sh --n_sweeps <int> [--mode <warmup|production|warmup_production>] [--L <int>] [--rho <value>] [options]

Required:
  --n_sweeps          number of sweeps for selected mode (production sweeps for warmup_production)
  --mode/--L/--rho    required unless --continue_run_id or --warmup_run_id is provided (warmup_production always requires them)

Optional:
  --request_memory    Condor request_memory value (e.g. "4 GB")
  --request_cpus      Condor request_cpus value
  --num_replicas      number of independent replicas to run and aggregate per job (default: 1)
  --replica_strategy  replica execution strategy on cluster: mp or dag (default: mp)
  --warmup_n_sweeps   warmup sweeps for --mode warmup_production
  --warmup_run_id     specific warmup run_id to initialize production from
  --continue_run_id   specific production run_id to continue from
  --warmup_state_dir  warmup state directory for production mode
  --ffr               fluctuation rate at origin bond (default: 1.0)
  --force_strength    forcing magnitude at origin bond (default: 1.0)
  --run_label         optional custom run label prefix
  -h, --help          show this help

Behavior:
  - warmup: submits without initial_state
  - production initialization: starts from warmup state (via --warmup_run_id, --warmup_state_dir, or auto-registry lookup)
  - production continuation: starts from latest state under --continue_run_id
  - warmup_production: submits one chained DAG
      1) warmup single-process run (no replicas)
      2) production run(s) initialized from that warmup state (num_replicas + replica_strategy)
  - with --replica_strategy mp and --num_replicas > 1: one Condor job uses Julia multi-process workers and saves one aggregated state
    default request_cpus is num_replicas + 1 (main process + workers)
  - with --replica_strategy dag and --num_replicas > 1: Condor DAG submits one job per replica, then an aggregation child job
    default request_cpus is 1 per replica node (unless --request_cpus is provided)
  - creates run folder under runs/single_origin_bond/<mode>/<run_id> with configs, submit files, logs, states, manifest, and run_info
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

WARMUP_SCRIPT="${SCRIPT_DIR}/submit_single_origin_bond_warmup.sh"
PRODUCTION_SCRIPT="${SCRIPT_DIR}/submit_single_origin_bond_production.sh"
if [[ ! -f "${WARMUP_SCRIPT}" || ! -f "${PRODUCTION_SCRIPT}" ]]; then
    echo "Could not find single-origin-bond submit scripts in ${SCRIPT_DIR}"
    exit 1
fi

registry_file="${REPO_ROOT}/runs/single_origin_bond/run_registry.csv"

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

submit_chained_warmup_production() {
    local self_script="${SCRIPT_DIR}/submit_single_origin_bond_meta.sh"
    if [[ ! -f "${self_script}" ]]; then
        self_script="${REPO_ROOT}/cluster_scripts/submit_single_origin_bond_meta.sh"
    fi
    if [[ ! -f "${self_script}" ]]; then
        echo "Could not locate submit_single_origin_bond_meta.sh for chained submission."
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
    local chain_base
    if [[ -n "${run_label}" ]]; then
        chain_base="$(local_slugify "${run_label}")"
    else
        chain_base="single_warmup_production_L${L_val}_rho${rho_tag}_wns${warmup_n_sweeps}_pns${n_sweeps_val}_f${force_strength_val}_ffr${ffr_val}"
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
        --ffr "${ffr_val}"
        --force_strength "${force_strength_val}"
        --run_label "${warmup_label}"
    )
    if [[ -n "${request_memory}" ]]; then
        warmup_cmd+=(--request_memory "${request_memory}")
    fi

    echo "Preparing chained warmup stage (single process, NO_SUBMIT)..."
    local warmup_output
    warmup_output="$(
        NO_SUBMIT=true "${warmup_cmd[@]}"
    )"
    printf "%s\n" "${warmup_output}"

    local warmup_run_info
    warmup_run_info="$(printf "%s\n" "${warmup_output}" | awk -F': ' '/^Run info: /{print $2}' | tail -n 1)"
    if [[ -z "${warmup_run_info}" || ! -f "${warmup_run_info}" ]]; then
        echo "Failed to resolve warmup run_info from chained warmup stage."
        exit 1
    fi
    local warmup_run_id_local warmup_state_dir_local warmup_submit_dir warmup_submit_file
    warmup_run_id_local="$(read_run_info_value "${warmup_run_info}" "run_id")"
    warmup_state_dir_local="$(read_run_info_value "${warmup_run_info}" "state_dir")"
    warmup_submit_dir="$(read_run_info_value "${warmup_run_info}" "submit_dir")"
    warmup_submit_file="${warmup_submit_dir}/single_origin_bond_warmup.sub"
    if [[ ! -f "${warmup_submit_file}" ]]; then
        echo "Expected warmup submit file not found: ${warmup_submit_file}"
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
        --ffr "${ffr_val}"
        --force_strength "${force_strength_val}"
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
        NO_SUBMIT=true DEFER_INITIAL_STATE_LOOKUP=true "${production_cmd[@]}"
    )"
    printf "%s\n" "${production_output}"

    local production_run_info
    production_run_info="$(printf "%s\n" "${production_output}" | awk -F': ' '/^Run info: /{print $2}' | tail -n 1)"
    if [[ -z "${production_run_info}" || ! -f "${production_run_info}" ]]; then
        echo "Failed to resolve production run_info from chained production stage."
        exit 1
    fi
    local production_run_id_local production_submit_dir production_submit_file production_dag_file
    production_run_id_local="$(read_run_info_value "${production_run_info}" "run_id")"
    production_submit_dir="$(read_run_info_value "${production_run_info}" "submit_dir")"
    production_submit_file="${production_submit_dir}/single_origin_bond_production.sub"
    production_dag_file="${production_submit_dir}/single_origin_bond_production.dag"

    local chain_run_id="${chain_base}_${chain_timestamp}"
    local chain_root="${REPO_ROOT}/runs/single_origin_bond/warmup_production/${chain_run_id}"
    local chain_submit_dir="${chain_root}/submit"
    local chain_log_dir="${chain_root}/logs"
    local chain_run_info_file="${chain_root}/run_info.txt"
    local chain_dag_file="${chain_submit_dir}/single_origin_bond_warmup_production.dag"
    mkdir -p "${chain_submit_dir}" "${chain_log_dir}"

    {
        echo "JOB WARMUP ${warmup_submit_file}"
        if [[ "${replica_strategy}" == "dag" && "${num_replicas}" -gt 1 ]]; then
            if [[ ! -f "${production_dag_file}" ]]; then
                echo "Expected production DAG not found: ${production_dag_file}"
                exit 1
            fi
            echo "SUBDAG EXTERNAL PRODUCTION ${production_dag_file}"
        else
            if [[ ! -f "${production_submit_file}" ]]; then
                echo "Expected production submit file not found: ${production_submit_file}"
                exit 1
            fi
            echo "JOB PRODUCTION ${production_submit_file}"
        fi
        echo "PARENT WARMUP CHILD PRODUCTION"
    } > "${chain_dag_file}"

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
ffr=${ffr_val}
force_strength=${force_strength_val}
warmup_run_id=${warmup_run_id_local}
warmup_run_info=${warmup_run_info}
warmup_state_dir=${warmup_state_dir_local}
production_run_id=${production_run_id_local}
production_run_info=${production_run_info}
chain_dag=${chain_dag_file}
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
continue_run_id=""
continue_state_dir=""
run_label=""
ffr_val="1.0"
force_strength_val="1.0"

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
        --warmup_run_id)
            warmup_run_id="${2:-}"
            shift 2
            ;;
        --continue_run_id)
            continue_run_id="${2:-}"
            shift 2
            ;;
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
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

if [[ -z "${n_sweeps_val}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if ! [[ "${n_sweeps_val}" =~ ^[0-9]+$ ]] || (( n_sweeps_val <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps_val}'."
    exit 1
fi

is_number() {
    local value="$1"
    [[ "${value}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]
}

if ! is_number "${ffr_val}"; then
    echo "--ffr must be numeric. Got '${ffr_val}'."
    exit 1
fi
if ! is_number "${force_strength_val}"; then
    echo "--force_strength must be numeric. Got '${force_strength_val}'."
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
if [[ -n "${warmup_run_id}" && -n "${continue_run_id}" ]]; then
    echo "Set only one of --warmup_run_id or --continue_run_id."
    exit 1
fi

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
    IFS=',' read -r cont_ts cont_run_id cont_mode cont_L cont_rho cont_ns cont_cpus cont_mem cont_run_root cont_log_dir cont_state_dir cont_warmup_state_dir cont_ffr cont_force <<< "${continue_registry_row}"
    if [[ "${cont_mode}" != "production" ]]; then
        echo "--continue_run_id expects a production run. Got mode='${cont_mode}'."
        exit 1
    fi
    if [[ -z "${mode}" ]]; then
        mode="${cont_mode}"
    elif [[ "${mode}" != "${cont_mode}" ]]; then
        echo "continue_run_id mismatch: source mode='${cont_mode}', requested mode='${mode}'."
        exit 1
    fi
    if [[ -z "${L_val}" ]]; then
        L_val="${cont_L}"
    elif [[ "${L_val}" != "${cont_L}" ]]; then
        echo "continue_run_id mismatch: source L='${cont_L}', requested L='${L_val}'."
        exit 1
    fi
    if [[ -z "${rho_val}" ]]; then
        rho_val="${cont_rho}"
    elif [[ "${rho_val}" != "${cont_rho}" ]]; then
        echo "continue_run_id mismatch: source rho='${cont_rho}', requested rho='${rho_val}'."
        exit 1
    fi
    continue_state_dir="${cont_state_dir}"
    if [[ -z "${continue_state_dir}" || ! -d "${continue_state_dir}" ]]; then
        echo "Resolved continue state_dir is invalid: ${continue_state_dir}"
        exit 1
    fi
fi

if [[ -n "${warmup_run_id}" ]]; then
    if [[ ! -f "${registry_file}" ]]; then
        echo "Cannot resolve warmup_run_id='${warmup_run_id}': registry file not found: ${registry_file}"
        exit 1
    fi
    warmup_registry_row="$(lookup_registry_row_by_run_id "${warmup_run_id}" "${registry_file}")"
    if [[ -z "${warmup_registry_row}" ]]; then
        echo "Cannot resolve warmup_run_id='${warmup_run_id}': not found in ${registry_file}"
        exit 1
    fi
    IFS=',' read -r warm_ts warm_run_id warm_mode warm_L warm_rho warm_ns warm_cpus warm_mem warm_run_root warm_log_dir warm_state_dir warm_warmup_state_dir warm_ffr warm_force <<< "${warmup_registry_row}"
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

if [[ "${mode}" != "production" && ( -n "${warmup_run_id}" || -n "${continue_run_id}" ) ]]; then
    echo "--warmup_run_id/--continue_run_id are valid only for --mode production."
    exit 1
fi

export L="${L_val}"
export RHO0="${rho_val}"
export FFR="${ffr_val}"
export FORCE_STRENGTH="${force_strength_val}"
export NUM_REPLICAS="${num_replicas}"
export REPLICA_STRATEGY="${replica_strategy}"

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

timestamp="$(date +%Y%m%d-%H%M%S)"
rho_tag="$(slugify "${rho_val}")"
if [[ -z "${run_label}" ]]; then
    run_label="single_${mode}_L${L_val}_rho${rho_tag}_ns${n_sweeps_val}_f${force_strength_val}_ffr${ffr_val}"
    if (( num_replicas > 1 )); then
        run_label="${run_label}_nr${num_replicas}_${replica_strategy}"
    fi
fi
run_label="$(slugify "${run_label}")"
run_id="${run_label}_${timestamp}"
run_root="${REPO_ROOT}/runs/single_origin_bond/${mode}/${run_id}"
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
request_memory_effective="${REQUEST_MEMORY:-2 GB}"

if [[ "${mode}" == "production" ]]; then
    if [[ -n "${continue_run_id}" ]]; then
        if [[ -z "${continue_state_dir}" || ! -d "${continue_state_dir}" ]]; then
            echo "Resolved continue state_dir is invalid: ${continue_state_dir}"
            exit 1
        fi
    elif [[ -n "${warmup_run_id}" ]]; then
        if [[ -z "${warmup_state_dir}" || ! -d "${warmup_state_dir}" ]]; then
            echo "Resolved warmup state_dir is invalid: ${warmup_state_dir}"
            exit 1
        fi
    elif [[ -z "${warmup_state_dir}" ]]; then
        if [[ -f "${registry_file}" ]]; then
            warmup_state_dir="$(
                awk -F, -v L="${L}" -v rho="${RHO0}" -v ffr="${FFR}" -v fmag="${FORCE_STRENGTH}" '
                    $3=="warmup" && $4==L && $5==rho {
                        if (NF >= 14 && ($13 != ffr || $14 != fmag)) next
                        state_dir=$11
                    }
                    END {
                        if (state_dir != "") print state_dir
                    }' "${registry_file}"
            )"
        fi
        if [[ -z "${warmup_state_dir}" ]]; then
            warmup_state_dir="${REPO_ROOT}/saved_states/single_origin_bond/warmup"
        fi
    fi
elif [[ "${mode}" == "warmup" && -z "${warmup_state_dir}" ]]; then
    warmup_state_dir="${run_state_dir}"
fi

echo "Preparing single-origin-bond run:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  L=${L}"
echo "  rho0=${RHO0}"
echo "  n_sweeps=${n_sweeps_val}"
echo "  num_replicas=${NUM_REPLICAS}"
echo "  replica_strategy=${REPLICA_STRATEGY}"
echo "  force_strength=${FORCE_STRENGTH}"
echo "  ffr=${FFR}"
echo "  request_cpus=${request_cpus_effective}"
echo "  request_memory=${request_memory_effective}"
echo "  run_root=${run_root}"
echo "  run_logs=${run_log_dir}"
echo "  run_states=${run_state_dir}"
if [[ -n "${warmup_run_id}" ]]; then
    echo "  warmup_run_id=${warmup_run_id}"
fi
if [[ -n "${continue_run_id}" ]]; then
    echo "  continue_run_id=${continue_run_id}"
    echo "  continue_state_dir=${continue_state_dir}"
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
force_strength=${FORCE_STRENGTH}
ffr=${FFR}
request_cpus=${request_cpus_effective}
request_memory=${request_memory_effective}
run_root=${run_root}
config_dir=${run_config_dir}
submit_dir=${run_submit_dir}
log_dir=${run_log_dir}
state_dir=${run_state_dir}
manifest=${run_manifest}
warmup_state_dir=${warmup_state_dir}
warmup_run_id=${warmup_run_id}
continue_run_id=${continue_run_id}
continue_state_dir=${continue_state_dir}
EOF

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,request_cpus,request_memory,run_root,log_dir,state_dir,warmup_state_dir,ffr,force_strength" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "${mode}" "${L}" "${RHO0}" "${n_sweeps_val}" \
    "${request_cpus_effective}" "${request_memory_effective}" \
    "${run_root}" "${run_log_dir}" "${run_state_dir}" "${warmup_state_dir}" "${FFR}" "${FORCE_STRENGTH}" >> "${registry_file}"

if [[ "${mode}" == "warmup" ]]; then
    export WARMUP_SWEEPS="${n_sweeps_val}"
    bash "${WARMUP_SCRIPT}"
else
    export PRODUCTION_SWEEPS="${n_sweeps_val}"
    if [[ -n "${continue_run_id}" ]]; then
        export CONTINUE_STATE_DIR="${continue_state_dir}"
        export REQUIRE_CONTINUE_STATE="true"
    else
        export REQUIRE_INITIAL_STATE="true"
        export WARMUP_STATE_DIR="${warmup_state_dir}"
    fi
    bash "${PRODUCTION_SCRIPT}"
fi

echo "Run manifest: ${run_manifest}"
echo "Run info: ${run_info}"
