#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_diffusive_2d_origin_bond_production_batch.sh \
      --source_run_id <warmup_or_previous_production_run_id> \
      --L <int> --rho <float> --n_sweeps <int> --nr <int> [options]

Alternative source form:
  bash submit_diffusive_2d_origin_bond_production_batch.sh \
      --source_state_dir <dir> \
      --source_tag_prefix <prefix> \
      --L <int> --rho <float> --n_sweeps <int> --nr <int> [options]

Required:
  --n_sweeps <int>             Production sweeps per replica

Source options:
  --source_run_id <id>         Warmup or previous production run id to continue from
  --source_kind <kind>         Source run kind: auto, warmup, or production (default: auto)
  --source_state_dir <dir>     Directory holding initial-state files
  --source_tag_prefix <prefix> Prefix for source state id tags, e.g. warmup_<run_id>_r
  --source_run_info <path>     Legacy/debug path; run_info.txt from a source batch
  --previous_cumulative_state <path>
                               Optional explicit cumulative aggregate to seed from

Physical options:
  --L <int>                    Even 2D lattice size (default: 64)
  --rho <float>                Density rho0 (default: 1000)
  --force_strength <float>     Bond force magnitude (default: 1.0)
  --ffr <float>                Fluctuating forcing rate (default: 1.0)
  --nr <int>                   Number of production replicas (default: 600)
  --num_replicas <int>         Alias for --nr

Cluster/options:
  --run_id <id>                Explicit run id
  --request_cpus <int>         Replica request_cpus (default: 1)
  --request_memory <value>     Replica request_memory (default: "6 GB")
  --aggregate_request_cpus <int>     Aggregate request_cpus (default: 1)
  --aggregate_request_memory <value> Aggregate request_memory (default: request_memory)
  --replica_retries <int>      DAG retry count for replica nodes (default: 2)
  --aggregate_retries <int>    DAG retry count for aggregate node (default: 1)
  --dag_maxjobs <int>          Optional DAGMan replica throttle, 0 disables (default: 0)
  --batch_name <name>          Condor batch_name (default: run id)
  --aggregate_root <dir>       Aggregate root override (default: runs/.../<run_id>/aggregated)
  --cumulative_tag <tag>       Stable cumulative id tag (default: physical-parameter tag)
  --no_submit                  Generate files only; do not call condor_submit_dag
  -h, --help                   Show this help

Behavior:
  - Resolves one terminal source state per replica.
  - Starts each production replica with --initial_state, so statistics are reset.
  - Adds a DAG child job that aggregates the production batch under the run folder
    and folds it into aggregate_root/current.
  - When continuing from a previous production batch, it seeds the new cumulative
    aggregate from the previous run's recorded aggregate state.
  - For the next manual production-batch DAG, pass this production run_id as
    --source_run_id.
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

RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_diffusive_origin_batch.sh"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"
for required_script in "${RUNNER_SCRIPT}" "${AGGREGATE_SCRIPT}" "${DAG_NOTIFY_UTILS}"; do
    if [[ ! -f "${required_script}" ]]; then
        echo "Missing required script: ${required_script}"
        exit 1
    fi
done
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

slugify() {
    printf "%s" "$1" | sed -E 's/[+]/p/g; s/-/m/g; s/[.]/p/g; s/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_//; s/_$//'
}

format_float() {
    awk -v x="$1" 'BEGIN { printf "%.12g", x }'
}

ensure_cluster_shared_dir_permissions() {
    local path="$1"
    local mode="$2"
    chmod "${mode}" "${path}" 2>/dev/null || true
}

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -v key="${key}" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "${run_info_path}"
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "--${name} must be a positive integer. Got '${value}'."
        exit 1
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
        echo "--${name} must be a non-negative integer. Got '${value}'."
        exit 1
    fi
}

require_positive_float() {
    local name="$1"
    local value="$2"
    if ! awk -v x="${value}" 'BEGIN { exit !(x > 0.0) }'; then
        echo "--${name} must be a positive number. Got '${value}'."
        exit 1
    fi
}

latest_state_for_id_tag() {
    local root_dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime

    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${root_dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

latest_cumulative_state_for_root() {
    local aggregate_root_path="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime current_dir

    if [[ "${aggregate_root_path}" != /* ]]; then
        aggregate_root_path="${REPO_ROOT}/${aggregate_root_path}"
    fi

    for current_dir in "${aggregate_root_path}/current" "${aggregate_root_path}"; do
        [[ -d "${current_dir}" ]] || continue
        while IFS= read -r -d '' candidate; do
            mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
            if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
                best_mtime="${mtime}"
                best_path="${candidate}"
            fi
        done < <(find "${current_dir}" -maxdepth 1 -type f -name "*_id-aggregated_${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)
        if [[ -n "${best_path}" ]]; then
            break
        fi
    done

    printf "%s" "${best_path}"
}

resolve_previous_cumulative_state_from_run_info() {
    local run_info_path="$1"
    local fallback_tag="${2:-}"
    local aggregate_state aggregate_root_from_info cumulative_tag_from_info

    [[ -f "${run_info_path}" ]] || { printf ""; return 0; }

    aggregate_state="$(read_run_info_value "${run_info_path}" "aggregate_state")"
    if [[ -n "${aggregate_state}" ]]; then
        if [[ "${aggregate_state}" != /* ]]; then
            aggregate_state="${REPO_ROOT}/${aggregate_state}"
        fi
        if [[ -f "${aggregate_state}" ]]; then
            printf "%s" "${aggregate_state}"
            return 0
        fi
    fi

    aggregate_root_from_info="$(read_run_info_value "${run_info_path}" "aggregate_root")"
    cumulative_tag_from_info="$(read_run_info_value "${run_info_path}" "cumulative_tag")"
    cumulative_tag_from_info="${cumulative_tag_from_info:-${fallback_tag}}"
    if [[ -n "${aggregate_root_from_info}" && -n "${cumulative_tag_from_info}" ]]; then
        latest_cumulative_state_for_root "${aggregate_root_from_info}" "${cumulative_tag_from_info}"
        return 0
    fi

    printf ""
}

source_run_id=""
source_kind="auto"
source_run_info=""
source_state_dir=""
source_tag_prefix=""
previous_cumulative_state=""
L=""
rho=""
force_strength=""
ffr=""
n_sweeps=""
num_replicas=""
run_id=""
request_cpus="1"
request_memory="6 GB"
aggregate_request_cpus="1"
aggregate_request_memory=""
replica_retries="2"
aggregate_retries="1"
dag_maxjobs="0"
batch_name=""
aggregate_root=""
cumulative_tag=""
no_submit="${NO_SUBMIT:-false}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_run_id)
            source_run_id="${2:-}"
            shift 2
            ;;
        --source_kind)
            source_kind="${2:-}"
            shift 2
            ;;
        --source_run_info)
            source_run_info="${2:-}"
            shift 2
            ;;
        --source_state_dir)
            source_state_dir="${2:-}"
            shift 2
            ;;
        --source_tag_prefix)
            source_tag_prefix="${2:-}"
            shift 2
            ;;
        --previous_cumulative_state)
            previous_cumulative_state="${2:-}"
            shift 2
            ;;
        --L)
            L="${2:-}"
            shift 2
            ;;
        --rho)
            rho="${2:-}"
            shift 2
            ;;
        --force_strength)
            force_strength="${2:-}"
            shift 2
            ;;
        --ffr)
            ffr="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --nr|--num_replicas)
            num_replicas="${2:-}"
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
        --aggregate_request_memory)
            aggregate_request_memory="${2:-}"
            shift 2
            ;;
        --replica_retries)
            replica_retries="${2:-}"
            shift 2
            ;;
        --aggregate_retries)
            aggregate_retries="${2:-}"
            shift 2
            ;;
        --dag_maxjobs)
            dag_maxjobs="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
            shift 2
            ;;
        --aggregate_root)
            aggregate_root="${2:-}"
            shift 2
            ;;
        --cumulative_tag)
            cumulative_tag="${2:-}"
            shift 2
            ;;
        --no_submit)
            no_submit="true"
            shift 1
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

if [[ -n "${source_run_info}" ]]; then
    if [[ ! -f "${source_run_info}" ]]; then
        echo "source_run_info not found: ${source_run_info}"
        exit 1
    fi
    source_state_dir="${source_state_dir:-$(read_run_info_value "${source_run_info}" raw_state_dir)}"
    source_state_dir="${source_state_dir:-$(read_run_info_value "${source_run_info}" state_dir)}"
    source_tag_prefix="${source_tag_prefix:-$(read_run_info_value "${source_run_info}" save_tag_prefix)}"
    source_tag_prefix="${source_tag_prefix:-$(read_run_info_value "${source_run_info}" replica_tag_prefix)}"
    num_replicas="${num_replicas:-$(read_run_info_value "${source_run_info}" num_replicas)}"
    L="${L:-$(read_run_info_value "${source_run_info}" L)}"
    rho="${rho:-$(read_run_info_value "${source_run_info}" rho0)}"
    force_strength="${force_strength:-$(read_run_info_value "${source_run_info}" force_strength)}"
    ffr="${ffr:-$(read_run_info_value "${source_run_info}" ffr)}"
    cumulative_tag="${cumulative_tag:-$(read_run_info_value "${source_run_info}" cumulative_tag)}"
fi

case "${source_kind}" in
    auto|warmup|production)
        ;;
    *)
        echo "--source_kind must be auto, warmup, or production. Got '${source_kind}'."
        exit 1
        ;;
esac

if [[ -n "${source_run_id}" ]]; then
    source_run_id_exact="${source_run_id}"
    source_run_id_slug="$(slugify "${source_run_id}")"
    resolved_source_run_id=""
    warmup_root_exact="${REPO_ROOT}/runs/diffusive_2d_origin_bond/warmup/${source_run_id_exact}"
    production_root_exact="${REPO_ROOT}/runs/diffusive_2d_origin_bond/production/${source_run_id_exact}"
    warmup_root_slug="${REPO_ROOT}/runs/diffusive_2d_origin_bond/warmup/${source_run_id_slug}"
    production_root_slug="${REPO_ROOT}/runs/diffusive_2d_origin_bond/production/${source_run_id_slug}"
    warmup_root=""
    production_root=""
    resolved_source_kind=""

    if [[ "${source_kind}" == "warmup" ]]; then
        resolved_source_kind="warmup"
    elif [[ "${source_kind}" == "production" ]]; then
        resolved_source_kind="production"
    else
        warmup_exists="false"
        production_exists="false"
        if [[ -d "${warmup_root_exact}" ]]; then
            warmup_exists="true"
            warmup_root="${warmup_root_exact}"
            resolved_source_run_id="${source_run_id_exact}"
        elif [[ -d "${warmup_root_slug}" ]]; then
            warmup_exists="true"
            warmup_root="${warmup_root_slug}"
            resolved_source_run_id="${source_run_id_slug}"
        fi
        if [[ -d "${production_root_exact}" ]]; then
            production_exists="true"
            production_root="${production_root_exact}"
            if [[ -z "${resolved_source_run_id}" ]]; then
                resolved_source_run_id="${source_run_id_exact}"
            fi
        elif [[ -d "${production_root_slug}" ]]; then
            production_exists="true"
            production_root="${production_root_slug}"
            if [[ -z "${resolved_source_run_id}" ]]; then
                resolved_source_run_id="${source_run_id_slug}"
            fi
        fi
        if [[ "${warmup_exists}" == "true" && "${production_exists}" == "true" ]]; then
            echo "source_run_id exists as both warmup and production. Add --source_kind warmup or --source_kind production."
            exit 1
        elif [[ "${warmup_exists}" == "true" ]]; then
            resolved_source_kind="warmup"
        elif [[ "${production_exists}" == "true" ]]; then
            resolved_source_kind="production"
        else
            echo "Could not find source_run_id '${source_run_id_exact}' under runs/diffusive_2d_origin_bond/{warmup,production}."
            exit 1
        fi
    fi

    if [[ "${resolved_source_kind}" == "warmup" ]]; then
        if [[ -z "${warmup_root}" ]]; then
            if [[ -d "${warmup_root_exact}" ]]; then
                warmup_root="${warmup_root_exact}"
                resolved_source_run_id="${source_run_id_exact}"
            elif [[ -d "${warmup_root_slug}" ]]; then
                warmup_root="${warmup_root_slug}"
                resolved_source_run_id="${source_run_id_slug}"
            fi
        fi
        if [[ ! -d "${warmup_root}" ]]; then
            echo "Warmup source run not found. Tried:"
            echo "  ${warmup_root_exact}"
            if [[ "${warmup_root_slug}" != "${warmup_root_exact}" ]]; then
                echo "  ${warmup_root_slug}"
            fi
            exit 1
        fi
        source_state_dir="${source_state_dir:-${warmup_root}/states}"
        source_tag_prefix="${source_tag_prefix:-warmup_${resolved_source_run_id}_r}"
        if [[ -z "${source_run_info}" && -f "${warmup_root}/run_info.txt" ]]; then
            source_run_info="${warmup_root}/run_info.txt"
        fi
    else
        if [[ -z "${production_root}" ]]; then
            if [[ -d "${production_root_exact}" ]]; then
                production_root="${production_root_exact}"
                resolved_source_run_id="${source_run_id_exact}"
            elif [[ -d "${production_root_slug}" ]]; then
                production_root="${production_root_slug}"
                resolved_source_run_id="${source_run_id_slug}"
            fi
        fi
        if [[ ! -d "${production_root}" ]]; then
            echo "Production source run not found. Tried:"
            echo "  ${production_root_exact}"
            if [[ "${production_root_slug}" != "${production_root_exact}" ]]; then
                echo "  ${production_root_slug}"
            fi
            exit 1
        fi
        source_state_dir="${source_state_dir:-${production_root}/states/raw}"
        source_tag_prefix="${source_tag_prefix:-prod_${resolved_source_run_id}_r}"
        if [[ -z "${source_run_info}" && -f "${production_root}/run_info.txt" ]]; then
            source_run_info="${production_root}/run_info.txt"
        fi
    fi
    source_run_id="${resolved_source_run_id}"
    source_kind="${resolved_source_kind}"
fi

L="${L:-64}"
rho="${rho:-1000}"
force_strength="${force_strength:-1.0}"
ffr="${ffr:-1.0}"
num_replicas="${num_replicas:-600}"
aggregate_request_memory="${aggregate_request_memory:-${request_memory}}"

if [[ -z "${n_sweeps}" ]]; then
    echo "--n_sweeps is required for production batches; the warmup length is known, but production length was not specified."
    usage
    exit 1
fi
if [[ -z "${source_state_dir}" || -z "${source_tag_prefix}" ]]; then
    echo "Provide --source_run_info, or both --source_state_dir and --source_tag_prefix."
    usage
    exit 1
fi
if [[ "${source_state_dir}" != /* ]]; then
    source_state_dir="${REPO_ROOT}/${source_state_dir}"
fi
if [[ ! -d "${source_state_dir}" ]]; then
    echo "source_state_dir not found: ${source_state_dir}"
    exit 1
fi

require_positive_int "L" "${L}"
require_positive_int "n_sweeps" "${n_sweeps}"
require_positive_int "num_replicas" "${num_replicas}"
require_positive_int "request_cpus" "${request_cpus}"
require_positive_int "aggregate_request_cpus" "${aggregate_request_cpus}"
require_nonnegative_int "replica_retries" "${replica_retries}"
require_nonnegative_int "aggregate_retries" "${aggregate_retries}"
require_nonnegative_int "dag_maxjobs" "${dag_maxjobs}"
require_positive_float "rho" "${rho}"
require_positive_float "force_strength" "${force_strength}"
require_positive_float "ffr" "${ffr}"
if (( L % 2 != 0 )); then
    echo "--L must be even so the origin bond maps to [L/2,L/2] -> [L/2+1,L/2]. Got L=${L}."
    exit 1
fi

rho_display="$(format_float "${rho}")"
force_display="$(format_float "${force_strength}")"
ffr_display="$(format_float "${ffr}")"
rho_slug="$(slugify "${rho_display}")"
force_slug="$(slugify "${force_display}")"
ffr_slug="$(slugify "${ffr_display}")"

if [[ -z "${cumulative_tag}" ]]; then
    cumulative_tag="cumulative_L${L}_rho${rho_slug}_f${force_slug}_ffr${ffr_slug}"
else
    cumulative_tag="$(slugify "${cumulative_tag}")"
fi
if [[ -n "${previous_cumulative_state}" && "${previous_cumulative_state}" != /* ]]; then
    previous_cumulative_state="${REPO_ROOT}/${previous_cumulative_state}"
fi
if [[ -z "${previous_cumulative_state}" && "${source_kind}" == "production" && -n "${source_run_info}" ]]; then
    previous_cumulative_state="$(resolve_previous_cumulative_state_from_run_info "${source_run_info}" "${cumulative_tag}")"
fi
if [[ -n "${previous_cumulative_state}" && ! -f "${previous_cumulative_state}" ]]; then
    echo "previous_cumulative_state not found: ${previous_cumulative_state}"
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${run_id}" ]]; then
    run_id="diffusive_2d_origin_bond_L${L}_rho${rho_slug}_f${force_slug}_ffr${ffr_slug}_prod_ns${n_sweeps}_nr${num_replicas}_${timestamp}"
else
    run_id="$(slugify "${run_id}")"
fi
job_batch_name="${batch_name:-${run_id}}"

run_root="${REPO_ROOT}/runs/diffusive_2d_origin_bond/production/${run_id}"
if [[ -z "${aggregate_root}" ]]; then
    aggregate_root="${run_root}/aggregated"
elif [[ "${aggregate_root}" != /* ]]; then
    aggregate_root="${REPO_ROOT}/${aggregate_root}"
fi
config_dir="${run_root}/configs"
submit_dir="${run_root}/submit"
log_dir="${run_root}/logs"
raw_state_dir="${run_root}/states/raw"
run_info="${run_root}/run_info.txt"
manifest="${run_root}/manifest.csv"
initial_states_file="${run_root}/initial_states.txt"
registry_file="${REPO_ROOT}/runs/diffusive_2d_origin_bond/run_registry.csv"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${raw_state_dir}" "${aggregate_root}"
ensure_cluster_shared_dir_permissions "${run_root}" 755
ensure_cluster_shared_dir_permissions "${config_dir}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777
ensure_cluster_shared_dir_permissions "${raw_state_dir}" 1777
ensure_cluster_shared_dir_permissions "${aggregate_root}" 1777

center=$(( L / 2 ))
next_x=$(( center + 1 ))
description_name="diffusive_2d_origin_xbond_L${L}_rho${rho_slug}_f${force_slug}_ffr${ffr_slug}_production"
runtime_config="${config_dir}/diffusive_2d_origin_bond_production.yaml"
cat > "${runtime_config}" <<EOF
# Generated by cluster_scripts/submit_diffusive_2d_origin_bond_production_batch.sh
dim_num: 2
L: ${L}
ρ₀: ${rho_display}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${n_sweeps}
warmup_sweeps: 0
performance_mode: true
description: "${description_name}"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [[${center}, ${center}], [${next_x}, ${center}]]
forcing_magnitudes: [${force_display}]
ffrs: [${ffr_display}]
forcing_direction_flags: [true]
forcing_fluctuation_type: "alternating_direction"
forcing_rate_scheme: "symmetric_normalized"
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "${raw_state_dir}"
progress_interval: 1000
EOF

: > "${initial_states_file}"
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    source_tag="${source_tag_prefix}${replica_idx}"
    initial_state="$(latest_state_for_id_tag "${source_state_dir}" "${source_tag}")"
    if [[ -z "${initial_state}" ]]; then
        echo "Missing source state for replica ${replica_idx}: tag=${source_tag} dir=${source_state_dir}"
        exit 1
    fi
    echo "${initial_state}" >> "${initial_states_file}"
done

save_tag_prefix="prod_${run_id}_r"
batch_tag="batch_${run_id}"
archive_stamp="${run_id}"
dag_file="${submit_dir}/diffusive_2d_origin_bond_production.dag"
aggregate_submit_file="${submit_dir}/production_batch_aggregate.sub"
aggregate_output_file="${log_dir}/production_batch_aggregate.out"
aggregate_error_file="${log_dir}/production_batch_aggregate.err"
aggregate_log_file="${log_dir}/production_batch_aggregate.log"

: > "${dag_file}"
if (( dag_maxjobs > 0 )); then
    printf "MAXJOBS REPLICAS %s\n" "${dag_maxjobs}" >> "${dag_file}"
fi
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag,initial_state" > "${manifest}"

replica_job_ids=()
replica_idx=0
while IFS= read -r initial_state; do
    replica_idx=$(( replica_idx + 1 ))
    replica_submit_file="${submit_dir}/production_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/production_r${replica_idx}.out"
    replica_error_file="${log_dir}/production_r${replica_idx}.err"
    replica_log_file="${log_dir}/production_r${replica_idx}.log"
    replica_tag="${save_tag_prefix}${replica_idx}"
    replica_runner_arguments="${RUNNER_SCRIPT} ${runtime_config} --initial_state ${initial_state} --save_tag ${replica_tag}"

    cat > "${replica_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${replica_runner_arguments}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${replica_output_file}
error      = ${replica_error_file}
log        = ${replica_log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${job_batch_name}
queue
EOF

    job_id="R${replica_idx}"
    replica_job_ids+=("${job_id}")
    printf "JOB %s %s\n" "${job_id}" "${replica_submit_file}" >> "${dag_file}"
    if (( replica_retries > 0 )); then
        printf "RETRY %s %s\n" "${job_id}" "${replica_retries}" >> "${dag_file}"
    fi
    if (( dag_maxjobs > 0 )); then
        printf "CATEGORY %s REPLICAS\n" "${job_id}" >> "${dag_file}"
    fi
    printf "replica,%s,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${replica_submit_file}" "${replica_output_file}" "${replica_error_file}" "${replica_log_file}" "${replica_tag}" "${initial_state}" \
        >> "${manifest}"
done < "${initial_states_file}"

aggregate_arguments="${AGGREGATE_SCRIPT} --config ${runtime_config} --raw_state_dir ${raw_state_dir} --num_replicas ${num_replicas} --raw_tag_prefix ${save_tag_prefix} --aggregate_root ${aggregate_root} --batch_tag ${batch_tag} --cumulative_tag ${cumulative_tag} --archive_stamp ${archive_stamp} --run_info ${run_info}"
if [[ -n "${previous_cumulative_state}" ]]; then
    aggregate_arguments="${aggregate_arguments} --previous_cumulative_state ${previous_cumulative_state}"
fi
cat > "${aggregate_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${aggregate_arguments}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${aggregate_output_file}
error      = ${aggregate_error_file}
log        = ${aggregate_log_file}
request_cpus = ${aggregate_request_cpus}
request_memory = ${aggregate_request_memory}
batch_name = ${job_batch_name}
queue
EOF

printf "JOB AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
if (( aggregate_retries > 0 )); then
    printf "RETRY AGG %s\n" "${aggregate_retries}" >> "${dag_file}"
fi
printf "PARENT %s CHILD AGG\n" "${replica_job_ids[*]}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s,%s\n" \
    "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${batch_tag}" "" \
    >> "${manifest}"

dag_append_final_notification_node "${dag_file}" "${submit_dir}" "${log_dir}" "${run_root}" "${run_id}" "diffusive_2d_origin_bond_production" "${REPO_ROOT}"

cluster_id=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated production DAG but not submitting: ${dag_file}"
    cluster_id="NO_SUBMIT"
else
    submit_output="$(condor_submit_dag "${dag_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=production
simulation=diffusive_2d_origin_bond
dim_num=2
L=${L}
rho0=${rho_display}
D=1.0
force_strength=${force_display}
ffr=${ffr_display}
forcing_rate_scheme=symmetric_normalized
n_sweeps=${n_sweeps}
num_replicas=${num_replicas}
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
aggregate_request_memory=${aggregate_request_memory}
replica_retries=${replica_retries}
aggregate_retries=${aggregate_retries}
dag_maxjobs=${dag_maxjobs}
job_batch_name=${job_batch_name}
source_run_id=${source_run_id}
source_kind=${source_kind}
source_run_info=${source_run_info}
source_state_dir=${source_state_dir}
source_tag_prefix=${source_tag_prefix}
previous_cumulative_state=${previous_cumulative_state}
initial_states_file=${initial_states_file}
run_root=${run_root}
config_dir=${config_dir}
submit_dir=${submit_dir}
log_dir=${log_dir}
raw_state_dir=${raw_state_dir}
state_dir=${raw_state_dir}
runtime_config=${runtime_config}
dag_file=${dag_file}
manifest=${manifest}
save_tag_prefix=${save_tag_prefix}
batch_tag=${batch_tag}
cumulative_tag=${cumulative_tag}
aggregate_root=${aggregate_root}
archive_stamp=${archive_stamp}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG:-}
cluster_id=${cluster_id}
next_step_hint=bash cluster_scripts/submit_diffusive_2d_origin_bond_production_batch.sh --source_run_id ${run_id} --L ${L} --rho ${rho_display} --n_sweeps ${n_sweeps} --nr ${num_replicas}
EOF

mkdir -p "$(dirname "${registry_file}")"
if [[ ! -f "${registry_file}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,config_path,save_tag_prefix,aggregate_root,cumulative_tag" > "${registry_file}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "production" "${L}" "${rho_display}" "${n_sweeps}" "${num_replicas}" \
    "${request_cpus}" "${request_memory}" "${run_root}" "${submit_dir}" "${log_dir}" "${raw_state_dir}" "${runtime_config}" \
    "${save_tag_prefix}" "${aggregate_root}" "${cumulative_tag}" \
    >> "${registry_file}"

echo "Prepared diffusive 2D origin-bond production DAG."
echo "  run_id=${run_id}"
echo "  dag_file=${dag_file}"
echo "  run_info=${run_info}"
echo "  raw_states=${raw_state_dir}"
echo "  aggregate_root=${aggregate_root}"
echo "  cumulative_tag=${cumulative_tag}"
echo "  previous_cumulative_state=${previous_cumulative_state}"
echo "  next batch source_run_id=${run_id}"
