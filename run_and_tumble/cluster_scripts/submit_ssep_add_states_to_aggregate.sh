#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_ssep_add_states_to_aggregate.sh \
      --run_id <id> \
      --nr <int> \
      --ns <int> \
      [options]

Required:
  --run_id <id>                    Existing SSEP production run_id
  --nr <int>                       Number of new replica states to add
  --ns <int>                       Sweeps for each new replica state

Options:
  --num_replicas <int>             Alias for --nr
  --n_sweeps <int>                 Alias for --ns
  --mode <auto|production>         How to resolve --run_id (default: auto)
  --request_cpus <int>             Replica request_cpus (default: 1)
  --request_memory <value>         Replica and aggregate request_memory (default: "5 GB")
  --aggregate_request_cpus <int>   Aggregate request_cpus (default: 1)
  --batch_name <name>              Condor batch_name (default: auto)
  --job_label <label>              Optional label in the top-up batch token
  --no_submit                      Generate files only; do not call condor_submit_dag
  -h, --help                       Show help

Behavior:
  - Resolves the target SSEP production run from --run_id
  - Chooses one random previously completed production replica state per new replica as --initial_state
  - Forces warmup_sweeps=0 and performance_mode=true for the new replicas
  - Writes new raw states under:
      <state_dir>/topup_batches/<job_token>/
  - Aggregates [current aggregate + new raw states] into the same aggregate save tag,
    then archives the superseded aggregate file under:
      <aggregated_dir>/archive/<job_token>/
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_ssep.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_ssep.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

RUNNER_SCRIPT="${SCRIPT_DIR}/run_ssep_from_config.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_ssep_topup_batch.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/ssep/single_center_bond/run_registry.csv"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing runner script: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregate helper: ${AGGREGATE_SCRIPT}"
    exit 1
fi
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

resolve_initial_state_pool_from_manifest() {
    local manifest_path="$1"
    local state_root="$2"

    [[ -f "${manifest_path}" ]] || return 1

    local tags=()
    local row_type job_name submit_file output_file error_file log_file save_tag extra
    while IFS=',' read -r row_type job_name submit_file output_file error_file log_file save_tag extra; do
        [[ "${row_type}" == "replica" ]] || continue
        [[ -n "${save_tag}" ]] || continue
        tags+=("${save_tag}")
    done < "${manifest_path}"

    (( ${#tags[@]} > 0 )) || return 1

    local matched_state
    local resolved_states=()
    for save_tag in "${tags[@]}"; do
        matched_state="$(latest_state_for_id_tag "${state_root}" "${save_tag}")"
        [[ -n "${matched_state}" ]] || continue
        resolved_states+=("${matched_state}")
    done

    (( ${#resolved_states[@]} > 0 )) || return 1
    printf '%s\n' "${resolved_states[@]}"
    return 0
}

find_run_info_by_run_id() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local candidate=""

    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/ssep/single_center_bond/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    if [[ -f "${REGISTRY_FILE}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${REGISTRY_FILE}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _warmup _numrep _cpus _mem reg_run_root _submit_dir _log_dir _state_dir _config_path _aggregate_run_id <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

resolve_target_production_run_info() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local resolved_info

    resolved_info="$(find_run_info_by_run_id "${lookup_run_id}" "${mode_hint}" || true)"
    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve SSEP production run_info for run_id='${lookup_run_id}' (mode=${mode_hint})." >&2
        return 1
    fi
    if [[ "$(read_run_info_value "${resolved_info}" "mode")" != "production" ]]; then
        echo "run_id='${lookup_run_id}' does not resolve to an SSEP production run." >&2
        return 1
    fi
    echo "${resolved_info}"
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
    done < <(find "${root_dir}" -maxdepth 1 \( -type f -o -type l \) -name "*_id-${id_tag}.jld2" -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

rewrite_runtime_config() {
    local source_config="$1"
    local target_config="$2"
    local save_dir="$3"
    local n_sweeps_val="$4"
    local warmup_sweeps_val="$5"
    local performance_mode_val="$6"
    local plot_final_val="$7"
    local save_final_plot_val="$8"
    local cluster_mode_val="$9"
    local save_dir_line n_sweeps_line warmup_sweeps_line performance_mode_line cluster_mode_line plot_final_line save_final_plot_line

    save_dir_line="save_dir: \"${save_dir}\""
    n_sweeps_line="n_sweeps: ${n_sweeps_val}"
    warmup_sweeps_line="warmup_sweeps: ${warmup_sweeps_val}"
    performance_mode_line="performance_mode: ${performance_mode_val}"
    cluster_mode_line="cluster_mode: ${cluster_mode_val}"
    plot_final_line="plot_final: ${plot_final_val}"
    save_final_plot_line="save_final_plot: ${save_final_plot_val}"

    awk \
    -v save_dir_line="${save_dir_line}" \
    -v n_sweeps_line="${n_sweeps_line}" \
    -v warmup_sweeps_line="${warmup_sweeps_line}" \
    -v performance_mode_line="${performance_mode_line}" \
    -v cluster_mode_line="${cluster_mode_line}" \
    -v plot_final_line="${plot_final_line}" \
    -v save_final_plot_line="${save_final_plot_line}" '
    BEGIN {
        seen_save=0
        seen_sweeps=0
        seen_warmup=0
        seen_performance=0
        seen_cluster=0
        seen_plot_final=0
        seen_save_final_plot=0
    }
    {
        if ($0 ~ /^n_sweeps:[[:space:]]*/) {
            print n_sweeps_line
            seen_sweeps=1
            next
        }
        if ($0 ~ /^warmup_sweeps:[[:space:]]*/) {
            print warmup_sweeps_line
            seen_warmup=1
            next
        }
        if ($0 ~ /^save_dir:[[:space:]]*/) {
            print save_dir_line
            seen_save=1
            next
        }
        if ($0 ~ /^performance_mode:[[:space:]]*/) {
            print performance_mode_line
            seen_performance=1
            next
        }
        if ($0 ~ /^cluster_mode:[[:space:]]*/) {
            print cluster_mode_line
            seen_cluster=1
            next
        }
        if ($0 ~ /^plot_final:[[:space:]]*/) {
            print plot_final_line
            seen_plot_final=1
            next
        }
        if ($0 ~ /^save_final_plot:[[:space:]]*/) {
            print save_final_plot_line
            seen_save_final_plot=1
            next
        }
        print
    }
    END {
        if (!seen_sweeps) print n_sweeps_line
        if (!seen_warmup) print warmup_sweeps_line
        if (!seen_save) print save_dir_line
        if (!seen_performance) print performance_mode_line
        if (!seen_cluster) print cluster_mode_line
        if (!seen_plot_final) print plot_final_line
        if (!seen_save_final_plot) print save_final_plot_line
    }' "${source_config}" > "${target_config}"
}

run_id=""
mode="auto"
num_replicas=""
n_sweeps=""
request_cpus="1"
request_memory="5 GB"
aggregate_request_cpus="1"
batch_name=""
job_label=""
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --nr|--num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --ns|--n_sweeps)
            n_sweeps="${2:-}"
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

if [[ -z "${run_id}" || -z "${num_replicas}" || -z "${n_sweeps}" ]]; then
    echo "--run_id, --nr/--num_replicas, and --ns/--n_sweeps are required."
    usage
    exit 1
fi

for numeric_name in num_replicas n_sweeps request_cpus aggregate_request_cpus; do
    value="${!numeric_name}"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        echo "--${numeric_name} must be a positive integer. Got '${value}'."
        exit 1
    fi
done

target_run_info="$(resolve_target_production_run_info "${run_id}" "${mode}")"
target_run_root="$(read_run_info_value "${target_run_info}" "run_root")"
state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
aggregated_dir="$(read_run_info_value "${target_run_info}" "aggregated_dir")"
config_path="$(read_run_info_value "${target_run_info}" "runtime_config")"
target_manifest="$(read_run_info_value "${target_run_info}" "manifest")"
if [[ -z "${config_path}" ]]; then
    config_path="$(read_run_info_value "${target_run_info}" "config_path")"
fi
aggregate_run_id="$(read_run_info_value "${target_run_info}" "aggregate_run_id")"

[[ -n "${target_run_root}" && -d "${target_run_root}" ]] || {
    echo "Could not resolve target run_root from ${target_run_info}"
    exit 1
}
[[ -n "${state_dir}" && -d "${state_dir}" ]] || {
    echo "Could not resolve source state_dir from ${target_run_info}"
    exit 1
}
if [[ -z "${aggregated_dir}" ]]; then
    aggregated_dir="${target_run_root}/aggregated"
fi
[[ -d "${aggregated_dir}" ]] || {
    echo "Could not resolve aggregated_dir from ${target_run_info}"
    exit 1
}
[[ -n "${config_path}" && -f "${config_path}" ]] || {
    echo "Could not resolve config_path/runtime_config from ${target_run_info}"
    exit 1
}
if [[ -z "${aggregate_run_id}" ]]; then
    aggregate_run_id="aggregated_${run_id}"
fi

current_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "${aggregate_run_id}")"
if [[ -z "${current_aggregate}" ]]; then
    echo "Could not find the latest aggregate for save_tag='${aggregate_run_id}' under ${aggregated_dir}"
    exit 1
fi

mapfile -t initial_state_pool < <(
    resolve_initial_state_pool_from_manifest "${target_manifest}" "${state_dir}" || true
)
if (( ${#initial_state_pool[@]} == 0 )); then
    mapfile -t initial_state_pool < <(
        find "${state_dir}" -maxdepth 1 \( -type f -o -type l \) -name '*.jld2' -size +0c \
            ! -name '*_id-aggregated_*.jld2' \
            | sort
    )
fi
if (( ${#initial_state_pool[@]} == 0 )); then
    echo "No non-aggregated saved states were found under ${state_dir}."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
job_token_base="topup_ns${n_sweeps}_nr${num_replicas}_${timestamp}"
if [[ -n "${job_label}" ]]; then
    job_token_base="${job_label}_${job_token_base}"
fi
job_token="$(sanitize_token "${job_token_base}")"
job_root="${target_run_root}/add_to_aggregate_jobs/${job_token}"
config_dir="${job_root}/configs"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
raw_state_dir="${state_dir}/topup_batches/${job_token}"
archive_dir="${aggregated_dir}/archive/${job_token}"
job_info="${job_root}/run_info.txt"
manifest="${job_root}/manifest.csv"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${raw_state_dir}" "${archive_dir}"
ensure_cluster_shared_dir_permissions "${target_run_root}" 755
ensure_cluster_shared_dir_permissions "${state_dir}" 1777
ensure_cluster_shared_dir_permissions "${aggregated_dir}" 1777
ensure_cluster_shared_dir_permissions "${job_root}" 755
ensure_cluster_shared_dir_permissions "${config_dir}" 755
ensure_cluster_shared_dir_permissions "${submit_dir}" 755
ensure_cluster_shared_dir_permissions "${log_dir}" 1777
ensure_cluster_shared_dir_permissions "${raw_state_dir}" 1777
ensure_cluster_shared_dir_permissions "${archive_dir}" 1777

runtime_config="${config_dir}/$(basename "${config_path%.yaml}")_topup_${job_token}.yaml"
rewrite_runtime_config "${config_path}" "${runtime_config}" "${raw_state_dir}" "${n_sweeps}" "0" "true" "false" "false" "true"

replica_tag_prefix="topup_${job_token}_r"
dag_file="${submit_dir}/ssep_add_states_to_aggregate.dag"
aggregate_submit_file="${submit_dir}/ssep_add_states_aggregate.sub"
aggregate_output_file="${log_dir}/ssep_add_states_aggregate.out"
aggregate_error_file="${log_dir}/ssep_add_states_aggregate.err"
aggregate_log_file="${log_dir}/ssep_add_states_aggregate.log"
aggregate_job_info="${job_root}/aggregate_result.txt"
job_batch_name="${batch_name:-${run_id}_topup_${job_token}}"

: > "${dag_file}"
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag,initial_state" > "${manifest}"

replica_job_ids=()
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_submit_file="${submit_dir}/ssep_topup_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/ssep_topup_r${replica_idx}.out"
    replica_error_file="${log_dir}/ssep_topup_r${replica_idx}.err"
    replica_log_file="${log_dir}/ssep_topup_r${replica_idx}.log"
    replica_tag="${replica_tag_prefix}${replica_idx}"
    pool_idx=$(( RANDOM % ${#initial_state_pool[@]} ))
    initial_state="${initial_state_pool[$pool_idx]}"
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
    printf "replica,%s,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${replica_submit_file}" "${replica_output_file}" "${replica_error_file}" "${replica_log_file}" "${replica_tag}" "${initial_state}" \
        >> "${manifest}"
done

aggregate_arguments="${AGGREGATE_SCRIPT} --config ${runtime_config} --target_run_info ${target_run_info} --raw_state_dir ${raw_state_dir} --num_replicas ${num_replicas} --replica_tag_prefix ${replica_tag_prefix} --save_tag ${aggregate_run_id} --archive_dir ${archive_dir} --job_info ${aggregate_job_info}"
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
request_memory = ${request_memory}
batch_name = ${job_batch_name}
queue
EOF

printf "JOB AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
printf "PARENT %s CHILD AGG\n" "${replica_job_ids[*]}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s,%s\n" \
    "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${aggregate_run_id}" "${current_aggregate}" \
    >> "${manifest}"
dag_append_final_notification_node "${dag_file}" "${submit_dir}" "${log_dir}" "${job_root}" "${run_id}" "ssep_add_states_to_aggregate" "${REPO_ROOT}"

cluster_id=""
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated DAG but not submitting: ${dag_file}"
    cluster_id="NO_SUBMIT"
else
    submit_output="$(condor_submit_dag "${dag_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

cat > "${job_info}" <<EOF
timestamp=${timestamp}
target_run_id=${run_id}
target_run_info=${target_run_info}
target_run_root=${target_run_root}
target_manifest=${target_manifest}
config_path=${config_path}
runtime_config=${runtime_config}
state_dir=${state_dir}
aggregated_dir=${aggregated_dir}
current_aggregate=${current_aggregate}
aggregate_run_id=${aggregate_run_id}
initial_state_pool_count=${#initial_state_pool[@]}
num_replicas=${num_replicas}
n_sweeps=${n_sweeps}
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
job_batch_name=${job_batch_name}
job_token=${job_token}
job_root=${job_root}
raw_state_dir=${raw_state_dir}
archive_dir=${archive_dir}
manifest=${manifest}
dag_file=${dag_file}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
cluster_id=${cluster_id}
EOF

echo "SSEP aggregate top-up prepared."
echo "  target_run_id=${run_id}"
echo "  current_aggregate=${current_aggregate}"
echo "  aggregate_run_id=${aggregate_run_id}"
echo "  num_replicas=${num_replicas}"
echo "  n_sweeps=${n_sweeps}"
echo "  raw_state_dir=${raw_state_dir}"
echo "  archive_dir=${archive_dir}"
echo "  dag_file=${dag_file}"
echo "  job_info=${job_info}"
