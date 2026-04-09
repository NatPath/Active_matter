#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_active_objects_add_states_to_histograms.sh \
      --run_id <id> \
      --nr <int> \
      --ns <int> \
      [options]

Required:
  --run_id <id>                    Existing active-objects histogram run_id
  --nr <int>                       Number of new replica states to add
  --ns <int>                       Sweeps for each new replica state

Options:
  --num_replicas <int>             Alias for --nr
  --n_sweeps <int>                 Alias for --ns
  --request_cpus <int>             Replica request_cpus (default: 1)
  --request_memory <value>         Replica and aggregate request_memory (default: "5 GB")
  --aggregate_request_cpus <int>   Aggregate request_cpus (default: 1)
  --tr <int>                       Override aggregate thermalization cutoff (default: inherit target run)
  --max_sweep <int>                Override aggregate max_sweep (default: inherit target run)
  --job_label <label>              Optional label inside the top-up batch token
  --no_submit                      Generate files only; do not call condor_submit_dag
  -h, --help                       Show help

Behavior:
  - Resolves the target active-object run from --run_id
  - Reuses the target run's runtime config as a template
  - Writes new raw states under:
      <state_dir>/topup_batches/<job_token>/
  - Requires an existing live aggregate under:
      <histogram_dir>/aggregated/
  - Aggregates the successfully written new top-up states on top of that latest
    live aggregate, even if some top-up replicas fail
  - Archives the previous live aggregate under:
      <histogram_dir>/aggregated/archive/<job_token>/
  - Top-up aggregation keeps the target run's tr/max_sweep window; use
      submit_active_objects_saved_states_into_histograms.sh
    if you need a full rebuild with a different analysis window
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_active_objects_from_config.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_active_object_histograms_from_tags.sh"
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing active-object runner wrapper: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing histogram aggregate helper: ${AGGREGATE_SCRIPT}"
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

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

latest_histogram_aggregate() {
    local aggregated_dir="$1"
    local save_tag="$2"
    [[ -d "${aggregated_dir}" ]] || return 0
    find "${aggregated_dir}" -maxdepth 1 -type f -name "${save_tag}*_steady_state_hist.jld2" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | awk '{ $1=""; sub(/^ /,""); print }'
}

rewrite_runtime_config() {
    local source_config="$1"
    local target_config="$2"
    local save_dir="$3"
    local n_sweeps_val="$4"

    local save_dir_line n_sweeps_line performance_mode_line cluster_mode_line plot_final_line
    local save_final_plot_line live_plot_line save_live_plot_line

    save_dir_line="save_dir: \"${save_dir}\""
    n_sweeps_line="n_sweeps: ${n_sweeps_val}"
    performance_mode_line="performance_mode: true"
    cluster_mode_line="cluster_mode: true"
    plot_final_line="plot_final: false"
    save_final_plot_line="save_final_plot: false"
    live_plot_line="live_plot: false"
    save_live_plot_line="save_live_plot: false"

    awk \
        -v save_dir_line="${save_dir_line}" \
        -v n_sweeps_line="${n_sweeps_line}" \
        -v performance_mode_line="${performance_mode_line}" \
        -v cluster_mode_line="${cluster_mode_line}" \
        -v plot_final_line="${plot_final_line}" \
        -v save_final_plot_line="${save_final_plot_line}" \
        -v live_plot_line="${live_plot_line}" \
        -v save_live_plot_line="${save_live_plot_line}" '
        BEGIN {
            seen_save = seen_sweeps = seen_perf = seen_cluster = 0
            seen_plot_final = seen_save_plot = seen_live_plot = seen_save_live_plot = 0
        }
        {
            if ($0 ~ /^[[:space:]]*save_dir:[[:space:]]*/) {
                print save_dir_line
                seen_save = 1
                next
            }
            if ($0 ~ /^[[:space:]]*n_sweeps:[[:space:]]*/) {
                print n_sweeps_line
                seen_sweeps = 1
                next
            }
            if ($0 ~ /^[[:space:]]*performance_mode:[[:space:]]*/) {
                print performance_mode_line
                seen_perf = 1
                next
            }
            if ($0 ~ /^[[:space:]]*cluster_mode:[[:space:]]*/) {
                print cluster_mode_line
                seen_cluster = 1
                next
            }
            if ($0 ~ /^[[:space:]]*plot_final:[[:space:]]*/) {
                print plot_final_line
                seen_plot_final = 1
                next
            }
            if ($0 ~ /^[[:space:]]*save_final_plot:[[:space:]]*/) {
                print save_final_plot_line
                seen_save_plot = 1
                next
            }
            if ($0 ~ /^[[:space:]]*live_plot:[[:space:]]*/) {
                print live_plot_line
                seen_live_plot = 1
                next
            }
            if ($0 ~ /^[[:space:]]*save_live_plot:[[:space:]]*/) {
                print save_live_plot_line
                seen_save_live_plot = 1
                next
            }
            print
        }
        END {
            if (!seen_save) print save_dir_line
            if (!seen_sweeps) print n_sweeps_line
            if (!seen_perf) print performance_mode_line
            if (!seen_cluster) print cluster_mode_line
            if (!seen_plot_final) print plot_final_line
            if (!seen_save_plot) print save_final_plot_line
            if (!seen_live_plot) print live_plot_line
            if (!seen_save_live_plot) print save_live_plot_line
        }' "${source_config}" > "${target_config}"
}

run_id=""
num_replicas=""
n_sweeps=""
request_cpus="1"
request_memory="5 GB"
aggregate_request_cpus="1"
tr_sweeps=""
max_sweep=""
job_label=""
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
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
        --tr|--min_sweep)
            tr_sweeps="${2:-}"
            shift 2
            ;;
        --max_sweep)
            max_sweep="${2:-}"
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
    echo "Missing required arguments."
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
if [[ -n "${tr_sweeps}" ]] && ! [[ "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    echo "--tr must be an integer. Got '${tr_sweeps}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer. Got '${max_sweep}'."
    exit 1
fi

target_run_root="${REPO_ROOT}/runs/active_objects/steady_state_histograms/${run_id}"
target_run_info="${target_run_root}/run_info.txt"
if [[ ! -f "${target_run_info}" ]]; then
    echo "Could not resolve active-object run_info for run_id='${run_id}'."
    exit 1
fi

target_runtime_config="$(read_run_info_value "${target_run_info}" "runtime_config")"
target_state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
target_hist_dir="$(read_run_info_value "${target_run_info}" "histogram_dir")"
target_tr_sweeps="$(read_run_info_value "${target_run_info}" "tr_sweeps")"
if [[ -z "${target_tr_sweeps}" ]]; then
    target_tr_sweeps="$(read_run_info_value "${target_run_info}" "min_sweep")"
fi
target_max_sweep="$(read_run_info_value "${target_run_info}" "max_sweep")"
aggregate_save_tag="aggregated_${run_id}"
target_aggregate_dir="${target_hist_dir}/aggregated"

[[ -n "${target_runtime_config}" && -f "${target_runtime_config}" ]] || {
    echo "Target runtime_config is missing for run_id='${run_id}'."
    exit 1
}
[[ -n "${target_state_dir}" && -d "${target_state_dir}" ]] || {
    echo "Target state_dir is missing for run_id='${run_id}'."
    exit 1
}
[[ -n "${target_hist_dir}" ]] || {
    echo "Target histogram_dir is missing for run_id='${run_id}'."
    exit 1
}

if [[ -z "${tr_sweeps}" ]]; then
    tr_sweeps="${target_tr_sweeps:-0}"
fi
if [[ -z "${max_sweep}" ]]; then
    max_sweep="${target_max_sweep:-}"
fi
if [[ "${tr_sweeps}" != "${target_tr_sweeps:-0}" ]]; then
    echo "Top-up aggregation must use the target run's existing tr_sweeps=${target_tr_sweeps:-0}."
    echo "Requested tr=${tr_sweeps} would invalidate incremental aggregation."
    echo "Use submit_active_objects_saved_states_into_histograms.sh for a full rebuild with a different tr."
    exit 1
fi
if [[ "${max_sweep}" != "${target_max_sweep:-}" ]]; then
    echo "Top-up aggregation must use the target run's existing max_sweep='${target_max_sweep:-}'."
    echo "Requested max_sweep='${max_sweep}' would invalidate incremental aggregation."
    echo "Use submit_active_objects_saved_states_into_histograms.sh for a full rebuild with a different max_sweep."
    exit 1
fi
if [[ "${n_sweeps}" =~ ^[0-9]+$ && "${tr_sweeps}" =~ ^-?[0-9]+$ ]]; then
    if (( tr_sweeps >= n_sweeps )); then
        echo "Resolved histogram tr=${tr_sweeps} is not valid for top-up n_sweeps=${n_sweeps}."
        echo "Pass --tr explicitly with a value smaller than --ns/--n_sweeps."
        exit 1
    fi
fi

current_live_aggregate="$(latest_histogram_aggregate "${target_aggregate_dir}" "${aggregate_save_tag}")"
if [[ -z "${current_live_aggregate}" ]]; then
    echo "Could not resolve an existing live aggregate under ${target_aggregate_dir} for save_tag=${aggregate_save_tag}."
    echo "Top-up aggregation requires a current aggregate to build on top of."
    echo "Run submit_active_objects_saved_states_into_histograms.sh first to create or repair the aggregate."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
job_token="topup_ns${n_sweeps}_nr${num_replicas}"
if [[ -n "${job_label}" ]]; then
    job_token="${job_token}_$(sanitize_token "${job_label}")"
fi
job_token="${job_token}_${timestamp}"

job_root="${REPO_ROOT}/runs/active_objects/add_to_histograms_jobs/${run_id}_${job_token}"
config_dir="${job_root}/configs"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
raw_state_dir="${target_state_dir}/topup_batches/${job_token}"
runtime_config="${config_dir}/$(basename "${target_runtime_config%.yaml}")_topup_${job_token}.yaml"
manifest="${job_root}/manifest.csv"
run_info="${job_root}/run_info.txt"
dag_file="${submit_dir}/active_objects_add_states_to_histograms.dag"
aggregate_submit_file="${submit_dir}/active_objects_add_states_aggregate.sub"
aggregate_output_file="${log_dir}/active_objects_add_states_aggregate.out"
aggregate_error_file="${log_dir}/active_objects_add_states_aggregate.err"
aggregate_log_file="${log_dir}/active_objects_add_states_aggregate.log"
aggregate_launcher_script="${submit_dir}/run_active_objects_topup_aggregate.sh"
archive_dir="${target_hist_dir}/aggregated/archive/${job_token}"
replica_tag_prefix="replica_addrep_${job_token}_r"
job_batch_name="${run_id}_topup_${job_token}"

mkdir -p "${config_dir}" "${submit_dir}" "${log_dir}" "${raw_state_dir}" "${target_hist_dir}" "${archive_dir}"
rewrite_runtime_config "${target_runtime_config}" "${runtime_config}" "${raw_state_dir}" "${n_sweeps}"

: > "${dag_file}"
echo "job_type,job_name,submit_file,output_file,error_file,log_file,save_tag" > "${manifest}"

replica_job_ids=()
for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
    replica_submit_file="${submit_dir}/active_objects_topup_replica_${replica_idx}.sub"
    replica_output_file="${log_dir}/active_objects_topup_r${replica_idx}.out"
    replica_error_file="${log_dir}/active_objects_topup_r${replica_idx}.err"
    replica_log_file="${log_dir}/active_objects_topup_r${replica_idx}.log"
    replica_tag="${replica_tag_prefix}${replica_idx}"

    cat > "${replica_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${RUNNER_SCRIPT} ${runtime_config} --save_tag ${replica_tag} --performance_mode
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
    printf "replica,%s,%s,%s,%s,%s,%s\n" \
        "${job_id}" "${replica_submit_file}" "${replica_output_file}" "${replica_error_file}" "${replica_log_file}" "${replica_tag}" \
        >> "${manifest}"
done

cat > "${aggregate_launcher_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

latest_histogram_aggregate() {
    local aggregated_dir="\$1"
    local save_tag="\$2"
    [[ -d "\${aggregated_dir}" ]] || return 0
    find "\${aggregated_dir}" -maxdepth 1 -type f -name "\${save_tag}*_steady_state_hist.jld2" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -n 1 | awk '{ \$1=""; sub(/^ /,""); print }'
}

stage_aggregate_family() {
    local aggregate_jld2="\$1"
    local staging_dir="\$2"
    local prefix="\${aggregate_jld2%.jld2}"
    mkdir -p "\${staging_dir}"
    while IFS= read -r -d '' artifact; do
        mv "\${artifact}" "\${staging_dir}/"
    done < <(find "\$(dirname "\${aggregate_jld2}")" -maxdepth 1 -type f -name "\$(basename "\${prefix}")*" -print0 2>/dev/null)
}

restore_staged_family() {
    local staging_dir="\$1"
    local live_dir="\$2"
    [[ -d "\${staging_dir}" ]] || return 0
    while IFS= read -r -d '' artifact; do
        mv "\${artifact}" "\${live_dir}/"
    done < <(find "\${staging_dir}" -maxdepth 1 -type f -print0 2>/dev/null)
}

archive_noncurrent_families() {
    local aggregated_dir="\$1"
    local archive_dir_local="\$2"
    local save_tag="\$3"
    local current_jld2="\$4"
    local current_prefix="\${current_jld2%.jld2}"
    while IFS= read -r -d '' artifact; do
        if [[ "\${artifact}" == "\${current_prefix}"* ]]; then
            continue
        fi
        mv "\${artifact}" "\${archive_dir_local}/"
    done < <(find "\${aggregated_dir}" -maxdepth 1 -type f -name "\${save_tag}*_steady_state_hist*" -print0 2>/dev/null)
}

cd $(printf '%q' "${REPO_ROOT}")
aggregate_dir=$(printf '%q' "${target_aggregate_dir}")
archive_dir=$(printf '%q' "${archive_dir}")
save_tag=$(printf '%q' "${aggregate_save_tag}")
raw_state_dir=$(printf '%q' "${raw_state_dir}")
hist_dir=$(printf '%q' "${target_hist_dir}")
staging_dir=$(printf '%q' "${job_root}/staged_previous_aggregate")
min_sweep=$(printf '%q' "${tr_sweeps}")
current_aggregate="\$(latest_histogram_aggregate "\${aggregate_dir}" "\${save_tag}")"
[[ -n "\${current_aggregate}" ]] || {
    echo "No current live aggregate found for save_tag=\${save_tag} under \${aggregate_dir}"
    exit 1
}
mkdir -p "\${archive_dir}"
stage_aggregate_family "\${current_aggregate}" "\${staging_dir}"
staged_base_aggregate="\${staging_dir}/\$(basename "\${current_aggregate}")"
aggregation_succeeded="false"
cleanup() {
    if [[ "\${aggregation_succeeded}" != "true" ]]; then
        find "\${aggregate_dir}" -maxdepth 1 -type f -name "\${save_tag}*_steady_state_hist*" -delete 2>/dev/null || true
        restore_staged_family "\${staging_dir}" "\${aggregate_dir}"
    fi
}
trap cleanup EXIT

cmd=(bash $(printf '%q' "${AGGREGATE_SCRIPT}")
    --state_dir "\${raw_state_dir}"
    --output_dir "\${hist_dir}"
    --save_tag "\${save_tag}"
    --all_states_recursive
    --base_aggregate "\${staged_base_aggregate}"
    --min_sweep "\${min_sweep}"
    --no_plot)
if [[ -n $(printf '%q' "${max_sweep}") ]]; then
    cmd+=(--max_sweep $(printf '%q' "${max_sweep}"))
fi
"\${cmd[@]}"

new_aggregate="\$(latest_histogram_aggregate "\${aggregate_dir}" "\${save_tag}")"
[[ -n "\${new_aggregate}" ]] || {
    echo "Top-up aggregation finished but no new aggregate was found for save_tag=\${save_tag}"
    exit 1
}

archive_noncurrent_families "\${aggregate_dir}" "\${archive_dir}" "\${save_tag}" "\${new_aggregate}"
restore_staged_family "\${staging_dir}" "\${archive_dir}"
rmdir "\${staging_dir}" 2>/dev/null || true
aggregation_succeeded="true"
echo "Top-up aggregate complete."
echo "  base_aggregate=\${staged_base_aggregate}"
echo "  new_aggregate=\${new_aggregate}"
echo "  archive_dir=\${archive_dir}"
EOF
chmod +x "${aggregate_launcher_script}"

cat > "${aggregate_submit_file}" <<EOF
Universe   = vanilla
Executable = ${aggregate_launcher_script}
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

printf "FINAL AGG %s\n" "${aggregate_submit_file}" >> "${dag_file}"
printf "aggregate,AGG,%s,%s,%s,%s,%s\n" \
    "${aggregate_submit_file}" "${aggregate_output_file}" "${aggregate_error_file}" "${aggregate_log_file}" "${aggregate_save_tag}" \
    >> "${manifest}"
dag_append_post_notification_script "${dag_file}" "AGG" "${submit_dir}" "${log_dir}" "${job_root}" "${run_id}" "active_objects_add_states_to_histograms" "${REPO_ROOT}"

cat > "${run_info}" <<EOF
target_run_id=${run_id}
target_run_root=${target_run_root}
target_runtime_config=${target_runtime_config}
target_state_dir=${target_state_dir}
target_histogram_dir=${target_hist_dir}
current_live_aggregate=${current_live_aggregate}
archive_dir=${archive_dir}
raw_state_dir=${raw_state_dir}
job_root=${job_root}
job_token=${job_token}
runtime_config=${runtime_config}
manifest=${manifest}
num_replicas=${num_replicas}
n_sweeps=${n_sweeps}
request_cpus=${request_cpus}
request_memory=${request_memory}
aggregate_request_cpus=${aggregate_request_cpus}
tr_sweeps=${tr_sweeps}
max_sweep=${max_sweep}
aggregate_save_tag=${aggregate_save_tag}
aggregate_launcher_script=${aggregate_launcher_script}
dag_file=${dag_file}
dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}
EOF

echo "Prepared active-object top-up DAG:"
echo "  target_run_id=${run_id}"
echo "  job_root=${job_root}"
echo "  raw_state_dir=${raw_state_dir}"
echo "  runtime_config=${runtime_config}"
echo "  histogram_dir=${target_hist_dir}"
echo "  current_live_aggregate=${current_live_aggregate}"
echo "  archive_dir=${archive_dir}"
echo "  dag_file=${dag_file}"

if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated DAG but not submitting: ${dag_file}"
    exit 0
fi

submit_output="$(condor_submit_dag "${dag_file}")"
cluster_id="$(printf '%s\n' "${submit_output}" | sed -nE 's/.*submitted to cluster ([0-9]+).*/\1/p' | tail -n 1)"
if [[ -n "${cluster_id}" ]]; then
    {
        echo "cluster_id=${cluster_id}"
        echo "submit_time=$(date +%Y-%m-%dT%H:%M:%S)"
    } >> "${run_info}"
fi
echo "${submit_output}"
