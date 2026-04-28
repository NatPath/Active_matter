#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_managed_diffusive_1d_center_bond_batch.sh \
      --run_id <managed_run_id> --slots <int> [options]

Options:
  --sweeps <int>              Sweeps per selected replica. Defaults to run_spec default_segment_sweeps.
  --checkpoint_interval <int> Defaults to run_spec checkpoint_interval_sweeps.
  --request_cpus <int>        Condor request_cpus (default: 1)
  --request_memory <value>    Condor request_memory (default: "6 GB")
  --dag_maxjobs <int>         Optional DAGMan throttle (default: 0)
  --batch_id <id>             Explicit batch id
  --no_submit                 Generate plan and DAG only; do not claim or submit
  -h, --help                  Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_1d_center_bond_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"
WORKER_SCRIPT="${SCRIPT_DIR}/run_managed_diffusive_1d_center_bond_replica.sh"

run_id=""
slots=""
sweeps=""
checkpoint_interval=""
request_cpus="1"
request_memory="6 GB"
dag_maxjobs="0"
batch_id=""
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id) run_id="${2:-}"; shift 2 ;;
        --slots) slots="${2:-}"; shift 2 ;;
        --sweeps) sweeps="${2:-}"; shift 2 ;;
        --checkpoint_interval) checkpoint_interval="${2:-}"; shift 2 ;;
        --request_cpus) request_cpus="${2:-}"; shift 2 ;;
        --request_memory) request_memory="${2:-}"; shift 2 ;;
        --dag_maxjobs) dag_maxjobs="${2:-}"; shift 2 ;;
        --batch_id) batch_id="${2:-}"; shift 2 ;;
        --no_submit) no_submit="true"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${run_id}" || -z "${slots}" ]]; then
    echo "Missing --run_id or --slots." >&2
    usage
    exit 1
fi
managed_require_positive_int "slots" "${slots}"
managed_require_positive_int "request_cpus" "${request_cpus}"
managed_require_positive_int "dag_maxjobs" "${dag_maxjobs}" 2>/dev/null || {
    if [[ "${dag_maxjobs}" != "0" ]]; then
        echo "--dag_maxjobs must be a non-negative integer. Got '${dag_maxjobs}'." >&2
        exit 1
    fi
}

run_id="$(managed_slugify "${run_id}")"
run_root="$(managed_run_root "${REPO_ROOT}" "${run_id}")"
run_spec="${run_root}/run_spec.yaml"
replicas_csv="${run_root}/replicas.csv"
lock_file="${run_root}/manager.lock"
if [[ ! -f "${run_spec}" || ! -f "${replicas_csv}" ]]; then
    echo "Managed run is not initialized: ${run_root}" >&2
    exit 1
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "Missing worker script: ${WORKER_SCRIPT}" >&2
    exit 1
fi

if [[ -z "${sweeps}" ]]; then
    sweeps="$(managed_yaml_value "${run_spec}" "default_segment_sweeps")"
fi
if [[ -z "${checkpoint_interval}" ]]; then
    checkpoint_interval="$(managed_yaml_value "${run_spec}" "checkpoint_interval_sweeps")"
fi
target_replicas="$(managed_yaml_value "${run_spec}" "target_replica_count")"
managed_require_positive_int "sweeps" "${sweeps}"
managed_require_positive_int "checkpoint_interval" "${checkpoint_interval}"
managed_require_positive_int "target_replica_count" "${target_replicas}"

timestamp="$(managed_timestamp)"
if [[ -z "${batch_id}" ]]; then
    batch_id="batch_${timestamp}_advance_ns${sweeps}_n${slots}"
else
    batch_id="$(managed_slugify "${batch_id}")"
fi

batch_root="${run_root}/batches/${batch_id}"
submit_dir="${batch_root}/submit"
log_dir="${batch_root}/logs"
config_dir="${batch_root}/configs"
plan_csv="${batch_root}/plan.csv"
dag_file="${submit_dir}/managed_diffusive_1d_center_bond.dag"
mkdir -p "${submit_dir}" "${log_dir}" "${config_dir}"

selected_ids_file="${batch_root}/selected_ids.txt"
new_rows_file="${batch_root}/new_rows.csv"
: > "${selected_ids_file}"
: > "${new_rows_file}"

(
    flock 9
    echo "replica_id,start_state,phase,elapsed_sweeps,statistics_sweeps,requested_sweeps,output_state,result_meta,segment_id,seed" > "${plan_csv}"

    candidates_file="${batch_root}/candidates.csv"
    awk -F, '
        NR == 1 { next }
        ($9 == "" || $9 == "idle") {
            if ($2 == "ready") {
                pri = 1
                metric = $4 + 0
            } else if ($2 == "production") {
                pri = 2
                metric = $4 + 0
            } else {
                pri = 3
                metric = -($3 + 0)
            }
            printf("%d,%020.0f,%s\n", pri, metric, $0)
        }
    ' "${replicas_csv}" | sort -t, -k1,1n -k2,2n -k3,3 > "${candidates_file}"

    selected=0
    while IFS=, read -r _pri _metric rid phase elapsed stats latest source_path source_tag claim status updated; do
        [[ -n "${rid}" ]] || continue
        if (( selected >= slots )); then
            break
        fi
        replica_dir="${run_root}/replicas/${rid}"
        output_state="${replica_dir}/current.jld2"
        result_meta="${replica_dir}/current.meta"
        segment_id="${batch_id}_${rid}"
        seed=$(( $(date +%s) + selected + 1000 ))
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${rid}" "${latest}" "${phase}" "${elapsed}" "${stats}" "${sweeps}" \
            "${output_state}" "${result_meta}" "${segment_id}" "${seed}" >> "${plan_csv}"
        echo "${rid}" >> "${selected_ids_file}"
        selected=$((selected + 1))
    done < "${candidates_file}"

    total_existing="$(awk -F, 'NR > 1 { c += 1 } END { print c + 0 }' "${replicas_csv}")"
    next_idx="$(managed_next_replica_index "${replicas_csv}")"
    while (( selected < slots && total_existing < target_replicas )); do
        rid="$(managed_replica_id "${next_idx}")"
        replica_dir="${run_root}/replicas/${rid}"
        output_state="${replica_dir}/current.jld2"
        result_meta="${replica_dir}/current.meta"
        segment_id="${batch_id}_${rid}"
        seed=$(( $(date +%s) + selected + 1000 ))
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${rid}" "" "warmup" "0" "0" "${sweeps}" \
            "${output_state}" "${result_meta}" "${segment_id}" "${seed}" >> "${plan_csv}"
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${rid}" "warmup" "0" "0" "" "" "new_random" "${batch_id}" "running" "${timestamp}" >> "${new_rows_file}"
        echo "${rid}" >> "${selected_ids_file}"
        selected=$((selected + 1))
        total_existing=$((total_existing + 1))
        next_idx=$((next_idx + 1))
    done

    if (( selected == 0 )); then
        echo "No available work was selected." >&2
        exit 2
    fi

    if [[ "${no_submit}" != "true" ]]; then
        selected_ids="$(tr '\n' ' ' < "${selected_ids_file}")"
        tmp_replicas="${replicas_csv}.tmp.$$"
        awk -F, -v OFS=',' -v ids="${selected_ids}" -v claim="${batch_id}" -v updated="${timestamp}" '
            BEGIN {
                n = split(ids, parts, " ")
                for (i = 1; i <= n; i++) if (parts[i] != "") selected[parts[i]] = 1
            }
            NR == 1 { print; next }
            ($1 in selected) {
                $8=claim; $9="running"; $10=updated
            }
            { print }
        ' "${replicas_csv}" > "${tmp_replicas}"
        cat "${new_rows_file}" >> "${tmp_replicas}"
        mv -f "${tmp_replicas}" "${replicas_csv}"
    fi
) 9>"${lock_file}"

: > "${dag_file}"
if (( dag_maxjobs > 0 )); then
    printf "MAXJOBS REPLICAS %s\n" "${dag_maxjobs}" >> "${dag_file}"
fi

job_idx=0
tail -n +2 "${plan_csv}" | while IFS=, read -r rid start_state phase elapsed stats requested output_state result_meta segment_id seed; do
    job_idx=$((job_idx + 1))
    job_id="R${job_idx}"
    submit_file="${submit_dir}/${rid}.sub"
    out_file="${log_dir}/${rid}.out"
    err_file="${log_dir}/${rid}.err"
    log_file="${log_dir}/${rid}.log"

    start_state_arg=""
    if [[ -n "${start_state}" ]]; then
        start_state_arg="--start_state ${start_state}"
    fi

    cat > "${submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${WORKER_SCRIPT} --run_root ${run_root} --run_spec ${run_spec} --replica_id ${rid} ${start_state_arg} --sweeps ${requested} --checkpoint_interval ${checkpoint_interval} --start_statistics_sweeps ${stats} --output_state ${output_state} --result_meta ${result_meta} --batch_id ${batch_id} --segment_id ${segment_id} --seed ${seed}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
getenv = True
output     = ${out_file}
error      = ${err_file}
log        = ${log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${run_id}_${batch_id}
queue
EOF
    printf "JOB %s %s\n" "${job_id}" "${submit_file}" >> "${dag_file}"
    printf "RETRY %s 1\n" "${job_id}" >> "${dag_file}"
    if (( dag_maxjobs > 0 )); then
        printf "CATEGORY %s REPLICAS\n" "${job_id}" >> "${dag_file}"
    fi
done

cluster_id="NO_SUBMIT"
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated managed batch DAG but did not claim or submit."
else
    if ! submit_output="$(condor_submit_dag "${dag_file}" 2>&1)"; then
        echo "${submit_output}" >&2
        echo "condor_submit_dag failed; clearing claims for batch ${batch_id}." >&2
        (
            flock 9
            tmp_replicas="${run_root}/replicas.csv.tmp.$$"
            awk -F, -v OFS=',' -v claim="${batch_id}" -v updated="$(managed_timestamp)" '
                NR == 1 { print; next }
                $8 == claim {
                    $8=""; $9="idle"; $10=updated
                }
                { print }
            ' "${run_root}/replicas.csv" > "${tmp_replicas}"
            mv -f "${tmp_replicas}" "${run_root}/replicas.csv"
        ) 9>"${lock_file}"
        exit 1
    fi
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

cat > "${batch_root}/batch_info.txt" <<EOF
run_id=${run_id}
batch_id=${batch_id}
timestamp=${timestamp}
sweeps=${sweeps}
checkpoint_interval=${checkpoint_interval}
slots_requested=${slots}
plan_csv=${plan_csv}
dag_file=${dag_file}
submit_dir=${submit_dir}
log_dir=${log_dir}
cluster_id=${cluster_id}
no_submit=${no_submit}
EOF

echo "Prepared managed diffusive 1D center-bond batch."
echo "  run_id=${run_id}"
echo "  batch_id=${batch_id}"
echo "  plan=${plan_csv}"
echo "  dag=${dag_file}"
echo "  no_submit=${no_submit}"
