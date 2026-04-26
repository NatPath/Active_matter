#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_managed_diffusive_1d_pmlr_aggregate.sh \
      --run_id <managed_run_id> [options]

Options:
  --include_running              Include latest committed checkpoints of running replicas
  --min_tstats <int>             Minimum statistics sweeps per replica (default: 1)
  --aggregate_tag <tag>          Explicit aggregate tag
  --request_cpus <int>           Condor request_cpus (default: 1)
  --request_memory <value>       Condor request_memory (default: "8 GB")
  --no_submit                    Generate submit file only
  -h, --help                     Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_1d_pmlr_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"
AGG_SCRIPT="${SCRIPT_DIR}/aggregate_managed_diffusive_1d_pmlr.sh"

run_id=""
include_running="false"
min_tstats="1"
aggregate_tag=""
request_cpus="1"
request_memory="8 GB"
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id) run_id="${2:-}"; shift 2 ;;
        --include_running) include_running="true"; shift 1 ;;
        --min_tstats) min_tstats="${2:-}"; shift 2 ;;
        --aggregate_tag) aggregate_tag="${2:-}"; shift 2 ;;
        --request_cpus) request_cpus="${2:-}"; shift 2 ;;
        --request_memory) request_memory="${2:-}"; shift 2 ;;
        --no_submit) no_submit="true"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${run_id}" ]]; then
    echo "Missing --run_id." >&2
    usage
    exit 1
fi
managed_require_positive_int "min_tstats" "${min_tstats}"
managed_require_positive_int "request_cpus" "${request_cpus}"

run_id="$(managed_slugify "${run_id}")"
run_root="$(managed_run_root "${REPO_ROOT}" "${run_id}")"
if [[ ! -f "${run_root}/run_spec.yaml" ]]; then
    echo "Managed run is not initialized: ${run_root}" >&2
    exit 1
fi

timestamp="$(managed_timestamp)"
if [[ -z "${aggregate_tag}" ]]; then
    aggregate_tag="managed_${timestamp}"
else
    aggregate_tag="$(managed_slugify "${aggregate_tag}")"
fi

submit_dir="${run_root}/aggregates/submit/${aggregate_tag}"
log_dir="${run_root}/aggregates/logs"
mkdir -p "${submit_dir}" "${log_dir}"
submit_file="${submit_dir}/aggregate.sub"
out_file="${log_dir}/${aggregate_tag}.out"
err_file="${log_dir}/${aggregate_tag}.err"
log_file="${log_dir}/${aggregate_tag}.log"

include_arg=""
if [[ "${include_running}" == "true" ]]; then
    include_arg="--include_running"
fi

cat > "${submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${AGG_SCRIPT} --run_id ${run_id} --min_tstats ${min_tstats} --aggregate_tag ${aggregate_tag} ${include_arg}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${out_file}
error      = ${err_file}
log        = ${log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${run_id}_aggregate
queue
EOF

cluster_id="NO_SUBMIT"
if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated aggregate submit file but did not submit."
else
    submit_output="$(condor_submit "${submit_file}")"
    echo "${submit_output}"
    cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
    cluster_id="${cluster_id:-NA}"
fi

cat > "${submit_dir}/aggregate_submit_info.txt" <<EOF
run_id=${run_id}
aggregate_tag=${aggregate_tag}
include_running=${include_running}
min_tstats=${min_tstats}
submit_file=${submit_file}
output=${out_file}
error=${err_file}
log=${log_file}
cluster_id=${cluster_id}
no_submit=${no_submit}
EOF

echo "Prepared managed aggregate job."
echo "  run_id=${run_id}"
echo "  aggregate_tag=${aggregate_tag}"
echo "  submit_file=${submit_file}"
echo "  no_submit=${no_submit}"
