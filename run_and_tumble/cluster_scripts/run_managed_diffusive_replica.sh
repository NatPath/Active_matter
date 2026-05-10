#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_managed_diffusive_replica.sh \
      --run_root <dir> --run_spec <yaml> --replica_id <id> \
      --sweeps <int> --checkpoint_interval <int> \
      --start_statistics_sweeps <int> \
      --output_state <path> --result_meta <path> \
      --batch_id <id> --segment_id <id> [--start_state <path>] [--seed <int>]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    REPO_ROOT="${SCRIPT_DIR}"
fi

run_root=""
run_spec=""
replica_id=""
start_state=""
sweeps=""
checkpoint_interval=""
start_statistics_sweeps="0"
output_state=""
result_meta=""
batch_id=""
segment_id=""
seed="1234"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_root) run_root="${2:-}"; shift 2 ;;
        --run_spec) run_spec="${2:-}"; shift 2 ;;
        --replica_id) replica_id="${2:-}"; shift 2 ;;
        --start_state) start_state="${2:-}"; shift 2 ;;
        --sweeps) sweeps="${2:-}"; shift 2 ;;
        --checkpoint_interval) checkpoint_interval="${2:-}"; shift 2 ;;
        --start_statistics_sweeps) start_statistics_sweeps="${2:-}"; shift 2 ;;
        --output_state) output_state="${2:-}"; shift 2 ;;
        --result_meta) result_meta="${2:-}"; shift 2 ;;
        --batch_id) batch_id="${2:-}"; shift 2 ;;
        --segment_id) segment_id="${2:-}"; shift 2 ;;
        --seed) seed="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${run_root}" || -z "${run_spec}" || -z "${replica_id}" || -z "${sweeps}" ||
      -z "${checkpoint_interval}" || -z "${output_state}" || -z "${result_meta}" ||
      -z "${batch_id}" || -z "${segment_id}" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

JULIA_SETUP_SCRIPT="${JULIA_SETUP_SCRIPT:-${CLUSTER_JULIA_SETUP_SCRIPT:-}}"
if [[ -n "${JULIA_SETUP_SCRIPT}" && -f "${JULIA_SETUP_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${JULIA_SETUP_SCRIPT}"
fi

JULIA_BIN="${JULIA_BIN:-julia}"
if ! command -v "${JULIA_BIN}" >/dev/null 2>&1; then
    echo "Julia executable '${JULIA_BIN}' not found in PATH." >&2
    exit 127
fi

meta_value() {
    local key="$1"
    [[ -f "${result_meta}" ]] || return 0
    awk -v key="${key}" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "${result_meta}"
}

lock_file="${run_root}/manager.lock"
replicas_csv="${run_root}/replicas.csv"
segments_csv="${run_root}/segments.csv"

mark_replica_finished() {
    local segment_status="$1"
    local phase elapsed_sweeps statistics_sweeps latest_state updated_at
    phase="$(meta_value phase)"
    elapsed_sweeps="$(meta_value elapsed_sweeps)"
    statistics_sweeps="$(meta_value statistics_sweeps)"
    latest_state="$(meta_value latest_state)"
    updated_at="$(meta_value updated_at)"
    if [[ -z "${updated_at}" ]]; then
        updated_at="$(date +%Y%m%d-%H%M%S)"
    fi

    (
        flock 9
        tmp_replicas="${replicas_csv}.tmp.$$"
        awk -F, -v OFS=',' \
            -v rid="${replica_id}" \
            -v phase="${phase}" \
            -v elapsed="${elapsed_sweeps}" \
            -v stats="${statistics_sweeps}" \
            -v latest="${latest_state}" \
            -v updated="${updated_at}" '
            NR == 1 { print; next }
            $1 == rid {
                if (phase != "") $2=phase
                if (elapsed != "") $3=elapsed
                if (stats != "") $4=stats
                if (latest != "") $5=latest
                $8=""; $9="idle"; $10=updated
            }
            { print }
        ' "${replicas_csv}" > "${tmp_replicas}"
        mv -f "${tmp_replicas}" "${replicas_csv}"

        if [[ ! -f "${segments_csv}" ]]; then
            echo "segment_id,replica_id,batch_id,kind,requested_sweeps,start_state,output_state,checkpoint_state,status,statistics_sweeps,submitted_at,completed_at" > "${segments_csv}"
        fi
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${segment_id}" "${replica_id}" "${batch_id}" "advance" "${sweeps}" "${start_state}" \
            "${latest_state}" "${latest_state}" "${segment_status}" "${statistics_sweeps}" "" "${updated_at}" \
            >> "${segments_csv}"
    ) 9>"${lock_file}"
}

cleanup_failed_claim() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 ]]; then
        set +e
        mark_replica_finished "failed"
        set -e
    fi
}
trap cleanup_failed_claim EXIT

cd "${REPO_ROOT}"
"${JULIA_BIN}" --startup-file=no utility_scripts/run_managed_diffusive_replica.jl \
  --run_spec "${run_spec}" \
  --replica_id "${replica_id}" \
  --start_state "${start_state}" \
  --sweeps "${sweeps}" \
  --checkpoint_interval "${checkpoint_interval}" \
  --start_statistics_sweeps "${start_statistics_sweeps}" \
  --output_state "${output_state}" \
  --result_meta "${result_meta}" \
  --seed "${seed}"

trap - EXIT
mark_replica_finished "completed"
