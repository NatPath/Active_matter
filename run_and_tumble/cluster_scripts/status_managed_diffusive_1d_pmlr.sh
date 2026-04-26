#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/status_managed_diffusive_1d_pmlr.sh --run_id <managed_run_id> [options]

Options:
  --min_tstats <int>   Eligibility threshold for aggregation counts (default: 1)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_1d_pmlr_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"

run_id=""
min_tstats="1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id) run_id="${2:-}"; shift 2 ;;
        --min_tstats) min_tstats="${2:-}"; shift 2 ;;
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

run_id="$(managed_slugify "${run_id}")"
run_root="$(managed_run_root "${REPO_ROOT}" "${run_id}")"
spec_path="${run_root}/run_spec.yaml"
replicas_csv="${run_root}/replicas.csv"
lock_file="${run_root}/manager.lock"

if [[ ! -f "${spec_path}" || ! -f "${replicas_csv}" ]]; then
    echo "Managed run not found or incomplete: ${run_root}" >&2
    exit 1
fi

echo "Managed diffusive 1D PmLr status"
echo "  run_id=${run_id}"
echo "  run_root=${run_root}"
echo "  target_replica_count=$(managed_yaml_value "${spec_path}" "target_replica_count")"
echo "  warmup_threshold_sweeps=$(managed_yaml_value "${spec_path}" "warmup_threshold_sweeps")"
echo "  default_segment_sweeps=$(managed_yaml_value "${spec_path}" "default_segment_sweeps")"
echo "  checkpoint_interval_sweeps=$(managed_yaml_value "${spec_path}" "checkpoint_interval_sweeps")"
echo "  aggregation_min_tstats=${min_tstats}"
echo

snapshot="$(mktemp)"
trap 'rm -f "${snapshot}"' EXIT
(
    flock 9
    cp "${replicas_csv}" "${snapshot}"
) 9>"${lock_file}"

meta_value() {
    local path="$1"
    local key="$2"
    [[ -f "${path}" ]] || return 0
    awk -v key="${key}" 'index($0, key "=") == 1 { print substr($0, length(key) + 2); exit }' "${path}"
}

declare -A phase_counts=()
declare -A status_counts=()
total=0
ledger_elapsed_sum=0
ledger_stats_sum=0
checkpoint_elapsed_sum=0
checkpoint_stats_sum=0
missing_latest_state_files=0
running_total=0
running_with_checkpoint=0
eligible_idle=0
eligible_running_checkpoint=0
eligible_running_without_checkpoint=0

while IFS=, read -r rid phase elapsed stats latest _source _source_tag _claim status _updated; do
    [[ "${rid}" != "replica_id" && -n "${rid}" ]] || continue
    status="${status:-idle}"
    phase="${phase:-unknown}"
    elapsed="${elapsed:-0}"
    stats="${stats:-0}"
    latest="${latest:-}"

    total=$((total + 1))
    phase_counts["${phase}"]=$(( ${phase_counts["${phase}"]:-0} + 1 ))
    status_counts["${status}"]=$(( ${status_counts["${status}"]:-0} + 1 ))
    ledger_elapsed_sum=$((ledger_elapsed_sum + elapsed))
    ledger_stats_sum=$((ledger_stats_sum + stats))

    live_elapsed="${elapsed}"
    live_stats="${stats}"
    live_latest="${latest}"
    checkpoint_seen="false"

    if [[ "${status}" == "running" ]]; then
        running_total=$((running_total + 1))
        meta_path="${run_root}/replicas/${rid}/current.meta"
        if [[ -f "${meta_path}" ]]; then
            checkpoint_seen="true"
            running_with_checkpoint=$((running_with_checkpoint + 1))
            meta_elapsed="$(meta_value "${meta_path}" "elapsed_sweeps")"
            meta_stats="$(meta_value "${meta_path}" "statistics_sweeps")"
            meta_latest="$(meta_value "${meta_path}" "latest_state")"
            [[ -n "${meta_elapsed}" ]] && live_elapsed="${meta_elapsed}"
            [[ -n "${meta_stats}" ]] && live_stats="${meta_stats}"
            [[ -n "${meta_latest}" ]] && live_latest="${meta_latest}"
        fi
    fi

    checkpoint_elapsed_sum=$((checkpoint_elapsed_sum + live_elapsed))
    checkpoint_stats_sum=$((checkpoint_stats_sum + live_stats))

    if [[ -n "${live_latest}" && ! -f "${live_latest}" ]]; then
        missing_latest_state_files=$((missing_latest_state_files + 1))
    fi

    if (( live_stats >= min_tstats )); then
        if [[ "${status}" == "running" ]]; then
            if [[ "${checkpoint_seen}" == "true" && -n "${live_latest}" && -f "${live_latest}" ]]; then
                eligible_running_checkpoint=$((eligible_running_checkpoint + 1))
            else
                eligible_running_without_checkpoint=$((eligible_running_without_checkpoint + 1))
            fi
        elif [[ "${status}" == "idle" || -z "${status}" ]]; then
            if [[ -n "${live_latest}" && -f "${live_latest}" ]]; then
                eligible_idle=$((eligible_idle + 1))
            fi
        fi
    fi
done < "${snapshot}"

echo "ledger_summary"
echo "  replicas_total=${total}"
echo "  elapsed_sweeps_sum=${ledger_elapsed_sum}"
echo "  statistics_sweeps_sum=${ledger_stats_sum}"
echo "  missing_latest_state_files=${missing_latest_state_files}"
echo
echo "checkpoint_aware_summary"
echo "  checkpoint_elapsed_sweeps_sum=${checkpoint_elapsed_sum}"
echo "  checkpoint_statistics_sweeps_sum=${checkpoint_stats_sum}"
echo "  running_total=${running_total}"
echo "  running_with_committed_checkpoint=${running_with_checkpoint}"
echo "  aggregation_eligible_idle=${eligible_idle}"
echo "  aggregation_eligible_with_running_checkpoints=$((eligible_idle + eligible_running_checkpoint))"
echo "  running_above_min_tstats_without_checkpoint=${eligible_running_without_checkpoint}"
echo
echo "phase_counts"
for key in "${!phase_counts[@]}"; do
    echo "  ${key}=${phase_counts[${key}]}"
done | sort
echo
echo "status_counts"
for key in "${!status_counts[@]}"; do
    echo "  ${key}=${status_counts[${key}]}"
done | sort

echo
echo "aggregate_current_files=$(find "${run_root}/aggregates/current" -maxdepth 1 -type f -name "*.jld2" 2>/dev/null | wc -l)"
latest_aggregate="$(find "${run_root}/aggregates/current" -maxdepth 1 -type f -name "*.jld2" -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 { $1=""; sub(/^ /, ""); print }')"
if [[ -n "${latest_aggregate}" ]]; then
    echo "latest_aggregate=${latest_aggregate}"
fi
