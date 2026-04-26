#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/reclaim_managed_diffusive_1d_pmlr_claims.sh \
      --run_id <managed_run_id> (--batch_id <id> | --all_running) [options]

Options:
  --dry_run       Show which replicas would be reclaimed
  -h, --help      Show this help

This is an explicit recovery tool. Use it only after confirming the selected
Condor jobs are not still running. It clears stale running claims so the next
managed submit can use those replicas again.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_1d_pmlr_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"

run_id=""
batch_id=""
all_running="false"
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id) run_id="${2:-}"; shift 2 ;;
        --batch_id) batch_id="${2:-}"; shift 2 ;;
        --all_running) all_running="true"; shift 1 ;;
        --dry_run) dry_run="true"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${run_id}" ]]; then
    echo "Missing --run_id." >&2
    usage
    exit 1
fi
if [[ -z "${batch_id}" && "${all_running}" != "true" ]]; then
    echo "Provide --batch_id or --all_running." >&2
    usage
    exit 1
fi
if [[ -n "${batch_id}" && "${all_running}" == "true" ]]; then
    echo "Use either --batch_id or --all_running, not both." >&2
    exit 1
fi

run_id="$(managed_slugify "${run_id}")"
run_root="$(managed_run_root "${REPO_ROOT}" "${run_id}")"
replicas_csv="${run_root}/replicas.csv"
lock_file="${run_root}/manager.lock"
if [[ ! -f "${replicas_csv}" ]]; then
    echo "Managed replicas.csv not found: ${replicas_csv}" >&2
    exit 1
fi

timestamp="$(managed_timestamp)"
reclaim_list="${run_root}/claims/reclaim_${timestamp}.csv"
mkdir -p "$(dirname "${reclaim_list}")"
echo "replica_id,old_phase,new_phase,old_elapsed,new_elapsed,old_tstats,new_tstats,latest_state" > "${reclaim_list}"

(
    flock 9
    tmp_replicas="${replicas_csv}.tmp.$$"
    awk -F, -v OFS=',' -v batch_id="${batch_id}" -v all_running="${all_running}" '
        NR == 1 { print; next }
        {
            should = ($9 == "running") && (all_running == "true" || $8 == batch_id)
            if (should) {
                print $0 > "/dev/stderr"
            }
            print
        }
    ' "${replicas_csv}" > "${tmp_replicas}.pre" 2>"${tmp_replicas}.targets"

    if [[ ! -s "${tmp_replicas}.targets" ]]; then
        rm -f "${tmp_replicas}" "${tmp_replicas}.pre" "${tmp_replicas}.targets"
        echo "No matching running claims found."
        exit 0
    fi

    if [[ "${dry_run}" == "true" ]]; then
        echo "Would reclaim:"
        cat "${tmp_replicas}.targets"
        rm -f "${tmp_replicas}" "${tmp_replicas}.pre" "${tmp_replicas}.targets"
        exit 0
    fi

    updates_file="${tmp_replicas}.updates"
    : > "${updates_file}"
    while IFS=, read -r rid old_phase old_elapsed old_stats old_latest _source _source_tag _claim _status _updated; do
        meta_path="${run_root}/replicas/${rid}/current.meta"
        new_phase="${old_phase}"
        new_elapsed="${old_elapsed}"
        new_stats="${old_stats}"
        new_latest="${old_latest}"

        if [[ -f "${meta_path}" ]]; then
            meta_phase="$(awk 'index($0, "phase=") == 1 { print substr($0, 7); exit }' "${meta_path}")"
            meta_elapsed="$(awk 'index($0, "elapsed_sweeps=") == 1 { print substr($0, 16); exit }' "${meta_path}")"
            meta_stats="$(awk 'index($0, "statistics_sweeps=") == 1 { print substr($0, 19); exit }' "${meta_path}")"
            meta_latest="$(awk 'index($0, "latest_state=") == 1 { print substr($0, 14); exit }' "${meta_path}")"
            [[ -n "${meta_phase}" ]] && new_phase="${meta_phase}"
            [[ -n "${meta_elapsed}" ]] && new_elapsed="${meta_elapsed}"
            [[ -n "${meta_stats}" ]] && new_stats="${meta_stats}"
            [[ -n "${meta_latest}" ]] && new_latest="${meta_latest}"
        fi

        printf "%s,%s,%s,%s,%s\n" "${rid}" "${new_phase}" "${new_elapsed}" "${new_stats}" "${new_latest}" >> "${updates_file}"
        printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${rid}" "${old_phase}" "${new_phase}" "${old_elapsed}" "${new_elapsed}" \
            "${old_stats}" "${new_stats}" "${new_latest}" >> "${reclaim_list}"
    done < "${tmp_replicas}.targets"

    awk -F, -v OFS=',' -v batch_id="${batch_id}" -v all_running="${all_running}" -v timestamp="${timestamp}" -v updates_file="${updates_file}" '
        BEGIN {
            while ((getline line < updates_file) > 0) {
                split(line, parts, ",")
                phase[parts[1]] = parts[2]
                elapsed[parts[1]] = parts[3]
                stats[parts[1]] = parts[4]
                latest[parts[1]] = parts[5]
            }
            close(updates_file)
        }
        NR == 1 { print; next }
        {
            should = ($9 == "running") && (all_running == "true" || $8 == batch_id)
            if (should) {
                if ($1 in phase) $2=phase[$1]
                if ($1 in elapsed) $3=elapsed[$1]
                if ($1 in stats) $4=stats[$1]
                if (($1 in latest) && latest[$1] != "") $5=latest[$1]
                $8=""; $9="idle"; $10=timestamp
            }
            print
        }
    ' "${replicas_csv}" > "${tmp_replicas}"
    mv -f "${tmp_replicas}" "${replicas_csv}"
    rm -f "${tmp_replicas}.pre" "${tmp_replicas}.targets" "${updates_file}"
) 9>"${lock_file}"

echo "Reclaim completed."
echo "  run_id=${run_id}"
echo "  batch_id=${batch_id:-ALL_RUNNING}"
echo "  dry_run=${dry_run}"
