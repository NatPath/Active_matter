#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/inspect_file_usage.sh [options]

Options:
  --root <path>          Root to inspect. Repeatable. If omitted, inspect known repo/cluster roots.
  --runs_root <path>     Override the project runs root used for repo-specific cleanup hints.
  --depth <int>          Depth for shallow inode summary via du --inodes (default: 2)
  --top <int>            Show at most this many rows per section (default: 20)
  --no_targeted          Skip repo-specific candidate scans under runs/
  -h, --help             Show help

Behavior:
  - Read-only cluster-local inspection. No SSH, no deletion.
  - Prefers shallow inode summaries from du --inodes to keep the first pass cheap.
  - Then scans repo-specific directories that commonly create many small files:
      * two_force_d add_repeats DAG scaffolding
      * active_objects per-run histograms and manual rebuild jobs
      * SSEP manual aggregate jobs and archived aggregates
      * raw top-up / repeat-batch state directories
  - "derived" means the directory is usually rebuildable from raw states + metadata.
  - "raw" means the directory usually contains irreducible saved states; delete only
    after confirming the latest aggregate/histogram already contains what you need.
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

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

declare -a requested_roots=()
runs_root_override=""
depth="2"
top="20"
no_targeted="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            requested_roots+=("${2:-}")
            shift 2
            ;;
        --runs_root)
            runs_root_override="${2:-}"
            shift 2
            ;;
        --depth)
            depth="${2:-}"
            shift 2
            ;;
        --top)
            top="${2:-}"
            shift 2
            ;;
        --no_targeted)
            no_targeted="true"
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

if ! [[ "${depth}" =~ ^[0-9]+$ ]]; then
    echo "--depth must be a non-negative integer. Got '${depth}'."
    exit 1
fi
if ! [[ "${top}" =~ ^[0-9]+$ ]] || (( top <= 0 )); then
    echo "--top must be a positive integer. Got '${top}'."
    exit 1
fi

declare -a roots=()

add_root() {
    local candidate="$1"
    local canonical existing
    [[ -n "${candidate}" && -d "${candidate}" ]] || return 0
    canonical="$(cd "${candidate}" && pwd)"
    for existing in "${roots[@]}"; do
        [[ "${existing}" == "${canonical}" ]] && return 0
    done
    roots+=("${canonical}")
}

if (( ${#requested_roots[@]} > 0 )); then
    for root in "${requested_roots[@]}"; do
        add_root "${root}"
    done
else
    add_root "${REPO_ROOT}/runs"
    if [[ -n "${CLUSTER_DATA_ROOT:-}" ]]; then
        add_root "${CLUSTER_DATA_ROOT}"
    fi
    if [[ -n "${CLUSTER_CODE_ROOT:-}" ]]; then
        if [[ -d "${CLUSTER_CODE_ROOT}/runs" ]]; then
            add_root "${CLUSTER_CODE_ROOT}/runs"
        else
            add_root "${CLUSTER_CODE_ROOT}"
        fi
    fi
fi

if (( ${#roots[@]} == 0 )); then
    echo "No readable roots were resolved."
    echo "Pass --root explicitly, or set CLUSTER_DATA_ROOT / CLUSTER_CODE_ROOT in cluster_env.sh."
    exit 1
fi

have_du_inodes="false"
if du --inodes -d 0 "${REPO_ROOT}" >/dev/null 2>&1; then
    have_du_inodes="true"
fi

print_shallow_inode_summary() {
    local root="$1"

    echo
    echo "Root: ${root}"
    if [[ "${have_du_inodes}" == "true" ]]; then
        echo "Shallow inode summary (depth=${depth}, higher means more files underneath):"
        du --inodes -x -d "${depth}" "${root}" 2>/dev/null \
            | sort -nr \
            | head -n "${top}" \
            | awk '
                {
                    count = $1
                    $1 = ""
                    sub(/^ /, "")
                    printf "  %8s  %s\n", count, $0
                }
            '
    else
        echo "du --inodes is not available on this machine; skipping shallow inode summary."
    fi
}

resolve_runs_root() {
    local inspected_root="$1"
    if [[ -n "${runs_root_override}" ]]; then
        if [[ -d "${runs_root_override}" ]]; then
            cd "${runs_root_override}" && pwd
        fi
        return 0
    fi
    if [[ -d "${inspected_root}/runs" ]]; then
        cd "${inspected_root}/runs" && pwd
        return 0
    fi
    if [[ "$(basename "${inspected_root}")" == "runs" ]]; then
        printf "%s\n" "${inspected_root}"
        return 0
    fi
    return 1
}

count_files_recursive() {
    local dir="$1"
    find "${dir}" -type f 2>/dev/null | wc -l | tr -d ' '
}

print_candidate_section() {
    local label="$1"
    local classification="$2"
    local hint="$3"
    local count_file="$4"
    [[ -s "${count_file}" ]] || return 0

    echo
    echo "[${classification}] ${label}"
    echo "  ${hint}"
    sort -nr "${count_file}" | head -n "${top}" | awk -F'\t' '{printf "  %8s  %s\n", $1, $2}'
}

collect_named_dirs() {
    local output_file="$1"
    shift
    local dir count
    : > "${output_file}"
    while IFS= read -r dir; do
        [[ -d "${dir}" ]] || continue
        count="$(count_files_recursive "${dir}")"
        [[ "${count}" =~ ^[0-9]+$ ]] || continue
        (( count > 0 )) || continue
        printf "%s\t%s\n" "${count}" "${dir}" >> "${output_file}"
    done
}

inspect_repo_specific_candidates() {
    local runs_root="$1"
    local tmp_dag tmp_submit tmp_manual_hist tmp_per_run tmp_manual_agg tmp_archive tmp_topup tmp_repeat

    echo
    echo "Repo-specific cleanup candidates under ${runs_root}:"

    tmp_dag="$(mktemp)"
    tmp_submit="$(mktemp)"
    tmp_manual_hist="$(mktemp)"
    tmp_per_run="$(mktemp)"
    tmp_manual_agg="$(mktemp)"
    tmp_archive="$(mktemp)"
    tmp_topup="$(mktemp)"
    tmp_repeat="$(mktemp)"
    trap 'rm -f "${tmp_dag}" "${tmp_submit}" "${tmp_manual_hist}" "${tmp_per_run}" "${tmp_manual_agg}" "${tmp_archive}" "${tmp_topup}" "${tmp_repeat}"' RETURN

    collect_named_dirs "${tmp_dag}" < <(
        find "${runs_root}/two_force_d/add_repeats_jobs" -mindepth 2 -maxdepth 2 -type d -name dag_snippets 2>/dev/null
    )
    print_candidate_section \
        "two_force_d add_repeats dag_snippets" \
        "derived" \
        "Pure DAG scaffolding. Usually safe to remove after the batch is finished and job_info/manifest are retained." \
        "${tmp_dag}"

    collect_named_dirs "${tmp_submit}" < <(
        find "${runs_root}" \
            \( -path "${runs_root}/two_force_d/add_repeats_jobs/*/submit" \
            -o -path "${runs_root}/two_force_d/*/*/submit" \
            -o -path "${runs_root}/active_objects/steady_state_histograms/*/submit" \
            -o -path "${runs_root}/ssep/single_center_bond/production/*/manual_aggregate_jobs/*/submit" \) \
            -type d 2>/dev/null
    )
    print_candidate_section \
        "submit directories" \
        "derived" \
        "Condor submit scaffolding. Usually safe after completion if run_info/manifest/job_info and the final outputs are kept." \
        "${tmp_submit}"

    collect_named_dirs "${tmp_manual_hist}" < <(
        find "${runs_root}/active_objects/steady_state_histograms" -mindepth 2 -maxdepth 2 -type d -name manual_histogram_jobs 2>/dev/null
    )
    print_candidate_section \
        "active_objects manual_histogram_jobs" \
        "derived" \
        "Rebuild-job bookkeeping. Keep recent diagnostics you care about; older completed jobs are usually disposable." \
        "${tmp_manual_hist}"

    collect_named_dirs "${tmp_per_run}" < <(
        find "${runs_root}/active_objects/steady_state_histograms" -mindepth 3 -maxdepth 3 -type d -path '*/histograms/per_run' 2>/dev/null
    )
    print_candidate_section \
        "active_objects histograms/per_run" \
        "derived" \
        "Per-state histogram artifacts. Rebuildable from saved states via submit_active_objects_saved_states_into_histograms.sh." \
        "${tmp_per_run}"

    collect_named_dirs "${tmp_manual_agg}" < <(
        find "${runs_root}/ssep/single_center_bond/production" -mindepth 2 -maxdepth 2 -type d -name manual_aggregate_jobs 2>/dev/null
    )
    print_candidate_section \
        "SSEP manual_aggregate_jobs" \
        "derived" \
        "Saved-state aggregation bookkeeping. Usually safe after you keep the latest live aggregate and any notes you need." \
        "${tmp_manual_agg}"

    collect_named_dirs "${tmp_archive}" < <(
        find "${runs_root}" \
            \( -path "${runs_root}/ssep/single_center_bond/production/*/aggregated/archive" \
            -o -path "${runs_root}/two_force_d/*/*/states/aggregated/archive" \) \
            -type d 2>/dev/null
    )
    print_candidate_section \
        "aggregate archives" \
        "derived" \
        "Superseded aggregate versions. Keep the current live aggregate; older archived versions are fallback copies, not primary knowledge." \
        "${tmp_archive}"

    collect_named_dirs "${tmp_topup}" < <(
        find "${runs_root}/active_objects/steady_state_histograms" -mindepth 3 -maxdepth 3 -type d -path '*/states/topup_batches' 2>/dev/null
    )
    print_candidate_section \
        "active_objects states/topup_batches" \
        "raw" \
        "Raw top-up states. Delete only after confirming the latest aggregated histograms already reflect them and you do not need per-state re-analysis." \
        "${tmp_topup}"

    collect_named_dirs "${tmp_repeat}" < <(
        find "${runs_root}/two_force_d" -mindepth 3 -maxdepth 4 -type d -path '*/states/repeat_batches' 2>/dev/null
    )
    print_candidate_section \
        "two_force_d states/repeat_batches" \
        "raw" \
        "Raw repeat-batch states. Delete only after confirming the latest aggregates under states/aggregated already include the batches you want to preserve." \
        "${tmp_repeat}"
}

echo "Read-only file usage inspection"
echo "  repo_root=${REPO_ROOT}"
echo "  roots=${#roots[@]}"
echo "  depth=${depth}"
echo "  top=${top}"

for root in "${roots[@]}"; do
    print_shallow_inode_summary "${root}"

    if [[ "${no_targeted}" == "true" ]]; then
        continue
    fi

    runs_root="$(resolve_runs_root "${root}" || true)"
    if [[ -n "${runs_root:-}" && -d "${runs_root}" ]]; then
        inspect_repo_specific_candidates "${runs_root}"
    fi
done
