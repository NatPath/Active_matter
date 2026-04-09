#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage:
  bash copy_data_from_cluster.sh [options]

Core options:
  --mode <warmup|production>              Run mode for --list/--latest (for --run_id, registry mode is used)
  --run_id <id>                           Exact run_id to fetch (searched in both registry roots)
  --latest                                Fetch latest run_id in selected run family (can combine with filters)
  --list                                  List run registry entries and exit
  --tail <N>                              Number of listed entries (default: 30)
  --run_family <two_force_d|single_origin_bond|ssep|active_objects|auto>
                                          Registry family for --list/--latest (default: two_force_d)

Run filters for --list / --latest:
  --L <int>                               Match L column
  --rho <value>                           Match rho0 column
  --n_sweeps <int>                        Match n_sweeps column
  --d_min <int>                           Match d_min column (two_force_d only)
  --d_max <int>                           Match d_max column (two_force_d only)
  --d_step <int>                          Match d_step column (two_force_d only)

Paths / connection:
  --remote_user <name>                    SSH user (default: CLUSTER_REMOTE_USER)
  --remote_host <host>                    SSH host (default: CLUSTER_REMOTE_HOST)
  --remote_root <path>                    Primary cluster root (default: CLUSTER_DATA_ROOT)
  --secondary_remote_root <path>          Secondary cluster root checked for --run_id
                                          (default: CLUSTER_CODE_ROOT)
  --local_root <path>                     Local root for downloaded data (default: <repo>/cluster_results)
  --sync_scope <auto|aggregation|full>    Data sync scope:
                                          auto (default): if run_id looks aggregated (_nr...), fetch aggregated artifacts only
                                          aggregation: fetch aggregated state files + run metadata
                                          (prefers aggregated/, then states/aggregated, with legacy fallback;
                                          excludes aggregated/archive and states/aggregated/archive)
                                          full: fetch full run directory

Post-processing:
  --plot                                  Run load_and_plot.jl after sync
  --aggregated_saved_only                 Restrict copy/plot to live *_id-aggregated_saved_*.jld2 only
                                          under aggregated/ (or states/aggregated/ / states/<state_subdir>/ when provided)
                                          and skip legacy fallback paths
  --state_subdir <name>                   Custom subdir under states/ for aggregation sync/plot
                                          (example: debug_new_raw_post_t8e9_20260325)
  --state_glob <pattern>                  File glob for state selection in aggregation sync/plot
                                          (default aggregation glob: *id-aggregated_*.jld2)
  --sample_count <N>                      For aggregation sync with --state_subdir, fetch only the latest N
                                          matching files from that remote subdir
  --skip_per_state_sweep                  Pass --skip_per_state_sweep to load_and_plot.jl
  --baseline_j2 <value>                   Baseline override for two_force_d analysis
                                          (default when omitted: automatic rho0^2-based baseline from load_and_plot.jl)
  --mode_plot <single|two_force_d>        load_and_plot mode; auto-selected from run family when omitted

Examples:
  bash copy_data_from_cluster.sh --list --run_family two_force_d --mode warmup
  bash copy_data_from_cluster.sh --latest --run_family single_origin_bond --mode production --L 128 --rho 100 --n_sweeps 1000000
  bash copy_data_from_cluster.sh --run_id single_production_L128_rho100_ns1000000_f1.0_ffr1.0_20260225-152138 --plot
  bash copy_data_from_cluster.sh --run_id two_force_production_L128_rho100_ns1000000_d2-32-s2_nr8_dag_20260225-180000 --sync_scope aggregation
  bash copy_data_from_cluster.sh --run_id ssep_ctmc_single_center_bond_L256_rho05_ns500000000_nr600_dag_20260328-120000
  bash copy_data_from_cluster.sh --run_id active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k5e-5_nr10_hist_20260331-120000 --plot
  bash copy_data_from_cluster.sh --run_id two_force_warmup_production_L256_rho100_..._production_20260226-032016 --aggregated_saved_only --plot

Notes:
  - If present, cluster_scripts/cluster_env.sh is sourced automatically for local defaults.
USAGE_EOF
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

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

mode="warmup"
mode_explicit="false"
run_id=""
latest="false"
list_only="false"
tail_n="30"
run_family="two_force_d"

filter_L=""
filter_rho=""
filter_n_sweeps=""
filter_d_min=""
filter_d_max=""
filter_d_step=""

remote_user="${REMOTE_USER:-${CLUSTER_REMOTE_USER:-}}"
remote_host="${REMOTE_HOST:-${CLUSTER_REMOTE_HOST:-}}"
remote_root="${REMOTE_ROOT:-${CLUSTER_DATA_ROOT:-}}"
secondary_remote_root="${SECONDARY_REMOTE_ROOT:-${CLUSTER_CODE_ROOT:-}}"
local_root="${LOCAL_RESULTS_ROOT:-${REPO_ROOT}/cluster_results}"
sync_scope="auto"

plot_after_sync="false"
aggregated_saved_only="false"
state_subdir=""
state_glob=""
state_glob_explicit="false"
skip_per_state="false"
baseline_j2=""
plot_mode="two_force_d"
plot_mode_explicit="false"
sample_count="0"

ssh_control_dir=""
ssh_control_path=""

declare -A REGISTRY_REL_MAP
declare -A RUN_ROOT_REL_MAP
REGISTRY_REL_MAP["two_force_d"]="runs/two_force_d/run_registry.csv"
REGISTRY_REL_MAP["single_origin_bond"]="runs/single_origin_bond/run_registry.csv"
REGISTRY_REL_MAP["ssep"]="runs/ssep/single_center_bond/run_registry.csv"
REGISTRY_REL_MAP["active_objects"]="runs/active_objects/steady_state_histograms/run_registry.csv"
RUN_ROOT_REL_MAP["two_force_d"]="runs/two_force_d"
RUN_ROOT_REL_MAP["single_origin_bond"]="runs/single_origin_bond"
RUN_ROOT_REL_MAP["ssep"]="runs/ssep/single_center_bond"
RUN_ROOT_REL_MAP["active_objects"]="runs/active_objects/steady_state_histograms"
ALL_FAMILIES=("two_force_d" "single_origin_bond" "ssep" "active_objects")

cleanup_ssh_master() {
    if [[ -n "${ssh_control_path}" ]]; then
        ssh -o ControlPath="${ssh_control_path}" -O exit "${remote_user}@${remote_host}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${ssh_control_dir}" && -d "${ssh_control_dir}" ]]; then
        rm -rf "${ssh_control_dir}"
    fi
}

open_ssh_master() {
    ssh_control_dir="$(mktemp -d /tmp/copy_from_cluster_XXXXXX)"
    ssh_control_path="${ssh_control_dir}/control.sock"
    ssh -MNf \
        -o ControlMaster=yes \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "${remote_user}@${remote_host}"
}

scp_with_master() {
    scp \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "$@"
}

rsync_with_master() {
    rsync \
        -e "ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=${ssh_control_path}" \
        "$@"
}

ssh_with_master() {
    ssh \
        -o ControlMaster=auto \
        -o ControlPersist=600 \
        -o ControlPath="${ssh_control_path}" \
        "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            mode="${2:-}"
            mode_explicit="true"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --latest)
            latest="true"
            shift 1
            ;;
        --list)
            list_only="true"
            shift 1
            ;;
        --tail)
            tail_n="${2:-}"
            shift 2
            ;;
        --run_family)
            run_family="${2:-}"
            shift 2
            ;;
        --L)
            filter_L="${2:-}"
            shift 2
            ;;
        --rho)
            filter_rho="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            filter_n_sweeps="${2:-}"
            shift 2
            ;;
        --d_min)
            filter_d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            filter_d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            filter_d_step="${2:-}"
            shift 2
            ;;
        --remote_user)
            remote_user="${2:-}"
            shift 2
            ;;
        --remote_host)
            remote_host="${2:-}"
            shift 2
            ;;
        --remote_root)
            remote_root="${2:-}"
            shift 2
            ;;
        --secondary_remote_root)
            secondary_remote_root="${2:-}"
            shift 2
            ;;
        --local_root)
            local_root="${2:-}"
            shift 2
            ;;
        --sync_scope)
            sync_scope="${2:-}"
            shift 2
            ;;
        --plot)
            plot_after_sync="true"
            shift 1
            ;;
        --aggregated_saved_only)
            aggregated_saved_only="true"
            shift 1
            ;;
        --state_subdir)
            state_subdir="${2:-}"
            shift 2
            ;;
        --state_glob)
            state_glob="${2:-}"
            state_glob_explicit="true"
            shift 2
            ;;
        --sample_count)
            sample_count="${2:-}"
            shift 2
            ;;
        --skip_per_state_sweep)
            skip_per_state="true"
            shift 1
            ;;
        --baseline_j2)
            baseline_j2="${2:-}"
            shift 2
            ;;
        --mode_plot)
            plot_mode="${2:-}"
            plot_mode_explicit="true"
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

if [[ "${mode}" != "warmup" && "${mode}" != "production" ]]; then
    echo "--mode must be warmup or production. Got '${mode}'."
    exit 1
fi
if ! [[ "${tail_n}" =~ ^[0-9]+$ ]] || (( tail_n <= 0 )); then
    echo "--tail must be a positive integer. Got '${tail_n}'."
    exit 1
fi
if [[ "${run_family}" != "two_force_d" && "${run_family}" != "single_origin_bond" && "${run_family}" != "ssep" && "${run_family}" != "active_objects" && "${run_family}" != "auto" ]]; then
    echo "--run_family must be two_force_d, single_origin_bond, ssep, active_objects, or auto. Got '${run_family}'."
    exit 1
fi
if [[ "${sync_scope}" != "auto" && "${sync_scope}" != "aggregation" && "${sync_scope}" != "full" ]]; then
    echo "--sync_scope must be auto, aggregation, or full. Got '${sync_scope}'."
    exit 1
fi
if [[ "${state_glob_explicit}" == "true" && -z "${state_glob}" ]]; then
    echo "--state_glob cannot be empty."
    exit 1
fi
if [[ -n "${state_subdir}" ]]; then
    if [[ "${state_subdir}" == /* || "${state_subdir}" == *".."* ]]; then
        echo "--state_subdir must be a relative subdir under states/. Got '${state_subdir}'."
        exit 1
    fi
fi
if ! [[ "${sample_count}" =~ ^[0-9]+$ ]]; then
    echo "--sample_count must be a non-negative integer. Got '${sample_count}'."
    exit 1
fi
if (( sample_count > 0 )); then
    if [[ "${sync_scope}" == "full" ]]; then
        echo "--sample_count is supported only with aggregation sync."
        exit 1
    fi
    if [[ -z "${state_subdir}" ]]; then
        echo "--sample_count requires --state_subdir so the remote sample set is well-defined."
        exit 1
    fi
fi
if [[ "${plot_mode_explicit}" == "true" && "${plot_mode}" != "single" && "${plot_mode}" != "two_force_d" ]]; then
    echo "--mode_plot must be single or two_force_d. Got '${plot_mode}'."
    exit 1
fi
if [[ -n "${baseline_j2}" && ! "${baseline_j2}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; then
    echo "--baseline_j2 must be numeric when provided. Got '${baseline_j2}'."
    exit 1
fi
if [[ "${aggregated_saved_only}" == "true" ]]; then
    sync_scope="aggregation"
    if [[ "${state_glob_explicit}" != "true" ]]; then
        state_glob="*id-aggregated_saved_*.jld2"
    fi
fi

missing_settings=()
[[ -n "${remote_user}" ]] || missing_settings+=("--remote_user or CLUSTER_REMOTE_USER")
[[ -n "${remote_host}" ]] || missing_settings+=("--remote_host or CLUSTER_REMOTE_HOST")
[[ -n "${remote_root}" ]] || missing_settings+=("--remote_root or CLUSTER_DATA_ROOT")
if (( ${#missing_settings[@]} > 0 )); then
    echo "Missing cluster connection settings:"
    printf '  %s\n' "${missing_settings[@]}"
    echo "Set them via flags, environment variables, or cluster_scripts/cluster_env.sh."
    echo "Use cluster_scripts/cluster_env.example.sh as a template."
    exit 1
fi

trap cleanup_ssh_master EXIT
open_ssh_master

sync_registry() {
    local family="$1"
    local remote_root_base="$2"
    local cache_label="$3"
    local registry_rel="${REGISTRY_REL_MAP[$family]}"
    local local_registry="${local_root}/.registry_cache/${cache_label}/${registry_rel}"

    mkdir -p "$(dirname "${local_registry}")"
    echo "Syncing ${family} run registry from cluster root ${remote_root_base}..." >&2
    if scp_with_master "${remote_user}@${remote_host}:${remote_root_base}/${registry_rel}" "${local_registry}"; then
        echo "${local_registry}"
        return 0
    fi

    if [[ -f "${local_registry}" ]]; then
        echo "WARNING: failed to download ${registry_rel}; using existing local copy at ${local_registry}" >&2
        echo "${local_registry}"
        return 0
    fi

    echo "WARNING: failed to download ${registry_rel} and no local copy exists" >&2
    return 1
}

find_row_by_run_id() {
    local registry_path="$1"
    local lookup_run_id="$2"
    awk -F, -v rid="${lookup_run_id}" '
        NR == 1 {next}
        $2 == rid {row = $0}
        END {print row}
    ' "${registry_path}"
}

latest_run_id_for_family() {
    local family="$1"
    local registry_path="$2"

    if [[ "${family}" == "two_force_d" ]]; then
        awk -F, \
            -v mode="${mode}" \
            -v fL="${filter_L}" \
            -v fRho="${filter_rho}" \
            -v fSweeps="${filter_n_sweeps}" \
            -v fDmin="${filter_d_min}" \
            -v fDmax="${filter_d_max}" \
            -v fDstep="${filter_d_step}" '
            NR==1 {next}
            {
                keep = 1
                if (mode != "" && $3 != mode) keep = 0
                if (fL != "" && $4 != fL) keep = 0
                if (fRho != "" && $5 != fRho) keep = 0
                if (fSweeps != "" && $6 != fSweeps) keep = 0
                if (fDmin != "" && $7 != fDmin) keep = 0
                if (fDmax != "" && $8 != fDmax) keep = 0
                if (fDstep != "" && $9 != fDstep) keep = 0
                if (keep) chosen = $2
            }
            END {print chosen}
        ' "${registry_path}"
    else
        awk -F, \
            -v mode="${mode}" \
            -v fL="${filter_L}" \
            -v fRho="${filter_rho}" \
            -v fSweeps="${filter_n_sweeps}" '
            NR==1 {next}
            {
                keep = 1
                if (mode != "" && $3 != mode) keep = 0
                if (fL != "" && $4 != fL) keep = 0
                if (fRho != "" && $5 != fRho) keep = 0
                if (fSweeps != "" && $6 != fSweeps) keep = 0
                if (keep) chosen = $2
            }
            END {print chosen}
        ' "${registry_path}"
    fi
}

prune_local_aggregation_matches() {
    local run_dir="$1"
    local file_glob="$2"
    local pruned_count=0
    local dir file

    local search_dirs=("${run_dir}/states/aggregated" "${run_dir}/states")
    if [[ "${aggregated_saved_only}" == "true" ]]; then
        if [[ -n "${state_subdir}" ]]; then
            search_dirs=("${run_dir}/states/${state_subdir}")
        else
            search_dirs=("${run_dir}/aggregated" "${run_dir}/states/aggregated")
        fi
    elif [[ -n "${state_subdir}" ]]; then
        search_dirs=("${run_dir}/states/${state_subdir}" "${run_dir}/aggregated" "${run_dir}/states/aggregated" "${run_dir}/states")
    else
        search_dirs=("${run_dir}/aggregated" "${run_dir}/states/aggregated" "${run_dir}/states")
    fi

    for dir in "${search_dirs[@]}"; do
        [[ -d "${dir}" ]] || continue
        while IFS= read -r -d '' file; do
            rm -f "${file}"
            pruned_count=$((pruned_count + 1))
        done < <(
            find "${dir}" -maxdepth 1 -type f \
                -name "${file_glob}" \
                ! -path '*/aggregated/archive/*' \
                -print0 2>/dev/null
        )
    done

    if (( pruned_count > 0 )); then
        echo "Pruned ${pruned_count} stale local cached state(s) matching ${file_glob} before sync."
    fi
}

select_latest_two_force_d_files() {
    declare -A best_file=()
    declare -A best_t=()
    declare -A best_mtime=()
    local file base d_val t_val mtime_val current_t current_mtime

    for file in "$@"; do
        base="$(basename "${file}")"
        if [[ ! "${base}" =~ two_force_d([0-9]+)_ ]]; then
            printf "%s\n" "${file}"
            continue
        fi

        d_val="${BASH_REMATCH[1]}"
        if [[ "${base}" =~ _t(-?[0-9]+)_id- ]]; then
            t_val="${BASH_REMATCH[1]}"
        else
            t_val="-9223372036854775808"
        fi
        mtime_val="$(stat -c %Y "${file}" 2>/dev/null || echo 0)"

        if [[ -z "${best_file[$d_val]+x}" ]]; then
            best_file["${d_val}"]="${file}"
            best_t["${d_val}"]="${t_val}"
            best_mtime["${d_val}"]="${mtime_val}"
            continue
        fi

        current_t="${best_t[$d_val]}"
        current_mtime="${best_mtime[$d_val]}"
        if (( t_val > current_t )) || { (( t_val == current_t )) && (( mtime_val >= current_mtime )); }; then
            best_file["${d_val}"]="${file}"
            best_t["${d_val}"]="${t_val}"
            best_mtime["${d_val}"]="${mtime_val}"
        fi
    done

    if (( ${#best_file[@]} == 0 )); then
        return 0
    fi

    while IFS= read -r d_val; do
        printf "%s\n" "${best_file[$d_val]}"
    done < <(printf "%s\n" "${!best_file[@]}" | sort -n)
}

write_filtered_registry() {
    local family="$1"
    local registry_path="$2"
    local out_path="$3"

    if [[ "${family}" == "two_force_d" ]]; then
        awk -F, \
            -v mode="${mode}" \
            -v fL="${filter_L}" \
            -v fRho="${filter_rho}" \
            -v fSweeps="${filter_n_sweeps}" \
            -v fDmin="${filter_d_min}" \
            -v fDmax="${filter_d_max}" \
            -v fDstep="${filter_d_step}" '
            NR==1 {print; next}
            {
                keep = 1
                if (mode != "" && $3 != mode) keep = 0
                if (fL != "" && $4 != fL) keep = 0
                if (fRho != "" && $5 != fRho) keep = 0
                if (fSweeps != "" && $6 != fSweeps) keep = 0
                if (fDmin != "" && $7 != fDmin) keep = 0
                if (fDmax != "" && $8 != fDmax) keep = 0
                if (fDstep != "" && $9 != fDstep) keep = 0
                if (keep) print
            }
        ' "${registry_path}" > "${out_path}"
    else
        awk -F, \
            -v mode="${mode}" \
            -v fL="${filter_L}" \
            -v fRho="${filter_rho}" \
            -v fSweeps="${filter_n_sweeps}" '
            NR==1 {print; next}
            {
                keep = 1
                if (mode != "" && $3 != mode) keep = 0
                if (fL != "" && $4 != fL) keep = 0
                if (fRho != "" && $5 != fRho) keep = 0
                if (fSweeps != "" && $6 != fSweeps) keep = 0
                if (keep) print
            }
        ' "${registry_path}" > "${out_path}"
    fi
}

resolved_family=""
resolved_registry=""
registry_row=""
resolved_remote_root=""

declare -a SEARCH_REMOTE_ROOTS=()
SEARCH_REMOTE_ROOTS+=("${remote_root}")
if [[ -n "${secondary_remote_root}" && "${secondary_remote_root}" != "${remote_root}" ]]; then
    SEARCH_REMOTE_ROOTS+=("${secondary_remote_root}")
fi

if [[ -z "${run_id}" ]]; then
    if [[ "${latest}" != "true" && "${list_only}" != "true" ]]; then
        echo "Provide --run_id <id> or use --latest or --list."
        exit 1
    fi

    selected_family="${run_family}"
    if [[ "${selected_family}" == "auto" ]]; then
        selected_family="two_force_d"
    fi

    if [[ "${selected_family}" != "two_force_d" ]]; then
        if [[ -n "${filter_d_min}" || -n "${filter_d_max}" || -n "${filter_d_step}" ]]; then
            echo "WARNING: --d_min/--d_max/--d_step filters are ignored for ${selected_family}." >&2
        fi
    fi

    if ! resolved_registry="$(sync_registry "${selected_family}" "${remote_root}" "primary")"; then
        echo "Could not access registry for run family '${selected_family}'."
        exit 1
    fi
    resolved_remote_root="${remote_root}"

    if [[ "${list_only}" == "true" ]]; then
        filtered="$(mktemp)"
        write_filtered_registry "${selected_family}" "${resolved_registry}" "${filtered}"

        echo "Registry family: ${selected_family}"
        echo "Registry: ${resolved_registry}"
        echo "Showing last ${tail_n} matching entries:"
        if command -v column >/dev/null 2>&1; then
            tail -n "${tail_n}" "${filtered}" | column -s, -t
        else
            tail -n "${tail_n}" "${filtered}"
        fi
        rm -f "${filtered}"
        exit 0
    fi

    run_id="$(latest_run_id_for_family "${selected_family}" "${resolved_registry}")"
    if [[ -z "${run_id}" ]]; then
        echo "No run matches the requested filters in '${selected_family}'."
        exit 1
    fi

    resolved_family="${selected_family}"
    registry_row="$(find_row_by_run_id "${resolved_registry}" "${run_id}")"
else
    found_count=0
    found_families=()
    found_registries=()
    found_roots=()

    for root_idx in "${!SEARCH_REMOTE_ROOTS[@]}"; do
        registry_root="${SEARCH_REMOTE_ROOTS[$root_idx]}"
        cache_label="root$((root_idx + 1))"
        for family in "${ALL_FAMILIES[@]}"; do
            if registry_path="$(sync_registry "${family}" "${registry_root}" "${cache_label}")"; then
                row="$(find_row_by_run_id "${registry_path}" "${run_id}")"
                if [[ -n "${row}" ]]; then
                    if (( found_count > 0 )) && [[ "${family}" == "${resolved_family}" ]]; then
                        continue
                    fi
                    found_count=$((found_count + 1))
                    found_families+=("${family}")
                    found_registries+=("${registry_path}")
                    found_roots+=("${registry_root}")
                    resolved_family="${family}"
                    resolved_registry="${registry_path}"
                    resolved_remote_root="${registry_root}"
                    registry_row="${row}"
                fi
            fi
        done
    done

    if (( found_count == 0 )); then
        echo "run_id '${run_id}' not found in available registries under:"
        for registry_root in "${SEARCH_REMOTE_ROOTS[@]}"; do
            echo "  ${registry_root}"
        done
        exit 1
    fi

    if (( found_count > 1 )); then
        echo "run_id '${run_id}' was found in multiple registries, cannot disambiguate automatically:"
        for idx in "${!found_families[@]}"; do
            echo "  family=${found_families[$idx]} root=${found_roots[$idx]} registry=${found_registries[$idx]}"
        done
        echo "Use a unique run_id or run with --run_family plus --latest/--list to select a specific registry."
        exit 1
    fi
fi

if [[ -z "${registry_row}" || -z "${resolved_family}" ]]; then
    echo "Internal error: failed to resolve run metadata for run_id='${run_id}'."
    exit 1
fi

if [[ "${resolved_family}" == "two_force_d" ]]; then
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_dmin reg_dmax reg_dstep reg_cpus reg_mem reg_run_root reg_log_dir reg_state_dir reg_warmup_state_dir <<< "${registry_row}"
elif [[ "${resolved_family}" == "ssep" ]]; then
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_warmup_sweeps reg_num_replicas reg_cpus reg_mem reg_run_root reg_submit_dir reg_log_dir reg_state_dir reg_config_path reg_aggregate_run_id <<< "${registry_row}"
elif [[ "${resolved_family}" == "active_objects" ]]; then
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_warmup_sweeps reg_num_replicas reg_cpus reg_mem reg_run_root reg_submit_dir reg_log_dir reg_state_dir reg_histogram_dir reg_config_path reg_aggregate_run_id <<< "${registry_row}"
else
    IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_cpus reg_mem reg_run_root reg_log_dir reg_state_dir reg_warmup_state_dir reg_ffr reg_force_strength <<< "${registry_row}"
fi

if [[ "${mode_explicit}" == "true" && "${mode}" != "${reg_mode}" ]]; then
    echo "WARNING: --mode='${mode}' does not match registry mode='${reg_mode}' for run_id='${run_id}'. Using registry mode."
fi
mode="${reg_mode}"

if [[ -n "${reg_run_root}" ]]; then
    remote_run_dir="${reg_run_root}"
else
    remote_run_dir="${resolved_remote_root}/${RUN_ROOT_REL_MAP[$resolved_family]}/${mode}/${run_id}"
fi

local_run_dir="${local_root}/${RUN_ROOT_REL_MAP[$resolved_family]}/${mode}/${run_id}"
mkdir -p "${local_run_dir}"

echo "Fetching run:"
echo "  run_id=${run_id}"
echo "  family=${resolved_family}"
echo "  from=${remote_user}@${remote_host}:${remote_run_dir}"
echo "  to=${local_run_dir}"

sync_scope_effective="${sync_scope}"
if [[ "${sync_scope_effective}" == "auto" ]]; then
    if [[ "${run_id}" =~ _nr[0-9]+(_|$) ]]; then
        sync_scope_effective="aggregation"
    else
        sync_scope_effective="full"
    fi
fi
echo "  sync_scope=${sync_scope_effective}"
if [[ -n "${state_subdir}" ]]; then
    echo "  state_subdir=${state_subdir}"
fi

aggregation_glob="*id-aggregated_*.jld2"
if [[ "${state_glob_explicit}" == "true" ]]; then
    aggregation_glob="${state_glob}"
elif [[ "${resolved_family}" == "active_objects" && -n "${reg_aggregate_run_id:-}" ]]; then
    aggregation_glob="${reg_aggregate_run_id}*_steady_state_hist.jld2"
elif [[ "${resolved_family}" == "ssep" && -n "${reg_aggregate_run_id:-}" ]]; then
    aggregation_glob="*id-${reg_aggregate_run_id}.jld2"
elif [[ -n "${state_glob}" ]]; then
    aggregation_glob="${state_glob}"
fi
if [[ "${sync_scope_effective}" == "aggregation" ]]; then
    echo "  aggregation_state_glob=${aggregation_glob}"
fi
if (( sample_count > 0 )); then
    echo "  sample_count=${sample_count}"
fi

if [[ "${sync_scope_effective}" == "aggregation" && "${aggregated_saved_only}" == "true" && "${resolved_family}" != "active_objects" ]]; then
    prune_local_aggregation_matches "${local_run_dir}" "${aggregation_glob}"
fi

sync_latest_ssep_aggregate_only="false"
if [[ "${sync_scope_effective}" == "aggregation" && "${resolved_family}" == "ssep" && -z "${state_subdir}" && "${aggregated_saved_only}" != "true" ]]; then
    sync_latest_ssep_aggregate_only="true"
fi

if [[ "${sync_scope_effective}" == "aggregation" ]]; then
    if [[ "${sync_latest_ssep_aggregate_only}" == "true" ]]; then
        remote_latest_cmd=$(
            cat <<EOF
cd $(printf '%q' "${remote_run_dir}") && for dir in aggregated states/aggregated states; do if [[ -d "\${dir}" ]]; then find "\${dir}" -maxdepth 1 -type f -name $(printf '%q' "${aggregation_glob}") ! -path '*/archive/*' -printf '%T@ %p\n'; fi; done | sort -nr | head -n 1
EOF
        )
        latest_remote_rel_path="$(
            ssh_with_master "${remote_user}@${remote_host}" "${remote_latest_cmd}" | awk '{ $1=\"\"; sub(/^ /,\"\"); print }' | tail -n 1
        )"
        if [[ -z "${latest_remote_rel_path}" ]]; then
            echo "No remote aggregate matched ${aggregation_glob} under aggregated/ or states/; nothing to sync."
        else
            files_from="$(mktemp)"
            {
                echo "run_info.txt"
                echo "manifest.csv"
                echo "${latest_remote_rel_path}"
            } > "${files_from}"
            rsync_with_master -av --progress --files-from="${files_from}" \
                "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"
            rm -f "${files_from}"
        fi
    elif (( sample_count > 0 )); then
        remote_state_dir="${remote_run_dir}/states/${state_subdir}"
        remote_list_cmd=$(
            cat <<EOF
cd $(printf '%q' "${remote_state_dir}") && find . -maxdepth 1 -type f -name $(printf '%q' "${aggregation_glob}") -printf '%T@ %f\n' | sort -nr | head -n $(printf '%q' "${sample_count}")
EOF
        )
        mapfile -t sampled_remote_entries < <(
            ssh_with_master "${remote_user}@${remote_host}" "${remote_list_cmd}" | awk '{ $1=""; sub(/^ /,""); print }'
        )
        if (( ${#sampled_remote_entries[@]} == 0 )); then
            echo "No remote files matched ${aggregation_glob} under states/${state_subdir}; nothing to sync."
        else
            files_from="$(mktemp)"
            {
                echo "run_info.txt"
                for sampled_file in "${sampled_remote_entries[@]}"; do
                    [[ -n "${sampled_file}" ]] || continue
                    echo "states/${state_subdir}/${sampled_file}"
                done
            } > "${files_from}"
            rsync_with_master -av --progress --files-from="${files_from}" \
                "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"
            rm -f "${files_from}"
        fi
    else
        rsync_args=(-av --progress --prune-empty-dirs
            --exclude='aggregated/archive/***'
            --exclude='states/aggregated/archive/***'
            --include='*/'
            --include='run_info.txt'
            --include='manifest.csv'
            --include='reports/***')
        if [[ "${resolved_family}" == "active_objects" ]]; then
            rsync_args+=(
                --include='histograms/'
                --include='histograms/aggregated/'
                --include="histograms/aggregated/${aggregation_glob}"
            )
        elif [[ -n "${state_subdir}" ]]; then
            rsync_args+=(--include="states/${state_subdir}/${aggregation_glob}")
        elif [[ "${aggregated_saved_only}" == "true" ]]; then
            rsync_args+=(--include="aggregated/${aggregation_glob}" --include="states/aggregated/${aggregation_glob}")
        elif [[ "${resolved_family}" == "ssep" ]]; then
            rsync_args+=(--include="aggregated/${aggregation_glob}")
        fi
        if [[ "${aggregated_saved_only}" != "true" && "${resolved_family}" != "active_objects" ]]; then
            rsync_args+=(
                --include="aggregated/${aggregation_glob}"
                --include="states/aggregated/${aggregation_glob}"
                --include="states/${aggregation_glob}"
                --include="${aggregation_glob}"
            )
        fi
        rsync_args+=(
            --exclude='*'
            "${remote_user}@${remote_host}:${remote_run_dir}/"
            "${local_run_dir}/"
        )
        rsync_with_master "${rsync_args[@]}"
    fi
else
    rsync_with_master -av --progress \
        --exclude='aggregated/archive/***' \
        --exclude='states/aggregated/archive/***' \
        "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"
fi

if [[ "${plot_after_sync}" == "true" ]]; then
    julia_setup="${JULIA_SETUP_SCRIPT:-${CLUSTER_JULIA_SETUP_SCRIPT:-}}"
    if [[ -n "${julia_setup}" && -f "${julia_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${julia_setup}"
    fi
    julia_bin="${JULIA_BIN:-julia}"
    if ! command -v "${julia_bin}" >/dev/null 2>&1; then
        echo "Julia executable '${julia_bin}' not found. Skipping plot."
        exit 0
    fi

    if [[ "${resolved_family}" == "active_objects" ]]; then
        hist_dir="${local_run_dir}/histograms/aggregated"
        if [[ ! -d "${hist_dir}" ]]; then
            echo "No aggregated histogram directory found at ${hist_dir}; skipping plot."
            exit 0
        fi

        mapfile -t histogram_files < <(find "${hist_dir}" -maxdepth 1 -type f -name "${aggregation_glob}" | sort)
        if (( ${#histogram_files[@]} == 0 )); then
            echo "No aggregated histogram files found in ${hist_dir} (pattern: ${aggregation_glob}); skipping plot."
            exit 0
        fi

        out_dir="${local_run_dir}/reports/active_object_histograms_$(date +%Y%m%d-%H%M%S)"
        mkdir -p "${out_dir}"
        cmd=("${julia_bin}" "${REPO_ROOT}/utility_scripts/plot_active_object_steady_state_histograms.jl")
        cmd+=("${histogram_files[@]}")
        cmd+=("--out_dir" "${out_dir}")

        echo "Running active-object histogram plotter on ${#histogram_files[@]} file(s)..."
        "${cmd[@]}"
        echo "Plots saved to: ${out_dir}"
        echo "Done. Local run folder: ${local_run_dir}"
        exit 0
    fi

    if [[ "${plot_mode_explicit}" != "true" ]]; then
        if [[ "${resolved_family}" == "two_force_d" ]]; then
            plot_mode="two_force_d"
        else
            plot_mode="single"
        fi
    fi

    plot_glob="*.jld2"
    if [[ "${sync_scope_effective}" == "aggregation" ]]; then
        plot_glob="${aggregation_glob}"
    fi

    find_state_files() {
        local dir="$1"
        if [[ "${sync_scope_effective}" == "aggregation" ]]; then
            find "${dir}" -maxdepth 1 -type f -name "${plot_glob}" ! -path '*/aggregated/archive/*' | sort
        else
            find "${dir}" -type f -name "${plot_glob}" ! -path '*/aggregated/archive/*' | sort
        fi
    }

    states_dir_custom="${local_run_dir}/states/${state_subdir}"
    states_dir_new_aggregated="${local_run_dir}/aggregated"
    states_dir_aggregated="${local_run_dir}/states/aggregated"
    states_dir_legacy="${local_run_dir}/states"
    states_dir="${states_dir_legacy}"
    if [[ "${aggregated_saved_only}" == "true" && -n "${state_subdir}" ]]; then
        states_dir="${states_dir_custom}"
    elif [[ "${aggregated_saved_only}" == "true" && -d "${states_dir_new_aggregated}" ]]; then
        states_dir="${states_dir_new_aggregated}"
    elif [[ "${aggregated_saved_only}" == "true" ]]; then
        states_dir="${states_dir_aggregated}"
    elif [[ "${sync_scope_effective}" == "aggregation" && -n "${state_subdir}" && -d "${states_dir_custom}" ]]; then
        states_dir="${states_dir_custom}"
    elif [[ "${sync_scope_effective}" == "aggregation" && -d "${states_dir_new_aggregated}" ]]; then
        states_dir="${states_dir_new_aggregated}"
    elif [[ "${sync_scope_effective}" == "aggregation" && -d "${states_dir_aggregated}" ]]; then
        states_dir="${states_dir_aggregated}"
    fi
    if [[ ! -d "${states_dir}" ]]; then
        echo "No states directory found at ${states_dir}; skipping plot."
        exit 0
    fi

    mapfile -t state_files < <(find_state_files "${states_dir}")
    if [[ "${aggregated_saved_only}" != "true" ]] && (( ${#state_files[@]} == 0 )) && [[ "${states_dir}" != "${states_dir_new_aggregated}" && -d "${states_dir_new_aggregated}" ]]; then
        mapfile -t state_files < <(find_state_files "${states_dir_new_aggregated}")
        if (( ${#state_files[@]} > 0 )); then
            states_dir="${states_dir_new_aggregated}"
        fi
    fi
    if [[ "${aggregated_saved_only}" != "true" ]] && (( ${#state_files[@]} == 0 )) && [[ "${states_dir}" != "${states_dir_aggregated}" && -d "${states_dir_aggregated}" ]]; then
        mapfile -t state_files < <(find_state_files "${states_dir_aggregated}")
        if (( ${#state_files[@]} > 0 )); then
            states_dir="${states_dir_aggregated}"
        fi
    fi
    if [[ "${aggregated_saved_only}" != "true" ]] && (( ${#state_files[@]} == 0 )) && [[ "${states_dir}" != "${states_dir_legacy}" && -d "${states_dir_legacy}" ]]; then
        mapfile -t state_files < <(find_state_files "${states_dir_legacy}")
        if (( ${#state_files[@]} > 0 )); then
            states_dir="${states_dir_legacy}"
        fi
    fi
    if [[ "${aggregated_saved_only}" == "true" && "${resolved_family}" == "two_force_d" && ${#state_files[@]} -gt 0 ]]; then
        original_state_count="${#state_files[@]}"
        mapfile -t latest_state_files < <(select_latest_two_force_d_files "${state_files[@]}")
        if (( ${#latest_state_files[@]} > 0 )); then
            state_files=("${latest_state_files[@]}")
        fi
        if (( ${#state_files[@]} < original_state_count )); then
            echo "Filtered ${original_state_count} matching state(s) down to ${#state_files[@]} latest-per-d state(s) for plotting."
        fi
    fi
    if (( ${#state_files[@]} == 0 )); then
        echo "No matching states found in ${states_dir} (pattern: ${plot_glob}); skipping plot."
        exit 0
    fi

    out_dir="${local_run_dir}/reports/load_and_plot_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${out_dir}"

    cmd=("${julia_bin}" "${REPO_ROOT}/load_and_plot.jl")
    cmd+=("${state_files[@]}")
    cmd+=("--mode" "${plot_mode}" "--out_dir" "${out_dir}")
    if [[ -n "${baseline_j2}" ]]; then
        cmd+=("--baseline_j2" "${baseline_j2}")
    fi
    if [[ "${skip_per_state}" == "true" ]]; then
        cmd+=("--skip_per_state_sweep")
    fi

    echo "Running load_and_plot on ${#state_files[@]} state(s) (pattern: ${plot_glob})..."
    "${cmd[@]}"
    echo "Plots saved to: ${out_dir}"
fi

echo "Done. Local run folder: ${local_run_dir}"
