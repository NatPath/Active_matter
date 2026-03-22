#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE_EOF'
Usage:
  bash copy_data_from_cluster.sh [options]

Core options:
  --mode <warmup|production>              Run mode for --list/--latest (for --run_id, registry mode is used)
  --run_id <id>                           Exact run_id to fetch (searched in both registries)
  --latest                                Fetch latest run_id in selected run family (can combine with filters)
  --list                                  List run registry entries and exit
  --tail <N>                              Number of listed entries (default: 30)
  --run_family <two_force_d|single_origin_bond|auto>
                                          Registry family for --list/--latest (default: two_force_d)

Run filters for --list / --latest:
  --L <int>                               Match L column
  --rho <value>                           Match rho0 column
  --n_sweeps <int>                        Match n_sweeps column
  --d_min <int>                           Match d_min column (two_force_d only)
  --d_max <int>                           Match d_max column (two_force_d only)
  --d_step <int>                          Match d_step column (two_force_d only)

Paths / connection:
  --remote_user <name>                    SSH user (default: nativmr)
  --remote_host <host>                    SSH host (default: tech-ui02.hep.technion.ac.il)
  --remote_root <path>                    Cluster repo root (default: /storage/ph_kafri/nativmr/run_and_tumble)
  --local_root <path>                     Local root for downloaded data (default: <repo>/cluster_results)
  --sync_scope <auto|aggregation|full>    Data sync scope:
                                          auto (default): if run_id looks aggregated (_nr...), fetch aggregated artifacts only
                                          aggregation: fetch aggregated state files + run metadata
                                          (prefers states/aggregated, with legacy fallback; excludes states/aggregated/archive)
                                          full: fetch full run directory

Post-processing:
  --plot                                  Run load_and_plot.jl after sync
  --aggregated_saved_only                 Restrict copy/plot to *_id-aggregated_saved_*.jld2
                                          (forces --sync_scope aggregation)
  --state_glob <pattern>                  File glob for state selection in aggregation sync/plot
                                          (default aggregation glob: *id-aggregated_*.jld2)
  --skip_per_state_sweep                  Pass --skip_per_state_sweep to load_and_plot.jl
  --baseline_j2 <value>                   Baseline override for two_force_d analysis
                                          (default when omitted: automatic rho0^2-based baseline from load_and_plot.jl)
  --mode_plot <single|two_force_d>        load_and_plot mode; auto-selected from run family when omitted

Examples:
  bash copy_data_from_cluster.sh --list --run_family two_force_d --mode warmup
  bash copy_data_from_cluster.sh --latest --run_family single_origin_bond --mode production --L 128 --rho 100 --n_sweeps 1000000
  bash copy_data_from_cluster.sh --run_id single_production_L128_rho100_ns1000000_f1.0_ffr1.0_20260225-152138 --plot
  bash copy_data_from_cluster.sh --run_id two_force_production_L128_rho100_ns1000000_d2-32-s2_nr8_dag_20260225-180000 --sync_scope aggregation
  bash copy_data_from_cluster.sh --run_id two_force_warmup_production_L256_rho100_..._production_20260226-032016 --aggregated_saved_only --plot
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

remote_user="${REMOTE_USER:-nativmr}"
remote_host="${REMOTE_HOST:-tech-ui02.hep.technion.ac.il}"
remote_root="${REMOTE_ROOT:-/storage/ph_kafri/nativmr/run_and_tumble}"
local_root="${LOCAL_RESULTS_ROOT:-${REPO_ROOT}/cluster_results}"
sync_scope="auto"

plot_after_sync="false"
aggregated_saved_only="false"
state_glob=""
state_glob_explicit="false"
skip_per_state="false"
baseline_j2=""
plot_mode="two_force_d"
plot_mode_explicit="false"

declare -A REGISTRY_REL_MAP
REGISTRY_REL_MAP["two_force_d"]="runs/two_force_d/run_registry.csv"
REGISTRY_REL_MAP["single_origin_bond"]="runs/single_origin_bond/run_registry.csv"
ALL_FAMILIES=("two_force_d" "single_origin_bond")

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
        --state_glob)
            state_glob="${2:-}"
            state_glob_explicit="true"
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
if [[ "${run_family}" != "two_force_d" && "${run_family}" != "single_origin_bond" && "${run_family}" != "auto" ]]; then
    echo "--run_family must be two_force_d, single_origin_bond, or auto. Got '${run_family}'."
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

sync_registry() {
    local family="$1"
    local registry_rel="${REGISTRY_REL_MAP[$family]}"
    local local_registry="${local_root}/${registry_rel}"

    mkdir -p "$(dirname "${local_registry}")"
    echo "Syncing ${family} run registry from cluster..." >&2
    if scp "${remote_user}@${remote_host}:${remote_root}/${registry_rel}" "${local_registry}"; then
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

    for dir in "${run_dir}/states/aggregated" "${run_dir}/states"; do
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

if [[ -z "${run_id}" ]]; then
    if [[ "${latest}" != "true" && "${list_only}" != "true" ]]; then
        echo "Provide --run_id <id> or use --latest or --list."
        exit 1
    fi

    selected_family="${run_family}"
    if [[ "${selected_family}" == "auto" ]]; then
        selected_family="two_force_d"
    fi

    if [[ "${selected_family}" == "single_origin_bond" ]]; then
        if [[ -n "${filter_d_min}" || -n "${filter_d_max}" || -n "${filter_d_step}" ]]; then
            echo "WARNING: --d_min/--d_max/--d_step filters are ignored for single_origin_bond." >&2
        fi
    fi

    if ! resolved_registry="$(sync_registry "${selected_family}")"; then
        echo "Could not access registry for run family '${selected_family}'."
        exit 1
    fi

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

    for family in "${ALL_FAMILIES[@]}"; do
        if registry_path="$(sync_registry "${family}")"; then
            row="$(find_row_by_run_id "${registry_path}" "${run_id}")"
            if [[ -n "${row}" ]]; then
                found_count=$((found_count + 1))
                found_families+=("${family}")
                found_registries+=("${registry_path}")
                resolved_family="${family}"
                resolved_registry="${registry_path}"
                registry_row="${row}"
            fi
        fi
    done

    if (( found_count == 0 )); then
        two_force_registry="${local_root}/${REGISTRY_REL_MAP[two_force_d]}"
        single_registry="${local_root}/${REGISTRY_REL_MAP[single_origin_bond]}"
        echo "run_id '${run_id}' not found in either registry:"
        echo "  ${two_force_registry}"
        echo "  ${single_registry}"
        exit 1
    fi

    if (( found_count > 1 )); then
        echo "run_id '${run_id}' was found in multiple registries, cannot disambiguate automatically:"
        for idx in "${!found_families[@]}"; do
            echo "  family=${found_families[$idx]} registry=${found_registries[$idx]}"
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
    remote_run_dir="${remote_root}/runs/${resolved_family}/${mode}/${run_id}"
fi

local_run_dir="${local_root}/runs/${resolved_family}/${mode}/${run_id}"
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

aggregation_glob="*id-aggregated_*.jld2"
if [[ "${state_glob_explicit}" == "true" ]]; then
    aggregation_glob="${state_glob}"
elif [[ -n "${state_glob}" ]]; then
    aggregation_glob="${state_glob}"
fi
if [[ "${sync_scope_effective}" == "aggregation" ]]; then
    echo "  aggregation_state_glob=${aggregation_glob}"
fi

if [[ "${sync_scope_effective}" == "aggregation" && "${aggregated_saved_only}" == "true" ]]; then
    prune_local_aggregation_matches "${local_run_dir}" "${aggregation_glob}"
fi

if [[ "${sync_scope_effective}" == "aggregation" ]]; then
    rsync -av --progress --prune-empty-dirs \
        --exclude='states/aggregated/archive/***' \
        --include='*/' \
        --include='run_info.txt' \
        --include='manifest.csv' \
        --include='reports/***' \
        --include="states/aggregated/${aggregation_glob}" \
        --include="states/${aggregation_glob}" \
        --include="${aggregation_glob}" \
        --exclude='*' \
        "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"
else
    rsync -av --progress \
        --exclude='states/aggregated/archive/***' \
        "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"
fi

if [[ "${plot_after_sync}" == "true" ]]; then
    if [[ "${plot_mode_explicit}" != "true" ]]; then
        if [[ "${resolved_family}" == "single_origin_bond" ]]; then
            plot_mode="single"
        else
            plot_mode="two_force_d"
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

    states_dir_aggregated="${local_run_dir}/states/aggregated"
    states_dir_legacy="${local_run_dir}/states"
    states_dir="${states_dir_legacy}"
    if [[ "${sync_scope_effective}" == "aggregation" && -d "${states_dir_aggregated}" ]]; then
        states_dir="${states_dir_aggregated}"
    fi
    if [[ ! -d "${states_dir}" ]]; then
        echo "No states directory found at ${states_dir}; skipping plot."
        exit 0
    fi

    mapfile -t state_files < <(find_state_files "${states_dir}")
    if (( ${#state_files[@]} == 0 )) && [[ "${states_dir}" != "${states_dir_legacy}" && -d "${states_dir_legacy}" ]]; then
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

    julia_setup="${JULIA_SETUP_SCRIPT:-/Local/ph_kafri/julia-1.7.2/bin/setup.sh}"
    if [[ -f "${julia_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${julia_setup}"
    fi
    julia_bin="${JULIA_BIN:-julia}"
    if ! command -v "${julia_bin}" >/dev/null 2>&1; then
        echo "Julia executable '${julia_bin}' not found. Skipping plot."
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
