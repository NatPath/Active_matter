#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_two_force_d_saved_files.sh --run_id <run_id> [options]

Required:
  --run_id <id>                     two_force_d run_id

Options:
  --mode <auto|warmup|production|warmup_production>
                                    Where to resolve run_id from (default: auto)
  --num_files <int>                 Files per d to aggregate; 0 means all (default: num_replicas from run_info)
  --state_dir <path>                Override state directory
  --config_dir <path>               Override config directory
  --extra_raw_dir <path>            Also include raw states from this top-level directory
                                    when building the aggregation input list
  --aggregated_subdir <name>        Save aggregated outputs under <state_dir>/<name>
                                    (default: save directly under state_dir)
  --exclude_aggregated_inputs       Do not use existing aggregated states as discovery inputs
  --incremental_from_existing_aggregate
                                    If a current aggregate exists for d, use it as the base input
                                    and add only raw files from --extra_raw_dir paths; otherwise
                                    fall back to top-level raw states plus --extra_raw_dir paths
  --archive_existing_aggregates     Before re-aggregating, move prior non-partial aggregates
                                    for that d into <aggregated_subdir>/<archive_subdir>/<stamp>/d_<d>
  --archive_subdir <name>           Archive subdirectory under aggregated_subdir (default: archive)
  --archive_stamp <token>           Optional archive stamp token (default: current timestamp)
  --d_min <int>                     Override d_min (requires --d_max and --d_step)
  --d_max <int>                     Override d_max (requires --d_min and --d_step)
  --d_step <int>                    Override d_step (requires --d_min and --d_max)
  --force                           Re-aggregate even if output already exists
  --dry_run                         Print actions only
  --keep_going                      Continue on per-d errors
  -h, --help                        Show help

Behavior:
  - Resolves run_info for run_id (and chain->production for warmup_production).
  - By default, uses the exact d list from run_info (d_values / d_spacing); a linear seq is used only
    when --d_min/--d_max/--d_step are explicitly provided together.
  - For each d, selects saved files by filename pattern:
      production: two_force_d<d>_prod_*.jld2
      warmup:     two_force_d<d>_warmup_*.jld2
  - By default, raw discovery uses only top-level files under --state_dir and top-level files under
    --aggregated_subdir. --extra_raw_dir adds more top-level raw search roots.
  - Aggregates selected files via:
      run_diffusive_no_activity_from_config.sh <config> --aggregate_state_list <list> --save_tag <tag>
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing helper script: ${RUNNER_SCRIPT}"
    exit 1
fi

SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Could not find spacing utils script: ${SPACING_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${SPACING_UTILS}"

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

find_run_info_by_run_id() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local candidate=""

    if [[ "${mode_hint}" == "warmup" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "warmup_production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup_production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    local registry_file="${REPO_ROOT}/runs/two_force_d/run_registry.csv"
    if [[ -f "${registry_file}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${registry_file}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _dmin _dmax _dstep _cpus _mem reg_run_root _log_dir _state_dir _warmup_state_dir <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

infer_spacing_from_run_id() {
    local run_id_val="$1"
    if [[ "${run_id_val}" == *"-lm"* || "${run_id_val}" == *"_lm"* ]]; then
        echo "log_midpoints"
    else
        echo "linear"
    fi
}

resolve_d_values_from_run_info() {
    local run_info_path="$1"
    local run_id_val="$2"
    local -n out_ref="$3"
    local d_values_csv d_spacing_local d_min_local d_max_local d_step_local

    out_ref=()
    d_values_csv="$(read_run_info_value "${run_info_path}" "d_values")"
    if [[ -n "${d_values_csv}" ]]; then
        if ! two_force_d_csv_to_array "${d_values_csv}" out_ref; then
            echo "Invalid d_values='${d_values_csv}' in ${run_info_path}" >&2
            return 1
        fi
        if (( ${#out_ref[@]} > 0 )); then
            return 0
        fi
    fi

    d_spacing_local="$(read_run_info_value "${run_info_path}" "d_spacing")"
    if [[ -z "${d_spacing_local}" ]]; then
        d_spacing_local="$(infer_spacing_from_run_id "${run_id_val}")"
    fi
    d_spacing_local="$(two_force_d_normalize_spacing_mode "${d_spacing_local}")" || {
        echo "Invalid d_spacing='${d_spacing_local}' in ${run_info_path}" >&2
        return 1
    }

    d_min_local="$(read_run_info_value "${run_info_path}" "d_min")"
    d_max_local="$(read_run_info_value "${run_info_path}" "d_max")"
    d_step_local="$(read_run_info_value "${run_info_path}" "d_step")"
    if ! [[ "${d_min_local}" =~ ^[0-9]+$ && "${d_max_local}" =~ ^[0-9]+$ && "${d_step_local}" =~ ^[0-9]+$ ]]; then
        echo "Invalid d-range in ${run_info_path}: d_min='${d_min_local}' d_max='${d_max_local}' d_step='${d_step_local}'" >&2
        return 1
    fi

    mapfile -t out_ref < <(two_force_d_generate_d_values "${d_spacing_local}" "${d_min_local}" "${d_max_local}" "${d_step_local}")
    return 0
}

make_short_tag_base() {
    local raw="$1"
    local cleaned hash
    cleaned="$(printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
    if (( ${#cleaned} > 36 )); then
        cleaned="${cleaned:0:36}"
    fi
    hash="$(printf "%s" "${raw}" | cksum | awk '{print $1}')"
    printf "%s_%s" "${cleaned}" "${hash}"
}

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
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
    done < <(find "${root_dir}" -type f -name "*_id-${id_tag}.jld2" ! -path '*/archive/*' ! -size 0 -print0 2>/dev/null)
    printf "%s" "${best_path}"
}

latest_state_for_id_tag_top_level() {
    local dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime
    [[ -d "${dir}" ]] || { printf ""; return 0; }
    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)
    printf "%s" "${best_path}"
}

move_state_to_aggregated_subdir() {
    local source_file="$1"
    if [[ -z "${aggregated_subdir}" ]]; then
        printf "%s" "${source_file}"
        return 0
    fi
    local target_file="${effective_aggregated_state_dir}/$(basename "${source_file}")"
    if [[ "${source_file}" == "${target_file}" ]]; then
        printf "%s" "${source_file}"
        return 0
    fi
    mkdir -p "${effective_aggregated_state_dir}"
    mv -f "${source_file}" "${target_file}"
    printf "%s" "${target_file}"
}

move_file_to_archive_dir() {
    local source_file="$1"
    local archive_dir="$2"
    local base_name target_path stem idx

    mkdir -p "${archive_dir}"
    base_name="$(basename "${source_file}")"
    target_path="${archive_dir}/${base_name}"
    if [[ -e "${target_path}" ]]; then
        stem="${base_name%.jld2}"
        idx=1
        while [[ -e "${archive_dir}/${stem}__arch${idx}.jld2" ]]; do
            idx=$((idx + 1))
        done
        target_path="${archive_dir}/${stem}__arch${idx}.jld2"
    fi
    mv -f "${source_file}" "${target_path}"
    printf "%s" "${target_path}"
}

stage_existing_aggregate_files_for_d() {
    local d_val="$1"
    local suffix_val="$2"
    local stage_map_file="$3"
    local preserve_path="${4:-}"
    local candidate archived_path

    [[ "${archive_existing_aggregates}" == "true" ]] || return 0
    : > "${stage_map_file}"

    while IFS= read -r candidate; do
        [[ -z "${candidate}" ]] && continue
        if [[ -n "${preserve_path}" && "${candidate}" == "${preserve_path}" ]]; then
            continue
        fi
        if [[ "${dry_run}" == "true" ]]; then
            echo "d=${d_val}: would stage existing aggregate ${candidate} -> ${effective_archive_root}/d_${d_val}/"
            continue
        fi
        archived_path="$(move_file_to_archive_dir "${candidate}" "${effective_archive_root}/d_${d_val}")"
        printf "%s\t%s\n" "${candidate}" "${archived_path}" >> "${stage_map_file}"
        echo "d=${d_val}: staged existing aggregate -> ${archived_path}"
    done < <({
        find "${effective_state_dir}" -maxdepth 1 -type f \
            -name "two_force_d${d_val}_${suffix_val}_*.jld2" \
            -name "*_id-aggregated_*" \
            ! -name "*_id-aggregated_partial_*" \
            -print 2>/dev/null
        if [[ "${effective_aggregated_state_dir}" != "${effective_state_dir}" && -d "${effective_aggregated_state_dir}" ]]; then
            find "${effective_aggregated_state_dir}" -maxdepth 1 -type f \
                -name "two_force_d${d_val}_${suffix_val}_*.jld2" \
                -name "*_id-aggregated_*" \
                ! -name "*_id-aggregated_partial_*" \
                -print 2>/dev/null
        fi
    } | awk '!seen[$0]++' | sort)
}

restore_staged_aggregate_files() {
    local d_val="$1"
    local stage_map_file="$2"
    local original_path archived_path

    [[ -f "${stage_map_file}" ]] || return 0
    [[ -s "${stage_map_file}" ]] || return 0

    while IFS=$'\t' read -r original_path archived_path; do
        [[ -z "${original_path}" || -z "${archived_path}" ]] && continue
        if [[ ! -e "${archived_path}" ]]; then
            continue
        fi
        mkdir -p "$(dirname "${original_path}")"
        mv -f "${archived_path}" "${original_path}"
        echo "d=${d_val}: restored staged aggregate -> ${original_path}"
    done < "${stage_map_file}"
}

run_id=""
mode="auto"
num_files=""
state_dir=""
config_dir=""
extra_raw_dirs=()
aggregated_subdir=""
exclude_aggregated_inputs="false"
incremental_from_existing_aggregate="false"
archive_existing_aggregates="false"
archive_subdir="archive"
archive_stamp=""
d_min=""
d_max=""
d_step=""
force_reaggregate="false"
dry_run="false"
keep_going="false"

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
        --num_files)
            num_files="${2:-}"
            shift 2
            ;;
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --config_dir)
            config_dir="${2:-}"
            shift 2
            ;;
        --extra_raw_dir)
            extra_raw_dirs+=("${2:-}")
            shift 2
            ;;
        --aggregated_subdir)
            aggregated_subdir="${2:-}"
            shift 2
            ;;
        --exclude_aggregated_inputs)
            exclude_aggregated_inputs="true"
            shift
            ;;
        --incremental_from_existing_aggregate)
            incremental_from_existing_aggregate="true"
            shift
            ;;
        --archive_existing_aggregates)
            archive_existing_aggregates="true"
            shift
            ;;
        --archive_subdir)
            archive_subdir="${2:-}"
            shift 2
            ;;
        --archive_stamp)
            archive_stamp="${2:-}"
            shift 2
            ;;
        --d_min)
            d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            d_step="${2:-}"
            shift 2
            ;;
        --force)
            force_reaggregate="true"
            shift
            ;;
        --dry_run)
            dry_run="true"
            shift
            ;;
        --keep_going)
            keep_going="true"
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

if [[ -z "${run_id}" ]]; then
    echo "--run_id is required."
    usage
    exit 1
fi

case "${mode}" in
    auto|warmup|production|warmup_production)
        ;;
    *)
        echo "--mode must be one of: auto, warmup, production, warmup_production."
        exit 1
        ;;
esac

if [[ -n "${aggregated_subdir}" ]] && ! [[ "${aggregated_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--aggregated_subdir must match [A-Za-z0-9._-]+ when provided. Got '${aggregated_subdir}'."
    exit 1
fi
for extra_dir in "${extra_raw_dirs[@]}"; do
    if [[ -z "${extra_dir}" || ! -d "${extra_dir}" ]]; then
        echo "--extra_raw_dir is invalid: ${extra_dir}"
        exit 1
    fi
done
if [[ "${archive_existing_aggregates}" == "true" && -z "${aggregated_subdir}" ]]; then
    echo "--archive_existing_aggregates requires --aggregated_subdir."
    exit 1
fi
if [[ -n "${archive_subdir}" ]] && ! [[ "${archive_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_subdir must match [A-Za-z0-9._-]+ when provided. Got '${archive_subdir}'."
    exit 1
fi
if [[ -n "${archive_stamp}" ]] && ! [[ "${archive_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_stamp must match [A-Za-z0-9._-]+ when provided. Got '${archive_stamp}'."
    exit 1
fi

run_info_path="$(find_run_info_by_run_id "${run_id}" "${mode}" || true)"
if [[ -z "${run_info_path}" || ! -f "${run_info_path}" ]]; then
    echo "Could not resolve run_info for run_id='${run_id}' (mode=${mode})."
    exit 1
fi

resolved_mode="$(read_run_info_value "${run_info_path}" "mode")"
target_run_info="${run_info_path}"
target_run_id="$(read_run_info_value "${target_run_info}" "run_id")"

if [[ "${resolved_mode}" == "warmup_production" ]]; then
    production_run_info="$(read_run_info_value "${run_info_path}" "production_run_info")"
    production_run_id="$(read_run_info_value "${run_info_path}" "production_run_id")"
    if [[ -n "${production_run_info}" && -f "${production_run_info}" ]]; then
        target_run_info="${production_run_info}"
    elif [[ -n "${production_run_id}" ]]; then
        target_run_info="$(find_run_info_by_run_id "${production_run_id}" "production" || true)"
    fi
    if [[ -z "${target_run_info}" || ! -f "${target_run_info}" ]]; then
        echo "Could not resolve production run_info from chain run_id='${run_id}'."
        exit 1
    fi
    target_run_id="$(read_run_info_value "${target_run_info}" "run_id")"
fi

target_mode="$(read_run_info_value "${target_run_info}" "mode")"
if [[ "${target_mode}" != "production" && "${target_mode}" != "warmup" ]]; then
    echo "Resolved target mode '${target_mode}' is not supported."
    exit 1
fi

run_root="$(read_run_info_value "${target_run_info}" "run_root")"
effective_state_dir="${state_dir:-$(read_run_info_value "${target_run_info}" "state_dir")}"
[[ -z "${effective_state_dir}" && -n "${run_root}" ]] && effective_state_dir="${run_root}/states"
if [[ -z "${effective_state_dir}" || ! -d "${effective_state_dir}" ]]; then
    echo "State directory is invalid: ${effective_state_dir}"
    exit 1
fi

effective_aggregated_state_dir="${effective_state_dir}"
if [[ -n "${aggregated_subdir}" ]]; then
    effective_aggregated_state_dir="${effective_state_dir}/${aggregated_subdir}"
    if [[ "${dry_run}" != "true" ]]; then
        mkdir -p "${effective_aggregated_state_dir}"
    fi
fi

effective_archive_root=""
if [[ "${archive_existing_aggregates}" == "true" ]]; then
    if [[ -n "${archive_stamp}" ]]; then
        effective_archive_stamp="${archive_stamp}"
    else
        effective_archive_stamp="$(date +%Y%m%d-%H%M%S)"
    fi
    effective_archive_root="${effective_aggregated_state_dir}/${archive_subdir}/${effective_archive_stamp}"
    if [[ "${dry_run}" != "true" ]]; then
        mkdir -p "${effective_archive_root}"
    fi
fi

effective_config_dir="${config_dir:-$(read_run_info_value "${target_run_info}" "config_dir")}"
[[ -z "${effective_config_dir}" && -n "${run_root}" ]] && effective_config_dir="${run_root}/configs"
if [[ -z "${effective_config_dir}" || ! -d "${effective_config_dir}" ]]; then
    echo "Config directory is invalid: ${effective_config_dir}"
    exit 1
fi

default_num_files="$(read_run_info_value "${target_run_info}" "num_replicas")"
effective_num_files="${num_files:-${default_num_files}}"
if [[ -z "${effective_num_files}" ]]; then
    effective_num_files="0"
fi
if ! [[ "${effective_num_files}" =~ ^[0-9]+$ ]]; then
    echo "--num_files must be integer >= 0. Got '${effective_num_files}'."
    exit 1
fi

effective_d_min="${d_min:-$(read_run_info_value "${target_run_info}" "d_min")}"
effective_d_max="${d_max:-$(read_run_info_value "${target_run_info}" "d_max")}"
effective_d_step="${d_step:-$(read_run_info_value "${target_run_info}" "d_step")}"
declare -a D_VALUES=()
if [[ -n "${d_min}" || -n "${d_max}" || -n "${d_step}" ]]; then
    if [[ -z "${d_min}" || -z "${d_max}" || -z "${d_step}" ]]; then
        echo "--d_min, --d_max, and --d_step must be provided together."
        exit 1
    fi
    if ! [[ "${effective_d_min}" =~ ^[0-9]+$ && "${effective_d_max}" =~ ^[0-9]+$ && "${effective_d_step}" =~ ^[0-9]+$ ]]; then
        echo "d-range is invalid: d_min='${effective_d_min}', d_max='${effective_d_max}', d_step='${effective_d_step}'"
        exit 1
    fi
    if (( effective_d_step <= 0 || effective_d_max < effective_d_min )); then
        echo "Invalid d-range values: ${effective_d_min}:${effective_d_step}:${effective_d_max}"
        exit 1
    fi
    mapfile -t D_VALUES < <(seq "${effective_d_min}" "${effective_d_step}" "${effective_d_max}")
else
    if ! resolve_d_values_from_run_info "${target_run_info}" "${target_run_id}" D_VALUES; then
        exit 1
    fi
    if (( ${#D_VALUES[@]} == 0 )); then
        echo "Resolved d-value list is empty for ${target_run_info}."
        exit 1
    fi
fi

suffix="prod"
if [[ "${target_mode}" == "warmup" ]]; then
    suffix="warmup"
fi

run_hash="$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
L_val="$(read_run_info_value "${target_run_info}" "L")"
rho_val="$(read_run_info_value "${target_run_info}" "rho0")"
rho_tag="$(sanitize_token "${rho_val}")"
if [[ -z "${L_val}" ]]; then
    L_val="NA"
fi
if [[ -z "${rho_tag}" ]]; then
    rho_tag="NA"
fi

echo "Resolved aggregation target:"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_mode=${target_mode}"
echo "  run_info=${target_run_info}"
echo "  state_dir=${effective_state_dir}"
echo "  aggregated_output_dir=${effective_aggregated_state_dir}"
echo "  config_dir=${effective_config_dir}"
if (( ${#extra_raw_dirs[@]} > 0 )); then
    echo "  extra_raw_dirs=$(IFS=:; echo "${extra_raw_dirs[*]}")"
fi
echo "  exclude_aggregated_inputs=${exclude_aggregated_inputs}"
echo "  incremental_from_existing_aggregate=${incremental_from_existing_aggregate}"
echo "  num_files_per_d=${effective_num_files} (0=all)"
echo "  d_values=$(IFS=,; echo "${D_VALUES[*]}")"
echo "  force_reaggregate=${force_reaggregate}"
echo "  dry_run=${dry_run}"
echo "  keep_going=${keep_going}"
echo "  archive_existing_aggregates=${archive_existing_aggregates}"
if [[ "${archive_existing_aggregates}" == "true" ]]; then
    echo "  archive_root=${effective_archive_root}"
fi
echo "  save_tag_template=aggregated_saved_r${run_hash}_d<d>"
echo "  legacy_save_tag_template=aggregated_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_d<d>"

ok_count=0
skip_count=0
fail_count=0

for d in "${D_VALUES[@]}"; do
    runtime_config="${effective_config_dir}/d_${d}.yaml"
    if [[ ! -f "${runtime_config}" ]]; then
        fallback_cfg="${REPO_ROOT}/configuration_files/two_force_d_sweep/${target_mode}/d_${d}.yaml"
        if [[ -f "${fallback_cfg}" ]]; then
            runtime_config="${fallback_cfg}"
        else
            echo "d=${d}: missing config ${effective_config_dir}/d_${d}.yaml"
            fail_count=$((fail_count + 1))
            [[ "${keep_going}" == "true" ]] && continue
            exit 1
        fi
    fi

    save_tag="aggregated_saved_r${run_hash}_d${d}"
    legacy_save_tag="aggregated_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_d${d}"
    existing_agg=""
    if [[ "${effective_aggregated_state_dir}" != "${effective_state_dir}" ]]; then
        existing_agg="$(latest_state_for_id_tag_top_level "${effective_aggregated_state_dir}" "${save_tag}")"
    fi
    if [[ -z "${existing_agg}" ]]; then
        existing_agg="$(latest_state_for_id_tag_top_level "${effective_state_dir}" "${save_tag}")"
    fi
    if [[ -z "${existing_agg}" ]]; then
        if [[ "${effective_aggregated_state_dir}" != "${effective_state_dir}" ]]; then
            existing_agg="$(latest_state_for_id_tag_top_level "${effective_aggregated_state_dir}" "${legacy_save_tag}")"
        fi
        if [[ -z "${existing_agg}" ]]; then
            existing_agg="$(latest_state_for_id_tag_top_level "${effective_state_dir}" "${legacy_save_tag}")"
        fi
    fi
    if [[ -n "${existing_agg}" && "${force_reaggregate}" != "true" ]]; then
        if [[ -n "${aggregated_subdir}" && "${existing_agg}" != "${effective_aggregated_state_dir}/"* ]]; then
            if [[ "${dry_run}" == "true" ]]; then
                echo "d=${d}: would move existing aggregate to ${effective_aggregated_state_dir}/"
            else
                existing_agg="$(move_state_to_aggregated_subdir "${existing_agg}")"
            fi
        fi
        echo "d=${d}: skip (already exists) -> ${existing_agg}"
        skip_count=$((skip_count + 1))
        continue
    fi

    mapfile -t root_candidates < <(
        find "${effective_state_dir}" -maxdepth 1 -type f \
            -name "two_force_d${d}_${suffix}_*.jld2" \
            ! -name "*_id-aggregated_*" \
            ! -size 0 \
            -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | awk '{ $1=""; sub(/^ /,""); print }'
    )
    extra_raw_candidates=()
    if (( ${#extra_raw_dirs[@]} > 0 )); then
        while IFS= read -r candidate_path; do
            [[ -n "${candidate_path}" ]] && extra_raw_candidates+=("${candidate_path}")
        done < <(
            for extra_dir in "${extra_raw_dirs[@]}"; do
                find "${extra_dir}" -maxdepth 1 -type f \
                    -name "two_force_d${d}_${suffix}_*.jld2" \
                    ! -name "*_id-aggregated_*" \
                    ! -size 0 \
                    -printf '%T@ %p\n' 2>/dev/null
            done | sort -nr | awk '{ $1=""; sub(/^ /,""); print }'
        )
    fi
    aggregated_input_candidates=()
    if [[ "${exclude_aggregated_inputs}" != "true" && "${incremental_from_existing_aggregate}" != "true" ]]; then
        while IFS= read -r candidate_path; do
            [[ -n "${candidate_path}" ]] && aggregated_input_candidates+=("${candidate_path}")
        done < <(
            if [[ "${effective_aggregated_state_dir}" != "${effective_state_dir}" && -d "${effective_aggregated_state_dir}" ]]; then
                find "${effective_aggregated_state_dir}" -maxdepth 1 -type f \
                    -name "two_force_d${d}_${suffix}_*.jld2" \
                    -name "*_id-aggregated_*" \
                    ! -name "*_id-aggregated_partial_*" \
                    ! -size 0 \
                    -printf '%T@ %p\n' 2>/dev/null \
                    | sort -nr | awk '{ $1=""; sub(/^ /,""); print }'
            fi
        )
    fi
    mapfile -t candidates < <(
        if [[ "${incremental_from_existing_aggregate}" == "true" && -n "${existing_agg}" ]]; then
            {
                printf '%s\n' "${existing_agg}"
                printf '%s\n' "${extra_raw_candidates[@]}"
            }
        else
            {
                printf '%s\n' "${root_candidates[@]}"
                printf '%s\n' "${extra_raw_candidates[@]}"
                printf '%s\n' "${aggregated_input_candidates[@]}"
            }
        fi | awk 'NF && !seen[$0]++' \
          | while IFS= read -r candidate_path; do
                printf '%s %s\n' "$(stat -c %Y "${candidate_path}" 2>/dev/null || echo 0)" "${candidate_path}"
            done \
          | sort -nr \
          | awk '{ $1=""; sub(/^ /,""); print }'
    )

    if (( ${#candidates[@]} == 0 )); then
        echo "d=${d}: no matching saved files found (pattern two_force_d${d}_${suffix}_*.jld2)."
        fail_count=$((fail_count + 1))
        [[ "${keep_going}" == "true" ]] && continue
        exit 1
    fi

    selected_count="${#candidates[@]}"
    if (( effective_num_files > 0 )); then
        selected_count="${effective_num_files}"
        if (( ${#candidates[@]} < effective_num_files )); then
            echo "d=${d}: only ${#candidates[@]} files found, expected ${effective_num_files}."
            fail_count=$((fail_count + 1))
            [[ "${keep_going}" == "true" ]] && continue
            exit 1
        fi
    fi

    state_list_file="$(mktemp)"
    stage_map_file="$(mktemp)"
    trap 'rm -f "${state_list_file}" "${stage_map_file}"' EXIT
    printf "%s\n" "${candidates[@]:0:${selected_count}}" > "${state_list_file}"

    base_aggregate_count=0
    if [[ "${incremental_from_existing_aggregate}" == "true" && -n "${existing_agg}" ]]; then
        base_aggregate_count=1
    fi
    echo "d=${d}: aggregating ${selected_count} input state(s) (root=${#root_candidates[@]}, extra=${#extra_raw_candidates[@]}, aggregated=${#aggregated_input_candidates[@]}, base_aggregate=${base_aggregate_count}) -> save_tag=${save_tag} (legacy=${legacy_save_tag})"
    if [[ "${dry_run}" == "true" ]]; then
        head -n 3 "${state_list_file}" | sed 's/^/  sample: /'
        rm -f "${state_list_file}"
        rm -f "${stage_map_file}"
        continue
    fi

    if bash "${RUNNER_SCRIPT}" "${runtime_config}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"; then
        saved="$(latest_state_for_id_tag_top_level "${effective_state_dir}" "${save_tag}")"
        if [[ -z "${saved}" ]]; then
            saved="$(latest_state_for_id_tag_top_level "${effective_state_dir}" "${legacy_save_tag}")"
        fi
        if [[ -n "${saved}" ]]; then
            stage_existing_aggregate_files_for_d "${d}" "${suffix}" "${stage_map_file}" "${saved}"
            if [[ -n "${aggregated_subdir}" ]]; then
                if ! saved="$(move_state_to_aggregated_subdir "${saved}")"; then
                    restore_staged_aggregate_files "${d}" "${stage_map_file}"
                    echo "d=${d}: failed to move freshly aggregated state into ${effective_aggregated_state_dir}"
                    fail_count=$((fail_count + 1))
                    if [[ "${keep_going}" != "true" ]]; then
                        rm -f "${state_list_file}"
                        rm -f "${stage_map_file}"
                        exit 1
                    fi
                    rm -f "${state_list_file}"
                    rm -f "${stage_map_file}"
                    continue
                fi
            fi
            echo "d=${d}: aggregated -> ${saved}"
            ok_count=$((ok_count + 1))
        else
            echo "d=${d}: aggregation command succeeded but no output found for save_tag=${save_tag}"
            restore_staged_aggregate_files "${d}" "${stage_map_file}"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" != "true" ]]; then
                rm -f "${state_list_file}"
                rm -f "${stage_map_file}"
                exit 1
            fi
        fi
    else
        echo "d=${d}: aggregation failed."
        restore_staged_aggregate_files "${d}" "${stage_map_file}"
        fail_count=$((fail_count + 1))
        if [[ "${keep_going}" != "true" ]]; then
            rm -f "${state_list_file}"
            rm -f "${stage_map_file}"
            exit 1
        fi
    fi

    rm -f "${state_list_file}"
    rm -f "${stage_map_file}"
done

echo "Aggregation summary:"
echo "  target_run_id=${target_run_id}"
echo "  state_dir=${effective_state_dir}"
echo "  aggregated_output_dir=${effective_aggregated_state_dir}"
echo "  success=${ok_count}"
echo "  skipped_existing=${skip_count}"
echo "  failed=${fail_count}"

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi
if (( fail_count > 0 )); then
    exit 1
fi
