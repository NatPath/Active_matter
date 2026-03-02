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
  --aggregated_subdir <name>        Save aggregated outputs under <state_dir>/<name>
                                    (default: save directly under state_dir)
  --d_min <int>                     Override d_min
  --d_max <int>                     Override d_max
  --d_step <int>                    Override d_step
  --force                           Re-aggregate even if output already exists
  --dry_run                         Print actions only
  --keep_going                      Continue on per-d errors
  -h, --help                        Show help

Behavior:
  - Resolves run_info for run_id (and chain->production for warmup_production).
  - For each d, selects saved files by filename pattern:
      production: two_force_d<d>_prod_*.jld2
      warmup:     two_force_d<d>_warmup_*.jld2
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
    done < <(find "${root_dir}" -type f -name "*_id-${id_tag}.jld2" -print0 2>/dev/null)
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

run_id=""
mode="auto"
num_files=""
state_dir=""
config_dir=""
aggregated_subdir=""
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
        --aggregated_subdir)
            aggregated_subdir="${2:-}"
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
if ! [[ "${effective_d_min}" =~ ^[0-9]+$ && "${effective_d_max}" =~ ^[0-9]+$ && "${effective_d_step}" =~ ^[0-9]+$ ]]; then
    echo "d-range is invalid: d_min='${effective_d_min}', d_max='${effective_d_max}', d_step='${effective_d_step}'"
    exit 1
fi
if (( effective_d_step <= 0 || effective_d_max < effective_d_min )); then
    echo "Invalid d-range values: ${effective_d_min}:${effective_d_step}:${effective_d_max}"
    exit 1
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
echo "  num_files_per_d=${effective_num_files} (0=all)"
echo "  d_range=${effective_d_min}:${effective_d_step}:${effective_d_max}"
echo "  force_reaggregate=${force_reaggregate}"
echo "  dry_run=${dry_run}"
echo "  keep_going=${keep_going}"
echo "  save_tag_template=aggregated_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_d<d>"

ok_count=0
skip_count=0
fail_count=0

for d in $(seq "${effective_d_min}" "${effective_d_step}" "${effective_d_max}"); do
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

    save_tag="aggregated_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_d${d}"
    existing_agg="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag}")"
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

    mapfile -t candidates < <(
        find "${effective_state_dir}" -type f \
            -name "two_force_d${d}_${suffix}_*.jld2" \
            ! -name "*_id-aggregated_*" \
            -printf '%T@ %p\n' 2>/dev/null \
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
    trap 'rm -f "${state_list_file}"' EXIT
    printf "%s\n" "${candidates[@]:0:${selected_count}}" > "${state_list_file}"

    echo "d=${d}: aggregating ${selected_count} saved files -> save_tag=${save_tag}"
    if [[ "${dry_run}" == "true" ]]; then
        head -n 3 "${state_list_file}" | sed 's/^/  sample: /'
        rm -f "${state_list_file}"
        continue
    fi

    if bash "${RUNNER_SCRIPT}" "${runtime_config}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"; then
        saved="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag}")"
        if [[ -n "${saved}" ]]; then
            if [[ -n "${aggregated_subdir}" ]]; then
                saved="$(move_state_to_aggregated_subdir "${saved}")"
            fi
            echo "d=${d}: aggregated -> ${saved}"
            ok_count=$((ok_count + 1))
        else
            echo "d=${d}: aggregation command succeeded but no output found for save_tag=${save_tag}"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" != "true" ]]; then
                rm -f "${state_list_file}"
                exit 1
            fi
        fi
    else
        echo "d=${d}: aggregation failed."
        fail_count=$((fail_count + 1))
        if [[ "${keep_going}" != "true" ]]; then
            rm -f "${state_list_file}"
            exit 1
        fi
    fi

    rm -f "${state_list_file}"
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
