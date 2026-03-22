#!/usr/bin/env bash
set -euo pipefail

script_start_epoch="$(date +%s.%N)"
script_start_epoch_int="${script_start_epoch%.*}"
script_start_stamp="$(date +%Y%m%d-%H%M%S)"

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_two_force_d_partial_snapshot.sh --run_id <run_id> [options]

Required:
  --run_id <id>                     two_force_d run_id (typically production)

Options:
  --mode <auto|warmup|production|warmup_production>
                                    Where to resolve run_id from (default: auto)
  --num_files <int>                 Max files per d to aggregate from the snapshot;
                                    0 means all available at snapshot time (default: 0)
  --state_dir <path>                Override state directory
  --config_dir <path>               Override config directory
  --d_min <int>                     Override d_min
  --d_max <int>                     Override d_max
  --d_step <int>                    Override d_step
  --keep_going                      Continue on per-d aggregation errors
  --dry_run                         Print actions only
  -h, --help                        Show help

Behavior:
  - Captures a fixed cutoff time at script start.
  - Aggregates only replica state files with mtime <= cutoff time.
  - Never uses files with *_id-aggregated_* in the name.
  - Saves outputs under:
      <state_dir>/aggregated_partial
  - Uses save tags prefixed with aggregated_partial_*, so it does not
    collide with DAG final aggregation tags (aggregated_saved_*).
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

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

run_id=""
mode="auto"
num_files="0"
state_dir=""
config_dir=""
d_min=""
d_max=""
d_step=""
keep_going="false"
dry_run="false"

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
        --keep_going)
            keep_going="true"
            shift
            ;;
        --dry_run)
            dry_run="true"
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

if ! [[ "${num_files}" =~ ^[0-9]+$ ]]; then
    echo "--num_files must be a non-negative integer. Got '${num_files}'."
    exit 1
fi
for numeric_name in d_min d_max d_step; do
    numeric_value="${!numeric_name}"
    if [[ -n "${numeric_value}" ]] && ! [[ "${numeric_value}" =~ ^[0-9]+$ ]]; then
        echo "--${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done

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

effective_config_dir="${config_dir:-$(read_run_info_value "${target_run_info}" "config_dir")}"
[[ -z "${effective_config_dir}" && -n "${run_root}" ]] && effective_config_dir="${run_root}/configs"
if [[ -z "${effective_config_dir}" || ! -d "${effective_config_dir}" ]]; then
    echo "Config directory is invalid: ${effective_config_dir}"
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

partial_subdir="${effective_state_dir}/aggregated_partial"
if [[ "${dry_run}" != "true" ]]; then
    mkdir -p "${partial_subdir}"
fi

suffix="prod"
if [[ "${target_mode}" == "warmup" ]]; then
    suffix="warmup"
fi

run_hash="$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
L_val="$(read_run_info_value "${target_run_info}" "L")"
rho_val="$(read_run_info_value "${target_run_info}" "rho0")"
rho_tag="$(sanitize_token "${rho_val}")"
[[ -z "${L_val}" ]] && L_val="NA"
[[ -z "${rho_tag}" ]] && rho_tag="NA"

echo "Resolved partial aggregation target:"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_mode=${target_mode}"
echo "  run_info=${target_run_info}"
echo "  state_dir=${effective_state_dir}"
echo "  output_dir=${partial_subdir}"
echo "  config_dir=${effective_config_dir}"
echo "  num_files_per_d=${num_files} (0=all available at snapshot)"
echo "  d_range=${effective_d_min}:${effective_d_step}:${effective_d_max}"
echo "  snapshot_epoch=${script_start_epoch}"
echo "  snapshot_stamp=${script_start_stamp}"
echo "  keep_going=${keep_going}"
echo "  dry_run=${dry_run}"
echo "  save_tag_template=aggregated_partial_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_snap${script_start_stamp}_d<d>"

ok_count=0
skip_count=0
fail_count=0
temp_files=()
cleanup() {
    for f in "${temp_files[@]}"; do
        [[ -f "${f}" ]] && rm -f "${f}"
    done
}
trap cleanup EXIT

for d in $(seq "${effective_d_min}" "${effective_d_step}" "${effective_d_max}"); do
    runtime_config="${effective_config_dir}/d_${d}.yaml"
    if [[ ! -f "${runtime_config}" ]]; then
        fallback_cfg="${REPO_ROOT}/configuration_files/two_force_d_sweep/${target_mode}/d_${d}.yaml"
        if [[ -f "${fallback_cfg}" ]]; then
            runtime_config="${fallback_cfg}"
        else
            echo "d=${d}: missing config ${effective_config_dir}/d_${d}.yaml"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" == "true" ]]; then
                continue
            fi
            exit 1
        fi
    fi

    mapfile -t candidates < <(
        find "${effective_state_dir}" -type f \
            -name "two_force_d${d}_${suffix}_*.jld2" \
            ! -name "*_id-aggregated_*" \
            -printf '%T@ %p\n' 2>/dev/null \
            | awk -v cutoff="${script_start_epoch}" '$1 <= cutoff' \
            | sort -nr \
            | awk '{ $1=""; sub(/^ /,""); print }'
    )

    if (( ${#candidates[@]} == 0 )); then
        echo "d=${d}: skip (no pre-snapshot files found)."
        skip_count=$((skip_count + 1))
        continue
    fi

    selected_count="${#candidates[@]}"
    if (( num_files > 0 && num_files < selected_count )); then
        selected_count="${num_files}"
    fi

    state_list_file="$(mktemp)"
    temp_files+=("${state_list_file}")
    printf "%s\n" "${candidates[@]:0:${selected_count}}" > "${state_list_file}"

    save_tag="aggregated_partial_saved_L${L_val}_rho${rho_tag}_rid${run_hash}_snap${script_start_stamp}_d${d}"
    echo "d=${d}: aggregating ${selected_count} files -> save_tag=${save_tag}"

    if [[ "${dry_run}" == "true" ]]; then
        head -n 3 "${state_list_file}" | sed 's/^/  sample: /'
        continue
    fi

    if bash "${RUNNER_SCRIPT}" "${runtime_config}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"; then
        saved="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag}")"
        if [[ -z "${saved}" ]]; then
            echo "d=${d}: aggregation command succeeded but no output found for save_tag=${save_tag}"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" != "true" ]]; then
                exit 1
            fi
            continue
        fi
        target_file="${partial_subdir}/$(basename "${saved}")"
        if [[ "${saved}" != "${target_file}" ]]; then
            mv -f "${saved}" "${target_file}"
            saved="${target_file}"
        fi
        echo "d=${d}: aggregated_partial -> ${saved}"
        ok_count=$((ok_count + 1))
    else
        echo "d=${d}: aggregation failed."
        fail_count=$((fail_count + 1))
        if [[ "${keep_going}" != "true" ]]; then
            exit 1
        fi
    fi
done

echo "Partial aggregation summary:"
echo "  target_run_id=${target_run_id}"
echo "  snapshot_epoch=${script_start_epoch}"
echo "  snapshot_epoch_int=${script_start_epoch_int}"
echo "  output_dir=${partial_subdir}"
echo "  success=${ok_count}"
echo "  skipped_no_presnapshot_files=${skip_count}"
echo "  failed=${fail_count}"

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi
if (( fail_count > 0 )); then
    exit 1
fi
