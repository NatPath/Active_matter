#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_two_force_d_run.sh --run_id <run_id> [options]

Required:
  --run_id <id>                     two_force_d run_id to aggregate

Options:
  --mode <auto|warmup|production|warmup_production>
                                    Where to resolve run_id from (default: auto)
  --num_replicas <int>              Override replicas count (default: from run_info)
  --state_dir <path>                Override state directory (default: from run_info)
  --config_dir <path>               Override config directory (default: from run_info)
  --d_min <int>                     Override d_min (default: from run_info)
  --d_max <int>                     Override d_max (default: from run_info)
  --d_step <int>                    Override d_step (default: from run_info)
  --force                           Re-aggregate even if aggregated state already exists
  --dry_run                         Print actions without running aggregation
  --keep_going                      Continue on per-d errors instead of exiting
  -h, --help                        Show this help

Behavior:
  - Resolves run_info from runs/two_force_d/<mode>/<run_id>/run_info.txt.
  - If run_id is a warmup_production chain run_id, auto-resolves production_run_id/run_info.
  - For each d, calls aggregate_replicas_from_tags.sh with:
      replica_tag_prefix=replica_<run_id>_d<d>_r
      save_tag=aggregated_<run_id>_d<d>
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

AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_replicas_from_tags.sh"
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing helper script: ${AGGREGATE_SCRIPT}"
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

has_all_replica_states_for_prefix() {
    local root_dir="$1"
    local prefix="$2"
    local n_replicas="$3"
    local idx
    for ((idx = 1; idx <= n_replicas; idx++)); do
        if [[ -z "$(latest_state_for_id_tag "${root_dir}" "${prefix}${idx}")" ]]; then
            return 1
        fi
    done
    return 0
}

detect_replica_prefix_for_d() {
    local root_dir="$1"
    local d_val="$2"
    local line
    local id_tag
    local prefix

    while IFS= read -r -d '' line; do
        id_tag="$(basename "${line}")"
        id_tag="${id_tag%.jld2}"
        id_tag="${id_tag##*_id-}"
        if [[ "${id_tag}" =~ ^(replica_.*_d${d_val}_r)[0-9]+$ ]]; then
            prefix="${BASH_REMATCH[1]}"
            printf "%s\n" "${prefix}"
        fi
    done < <(find "${root_dir}" -type f -name "*_id-replica_*_d${d_val}_r*.jld2" -print0 2>/dev/null)
}

make_short_tag_base() {
    local raw="$1"
    local cleaned hash
    cleaned="$(printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
    if (( ${#cleaned} > 40 )); then
        cleaned="${cleaned:0:40}"
    fi
    hash="$(printf "%s" "${raw}" | cksum | awk '{print $1}')"
    printf "%s_%s" "${cleaned}" "${hash}"
}

collect_latest_states_for_d() {
    local root_dir="$1"
    local d_val="$2"
    local mode_val="$3"
    local needed="$4"
    local desc_tag
    local pattern_desc
    local -a picked=()
    local line path

    if [[ "${mode_val}" == "warmup" ]]; then
        desc_tag="warmup"
    else
        desc_tag="prod"
    fi
    pattern_desc="two_force_d${d_val}_${desc_tag}_*.jld2"

    while IFS= read -r line; do
        path="${line#* }"
        picked+=("${path}")
        if (( ${#picked[@]} >= needed )); then
            break
        fi
    done < <(find "${root_dir}" -type f -name "${pattern_desc}" -printf '%T@ %p\n' 2>/dev/null | sort -nr)

    if (( ${#picked[@]} < needed )); then
        while IFS= read -r line; do
            path="${line#* }"
            if [[ "${path}" == *"_id-aggregated_"* ]]; then
                continue
            fi
            if [[ "${path}" == *"_id-replica_"* ]]; then
                continue
            fi
            picked+=("${path}")
            if (( ${#picked[@]} >= needed )); then
                break
            fi
        done < <(find "${root_dir}" -type f -name "*_fdist-${d_val}_*.jld2" -printf '%T@ %p\n' 2>/dev/null | sort -nr)
    fi

    printf "%s\n" "${picked[@]}"
}

find_run_info_by_run_id() {
    local lookup_run_id="$1"
    local mode_hint="$2"
    local candidate=""

    if [[ "${mode_hint}" == "warmup" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup/${lookup_run_id}/run_info.txt"
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    fi
    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/production/${lookup_run_id}/run_info.txt"
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    fi
    if [[ "${mode_hint}" == "warmup_production" || "${mode_hint}" == "auto" ]]; then
        candidate="${REPO_ROOT}/runs/two_force_d/warmup_production/${lookup_run_id}/run_info.txt"
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    fi

    local registry_file="${REPO_ROOT}/runs/two_force_d/run_registry.csv"
    if [[ -f "${registry_file}" ]]; then
        local registry_row
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${registry_file}")"
        if [[ -n "${registry_row}" ]]; then
            local reg_run_root
            IFS=',' read -r _ts _rid _mode _L _rho _ns _dmin _dmax _dstep _cpus _mem reg_run_root _log_dir _state_dir _warmup_state_dir <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

run_id=""
mode="auto"
num_replicas=""
state_dir=""
config_dir=""
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
        --num_replicas)
            num_replicas="${2:-}"
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
    echo "Resolved target mode '${target_mode}' is not supported for aggregation."
    echo "Target run_info: ${target_run_info}"
    exit 1
fi
target_replica_strategy="$(read_run_info_value "${target_run_info}" "replica_strategy")"

run_root="$(read_run_info_value "${target_run_info}" "run_root")"

effective_state_dir="${state_dir:-$(read_run_info_value "${target_run_info}" "state_dir")}"
if [[ -z "${effective_state_dir}" && -n "${run_root}" ]]; then
    effective_state_dir="${run_root}/states"
fi
if [[ -z "${effective_state_dir}" || ! -d "${effective_state_dir}" ]]; then
    echo "State directory is invalid: ${effective_state_dir}"
    exit 1
fi

effective_config_dir="${config_dir:-$(read_run_info_value "${target_run_info}" "config_dir")}"
if [[ -z "${effective_config_dir}" && -n "${run_root}" ]]; then
    effective_config_dir="${run_root}/configs"
fi
if [[ -z "${effective_config_dir}" || ! -d "${effective_config_dir}" ]]; then
    echo "Config directory is invalid: ${effective_config_dir}"
    exit 1
fi

effective_num_replicas="${num_replicas:-$(read_run_info_value "${target_run_info}" "num_replicas")}"
if [[ -z "${effective_num_replicas}" ]]; then
    inferred_nr="$(printf "%s" "${target_run_id}" | sed -nE 's/.*_nr([0-9]+)_.*/\1/p' | tail -n 1)"
    if [[ -n "${inferred_nr}" ]]; then
        effective_num_replicas="${inferred_nr}"
    else
        echo "WARNING: num_replicas missing in run_info; defaulting to 1."
        effective_num_replicas="1"
    fi
fi
if ! [[ "${effective_num_replicas}" =~ ^[0-9]+$ ]] || (( effective_num_replicas <= 0 )); then
    echo "num_replicas is invalid: ${effective_num_replicas}"
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

echo "Resolved aggregation target:"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_mode=${target_mode}"
echo "  run_info=${target_run_info}"
echo "  state_dir=${effective_state_dir}"
echo "  config_dir=${effective_config_dir}"
echo "  num_replicas=${effective_num_replicas}"
echo "  replica_strategy=${target_replica_strategy:-unknown}"
echo "  d_range=${effective_d_min}:${effective_d_step}:${effective_d_max}"
echo "  force_reaggregate=${force_reaggregate}"
echo "  dry_run=${dry_run}"
echo "  keep_going=${keep_going}"

if (( effective_num_replicas == 1 )); then
    echo "num_replicas=1 for this run; aggregation is usually not needed."
fi
if [[ "${target_replica_strategy}" == "mp" && "${effective_num_replicas}" -gt 1 ]]; then
    echo "WARNING: resolved run uses replica_strategy=mp; per-replica files may not exist."
    echo "WARNING: this recovery script expects replica-tagged files (typically DAG runs)."
fi

ok_count=0
skip_count=0
fail_count=0
agg_tag_base="$(make_short_tag_base "${target_run_id}")"

for d in $(seq "${effective_d_min}" "${effective_d_step}" "${effective_d_max}"); do
    runtime_config="${effective_config_dir}/d_${d}.yaml"
    if [[ ! -f "${runtime_config}" ]]; then
        fallback_cfg="${REPO_ROOT}/configuration_files/two_force_d_sweep/${target_mode}/d_${d}.yaml"
        if [[ -f "${fallback_cfg}" ]]; then
            runtime_config="${fallback_cfg}"
        else
            echo "Missing config for d=${d}: ${effective_config_dir}/d_${d}.yaml"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" == "true" ]]; then
                continue
            fi
            exit 1
        fi
    fi

    save_tag_short="aggregated_${agg_tag_base}_d${d}"
    save_tag_legacy="aggregated_${target_run_id}_d${d}"
    save_tag="${save_tag_short}"
    existing_agg="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag_short}")"
    if [[ -z "${existing_agg}" ]]; then
        existing_agg="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag_legacy}")"
    fi
    if [[ -n "${existing_agg}" && "${force_reaggregate}" != "true" ]]; then
        echo "d=${d}: skip (already exists) -> ${existing_agg}"
        skip_count=$((skip_count + 1))
        continue
    fi

    replica_prefix_default="replica_${target_run_id}_d${d}_r"
    replica_prefix="${replica_prefix_default}"
    use_tag_mode="true"
    if ! has_all_replica_states_for_prefix "${effective_state_dir}" "${replica_prefix}" "${effective_num_replicas}"; then
        detected_prefix="$(
            detect_replica_prefix_for_d "${effective_state_dir}" "${d}" \
                | sort \
                | uniq -c \
                | sort -nr \
                | awk 'NR==1 {print $2}'
        )"
        if [[ -n "${detected_prefix}" ]] && has_all_replica_states_for_prefix "${effective_state_dir}" "${detected_prefix}" "${effective_num_replicas}"; then
            if [[ "${detected_prefix}" != "${replica_prefix_default}" ]]; then
                echo "d=${d}: using auto-detected replica prefix '${detected_prefix}' (default not found)."
            fi
            replica_prefix="${detected_prefix}"
        fi
    fi

    if ! has_all_replica_states_for_prefix "${effective_state_dir}" "${replica_prefix}" "${effective_num_replicas}"; then
        use_tag_mode="false"
    fi

    if [[ "${use_tag_mode}" == "true" ]]; then
        if [[ "${replica_prefix}" =~ ^replica_(.*)_d${d}_r$ ]]; then
            save_tag="aggregated_${BASH_REMATCH[1]}_d${d}"
        fi
        if (( ${#save_tag} > 90 )); then
            save_tag="${save_tag_short}"
        fi
        cmd=(
            bash "${AGGREGATE_SCRIPT}"
            --config "${runtime_config}"
            --state_dir "${effective_state_dir}"
            --num_replicas "${effective_num_replicas}"
            --replica_tag_prefix "${replica_prefix}"
            --save_tag "${save_tag}"
        )
        echo "d=${d}: ${cmd[*]}"
        if [[ "${dry_run}" == "true" ]]; then
            continue
        fi

        if "${cmd[@]}"; then
            saved="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag}")"
            if [[ -n "${saved}" ]]; then
                echo "d=${d}: aggregated -> ${saved}"
                ok_count=$((ok_count + 1))
            else
                echo "d=${d}: aggregation command succeeded but no aggregated file found for save_tag=${save_tag}"
                fail_count=$((fail_count + 1))
                if [[ "${keep_going}" != "true" ]]; then
                    exit 1
                fi
            fi
        else
            echo "d=${d}: aggregation command failed."
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" != "true" ]]; then
                exit 1
            fi
        fi
        continue
    fi

    echo "d=${d}: no replica-tagged files found; falling back to d-pattern selection."
    selected_states="$(collect_latest_states_for_d "${effective_state_dir}" "${d}" "${target_mode}" "${effective_num_replicas}")"
    selected_count="$(printf "%s\n" "${selected_states}" | sed '/^$/d' | wc -l | awk '{print $1}')"
    if (( selected_count < effective_num_replicas )); then
        echo "d=${d}: fallback found only ${selected_count}/${effective_num_replicas} states."
        echo "d=${d}: looked for patterns like two_force_d${d}_*_*.jld2 and *_fdist-${d}_*.jld2."
        fail_count=$((fail_count + 1))
        if [[ "${keep_going}" == "true" ]]; then
            continue
        fi
        exit 1
    fi

    state_list_file="$(mktemp)"
    printf "%s\n" "${selected_states}" | sed '/^$/d' | head -n "${effective_num_replicas}" > "${state_list_file}"
    echo "d=${d}: aggregating ${effective_num_replicas} states via fallback selection (save_tag=${save_tag})."
    if [[ "${dry_run}" == "true" ]]; then
        rm -f "${state_list_file}"
        continue
    fi

    if bash "${RUNNER_SCRIPT}" "${runtime_config}" --aggregate_state_list "${state_list_file}" --save_tag "${save_tag}"; then
        saved="$(latest_state_for_id_tag "${effective_state_dir}" "${save_tag}")"
        if [[ -n "${saved}" ]]; then
            echo "d=${d}: aggregated -> ${saved}"
            ok_count=$((ok_count + 1))
        else
            echo "d=${d}: aggregation command succeeded but no aggregated file found for save_tag=${save_tag}"
            fail_count=$((fail_count + 1))
            if [[ "${keep_going}" != "true" ]]; then
                exit 1
            fi
        fi
    else
        echo "d=${d}: fallback aggregation command failed."
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
echo "  success=${ok_count}"
echo "  skipped_existing=${skip_count}"
echo "  failed=${fail_count}"

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi

if (( fail_count > 0 )); then
    exit 1
fi
