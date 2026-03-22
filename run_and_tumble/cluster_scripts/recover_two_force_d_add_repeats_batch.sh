#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash recover_two_force_d_add_repeats_batch.sh \
      --run_id <id> \
      --repeat_batch <token-or-path> \
      --d_values <csv> \
      [options]

Required:
  --run_id <id>                      Existing two_force_d production run_id or chain run_id
  --repeat_batch <token-or-path>     Repeat-batch token under <state_dir>/<raw_subdir>/,
                                     or an explicit directory path
  --d_values <csv>                   Comma-separated d values to recover, for example: 24,64,96,128

Options:
  --cluster_repo_root <path>         Target cluster checkout to patch/recover
                                     (default: /storage/ph_kafri/nativmr/run_and_tumble)
  --mode <auto|production|warmup_production>
                                     How to resolve --run_id on the target checkout (default: auto)
  --raw_subdir <name>                Raw repeat-batch subdir under state_dir (default: repeat_batches)
  --aggregated_subdir <name>         Aggregated output subdir under state_dir (default: aggregated)
  --archive_subdir <name>            Archive subdir under aggregated_subdir (default: archive)
  --expected_batch_repeats <int>     Forwarded to recovery script (default: 0)
  --request_cpus <int>               Forwarded to recovery script
  --request_memory <value>           Forwarded to recovery script
  --julia_num_procs_aggregate <int>  Forwarded to recovery script
  --batch_name <name>                Forwarded to recovery script
  --job_label <label>                Forwarded to recovery script
  --archive_stamp <token>            Forwarded to recovery script
  --no_submit                        Generate recovery submit files only; do not submit
  --dry_run                          Print actions without changing files or submitting
  -h, --help                         Show help

Behavior:
  1) Resolves the production state_dir for --run_id inside the target cluster checkout.
  2) Moves zero-byte .jld2 files from the repeat batch into:
       <repeat_batch_dir>/quarantined_zero_size_<timestamp>/
  3) Copies the locally patched files from the current checkout into the target checkout:
       run_diffusive_no_activity.jl
       cluster_scripts/aggregate_two_force_d_saved_files.sh
       cluster_scripts/recover_two_force_d_missing_aggregates.sh
  4) Runs recover_two_force_d_missing_aggregates.sh on the target checkout.

Typical use for "best available non-zero files":
  bash recover_two_force_d_add_repeats_batch.sh \
      --run_id <id> \
      --repeat_batch <token> \
      --d_values 24,64,96,128 \
      --expected_batch_repeats 0
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    SOURCE_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    SOURCE_REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate source repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

canonical_path() {
    local raw_path="$1"
    if [[ -d "${raw_path}" ]]; then
        (cd "${raw_path}" && pwd -P)
    else
        local parent_dir
        parent_dir="$(cd "$(dirname "${raw_path}")" && pwd -P)"
        printf "%s/%s\n" "${parent_dir}" "$(basename "${raw_path}")"
    fi
}

resolve_target_script_path() {
    local repo_root="$1"
    local script_name="$2"
    local preferred_path="${repo_root}/cluster_scripts/${script_name}"
    local flat_path="${repo_root}/${script_name}"

    if [[ -e "${preferred_path}" ]]; then
        echo "${preferred_path}"
        return 0
    fi
    if [[ -e "${flat_path}" ]]; then
        echo "${flat_path}"
        return 0
    fi

    # Default to cluster_scripts/ for new files when neither path exists yet.
    echo "${preferred_path}"
}

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

find_run_info_by_run_id() {
    local repo_root="$1"
    local lookup_run_id="$2"
    local mode_hint="$3"
    local candidate=""
    local registry_file="${repo_root}/runs/two_force_d/run_registry.csv"

    if [[ "${mode_hint}" == "warmup" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/warmup/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "warmup_production" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/warmup_production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

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

resolve_target_production_run_info() {
    local repo_root="$1"
    local lookup_run_id="$2"
    local mode_hint="$3"
    local resolved_info resolved_mode production_run_info production_run_id

    resolved_info="$(find_run_info_by_run_id "${repo_root}" "${lookup_run_id}" "${mode_hint}" || true)"
    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve run_info for run_id='${lookup_run_id}' (mode=${mode_hint})." >&2
        return 1
    fi

    resolved_mode="$(read_run_info_value "${resolved_info}" "mode")"
    if [[ "${resolved_mode}" == "warmup_production" ]]; then
        production_run_info="$(read_run_info_value "${resolved_info}" "production_run_info")"
        production_run_id="$(read_run_info_value "${resolved_info}" "production_run_id")"
        if [[ -n "${production_run_info}" && -f "${production_run_info}" ]]; then
            resolved_info="${production_run_info}"
        elif [[ -n "${production_run_id}" ]]; then
            resolved_info="$(find_run_info_by_run_id "${repo_root}" "${production_run_id}" "production" || true)"
        fi
    fi

    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve production run_info for run_id='${lookup_run_id}'." >&2
        return 1
    fi
    if [[ "$(read_run_info_value "${resolved_info}" "mode")" != "production" ]]; then
        echo "run_id='${lookup_run_id}' does not resolve to a production run." >&2
        return 1
    fi

    echo "${resolved_info}"
}

resolve_repeat_batch_dir() {
    local repeat_batch_raw="$1"
    local state_dir_root="$2"
    local raw_subdir_name="$3"
    local candidate=""

    if [[ -d "${repeat_batch_raw}" ]]; then
        canonical_path "${repeat_batch_raw}"
        return 0
    fi

    if [[ "${repeat_batch_raw}" == /* ]]; then
        echo "Repeat batch directory does not exist: ${repeat_batch_raw}" >&2
        return 1
    fi

    candidate="${state_dir_root}/${raw_subdir_name}/${repeat_batch_raw}"
    if [[ -d "${candidate}" ]]; then
        echo "${candidate}"
        return 0
    fi

    candidate="${state_dir_root}/${repeat_batch_raw}"
    if [[ -d "${candidate}" ]]; then
        echo "${candidate}"
        return 0
    fi

    echo "Could not resolve repeat batch '${repeat_batch_raw}' under ${state_dir_root}." >&2
    return 1
}

copy_if_needed() {
    local src="$1"
    local dst="$2"
    local src_abs dst_abs

    src_abs="$(canonical_path "${src}")"
    dst_abs="$(canonical_path "${dst}")"
    if [[ "${src_abs}" == "${dst_abs}" ]]; then
        echo "Sync skip (same path): ${dst_abs}"
        return 0
    fi

    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
    echo "Synced: ${src_abs} -> ${dst_abs}"
}

run_id=""
repeat_batch=""
d_values_csv=""
cluster_repo_root="/storage/ph_kafri/nativmr/run_and_tumble"
mode="auto"
raw_subdir="repeat_batches"
aggregated_subdir="aggregated"
archive_subdir="archive"
expected_batch_repeats="0"
request_cpus=""
request_memory=""
julia_num_procs_aggregate=""
batch_name=""
job_label=""
archive_stamp=""
no_submit="false"
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --repeat_batch)
            repeat_batch="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --cluster_repo_root)
            cluster_repo_root="${2:-}"
            shift 2
            ;;
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --raw_subdir)
            raw_subdir="${2:-}"
            shift 2
            ;;
        --aggregated_subdir)
            aggregated_subdir="${2:-}"
            shift 2
            ;;
        --archive_subdir)
            archive_subdir="${2:-}"
            shift 2
            ;;
        --expected_batch_repeats)
            expected_batch_repeats="${2:-}"
            shift 2
            ;;
        --request_cpus)
            request_cpus="${2:-}"
            shift 2
            ;;
        --request_memory)
            request_memory="${2:-}"
            shift 2
            ;;
        --julia_num_procs_aggregate)
            julia_num_procs_aggregate="${2:-}"
            shift 2
            ;;
        --batch_name)
            batch_name="${2:-}"
            shift 2
            ;;
        --job_label)
            job_label="${2:-}"
            shift 2
            ;;
        --archive_stamp)
            archive_stamp="${2:-}"
            shift 2
            ;;
        --no_submit)
            no_submit="true"
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

if [[ -z "${run_id}" || -z "${repeat_batch}" || -z "${d_values_csv}" ]]; then
    echo "--run_id, --repeat_batch, and --d_values are required."
    usage
    exit 1
fi

case "${mode}" in
    auto|production|warmup_production)
        ;;
    *)
        echo "--mode must be one of: auto, production, warmup_production."
        exit 1
        ;;
esac

if [[ ! -d "${cluster_repo_root}" ]]; then
    echo "Cluster repo root does not exist: ${cluster_repo_root}"
    exit 1
fi
if ! [[ "${expected_batch_repeats}" =~ ^[0-9]+$ ]]; then
    echo "--expected_batch_repeats must be an integer >= 0. Got '${expected_batch_repeats}'."
    exit 1
fi

target_run_info="$(resolve_target_production_run_info "${cluster_repo_root}" "${run_id}" "${mode}")"
target_run_id="$(read_run_info_value "${target_run_info}" "run_id")"
run_root="$(read_run_info_value "${target_run_info}" "run_root")"
target_state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
[[ -z "${target_state_dir}" && -n "${run_root}" ]] && target_state_dir="${run_root}/states"
if [[ -z "${target_state_dir}" || ! -d "${target_state_dir}" ]]; then
    echo "State directory is invalid: ${target_state_dir}"
    exit 1
fi

repeat_batch_dir="$(resolve_repeat_batch_dir "${repeat_batch}" "${target_state_dir}" "${raw_subdir}")"
timestamp="$(date +%Y%m%d-%H%M%S)"
quarantine_dir="${repeat_batch_dir}/quarantined_zero_size_${timestamp}"

runtime_src="${SOURCE_REPO_ROOT}/run_diffusive_no_activity.jl"
aggregate_src="${SOURCE_REPO_ROOT}/cluster_scripts/aggregate_two_force_d_saved_files.sh"
recover_src="${SOURCE_REPO_ROOT}/cluster_scripts/recover_two_force_d_missing_aggregates.sh"
batch_helper_src="${SOURCE_REPO_ROOT}/cluster_scripts/recover_two_force_d_add_repeats_batch.sh"

runtime_dst="${cluster_repo_root}/run_diffusive_no_activity.jl"
aggregate_dst="$(resolve_target_script_path "${cluster_repo_root}" "aggregate_two_force_d_saved_files.sh")"
recover_dst="$(resolve_target_script_path "${cluster_repo_root}" "recover_two_force_d_missing_aggregates.sh")"
batch_helper_dst="$(resolve_target_script_path "${cluster_repo_root}" "recover_two_force_d_add_repeats_batch.sh")"

echo "Resolved recovery flow:"
echo "  source_repo_root=${SOURCE_REPO_ROOT}"
echo "  cluster_repo_root=${cluster_repo_root}"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${target_run_info}"
echo "  target_state_dir=${target_state_dir}"
echo "  repeat_batch_dir=${repeat_batch_dir}"
echo "  quarantine_dir=${quarantine_dir}"
echo "  d_values=${d_values_csv}"
echo "  expected_batch_repeats=${expected_batch_repeats}"
echo "  no_submit=${no_submit}"
echo "  dry_run=${dry_run}"

mapfile -t zero_files < <(find "${repeat_batch_dir}" -maxdepth 1 -type f -name '*.jld2' -size 0 | sort)
if (( ${#zero_files[@]} == 0 )); then
    echo "Zero-size batch files: none"
else
    echo "Zero-size batch files (${#zero_files[@]}):"
    printf "  %s\n" "${zero_files[@]}"
fi

if [[ "${dry_run}" == "true" ]]; then
    echo "Would sync:"
    echo "  ${runtime_src} -> ${runtime_dst}"
    echo "  ${aggregate_src} -> ${aggregate_dst}"
    echo "  ${recover_src} -> ${recover_dst}"
    echo "  ${batch_helper_src} -> ${batch_helper_dst}"
else
    if (( ${#zero_files[@]} > 0 )); then
        mkdir -p "${quarantine_dir}"
        for zero_file in "${zero_files[@]}"; do
            mv -v "${zero_file}" "${quarantine_dir}/"
        done
    fi

    copy_if_needed "${runtime_src}" "${runtime_dst}"
    copy_if_needed "${aggregate_src}" "${aggregate_dst}"
    copy_if_needed "${recover_src}" "${recover_dst}"
    copy_if_needed "${batch_helper_src}" "${batch_helper_dst}"
    chmod +x "${aggregate_dst}" "${recover_dst}" "${batch_helper_dst}"
fi

recovery_cmd=(
    bash "${recover_dst}"
    --run_id "${target_run_id}"
    --mode production
    --repeat_batch "${repeat_batch_dir}"
    --d_values "${d_values_csv}"
    --raw_subdir "${raw_subdir}"
    --aggregated_subdir "${aggregated_subdir}"
    --archive_subdir "${archive_subdir}"
    --expected_batch_repeats "${expected_batch_repeats}"
)

if [[ -n "${request_cpus}" ]]; then
    recovery_cmd+=(--request_cpus "${request_cpus}")
fi
if [[ -n "${request_memory}" ]]; then
    recovery_cmd+=(--request_memory "${request_memory}")
fi
if [[ -n "${julia_num_procs_aggregate}" ]]; then
    recovery_cmd+=(--julia_num_procs_aggregate "${julia_num_procs_aggregate}")
fi
if [[ -n "${batch_name}" ]]; then
    recovery_cmd+=(--batch_name "${batch_name}")
fi
if [[ -n "${job_label}" ]]; then
    recovery_cmd+=(--job_label "${job_label}")
fi
if [[ -n "${archive_stamp}" ]]; then
    recovery_cmd+=(--archive_stamp "${archive_stamp}")
fi
if [[ "${no_submit}" == "true" ]]; then
    recovery_cmd+=(--no_submit)
fi

echo "Recovery command:"
printf '  %q' "${recovery_cmd[@]}"
printf '\n'

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi

(
    cd "${cluster_repo_root}"
    "${recovery_cmd[@]}"
)
