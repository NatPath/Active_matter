#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash recover_two_force_d_missing_aggregates.sh \
      --run_id <id> \
      --repeat_batch <token-or-path> \
      --d_values <csv> \
      [options]

Required:
  --run_id <id>                      Existing two_force_d production run_id or chain run_id
  --repeat_batch <token-or-path>     Repeat-batch token under <state_dir>/<raw_subdir>/,
                                     an explicit directory path, or 'auto' if exactly one batch exists
  --d_values <csv>                   Comma-separated d values to recover, for example: 64,96,128

Options:
  --mode <auto|production|warmup_production>
                                     How to resolve --run_id (default: auto)
  --state_dir <path>                 Override resolved production state_dir
  --config_dir <path>                Override resolved production config_dir
  --raw_subdir <name>                Raw add-repeat subdir under state_dir (default: repeat_batches)
  --aggregated_subdir <name>         Aggregated output subdir under state_dir (default: aggregated)
  --archive_subdir <name>            Archive subdir under aggregated_subdir (default: archive)
  --expected_batch_repeats <int>     Expected raw files per d inside repeat_batch; 0 disables the exact-count check
                                     (default: 300)
  --archive_stamp <token>            Archive stamp token (default: recover_missing_agg_<timestamp>)
  --request_cpus <int>               Condor request_cpus for aggregation jobs (default: 1)
  --request_memory <value>           Condor request_memory for aggregation jobs (default: "5 GB")
  --julia_num_procs_aggregate <int>  JULIA_NUM_PROCS_AGGREGATE for aggregation jobs (default: 1)
  --batch_name <name>                Condor batch_name (default: auto)
  --job_label <label>                Optional label added to the recovery job folder name
  --no_submit                        Generate submit artifacts only; do not call condor_submit
  -h, --help                         Show help

Behavior:
  - Resolves the production run_info from --run_id.
  - Resolves the repeat batch directory.
  - Verifies the repeat batch contains the expected raw files for each requested d.
  - Submits one Condor aggregation job per d.
  - Each aggregation job archives the old aggregate under:
      <state_dir>/<aggregated_subdir>/<archive_subdir>/<archive_stamp>/d_<d>/
    and writes the refreshed aggregate under:
      <state_dir>/<aggregated_subdir>/
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

AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_two_force_d_saved_files.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/two_force_d/run_registry.csv"

if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing aggregation script: ${AGGREGATE_SCRIPT}"
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

    if [[ -f "${REGISTRY_FILE}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${REGISTRY_FILE}")"
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
    local lookup_run_id="$1"
    local mode_hint="$2"
    local resolved_info resolved_mode production_run_info production_run_id

    resolved_info="$(find_run_info_by_run_id "${lookup_run_id}" "${mode_hint}" || true)"
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
            resolved_info="$(find_run_info_by_run_id "${production_run_id}" "production" || true)"
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

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

trim_spaces() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

parse_d_values_csv() {
    local raw_csv="$1"
    local -n out_ref="$2"
    local seen=" "
    local raw value

    out_ref=()
    IFS=',' read -r -a raw_values <<< "${raw_csv}"
    for raw in "${raw_values[@]}"; do
        value="$(trim_spaces "${raw}")"
        [[ -z "${value}" ]] && continue
        if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
            echo "Invalid d value '${value}' in --d_values." >&2
            return 1
        fi
        if [[ "${seen}" != *" ${value} "* ]]; then
            out_ref+=("${value}")
            seen="${seen}${value} "
        fi
    done

    if (( ${#out_ref[@]} == 0 )); then
        echo "--d_values did not contain any valid integers." >&2
        return 1
    fi
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

resolve_repeat_batch_dir() {
    local repeat_batch_raw="$1"
    local state_dir_root="$2"
    local raw_subdir_name="$3"
    local candidate=""
    local raw_root="${state_dir_root}/${raw_subdir_name}"
    local -a candidates=()

    if [[ "${repeat_batch_raw}" == "auto" ]]; then
        if [[ ! -d "${raw_root}" ]]; then
            echo "Raw repeat-batch root does not exist: ${raw_root}" >&2
            return 1
        fi
        mapfile -t candidates < <(find "${raw_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
        if (( ${#candidates[@]} == 1 )); then
            echo "${raw_root}/${candidates[0]}"
            return 0
        fi
        if (( ${#candidates[@]} == 0 )); then
            echo "No repeat batch directories found under ${raw_root}" >&2
        else
            echo "Multiple repeat batch directories found under ${raw_root}. Use one of:" >&2
            printf '  %s\n' "${candidates[@]}" >&2
        fi
        return 1
    fi

    if [[ -d "${repeat_batch_raw}" ]]; then
        echo "$(cd "${repeat_batch_raw}" && pwd)"
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
    if [[ -d "${raw_root}" ]]; then
        mapfile -t candidates < <(find "${raw_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort)
        if (( ${#candidates[@]} > 0 )); then
            echo "Available repeat batch tokens under ${raw_root}:" >&2
            printf '  %s\n' "${candidates[@]}" >&2
        fi
    fi
    return 1
}

run_id=""
mode="auto"
repeat_batch=""
d_values_csv=""

state_dir=""
config_dir=""
raw_subdir="repeat_batches"
aggregated_subdir="aggregated"
archive_subdir="archive"
expected_batch_repeats="300"
archive_stamp=""

request_cpus="1"
request_memory="5 GB"
julia_num_procs_aggregate="1"
batch_name=""
job_label=""
no_submit="false"

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
        --repeat_batch)
            repeat_batch="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
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
        --archive_stamp)
            archive_stamp="${2:-}"
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
        --no_submit)
            no_submit="true"
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

for token_name in raw_subdir aggregated_subdir archive_subdir; do
    token_value="${!token_name}"
    if [[ -z "${token_value}" || ! "${token_value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "--${token_name} contains unsupported characters: '${token_value}'."
        exit 1
    fi
done
if ! [[ "${expected_batch_repeats}" =~ ^[0-9]+$ ]]; then
    echo "--expected_batch_repeats must be an integer >= 0. Got '${expected_batch_repeats}'."
    exit 1
fi
if ! [[ "${request_cpus}" =~ ^[0-9]+$ ]] || (( request_cpus <= 0 )); then
    echo "--request_cpus must be a positive integer. Got '${request_cpus}'."
    exit 1
fi
if ! [[ "${julia_num_procs_aggregate}" =~ ^[0-9]+$ ]] || (( julia_num_procs_aggregate <= 0 )); then
    echo "--julia_num_procs_aggregate must be a positive integer. Got '${julia_num_procs_aggregate}'."
    exit 1
fi
if [[ -n "${archive_stamp}" ]] && ! [[ "${archive_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_stamp must match [A-Za-z0-9._-]+ when provided. Got '${archive_stamp}'."
    exit 1
fi

declare -a D_VALUES=()
parse_d_values_csv "${d_values_csv}" D_VALUES

target_run_info="$(resolve_target_production_run_info "${run_id}" "${mode}")"
target_run_id="$(read_run_info_value "${target_run_info}" "run_id")"
run_root="$(read_run_info_value "${target_run_info}" "run_root")"
target_state_dir="${state_dir:-$(read_run_info_value "${target_run_info}" "state_dir")}"
[[ -z "${target_state_dir}" && -n "${run_root}" ]] && target_state_dir="${run_root}/states"
target_config_dir="${config_dir:-$(read_run_info_value "${target_run_info}" "config_dir")}"
[[ -z "${target_config_dir}" && -n "${run_root}" ]] && target_config_dir="${run_root}/configs"

if [[ -z "${target_state_dir}" || ! -d "${target_state_dir}" ]]; then
    echo "State directory is invalid: ${target_state_dir}"
    exit 1
fi
if [[ -z "${target_config_dir}" || ! -d "${target_config_dir}" ]]; then
    echo "Config directory is invalid: ${target_config_dir}"
    exit 1
fi

repeat_batch_dir="$(resolve_repeat_batch_dir "${repeat_batch}" "${target_state_dir}" "${raw_subdir}")"
aggregated_dir="${target_state_dir}/${aggregated_subdir}"
mkdir -p "${aggregated_dir}"

run_hash="$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
timestamp="$(date +%Y%m%d-%H%M%S)"
if [[ -z "${archive_stamp}" ]]; then
    archive_stamp="recover_missing_agg_${timestamp}"
fi

safe_target_run_id="$(sanitize_token "${target_run_id}")"
if [[ -n "${job_label}" ]]; then
    job_slug="$(sanitize_token "${job_label}")"
else
    d_slug="$(IFS=_; echo "${D_VALUES[*]}")"
    job_slug="recover_d_${d_slug}_nr${expected_batch_repeats}"
fi
job_root="${REPO_ROOT}/runs/two_force_d/aggregation_jobs/${safe_target_run_id}_${job_slug}_${timestamp}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
manifest="${job_root}/preflight_manifest.csv"
job_info="${job_root}/job_info.txt"
mkdir -p "${submit_dir}" "${log_dir}"

if [[ -z "${batch_name}" ]]; then
    batch_name="two_force_d_recover_agg_${run_hash}"
fi

echo "Resolved recovery target:"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${target_run_info}"
echo "  target_state_dir=${target_state_dir}"
echo "  target_config_dir=${target_config_dir}"
echo "  repeat_batch_dir=${repeat_batch_dir}"
echo "  aggregated_dir=${aggregated_dir}"
echo "  archive_root=${aggregated_dir}/${archive_subdir}/${archive_stamp}"
echo "  d_values=$(IFS=,; echo "${D_VALUES[*]}")"
echo "  expected_batch_repeats=${expected_batch_repeats}"
echo "  request_cpus=${request_cpus}"
echo "  request_memory=${request_memory}"
echo "  JULIA_NUM_PROCS_AGGREGATE=${julia_num_procs_aggregate}"

echo "d,batch_count,total_raw_count,current_aggregate,config_path" > "${manifest}"

for d_val in "${D_VALUES[@]}"; do
    runtime_config="${target_config_dir}/d_${d_val}.yaml"
    if [[ ! -f "${runtime_config}" ]]; then
        fallback_cfg="${REPO_ROOT}/configuration_files/two_force_d_sweep/production/d_${d_val}.yaml"
        if [[ -f "${fallback_cfg}" ]]; then
            runtime_config="${fallback_cfg}"
        else
            echo "Missing production config for d=${d_val}: ${target_config_dir}/d_${d_val}.yaml"
            exit 1
        fi
    fi

    batch_count="$(
        find "${repeat_batch_dir}" -type f \
            -name "two_force_d${d_val}_prod_*.jld2" \
            ! -name "*_id-aggregated_*" \
            ! -size 0 \
            | wc -l | awk '{print $1}'
    )"
    total_raw_count="$(
        find "${target_state_dir}" -type f \
            -name "two_force_d${d_val}_prod_*.jld2" \
            ! -name "*_id-aggregated_*" \
            ! -size 0 \
            | wc -l | awk '{print $1}'
    )"
    current_aggregate="$(latest_state_for_id_tag "${aggregated_dir}" "aggregated_saved_r${run_hash}_d${d_val}")"
    current_aggregate="${current_aggregate:-NONE}"

    echo "Preflight d=${d_val}: batch_count=${batch_count} total_raw_count=${total_raw_count} current_aggregate=${current_aggregate}"
    printf "%s,%s,%s,%s,%s\n" \
        "${d_val}" "${batch_count}" "${total_raw_count}" "${current_aggregate}" "${runtime_config}" \
        >> "${manifest}"

    if (( expected_batch_repeats > 0 )) && [[ "${batch_count}" != "${expected_batch_repeats}" ]]; then
        echo "Expected ${expected_batch_repeats} repeat-batch files for d=${d_val}, found ${batch_count}. Aborting."
        exit 1
    fi
done

launcher_script="${submit_dir}/run_aggregate_one_d.sh"
submit_file="${submit_dir}/recover_missing_aggregates.sub"

{
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo 'DVAL="${1:?missing d value}"'
    echo "cd $(printf '%q' "${REPO_ROOT}")"
    echo "export JULIA_NUM_PROCS_AGGREGATE=$(printf '%q' "${julia_num_procs_aggregate}")"
    printf "bash %q --run_id %q --mode production --state_dir %q --config_dir %q --extra_raw_dir %q --aggregated_subdir %q --exclude_aggregated_inputs --incremental_from_existing_aggregate --archive_existing_aggregates --archive_subdir %q --archive_stamp %q --d_min \"\${DVAL}\" --d_max \"\${DVAL}\" --d_step 1 --num_files 0 --force\n" \
        "${AGGREGATE_SCRIPT}" \
        "${target_run_id}" \
        "${target_state_dir}" \
        "${target_config_dir}" \
        "${repeat_batch_dir}" \
        "${aggregated_subdir}" \
        "${archive_subdir}" \
        "${archive_stamp}"
} > "${launcher_script}"
chmod +x "${launcher_script}"

{
    echo "Universe   = vanilla"
    echo "Executable = /bin/bash"
    echo "arguments  = ${launcher_script} \$(DVAL)"
    echo "initialdir = ${REPO_ROOT}"
    echo "should_transfer_files = NO"
    echo "output     = ${log_dir}/d_\$(DVAL).out"
    echo "error      = ${log_dir}/d_\$(DVAL).err"
    echo "log        = ${log_dir}/d_\$(DVAL).log"
    echo "request_cpus = ${request_cpus}"
    echo "request_memory = ${request_memory}"
    echo "batch_name = ${batch_name}"
    echo "queue DVAL from ("
    for d_val in "${D_VALUES[@]}"; do
        echo "${d_val}"
    done
    echo ")"
} > "${submit_file}"

{
    echo "timestamp=${timestamp}"
    echo "requested_run_id=${run_id}"
    echo "target_run_id=${target_run_id}"
    echo "target_run_info=${target_run_info}"
    echo "target_state_dir=${target_state_dir}"
    echo "target_config_dir=${target_config_dir}"
    echo "repeat_batch=${repeat_batch}"
    echo "repeat_batch_dir=${repeat_batch_dir}"
    echo "raw_subdir=${raw_subdir}"
    echo "aggregated_subdir=${aggregated_subdir}"
    echo "archive_subdir=${archive_subdir}"
    echo "archive_stamp=${archive_stamp}"
    echo "expected_batch_repeats=${expected_batch_repeats}"
    echo "d_values=$(IFS=,; echo "${D_VALUES[*]}")"
    echo "request_cpus=${request_cpus}"
    echo "request_memory=${request_memory}"
    echo "julia_num_procs_aggregate=${julia_num_procs_aggregate}"
    echo "batch_name=${batch_name}"
    echo "job_root=${job_root}"
    echo "submit_file=${submit_file}"
    echo "launcher_script=${launcher_script}"
    echo "manifest=${manifest}"
} > "${job_info}"

echo "Prepared recovery submit artifacts:"
echo "  job_root=${job_root}"
echo "  submit_file=${submit_file}"
echo "  launcher_script=${launcher_script}"
echo "  manifest=${manifest}"
echo "  archive_stamp=${archive_stamp}"

if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated submit file but did not submit."
    echo "Submit manually with:"
    echo "  condor_submit ${submit_file}"
    exit 0
fi

submit_output="$(condor_submit "${submit_file}")"
echo "${submit_output}"
cluster_id="$(echo "${submit_output}" | grep -Eo 'cluster [0-9]+' | awk '{print $2}' | tail -n 1 || true)"
cluster_id="${cluster_id:-NA}"
echo "Submitted recovery aggregation jobs:"
echo "  cluster_id=${cluster_id}"
echo "  submit_file=${submit_file}"
echo "  logs=${log_dir}"
