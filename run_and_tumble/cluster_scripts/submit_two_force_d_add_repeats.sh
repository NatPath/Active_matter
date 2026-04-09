#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash submit_two_force_d_add_repeats.sh \
      --run_id <id> \
      [--d_threshold <int> | --d_values <csv>] \
      --n_sweeps <int> \
      --num_repeats <int> \
      [options]

Required:
  --run_id <id>                      Existing two_force_d production run_id or chain run_id
  --d_threshold <int>                Select all d values strictly larger than this threshold
  --d_values <csv>                   Explicit d values to repeat; overrides --d_threshold selection
  --n_sweeps <int>                   Production sweeps for each new replica
  --num_repeats <int>                Number of new replicas per selected d

Options:
  --mode <auto|production|warmup_production>
                                     How to resolve --run_id (default: auto)
  --warmup_run_id <id>               Override warmup source via explicit warmup run_id
  --warmup_state_dir <path>          Override warmup source via explicit state directory
  --request_cpus <int>               Condor request_cpus for replica nodes (default: 1)
  --request_memory <value>           Condor request_memory for replica/aggregate nodes (default: "5 GB")
  --aggregate_request_cpus <int>     Condor request_cpus for aggregate nodes (default: 1)
  --julia_num_procs_aggregate <int>  JULIA_NUM_PROCS_AGGREGATE for aggregate nodes (default: 1)
  --replica_retries <int>            DAG retry count for replica nodes on transient failures
                                     (default: 2)
  --estimate_runtime                 Enable runtime estimation prints inside replica jobs
  --estimate_sample_size <int>       Sample sweeps used when --estimate_runtime is enabled
                                     (default: 100)
  --segment_sweeps <int>             Split each replica into chained continuation segments of this size
                                     and save only the final segment into the raw repeat batch
  --aggregated_subdir <name>         Latest aggregate output dir under source state_dir (default: aggregated)
  --archive_subdir <name>            Archive dir under aggregated_subdir (default: archive)
  --raw_subdir <name>                Raw add-repeat output dir under source state_dir (default: repeat_batches)
  --aggregate_only_new_raw           Aggregate only the current repeat-batch raw dir and any --extra_raw_dir
                                     inputs, ignoring top-level raw states in target state_dir
  --extra_raw_dir <path>             Additional top-level raw state directory to include in aggregation
                                     (repeatable)
  --dag_maxjobs <int>                Forwarded to condor_submit_dag -maxjobs
                                     (default: 0, meaning no DAGMan submitted-node throttle)
  --dag_maxidle <int>                Forwarded to condor_submit_dag -maxidle
                                     (default: 0, meaning no DAGMan idle-proc throttle)
  --batch_name <name>                Condor batch_name (default: auto)
  --job_label <label>                Optional label used in the add-repeats job folder name
  --no_submit                        Generate submit files only; do not call condor_submit_dag
  -h, --help                         Show help

Behavior:
  - Resolves the target production run/config/state dir from --run_id.
  - Resolves a matching warmup state dir from:
      1) --warmup_state_dir
      2) --warmup_run_id
      3) inherited warmup_state_dir/continue chain on the production run
      4) latest compatible warmup in runs/two_force_d/run_registry.csv
  - Selects d values from the source run and keeps only d > d_threshold.
    If --d_values is provided, those exact d values are used instead.
  - Writes new raw replica states under:
      <source_state_dir>/<raw_subdir>/<job_token>/
    If --segment_sweeps is provided and smaller than --n_sweeps, intermediate continuation
    states are written under the add-repeats job folder, while only the final segment state
    is written under the raw repeat batch directory.
  - After each d finishes, re-aggregates all raw production states for that d into:
      <source_state_dir>/<aggregated_subdir>/
    by using the current aggregate for that d as the base input when present and
    adding the newly written raw files from the current repeat batch and any --extra_raw_dir inputs
    and archives previous non-partial aggregates for that d under:
      <source_state_dir>/<aggregated_subdir>/<archive_subdir>/<job_token>/d_<d>/
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
LATEST_STATE_RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_latest_state.sh"
AGGREGATE_SCRIPT="${SCRIPT_DIR}/aggregate_two_force_d_saved_files.sh"
SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
REGISTRY_FILE="${REPO_ROOT}/runs/two_force_d/run_registry.csv"

if [[ ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Missing helper script: ${RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${AGGREGATE_SCRIPT}" ]]; then
    echo "Missing helper script: ${AGGREGATE_SCRIPT}"
    exit 1
fi
if [[ ! -f "${LATEST_STATE_RUNNER_SCRIPT}" ]]; then
    echo "Missing helper script: ${LATEST_STATE_RUNNER_SCRIPT}"
    exit 1
fi
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Missing helper script: ${SPACING_UTILS}"
    exit 1
fi
DAG_NOTIFY_UTILS="${SCRIPT_DIR}/dag_notification_utils.sh"
if [[ ! -f "${DAG_NOTIFY_UTILS}" ]]; then
    echo "Missing DAG notification utils: ${DAG_NOTIFY_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${DAG_NOTIFY_UTILS}"
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

sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

dag_vars_escape() {
    local raw="$1"
    printf "%s" "${raw}" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
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
    local -a raw_values=()

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

infer_spacing_from_run_id() {
    local rid="$1"
    if [[ "${rid}" =~ -lm(_|$) ]]; then
        echo "log_midpoints"
    else
        echo "linear"
    fi
}

latest_matching_state() {
    local base_dir="$1"
    shift
    local pattern candidate
    for pattern in "$@"; do
        candidate="$(ls -1t "${base_dir}"/${pattern} 2>/dev/null | head -n 1 || true)"
        if [[ -n "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

rewrite_runtime_config() {
    local source_config="$1"
    local target_config="$2"
    local save_dir="$3"
    local n_sweeps_val="$4"
    local performance_mode_val="$5"
    local estimate_runtime_val="$6"
    local estimate_sample_size_val="$7"
    local save_dir_line n_sweeps_line performance_mode_line cluster_mode_line estimate_runtime_line estimate_sample_size_line

    save_dir_line="save_dir: \"${save_dir}\""
    n_sweeps_line="n_sweeps: ${n_sweeps_val}"
    performance_mode_line="performance_mode: ${performance_mode_val}"
    cluster_mode_line="cluster_mode: ${performance_mode_val}"
    estimate_runtime_line="estimate_runtime: ${estimate_runtime_val}"
    estimate_sample_size_line="estimate_sample_size: ${estimate_sample_size_val}"
    awk \
    -v save_dir_line="${save_dir_line}" \
    -v n_sweeps_line="${n_sweeps_line}" \
    -v performance_mode_line="${performance_mode_line}" \
    -v cluster_mode_line="${cluster_mode_line}" \
    -v estimate_runtime_line="${estimate_runtime_line}" \
    -v estimate_sample_size_line="${estimate_sample_size_line}" '
    BEGIN {
        seen_save=0
        seen_sweeps=0
        seen_performance=0
        seen_cluster=0
        seen_estimate_runtime=0
        seen_estimate_sample_size=0
    }
    {
        if ($0 ~ /^n_sweeps:[[:space:]]*/) {
            print n_sweeps_line
            seen_sweeps=1
            next
        }
        if ($0 ~ /^save_dir:[[:space:]]*/) {
            print save_dir_line
            seen_save=1
            next
        }
        if ($0 ~ /^performance_mode:[[:space:]]*/) {
            print performance_mode_line
            seen_performance=1
            next
        }
        if ($0 ~ /^cluster_mode:[[:space:]]*/) {
            print cluster_mode_line
            seen_cluster=1
            next
        }
        if ($0 ~ /^estimate_runtime:[[:space:]]*/) {
            print estimate_runtime_line
            seen_estimate_runtime=1
            next
        }
        if ($0 ~ /^estimate_sample_size:[[:space:]]*/) {
            print estimate_sample_size_line
            seen_estimate_sample_size=1
            next
        }
        print
    }
    END {
        if (!seen_sweeps) print n_sweeps_line
        if (!seen_save) print save_dir_line
        if (!seen_performance) print performance_mode_line
        if (!seen_cluster) print cluster_mode_line
        if (!seen_estimate_runtime) print estimate_runtime_line
        if (!seen_estimate_sample_size) print estimate_sample_size_line
    }' "${source_config}" > "${target_config}"
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

resolve_d_values_from_run_info() {
    local run_info_path="$1"
    local run_id_val="$2"
    local -n out_ref="$3"
    local d_values_csv d_spacing d_min d_max d_step

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

    d_spacing="$(read_run_info_value "${run_info_path}" "d_spacing")"
    if [[ -z "${d_spacing}" ]]; then
        d_spacing="$(infer_spacing_from_run_id "${run_id_val}")"
    fi
    d_spacing="$(two_force_d_normalize_spacing_mode "${d_spacing}")" || {
        echo "Invalid d_spacing='${d_spacing}' in ${run_info_path}" >&2
        return 1
    }

    d_min="$(read_run_info_value "${run_info_path}" "d_min")"
    d_max="$(read_run_info_value "${run_info_path}" "d_max")"
    d_step="$(read_run_info_value "${run_info_path}" "d_step")"
    if ! [[ "${d_min}" =~ ^[0-9]+$ && "${d_max}" =~ ^[0-9]+$ && "${d_step}" =~ ^[0-9]+$ ]]; then
        echo "Invalid d-range in ${run_info_path}: d_min='${d_min}' d_max='${d_max}' d_step='${d_step}'" >&2
        return 1
    fi

    mapfile -t out_ref < <(two_force_d_generate_d_values "${d_spacing}" "${d_min}" "${d_max}" "${d_step}")
    return 0
}

lookup_latest_matching_warmup_state_dir() {
    local prod_run_info="$1"
    local L_val rho_val d_min d_max d_step d_spacing result

    [[ -f "${REGISTRY_FILE}" ]] || return 1

    L_val="$(read_run_info_value "${prod_run_info}" "L")"
    rho_val="$(read_run_info_value "${prod_run_info}" "rho0")"
    d_min="$(read_run_info_value "${prod_run_info}" "d_min")"
    d_max="$(read_run_info_value "${prod_run_info}" "d_max")"
    d_step="$(read_run_info_value "${prod_run_info}" "d_step")"
    d_spacing="$(read_run_info_value "${prod_run_info}" "d_spacing")"
    if [[ -z "${d_spacing}" ]]; then
        d_spacing="$(infer_spacing_from_run_id "$(read_run_info_value "${prod_run_info}" "run_id")")"
    fi

    result="$(
        awk -F, -v L="${L_val}" -v rho="${rho_val}" -v dmin="${d_min}" -v dmax="${d_max}" -v dstep="${d_step}" -v spacing="${d_spacing}" '
            NR == 1 {next}
            $3 == "warmup" && $4 == L && $5 == rho && $7 == dmin && $8 == dmax && $9 == dstep {
                if (spacing == "log_midpoints" && $2 !~ /-lm(_|$)/) next
                if (spacing != "log_midpoints" && $2 ~ /-lm(_|$)/) next
                state_dir = $14
            }
            END {print state_dir}
        ' "${REGISTRY_FILE}"
    )"
    if [[ -n "${result}" && -d "${result}" ]]; then
        echo "${result}"
        return 0
    fi
    return 1
}

resolve_warmup_state_dir_for_run() {
    local prod_run_info="$1"
    local current_info="$1"
    local depth=0
    local candidate warmup_run_id continue_run_id warmup_info

    while (( depth < 12 )); do
        candidate="$(read_run_info_value "${current_info}" "warmup_state_dir")"
        if [[ -n "${candidate}" && -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi

        warmup_run_id="$(read_run_info_value "${current_info}" "warmup_run_id")"
        if [[ -n "${warmup_run_id}" ]]; then
            warmup_info="$(find_run_info_by_run_id "${warmup_run_id}" "warmup" || true)"
            if [[ -n "${warmup_info}" && -f "${warmup_info}" ]]; then
                candidate="$(read_run_info_value "${warmup_info}" "state_dir")"
                if [[ -n "${candidate}" && -d "${candidate}" ]]; then
                    echo "${candidate}"
                    return 0
                fi
            fi
        fi

        continue_run_id="$(read_run_info_value "${current_info}" "continue_run_id")"
        if [[ -z "${continue_run_id}" ]]; then
            break
        fi
        current_info="$(find_run_info_by_run_id "${continue_run_id}" "production" || true)"
        if [[ -z "${current_info}" || ! -f "${current_info}" ]]; then
            break
        fi
        depth=$((depth + 1))
    done

    lookup_latest_matching_warmup_state_dir "${prod_run_info}" || return 1
}

run_id=""
mode="auto"
d_threshold=""
d_values_csv=""
n_sweeps=""
num_repeats=""
warmup_run_id=""
warmup_state_dir=""
request_cpus="1"
request_memory="5 GB"
aggregate_request_cpus="1"
julia_num_procs_aggregate="1"
replica_retries="2"
estimate_runtime="false"
estimate_sample_size="100"
segment_sweeps=""
aggregated_subdir="aggregated"
archive_subdir="archive"
raw_subdir="repeat_batches"
aggregate_only_new_raw="false"
extra_raw_dirs=()
dag_maxjobs="0"
dag_maxidle="0"
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
        --d_threshold)
            d_threshold="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --num_repeats)
            num_repeats="${2:-}"
            shift 2
            ;;
        --warmup_run_id)
            warmup_run_id="${2:-}"
            shift 2
            ;;
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
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
        --aggregate_request_cpus)
            aggregate_request_cpus="${2:-}"
            shift 2
            ;;
        --julia_num_procs_aggregate)
            julia_num_procs_aggregate="${2:-}"
            shift 2
            ;;
        --replica_retries)
            replica_retries="${2:-}"
            shift 2
            ;;
        --estimate_runtime)
            estimate_runtime="true"
            shift
            ;;
        --estimate_sample_size)
            estimate_sample_size="${2:-}"
            shift 2
            ;;
        --segment_sweeps)
            segment_sweeps="${2:-}"
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
        --raw_subdir)
            raw_subdir="${2:-}"
            shift 2
            ;;
        --aggregate_only_new_raw)
            aggregate_only_new_raw="true"
            shift
            ;;
        --extra_raw_dir)
            extra_raw_dirs+=("${2:-}")
            shift 2
            ;;
        --dag_maxjobs)
            dag_maxjobs="${2:-}"
            shift 2
            ;;
        --dag_maxidle)
            dag_maxidle="${2:-}"
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

if [[ -z "${run_id}" || -z "${n_sweeps}" || -z "${num_repeats}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi
if [[ -z "${d_threshold}" && -z "${d_values_csv}" ]]; then
    echo "Set either --d_threshold or --d_values."
    usage
    exit 1
fi
if [[ -n "${d_threshold}" && -n "${d_values_csv}" ]]; then
    echo "Set only one of --d_threshold or --d_values."
    exit 1
fi

case "${mode}" in
    auto|production|warmup_production)
        ;;
    *)
        echo "--mode must be auto, production, or warmup_production. Got '${mode}'."
        exit 1
        ;;
esac

for numeric_name in n_sweeps num_repeats request_cpus aggregate_request_cpus julia_num_procs_aggregate estimate_sample_size; do
    numeric_value="${!numeric_name}"
    if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
        echo "--${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done
if ! [[ "${replica_retries}" =~ ^[0-9]+$ ]]; then
    echo "--replica_retries must be a non-negative integer. Got '${replica_retries}'."
    exit 1
fi
if [[ -n "${d_threshold}" ]]; then
    if ! [[ "${d_threshold}" =~ ^[0-9]+$ ]] || (( d_threshold <= 0 )); then
        echo "--d_threshold must be a positive integer. Got '${d_threshold}'."
        exit 1
    fi
fi
if [[ -n "${segment_sweeps}" ]]; then
    if ! [[ "${segment_sweeps}" =~ ^[0-9]+$ ]] || (( segment_sweeps <= 0 )); then
        echo "--segment_sweeps must be a positive integer. Got '${segment_sweeps}'."
        exit 1
    fi
fi
if [[ -n "${dag_maxjobs}" ]]; then
    if ! [[ "${dag_maxjobs}" =~ ^[0-9]+$ ]]; then
        echo "--dag_maxjobs must be a non-negative integer. Got '${dag_maxjobs}'."
        exit 1
    fi
fi
if [[ -n "${dag_maxidle}" ]]; then
    if ! [[ "${dag_maxidle}" =~ ^[0-9]+$ ]]; then
        echo "--dag_maxidle must be a non-negative integer. Got '${dag_maxidle}'."
        exit 1
    fi
fi

for token_name in aggregated_subdir archive_subdir raw_subdir; do
    token_value="${!token_name}"
    if [[ -z "${token_value}" || ! "${token_value}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "--${token_name} must match [A-Za-z0-9._-]+. Got '${token_value}'."
        exit 1
    fi
done

if [[ -n "${warmup_state_dir}" && ! -d "${warmup_state_dir}" ]]; then
    echo "--warmup_state_dir does not exist: ${warmup_state_dir}"
    exit 1
fi
for extra_dir in "${extra_raw_dirs[@]}"; do
    if [[ -z "${extra_dir}" || ! -d "${extra_dir}" ]]; then
        echo "--extra_raw_dir is invalid: ${extra_dir}"
        exit 1
    fi
done

target_run_info="$(resolve_target_production_run_info "${run_id}" "${mode}")"
target_run_id="$(read_run_info_value "${target_run_info}" "run_id")"
target_state_dir="$(read_run_info_value "${target_run_info}" "state_dir")"
target_config_dir="$(read_run_info_value "${target_run_info}" "config_dir")"
target_replica_strategy="$(read_run_info_value "${target_run_info}" "replica_strategy")"
target_num_replicas="$(read_run_info_value "${target_run_info}" "num_replicas")"

if [[ -z "${target_state_dir}" || ! -d "${target_state_dir}" ]]; then
    echo "Target state_dir is invalid: ${target_state_dir}"
    exit 1
fi
if [[ -z "${target_config_dir}" || ! -d "${target_config_dir}" ]]; then
    echo "Target config_dir is invalid: ${target_config_dir}"
    exit 1
fi

if [[ -n "${warmup_run_id}" && -n "${warmup_state_dir}" ]]; then
    echo "Set only one of --warmup_run_id or --warmup_state_dir."
    exit 1
fi
if [[ -n "${warmup_run_id}" ]]; then
    warmup_run_info="$(find_run_info_by_run_id "${warmup_run_id}" "warmup" || true)"
    if [[ -z "${warmup_run_info}" || ! -f "${warmup_run_info}" ]]; then
        echo "Could not resolve warmup run_info for warmup_run_id='${warmup_run_id}'."
        exit 1
    fi
    warmup_state_dir="$(read_run_info_value "${warmup_run_info}" "state_dir")"
    if [[ -z "${warmup_state_dir}" || ! -d "${warmup_state_dir}" ]]; then
        echo "Resolved warmup state_dir is invalid: ${warmup_state_dir}"
        exit 1
    fi
elif [[ -z "${warmup_state_dir}" ]]; then
    warmup_state_dir="$(resolve_warmup_state_dir_for_run "${target_run_info}" || true)"
    if [[ -z "${warmup_state_dir}" || ! -d "${warmup_state_dir}" ]]; then
        echo "Could not resolve a warmup_state_dir for target run '${target_run_id}'."
        echo "Pass --warmup_run_id or --warmup_state_dir explicitly."
        exit 1
    fi
fi

all_d_values=()
resolve_d_values_from_run_info "${target_run_info}" "${target_run_id}" all_d_values
selected_d_values=()
if [[ -n "${d_values_csv}" ]]; then
    requested_d_values=()
    parse_d_values_csv "${d_values_csv}" requested_d_values
    for requested_d in "${requested_d_values[@]}"; do
        found_match="false"
        for d_val in "${all_d_values[@]}"; do
            if [[ "${d_val}" == "${requested_d}" ]]; then
                selected_d_values+=("${requested_d}")
                found_match="true"
                break
            fi
        done
        if [[ "${found_match}" != "true" ]]; then
            echo "Requested d=${requested_d} is not present in run '${target_run_id}'."
            exit 1
        fi
    done
else
    for d_val in "${all_d_values[@]}"; do
        if (( d_val > d_threshold )); then
            selected_d_values+=("${d_val}")
        fi
    done
fi
if (( ${#selected_d_values[@]} == 0 )); then
    if [[ -n "${d_values_csv}" ]]; then
        echo "No d values were selected by --d_values='${d_values_csv}'."
    else
        echo "No d values in run '${target_run_id}' are larger than d_threshold=${d_threshold}."
    fi
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
timestamp_compact="$(date +%Y%m%d%H%M%S)"
safe_target_run_id="$(sanitize_token "${target_run_id}")"
if [[ -n "${job_label}" ]]; then
    job_slug="$(sanitize_token "${job_label}")"
elif [[ -n "${d_values_csv}" ]]; then
    job_slug="d$(printf "%s" "${d_values_csv}" | sed -E 's/[[:space:]]+//g; s/,/-/g')_ns${n_sweeps}_nr${num_repeats}"
else
    job_slug="gt${d_threshold}_ns${n_sweeps}_nr${num_repeats}"
fi
job_token="${job_slug}_${timestamp}"
job_root="${REPO_ROOT}/runs/two_force_d/add_repeats_jobs/${safe_target_run_id}_${job_token}"
submit_dir="${job_root}/submit"
log_dir="${job_root}/logs"
config_dir="${job_root}/configs"
segment_state_root="${job_root}/segment_states"
manifest="${job_root}/manifest.csv"
job_info="${job_root}/job_info.txt"
dag_file="${submit_dir}/add_repeats.dag"
jobs_section_file="${submit_dir}/add_repeats.jobs.tmp"
deps_section_file="${submit_dir}/add_repeats.deps.tmp"
raw_state_dir="${target_state_dir}/${raw_subdir}/${job_token}"
mkdir -p "${submit_dir}" "${log_dir}" "${config_dir}" "${segment_state_root}" "${raw_state_dir}"
: > "${dag_file}"
: > "${jobs_section_file}"
: > "${deps_section_file}"

if [[ -z "${batch_name}" ]]; then
    batch_name="two_force_d_add_repeats_$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
fi

run_hash="$(printf "%s" "${target_run_id}" | cksum | awk '{print $1}')"
selected_d_csv="$(IFS=,; echo "${selected_d_values[*]}")"
declare -A replica_template_submit_by_d=()

echo "d,new_repeats,existing_raw_states,warmup_state,config_path,raw_state_dir" > "${manifest}"

echo "Preparing two_force_d add-repeats DAG:"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${target_run_info}"
echo "  target_state_dir=${target_state_dir}"
echo "  target_config_dir=${target_config_dir}"
echo "  warmup_state_dir=${warmup_state_dir}"
echo "  d_threshold=${d_threshold}"
echo "  selected_d_values=${selected_d_csv}"
echo "  n_sweeps=${n_sweeps}"
echo "  segment_sweeps=${segment_sweeps:-none}"
echo "  num_repeats=${num_repeats}"
echo "  request_cpus=${request_cpus}"
echo "  aggregate_request_cpus=${aggregate_request_cpus}"
echo "  request_memory=${request_memory}"
echo "  JULIA_NUM_PROCS_AGGREGATE=${julia_num_procs_aggregate}"
echo "  replica_retries=${replica_retries}"
echo "  performance_mode=true"
echo "  estimate_runtime=${estimate_runtime}"
echo "  estimate_sample_size=${estimate_sample_size}"
echo "  aggregate_only_new_raw=${aggregate_only_new_raw}"
echo "  raw_state_dir=${raw_state_dir}"
if (( ${#extra_raw_dirs[@]} > 0 )); then
    echo "  extra_raw_dirs=$(IFS=:; echo "${extra_raw_dirs[*]}")"
fi
echo "  latest_aggregates_dir=${target_state_dir}/${aggregated_subdir}"
echo "  archive_root=${target_state_dir}/${aggregated_subdir}/${archive_subdir}/${job_token}"
echo "  job_root=${job_root}"
echo "  dag_maxjobs=${dag_maxjobs}"
echo "  dag_maxidle=${dag_maxidle}"
if [[ "${target_replica_strategy}" == "mp" && "${target_num_replicas}" =~ ^[0-9]+$ && "${target_num_replicas}" -gt 1 ]]; then
    echo "WARNING: source run used replica_strategy=mp with num_replicas=${target_num_replicas}."
    echo "WARNING: old per-replica SEM cannot be reconstructed unless raw prod states are still present."
fi

for d_val in "${selected_d_values[@]}"; do
    echo "  generating submit files for d=${d_val}"
    source_config="${target_config_dir}/d_${d_val}.yaml"
    if [[ ! -f "${source_config}" ]]; then
        fallback_cfg="${REPO_ROOT}/configuration_files/two_force_d_sweep/production/d_${d_val}.yaml"
        if [[ -f "${fallback_cfg}" ]]; then
            source_config="${fallback_cfg}"
        else
            echo "Missing production config for d=${d_val}: ${target_config_dir}/d_${d_val}.yaml"
            exit 1
        fi
    fi

    runtime_config="${config_dir}/d_${d_val}.yaml"
    rewrite_runtime_config "${source_config}" "${runtime_config}" "${raw_state_dir}" "${n_sweeps}" "true" "${estimate_runtime}" "${estimate_sample_size}"

    warmup_state="$(latest_matching_state "${warmup_state_dir}" \
        "aggregated/two_force_d${d_val}_warmup_*.jld2" \
        "two_force_d${d_val}_warmup_*.jld2" \
        "aggregated/two_force_d${d_val}_*.jld2" \
        "two_force_d${d_val}_*.jld2" || true)"
    if [[ -z "${warmup_state}" ]]; then
        echo "No warmup state found for d=${d_val} under ${warmup_state_dir}"
        exit 1
    fi

    if [[ -z "${segment_sweeps}" || ! "${segment_sweeps}" =~ ^[0-9]+$ || "${segment_sweeps}" -ge "${n_sweeps}" ]]; then
        replica_template_submit="${submit_dir}/d_${d_val}_replica_template.sub"
        cat > "${replica_template_submit}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${RUNNER_SCRIPT} ${runtime_config} --initial_state ${warmup_state} --save_tag \$(save_tag)
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${log_dir}/d_${d_val}_r\$(replica_idx).out
error      = ${log_dir}/d_${d_val}_r\$(replica_idx).err
log        = ${log_dir}/d_${d_val}_r\$(replica_idx).log
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${batch_name}
queue
EOF
        replica_template_submit_by_d["${d_val}"]="${replica_template_submit}"
    fi

    existing_raw_count="$(
        find "${target_state_dir}" -maxdepth 1 -type f \
            -name "two_force_d${d_val}_prod_*.jld2" \
            ! -name "*_id-aggregated_*" \
            | wc -l | awk '{print $1}'
    )"
    printf "%s,%s,%s,%s,%s,%s\n" \
        "${d_val}" "${num_repeats}" "${existing_raw_count}" "${warmup_state}" "${runtime_config}" "${raw_state_dir}" \
        >> "${manifest}"

    for ((replica_idx = 1; replica_idx <= num_repeats; replica_idx++)); do
        replica_tag="replica_addrep_${run_hash}_${timestamp_compact}_d${d_val}_r${replica_idx}"
        replica_output_file="${log_dir}/d_${d_val}_r${replica_idx}.out"
        replica_error_file="${log_dir}/d_${d_val}_r${replica_idx}.err"
        replica_log_file="${log_dir}/d_${d_val}_r${replica_idx}.log"
        replica_job_id="D${d_val}R${replica_idx}"

        if [[ -n "${segment_sweeps}" && "${segment_sweeps}" =~ ^[0-9]+$ && "${segment_sweeps}" -lt "${n_sweeps}" ]]; then
            replica_submit_file="${submit_dir}/d_${d_val}_replica_${replica_idx}.sub"
            remaining_sweeps="${n_sweeps}"
            segment_idx=1
            previous_state_dir=""
            previous_save_tag=""
            replica_segment_dir="${segment_state_root}/d_${d_val}/r_${replica_idx}"
            mkdir -p "${replica_segment_dir}"

            while (( remaining_sweeps > 0 )); do
                current_segment_sweeps="${remaining_sweeps}"
                if (( current_segment_sweeps > segment_sweeps )); then
                    current_segment_sweeps="${segment_sweeps}"
                fi
                remaining_sweeps=$((remaining_sweeps - current_segment_sweeps))
                is_final_segment="false"
                if (( remaining_sweeps == 0 )); then
                    is_final_segment="true"
                fi

                if [[ "${is_final_segment}" == "true" ]]; then
                    stage_config="${config_dir}/d_${d_val}_replica_${replica_idx}.yaml"
                    stage_save_dir="${raw_state_dir}"
                    stage_save_tag="${replica_tag}"
                    stage_submit_file="${replica_submit_file}"
                    stage_output_file="${replica_output_file}"
                    stage_error_file="${replica_error_file}"
                    stage_log_file="${replica_log_file}"
                    stage_job_id="${replica_job_id}"
                else
                    stage_config="${config_dir}/d_${d_val}_r${replica_idx}_seg_${segment_idx}.yaml"
                    stage_save_dir="${replica_segment_dir}"
                    stage_save_tag="${replica_tag}_seg${segment_idx}"
                    stage_submit_file="${submit_dir}/seg_d_${d_val}_r_${replica_idx}_s_${segment_idx}.sub"
                    stage_output_file="${log_dir}/d_${d_val}_r${replica_idx}_s${segment_idx}.out"
                    stage_error_file="${log_dir}/d_${d_val}_r${replica_idx}_s${segment_idx}.err"
                    stage_log_file="${log_dir}/d_${d_val}_r${replica_idx}_s${segment_idx}.log"
                    stage_job_id="D${d_val}R${replica_idx}S${segment_idx}"
                fi

                rewrite_runtime_config "${source_config}" "${stage_config}" "${stage_save_dir}" "${current_segment_sweeps}" "true" "${estimate_runtime}" "${estimate_sample_size}"

                if (( segment_idx == 1 )); then
                    stage_runner_arguments="${RUNNER_SCRIPT} ${stage_config} --initial_state ${warmup_state} --save_tag ${stage_save_tag}"
                else
                    stage_runner_arguments="${LATEST_STATE_RUNNER_SCRIPT} --runner_script ${RUNNER_SCRIPT} --config ${stage_config} --state_dir ${previous_state_dir} --pattern *_id-${previous_save_tag}.jld2 --state_arg_name continue --save_tag ${stage_save_tag}"
                fi

                cat > "${stage_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/bash
arguments  = ${stage_runner_arguments}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${stage_output_file}
error      = ${stage_error_file}
log        = ${stage_log_file}
request_cpus = ${request_cpus}
request_memory = ${request_memory}
batch_name = ${batch_name}
queue
EOF

                previous_state_dir="${stage_save_dir}"
                previous_save_tag="${stage_save_tag}"
                segment_idx=$((segment_idx + 1))
            done
        fi
    done

    aggregate_launcher="${submit_dir}/aggregate_d_${d_val}.sh"
    aggregate_submit_file="${submit_dir}/d_${d_val}_aggregate.sub"
    aggregate_output_file="${log_dir}/d_${d_val}_aggregate.out"
    aggregate_error_file="${log_dir}/d_${d_val}_aggregate.err"
    aggregate_log_file="${log_dir}/d_${d_val}_aggregate.log"

    {
        echo "#!/usr/bin/env bash"
        echo "set -euo pipefail"
        echo "cd $(printf '%q' "${REPO_ROOT}")"
        echo "export JULIA_NUM_PROCS_AGGREGATE=$(printf '%q' "${julia_num_procs_aggregate}")"
        printf "bash %q --run_id %q --mode production --state_dir %q --config_dir %q --extra_raw_dir %q" \
            "${AGGREGATE_SCRIPT}" \
            "${target_run_id}" \
            "${target_state_dir}" \
            "${target_config_dir}" \
            "${raw_state_dir}"
        for extra_raw_dir in "${extra_raw_dirs[@]}"; do
            printf " --extra_raw_dir %q" "${extra_raw_dir}"
        done
        if [[ "${aggregate_only_new_raw}" == "true" ]]; then
            printf " --only_extra_raw_inputs"
        fi
        printf " --aggregated_subdir %q --exclude_aggregated_inputs --incremental_from_existing_aggregate --archive_existing_aggregates --archive_subdir %q --archive_stamp %q --d_min %q --d_max %q --d_step 1 --num_files 0 --force\n" \
            "${aggregated_subdir}" \
            "${archive_subdir}" \
            "${job_token}" \
            "${d_val}" \
            "${d_val}"
    } > "${aggregate_launcher}"
    chmod +x "${aggregate_launcher}"

    cat > "${aggregate_submit_file}" <<EOF
Universe   = vanilla
Executable = ${aggregate_launcher}
initialdir = ${REPO_ROOT}
should_transfer_files = NO
output     = ${aggregate_output_file}
error      = ${aggregate_error_file}
log        = ${aggregate_log_file}
request_cpus = ${aggregate_request_cpus}
request_memory = ${request_memory}
batch_name = ${batch_name}
queue
EOF

    aggregate_job_id="D${d_val}A"
done

: > "${dag_file}"
echo "  assembling DAG sections"
for ((replica_idx = 1; replica_idx <= num_repeats; replica_idx++)); do
    for d_val in "${selected_d_values[@]}"; do
        replica_job_id="D${d_val}R${replica_idx}"
        if [[ -n "${segment_sweeps}" && "${segment_sweeps}" =~ ^[0-9]+$ && "${segment_sweeps}" -lt "${n_sweeps}" ]]; then
            replica_submit_file="${submit_dir}/d_${d_val}_replica_${replica_idx}.sub"
            remaining_sweeps="${n_sweeps}"
            segment_idx=1
            previous_job_id=""
            while (( remaining_sweeps > 0 )); do
                current_segment_sweeps="${remaining_sweeps}"
                if (( current_segment_sweeps > segment_sweeps )); then
                    current_segment_sweeps="${segment_sweeps}"
                fi
                remaining_sweeps=$((remaining_sweeps - current_segment_sweeps))
                if (( remaining_sweeps == 0 )); then
                    stage_job_id="${replica_job_id}"
                    stage_submit_file="${replica_submit_file}"
                else
                    stage_job_id="D${d_val}R${replica_idx}S${segment_idx}"
                    stage_submit_file="${submit_dir}/seg_d_${d_val}_r_${replica_idx}_s_${segment_idx}.sub"
                fi
                printf "JOB %s %s\n" "${stage_job_id}" "${stage_submit_file}" >> "${jobs_section_file}"
                if (( replica_retries > 0 )); then
                    printf "RETRY %s %s\n" "${stage_job_id}" "${replica_retries}" >> "${jobs_section_file}"
                fi
                if [[ -n "${previous_job_id}" ]]; then
                    printf "PARENT %s CHILD %s\n" "${previous_job_id}" "${stage_job_id}" >> "${deps_section_file}"
                fi
                previous_job_id="${stage_job_id}"
                segment_idx=$((segment_idx + 1))
            done
        else
            replica_tag="replica_addrep_${run_hash}_${timestamp_compact}_d${d_val}_r${replica_idx}"
            printf "JOB %s %s\n" "${replica_job_id}" "${replica_template_submit_by_d[${d_val}]}" >> "${jobs_section_file}"
            if (( replica_retries > 0 )); then
                printf "RETRY %s %s\n" "${replica_job_id}" "${replica_retries}" >> "${jobs_section_file}"
            fi
            printf 'VARS %s save_tag="%s" replica_idx="%s"\n' \
                "${replica_job_id}" \
                "$(dag_vars_escape "${replica_tag}")" \
                "${replica_idx}" \
                >> "${jobs_section_file}"
        fi
    done
done

for d_val in "${selected_d_values[@]}"; do
    aggregate_job_id="D${d_val}A"
    aggregate_submit_file="${submit_dir}/d_${d_val}_aggregate.sub"
    printf "JOB %s %s\n" "${aggregate_job_id}" "${aggregate_submit_file}" >> "${jobs_section_file}"
done

for d_val in "${selected_d_values[@]}"; do
    aggregate_job_id="D${d_val}A"
    replica_job_ids=()
    for ((replica_idx = 1; replica_idx <= num_repeats; replica_idx++)); do
        replica_job_ids+=("D${d_val}R${replica_idx}")
    done
    printf "PARENT %s CHILD %s\n" "${replica_job_ids[*]}" "${aggregate_job_id}" >> "${deps_section_file}"
done

cat "${jobs_section_file}" >> "${dag_file}"
cat "${deps_section_file}" >> "${dag_file}"
dag_append_final_notification_node "${dag_file}" "${submit_dir}" "${log_dir}" "${job_root}" "${target_run_id}" "two_force_d_add_repeats" "${REPO_ROOT}"

{
    echo "timestamp=${timestamp}"
    echo "requested_run_id=${run_id}"
    echo "target_run_id=${target_run_id}"
    echo "target_run_info=${target_run_info}"
    echo "target_state_dir=${target_state_dir}"
    echo "target_config_dir=${target_config_dir}"
    echo "warmup_state_dir=${warmup_state_dir}"
    echo "d_threshold=${d_threshold}"
    echo "selected_d_values=${selected_d_csv}"
    echo "n_sweeps=${n_sweeps}"
    echo "segment_sweeps=${segment_sweeps}"
    echo "num_repeats=${num_repeats}"
    echo "request_cpus=${request_cpus}"
    echo "aggregate_request_cpus=${aggregate_request_cpus}"
    echo "request_memory=${request_memory}"
    echo "julia_num_procs_aggregate=${julia_num_procs_aggregate}"
    echo "replica_retries=${replica_retries}"
    echo "performance_mode=true"
    echo "estimate_runtime=${estimate_runtime}"
    echo "estimate_sample_size=${estimate_sample_size}"
    echo "aggregate_only_new_raw=${aggregate_only_new_raw}"
    echo "aggregated_subdir=${aggregated_subdir}"
    echo "archive_subdir=${archive_subdir}"
    echo "raw_subdir=${raw_subdir}"
    echo "raw_state_dir=${raw_state_dir}"
    if (( ${#extra_raw_dirs[@]} > 0 )); then
        echo "extra_raw_dirs=$(IFS=:; echo "${extra_raw_dirs[*]}")"
    fi
    echo "dag_maxjobs=${dag_maxjobs}"
    echo "dag_maxidle=${dag_maxidle}"
    echo "batch_name=${batch_name}"
    echo "job_root=${job_root}"
    echo "dag_file=${dag_file}"
    echo "dag_notification_status_log=${DAG_NOTIFICATION_STATUS_LOG}"
    echo "manifest=${manifest}"
} > "${job_info}"

echo "Prepared add-repeats DAG:"
echo "  dag_file=${dag_file}"
echo "  manifest=${manifest}"
echo "  job_info=${job_info}"

if [[ "${no_submit}" == "true" ]]; then
    echo "NO_SUBMIT=true; generated DAG but did not submit."
    echo "Submit manually with:"
    echo "  condor_submit_dag -maxidle ${dag_maxidle} -maxjobs ${dag_maxjobs} ${dag_file}"
    exit 0
fi

submit_cmd=(condor_submit_dag)
submit_cmd+=(-maxidle "${dag_maxidle}")
submit_cmd+=(-maxjobs "${dag_maxjobs}")
submit_cmd+=("${dag_file}")
submit_output="$("${submit_cmd[@]}")"
echo "${submit_output}"
echo "Submitted add-repeats DAG:"
echo "  dag_file=${dag_file}"
echo "  logs=${log_dir}"
