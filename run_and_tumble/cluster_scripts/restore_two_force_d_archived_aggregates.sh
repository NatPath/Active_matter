#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash restore_two_force_d_archived_aggregates.sh \
      --run_id <id> \
      --d_values <csv> \
      [options]

Required:
  --run_id <id>                      Existing two_force_d production run_id or chain run_id
  --d_values <csv>                   Comma-separated d values, for example: 96,128

Options:
  --mode <auto|production|warmup_production>
                                     How to resolve --run_id (default: auto)
  --state_dir <path>                 Override resolved production state_dir
  --aggregated_subdir <name>         Aggregated output subdir under state_dir (default: aggregated)
  --archive_subdir <name>            Archive subdirectory under aggregated_subdir (default: archive)
  --target_t <int>                   Exact target aggregate t to restore (default: 8000000000)
  --source_archive_stamp <token>     Restrict restore candidates to a specific archive stamp
  --rewind_stamp <token>             Archive stamp for staging the current broken files
                                     (default: rewind_to_t<target_t>_<timestamp>)
  --dry_run                          Print planned actions only; do not modify files
  -h, --help                         Show help

Behavior:
  - Resolves the production run from --run_id.
  - Finds one archived aggregate per requested d with exact t=<target_t>.
  - Moves the current aggregate file(s) for that d from:
      <state_dir>/<aggregated_subdir>/
    into:
      <state_dir>/<aggregated_subdir>/<archive_subdir>/<rewind_stamp>/d_<d>/
  - Copies the archived target aggregate back into:
      <state_dir>/<aggregated_subdir>/

Safety:
  - Refuses to continue if the archived candidate is ambiguous.
  - Never deletes the archived source file; it copies it back into the live aggregate dir.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_FILE="${SCRIPT_DIR}/two_force_d_add_repeats_utils.sh"
if [[ ! -f "${UTILS_FILE}" ]]; then
    echo "Missing helper script: ${UTILS_FILE}"
    exit 1
fi
source "${UTILS_FILE}"

REPO_ROOT="$(two_force_repo_root_from_script_dir "${SCRIPT_DIR}")"

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
        while [[ -e "${archive_dir}/${stem}__rewind${idx}.jld2" ]]; do
            idx=$((idx + 1))
        done
        target_path="${archive_dir}/${stem}__rewind${idx}.jld2"
    fi
    mv -f "${source_file}" "${target_path}"
    printf "%s" "${target_path}"
}

find_restore_candidate_for_d() {
    local d_val="$1"
    local archive_root="$2"
    local state_root="$3"
    local aggregated_root="$4"
    local target_t_val="$5"
    local source_stamp="${6:-}"
    local pattern="*_t${target_t_val}_*_d${d_val}.jld2"
    local -a candidates=()
    local fallback=""

    if [[ -n "${source_stamp}" ]]; then
        mapfile -t candidates < <(
            find "${archive_root}/${source_stamp}/d_${d_val}" -maxdepth 1 -type f -name "${pattern}" -print 2>/dev/null | sort
        )
    else
        mapfile -t candidates < <(
            find "${archive_root}" -type f -path "*/d_${d_val}/*" -name "${pattern}" -print 2>/dev/null | sort
        )
        if (( ${#candidates[@]} == 0 )); then
            mapfile -t candidates < <(
                find "${state_root}" -type f -path "*/archive/*" -name "${pattern}" -print 2>/dev/null | sort
            )
        fi
        if (( ${#candidates[@]} == 0 )); then
            mapfile -t candidates < <(
                find "${state_root}" -type f ! -path "${aggregated_root}/*" -name "${pattern}" -print 2>/dev/null | sort
            )
            fallback="true"
        fi
    fi

    if (( ${#candidates[@]} == 1 )); then
        printf "%s" "${candidates[0]}"
        return 0
    fi

    if (( ${#candidates[@]} == 0 )); then
        echo "No archived candidate found for d=${d_val} with t=${target_t_val}." >&2
        return 1
    fi

    echo "Ambiguous archived candidates for d=${d_val} with t=${target_t_val}." >&2
    if [[ -n "${fallback}" ]]; then
        echo "Search had to fall back beyond ${archive_root}." >&2
    fi
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Use --source_archive_stamp to disambiguate." >&2
    return 1
}

run_id=""
mode="auto"
state_dir=""
aggregated_subdir="aggregated"
archive_subdir="archive"
target_t="8000000000"
d_values_csv=""
source_archive_stamp=""
rewind_stamp=""
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
        --state_dir)
            state_dir="${2:-}"
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
        --target_t)
            target_t="${2:-}"
            shift 2
            ;;
        --d_values)
            d_values_csv="${2:-}"
            shift 2
            ;;
        --source_archive_stamp)
            source_archive_stamp="${2:-}"
            shift 2
            ;;
        --rewind_stamp)
            rewind_stamp="${2:-}"
            shift 2
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
if [[ -z "${d_values_csv}" ]]; then
    echo "--d_values is required."
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
if [[ -z "${aggregated_subdir}" ]] || ! [[ "${aggregated_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--aggregated_subdir must match [A-Za-z0-9._-]+. Got '${aggregated_subdir}'."
    exit 1
fi
if [[ -n "${archive_subdir}" ]] && ! [[ "${archive_subdir}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--archive_subdir must match [A-Za-z0-9._-]+ when provided. Got '${archive_subdir}'."
    exit 1
fi
if ! [[ "${target_t}" =~ ^[0-9]+$ ]]; then
    echo "--target_t must be a non-negative integer. Got '${target_t}'."
    exit 1
fi
if [[ -n "${source_archive_stamp}" ]] && ! [[ "${source_archive_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--source_archive_stamp must match [A-Za-z0-9._-]+ when provided. Got '${source_archive_stamp}'."
    exit 1
fi
if [[ -n "${rewind_stamp}" ]] && ! [[ "${rewind_stamp}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--rewind_stamp must match [A-Za-z0-9._-]+ when provided. Got '${rewind_stamp}'."
    exit 1
fi

declare -a d_values=()
two_force_parse_csv_ints "${d_values_csv}" d_values
if (( ${#d_values[@]} == 0 )); then
    echo "--d_values did not contain any valid integers."
    exit 1
fi

prod_run_info="$(two_force_resolve_target_production_run_info "${REPO_ROOT}" "${run_id}" "${mode}")"
target_run_id="$(two_force_read_key_value "${prod_run_info}" "run_id")"
if [[ -z "${state_dir}" ]]; then
    state_dir="$(two_force_read_key_value "${prod_run_info}" "state_dir")"
fi
if [[ -z "${state_dir}" || ! -d "${state_dir}" ]]; then
    echo "Resolved state_dir is invalid: ${state_dir}"
    exit 1
fi

effective_agg_dir="${state_dir}/${aggregated_subdir}"
effective_archive_root="${effective_agg_dir}/${archive_subdir}"
if [[ -z "${rewind_stamp}" ]]; then
    rewind_stamp="rewind_to_t${target_t}_$(date +%Y%m%d-%H%M%S)"
fi

declare -A restore_candidates=()
declare -A current_candidates=()
declare -A current_count=()

for d in "${d_values[@]}"; do
    restore_candidates["${d}"]="$(find_restore_candidate_for_d "${d}" "${effective_archive_root}" "${state_dir}" "${effective_agg_dir}" "${target_t}" "${source_archive_stamp}")"
    mapfile -t current_matches < <(
        find "${effective_agg_dir}" -maxdepth 1 -type f \
            -name "*_id-aggregated_*" \
            ! -name "*_id-aggregated_partial_*" \
            -name "*_d${d}.jld2" \
            -print 2>/dev/null | sort
    )
    current_count["${d}"]="${#current_matches[@]}"
    current_candidates["${d}"]="$(printf "%s\n" "${current_matches[@]}")"
done

echo "Rewind two_force_d aggregates"
echo "  requested_run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${prod_run_info}"
echo "  state_dir=${state_dir}"
echo "  aggregated_dir=${effective_agg_dir}"
echo "  archive_root=${effective_archive_root}"
echo "  rewind_stamp=${rewind_stamp}"
echo "  target_t=${target_t}"
if [[ -n "${source_archive_stamp}" ]]; then
    echo "  source_archive_stamp=${source_archive_stamp}"
fi
echo "  selected_d_values=$(IFS=,; echo "${d_values[*]}")"
if [[ "${dry_run}" == "true" ]]; then
    echo "  dry_run=true"
fi

for d in "${d_values[@]}"; do
    echo "d=${d}: restore_candidate=${restore_candidates[${d}]}"
    if (( current_count["${d}"] > 0 )); then
        while IFS= read -r current_file; do
            [[ -z "${current_file}" ]] && continue
            echo "d=${d}: current_live=${current_file}"
        done <<< "${current_candidates[${d}]}"
    else
        echo "d=${d}: current_live=(none)"
    fi
done

if [[ "${dry_run}" == "true" ]]; then
    exit 0
fi

mkdir -p "${effective_agg_dir}"

for d in "${d_values[@]}"; do
    while IFS= read -r current_file; do
        [[ -z "${current_file}" ]] && continue
        archived_current="$(move_file_to_archive_dir "${current_file}" "${effective_archive_root}/${rewind_stamp}/d_${d}")"
        echo "d=${d}: staged current aggregate -> ${archived_current}"
    done <<< "${current_candidates[${d}]}"

    restored_path="${effective_agg_dir}/$(basename "${restore_candidates[${d}]}")"
    cp -fp "${restore_candidates[${d}]}" "${restored_path}"
    echo "d=${d}: restored archived aggregate -> ${restored_path}"
done
