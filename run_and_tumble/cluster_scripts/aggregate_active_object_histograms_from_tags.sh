#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash aggregate_active_object_histograms_from_tags.sh \
      --state_dir <path> \
      --output_dir <path> \
      --save_tag <tag> \
      [--num_replicas <int> --replica_tag_prefix <prefix>] \
      [--all_states_recursive] \
      [--min_sweep <int>] \
      [--max_sweep <int>] \
      [--plot_per_run] \
      [--no_plot]

Behavior:
  - default mode resolves one saved state per replica using expected IDs:
      *_id-${replica_tag_prefix}<replica_index>.jld2
  - with --all_states_recursive, resolves the latest state for every unique id tag
    under <state_dir> recursively
  - writes a temporary state list file
  - runs utility_scripts/active_object_steady_state_histograms.jl in multi-state mode
  - writes one per-run histogram artifact under <output_dir>/per_run/
  - writes one aggregated histogram artifact under <output_dir>/aggregated/
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HISTOGRAM_SCRIPT="${REPO_ROOT}/utility_scripts/active_object_steady_state_histograms.jl"

if [[ ! -f "${HISTOGRAM_SCRIPT}" ]]; then
    echo "Missing histogram utility: ${HISTOGRAM_SCRIPT}"
    exit 1
fi

state_dir=""
output_dir=""
num_replicas=""
replica_tag_prefix=""
save_tag=""
min_sweep="0"
max_sweep=""
plot_per_run="false"
no_plot="false"
all_states_recursive="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --output_dir)
            output_dir="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --replica_tag_prefix)
            replica_tag_prefix="${2:-}"
            shift 2
            ;;
        --save_tag)
            save_tag="${2:-}"
            shift 2
            ;;
        --all_states_recursive)
            all_states_recursive="true"
            shift
            ;;
        --min_sweep)
            min_sweep="${2:-}"
            shift 2
            ;;
        --max_sweep)
            max_sweep="${2:-}"
            shift 2
            ;;
        --plot_per_run)
            plot_per_run="true"
            shift
            ;;
        --no_plot)
            no_plot="true"
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

if [[ -z "${state_dir}" || -z "${output_dir}" || -z "${save_tag}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ ! -d "${state_dir}" ]]; then
    echo "State directory not found: ${state_dir}"
    exit 1
fi
if [[ "${all_states_recursive}" != "true" ]]; then
    if [[ -z "${num_replicas}" || -z "${replica_tag_prefix}" ]]; then
        echo "Default mode requires --num_replicas and --replica_tag_prefix."
        exit 1
    fi
    if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
        echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
        exit 1
    fi
fi
if ! [[ "${min_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--min_sweep must be an integer. Got '${min_sweep}'."
    exit 1
fi
if [[ -n "${max_sweep}" ]] && ! [[ "${max_sweep}" =~ ^-?[0-9]+$ ]]; then
    echo "--max_sweep must be an integer when provided. Got '${max_sweep}'."
    exit 1
fi

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
    done < <(find -L "${root_dir}" -type f -name "*_id-${id_tag}.jld2" -print0 2>/dev/null)

    printf "%s" "${best_path}"
}

state_list_file="$(mktemp)"
trap 'rm -f "${state_list_file}"' EXIT

if [[ "${all_states_recursive}" == "true" ]]; then
    declare -A best_path_by_id=()
    declare -A best_mtime_by_id=()
    while IFS= read -r -d '' candidate; do
        base_name="$(basename "${candidate}")"
        if [[ "${base_name}" =~ _id-([^.]+)\.jld2$ ]]; then
            id_tag="${BASH_REMATCH[1]}"
        else
            id_tag="${base_name%.jld2}"
        fi
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ -z "${best_mtime_by_id[${id_tag}]:-}" ]] || (( mtime >= best_mtime_by_id[${id_tag}] )); then
            best_mtime_by_id["${id_tag}"]="${mtime}"
            best_path_by_id["${id_tag}"]="${candidate}"
        fi
    done < <(find -L "${state_dir}" -type f -name "*.jld2" -print0 2>/dev/null)

    if (( ${#best_path_by_id[@]} == 0 )); then
        echo "No saved active-object states found under ${state_dir}"
        exit 1
    fi

    printf "%s\n" "${!best_path_by_id[@]}" | sort | while IFS= read -r id_tag; do
        [[ -n "${id_tag}" ]] || continue
        echo "${best_path_by_id[${id_tag}]}" >> "${state_list_file}"
    done
else
    for ((replica_idx = 1; replica_idx <= num_replicas; replica_idx++)); do
        replica_tag="${replica_tag_prefix}${replica_idx}"
        matched_state="$(latest_state_for_id_tag "${state_dir}" "${replica_tag}")"
        if [[ -z "${matched_state}" ]]; then
            echo "Missing replica state for replica ${replica_idx} (tag=${replica_tag}) in ${state_dir}"
            exit 1
        fi
        echo "${matched_state}" >> "${state_list_file}"
    done
fi

cmd=(julia --startup-file=no "${HISTOGRAM_SCRIPT}"
    --state_list "${state_list_file}"
    --output_dir "${output_dir}"
    --save_tag "${save_tag}"
    --min_sweep "${min_sweep}"
    --write_per_run)
if [[ -n "${max_sweep}" ]]; then
    cmd+=(--max_sweep "${max_sweep}")
fi
if [[ "${plot_per_run}" == "true" ]]; then
    cmd+=(--plot_per_run)
fi
if [[ "${no_plot}" == "true" ]]; then
    cmd+=(--no_plot)
fi

if [[ "${all_states_recursive}" == "true" ]]; then
    echo "Aggregating all unique active-object states under ${state_dir}"
else
    echo "Aggregating ${num_replicas} active-object steady-state histograms from ${state_dir}"
fi
echo "Output dir: ${output_dir}"
echo "Save tag: ${save_tag}"
"${cmd[@]}"
