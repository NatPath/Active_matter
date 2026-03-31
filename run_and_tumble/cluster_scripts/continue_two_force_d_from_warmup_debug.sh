#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash continue_two_force_d_from_warmup_debug.sh \
      --run_id <id> \
      --d_value <int> \
      --continue_sweeps <int> \
      [options]

Required:
  --run_id <id>              Existing two_force_d production or warmup_production run_id
  --d_value <int>            Target d value to continue from warmup
  --continue_sweeps <int>    Number of sweeps for the debug continuation

Options:
  --mode <auto|production|warmup_production>
                             How to resolve --run_id (default: auto)
  --out_root <path>          Root directory for debug outputs
                             (default: runs/two_force_d/warmup_continue_debug)
  --save_tag <tag>           Save tag override
  --cluster_mode <true|false>
                             Whether to force cluster_mode in the debug config
                             (default: true, for lean headless continuation)
  -h, --help                 Show help

Behavior:
  - Resolves the target production run/config directory from --run_id.
  - Resolves the matching warmup state directory for that run.
  - Picks the latest warmup state for --d_value.
  - Writes a dedicated debug config and output directory.
  - Continues the warmup state through the real --initial_state path.
  - Inspects the resulting saved state with inspect_two_force_raw_states.jl.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UTILS="${SCRIPT_DIR}/two_force_d_add_repeats_utils.sh"
RUNNER="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
INSPECTOR="${SCRIPT_DIR}/inspect_two_force_raw_states.jl"

if [[ ! -f "${UTILS}" || ! -f "${RUNNER}" || ! -f "${INSPECTOR}" ]]; then
    echo "Missing required helper(s) under ${SCRIPT_DIR}."
    exit 1
fi

# shellcheck disable=SC1090
source "${UTILS}"

run_id=""
mode="auto"
d_value=""
continue_sweeps=""
out_root="${REPO_ROOT}/runs/two_force_d/warmup_continue_debug"
save_tag=""
cluster_mode_override="true"

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
        --d_value)
            d_value="${2:-}"
            shift 2
            ;;
        --continue_sweeps)
            continue_sweeps="${2:-}"
            shift 2
            ;;
        --out_root)
            out_root="${2:-}"
            shift 2
            ;;
        --save_tag)
            save_tag="${2:-}"
            shift 2
            ;;
        --cluster_mode)
            cluster_mode_override="${2:-}"
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

if [[ -z "${run_id}" || -z "${d_value}" || -z "${continue_sweeps}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi
if ! [[ "${d_value}" =~ ^[0-9]+$ ]]; then
    echo "--d_value must be an integer."
    exit 1
fi
if ! [[ "${continue_sweeps}" =~ ^[0-9]+$ ]] || (( continue_sweeps <= 0 )); then
    echo "--continue_sweeps must be a positive integer."
    exit 1
fi
case "${mode}" in
    auto|production|warmup_production) ;;
    *)
        echo "--mode must be one of: auto, production, warmup_production."
        exit 1
        ;;
esac
case "$(printf "%s" "${cluster_mode_override}" | tr '[:upper:]' '[:lower:]')" in
    true|false|1|0|yes|no|on|off) ;;
    *)
        echo "--cluster_mode must be boolean-like."
        exit 1
        ;;
esac

rewrite_debug_config() {
    local source_config="$1"
    local target_config="$2"
    local save_dir="$3"
    local sweeps="$4"
    local cluster_mode_value="$5"
    awk -v save_dir_line="save_dir: \"${save_dir}\"" \
        -v sweeps_line="n_sweeps: ${sweeps}" \
        -v cluster_mode_line="cluster_mode: ${cluster_mode_value}" \
        '
        BEGIN {seen_save=0; seen_sweeps=0; seen_cluster=0; seen_save_times=0}
        {
            if ($0 ~ /^n_sweeps:[[:space:]]*/) {
                print sweeps_line
                seen_sweeps=1
                next
            }
            if ($0 ~ /^save_dir:[[:space:]]*/) {
                print save_dir_line
                seen_save=1
                next
            }
            if ($0 ~ /^cluster_mode:[[:space:]]*/) {
                print cluster_mode_line
                seen_cluster=1
                next
            }
            if ($0 ~ /^save_times:[[:space:]]*/) {
                print "save_times: []"
                seen_save_times=1
                next
            }
            print
        }
        END {
            if (!seen_sweeps) print sweeps_line
            if (!seen_save) print save_dir_line
            if (!seen_cluster) print cluster_mode_line
            if (!seen_save_times) print "save_times: []"
        }' "${source_config}" > "${target_config}"
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

target_run_info="$(two_force_resolve_target_production_run_info "${REPO_ROOT}" "${run_id}" "${mode}")"
target_run_id="$(two_force_read_key_value "${target_run_info}" "run_id")"
target_config_dir="$(two_force_read_key_value "${target_run_info}" "config_dir")"
warmup_state_dir="$(two_force_resolve_warmup_state_dir_for_run "${REPO_ROOT}" "${target_run_info}")"

source_config="${target_config_dir}/d_${d_value}.yaml"
if [[ ! -f "${source_config}" ]]; then
    fallback_config="${REPO_ROOT}/configuration_files/two_force_d_sweep/production/d_${d_value}.yaml"
    if [[ -f "${fallback_config}" ]]; then
        source_config="${fallback_config}"
    else
        echo "Could not locate config for d=${d_value}."
        exit 1
    fi
fi

warmup_state="$(latest_matching_state "${warmup_state_dir}" \
    "aggregated/two_force_d${d_value}_warmup_*.jld2" \
    "two_force_d${d_value}_warmup_*.jld2" \
    "aggregated/two_force_d${d_value}_*.jld2" \
    "two_force_d${d_value}_*.jld2" || true)"
if [[ -z "${warmup_state}" ]]; then
    echo "Could not resolve a warmup state for d=${d_value} under ${warmup_state_dir}."
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
safe_run_id="$(two_force_sanitize_token "${target_run_id}")"
job_root="${out_root}/${safe_run_id}/d_${d_value}/${timestamp}"
config_dir="${job_root}/configs"
state_dir="${job_root}/states"
mkdir -p "${config_dir}" "${state_dir}"

runtime_config="${config_dir}/d_${d_value}_warmup_continue_debug.yaml"
rewrite_debug_config "${source_config}" "${runtime_config}" "${state_dir}" "${continue_sweeps}" "${cluster_mode_override}"

if [[ -z "${save_tag}" ]]; then
    save_tag="warmup_continue_debug_${timestamp}_d${d_value}"
fi

echo "Warmup continuation debug run:"
echo "  run_id=${run_id}"
echo "  target_run_id=${target_run_id}"
echo "  target_run_info=${target_run_info}"
echo "  warmup_state_dir=${warmup_state_dir}"
echo "  warmup_state=${warmup_state}"
echo "  source_config=${source_config}"
echo "  runtime_config=${runtime_config}"
echo "  continue_sweeps=${continue_sweeps}"
echo "  cluster_mode=${cluster_mode_override}"
echo "  save_tag=${save_tag}"
echo "  output_root=${job_root}"

bash "${RUNNER}" "${runtime_config}" \
    --initial_state "${warmup_state}" \
    --continue_sweeps "${continue_sweeps}" \
    --save_tag "${save_tag}"

result_state="$(ls -1t "${state_dir}"/*_id-${save_tag}.jld2 2>/dev/null | head -n 1 || true)"
if [[ -z "${result_state}" ]]; then
    echo "Continuation finished but no saved state matching save_tag='${save_tag}' was found in ${state_dir}."
    exit 1
fi

echo
echo "Inspecting result state:"
echo "  result_state=${result_state}"
julia --startup-file=no "${INSPECTOR}" "${result_state}"
