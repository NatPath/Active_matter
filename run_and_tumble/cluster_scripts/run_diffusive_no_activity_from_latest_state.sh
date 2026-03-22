#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_diffusive_no_activity_from_latest_state.sh \
      --runner_script <path> \
      --config <path> \
      --state_dir <path> \
      --pattern <glob> [--pattern <glob> ...] \
      [--state_arg_name <initial_state|continue>] \
      [--num_runs <int>] \
      [--save_tag <tag>]

Behavior:
  - resolves the latest state file under --state_dir using patterns in order
  - runs the configured simulation with --<state_arg_name> <resolved_file>
  - forwards optional --num_runs / --save_tag to run_diffusive_no_activity.jl wrapper
EOF
}

runner_script=""
config_path=""
state_dir=""
state_arg_name="initial_state"
num_runs=""
save_tag=""
patterns=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner_script)
            runner_script="${2:-}"
            shift 2
            ;;
        --config)
            config_path="${2:-}"
            shift 2
            ;;
        --state_dir)
            state_dir="${2:-}"
            shift 2
            ;;
        --state_arg_name)
            state_arg_name="${2:-}"
            shift 2
            ;;
        --pattern)
            patterns+=("${2:-}")
            shift 2
            ;;
        --num_runs)
            num_runs="${2:-}"
            shift 2
            ;;
        --save_tag)
            save_tag="${2:-}"
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

if [[ -z "${runner_script}" || -z "${config_path}" || -z "${state_dir}" || "${#patterns[@]}" -eq 0 ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ ! -f "${runner_script}" ]]; then
    echo "Runner script not found: ${runner_script}"
    exit 1
fi
if [[ ! -f "${config_path}" ]]; then
    echo "Config file not found: ${config_path}"
    exit 1
fi
if [[ ! -d "${state_dir}" ]]; then
    echo "State directory not found: ${state_dir}"
    exit 1
fi
case "${state_arg_name}" in
    initial_state|continue)
        ;;
    *)
        echo "--state_arg_name must be 'initial_state' or 'continue'. Got '${state_arg_name}'."
        exit 1
        ;;
esac
if [[ -n "${num_runs}" ]]; then
    if ! [[ "${num_runs}" =~ ^[0-9]+$ ]] || (( num_runs <= 0 )); then
        echo "--num_runs must be a positive integer. Got '${num_runs}'."
        exit 1
    fi
fi

latest_state=""
for pattern in "${patterns[@]}"; do
    candidate="$(ls -1t "${state_dir}"/${pattern} 2>/dev/null | head -n 1 || true)"
    if [[ -n "${candidate}" ]]; then
        latest_state="${candidate}"
        break
    fi
done

if [[ -z "${latest_state}" ]]; then
    echo "Could not resolve any state under ${state_dir} for patterns: ${patterns[*]}"
    exit 1
fi

echo "Resolved state: ${latest_state}"

runner_args=("${config_path}" "--${state_arg_name}" "${latest_state}")
if [[ -n "${num_runs}" && "${num_runs}" != "1" ]]; then
    runner_args+=("--num_runs" "${num_runs}")
fi
if [[ -n "${save_tag}" ]]; then
    runner_args+=("--save_tag" "${save_tag}")
fi

bash "${runner_script}" "${runner_args[@]}"
