#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash fetch_single_diffusive_1d_pmlr_state_and_plot.sh [options]

Options:
  --run_id <id>                 diffusive_1d_pmlr run id
                                (default: diffusive_1d_pmlr_L512_rho100_gamma1_V16_prod_ns1000000_nr600_20260420-032120)
  --mode <production|warmup>    Which run subtree to fetch from (default: production)
  --save_tag <id>               Exact state id tag to fetch, e.g. p20260421-153000_r17
  --state_glob <pattern>        Glob for state selection when --save_tag is omitted
                                (default: *.jld2, latest match by mtime is fetched)
  --remote_user <name>          SSH user (default: CLUSTER_REMOTE_USER)
  --remote_host <host>          SSH host (default: CLUSTER_REMOTE_HOST)
  --remote_root <path>          Primary cluster root to probe first (default: CLUSTER_CODE_ROOT)
  --secondary_remote_root <path>
                                Secondary cluster root to probe if the first misses
                                (default: CLUSTER_DATA_ROOT)
  --local_root <path>           Local root for downloaded data
                                (default: <repo>/cluster_results)
  --julia_bin <path>            Julia executable for plotting (default: julia)
  --no_plot                     Fetch only; skip load_and_plot.jl
  -h, --help                    Show help

Behavior:
  - Fetches exactly one non-aggregated state from the cluster.
  - For production runs it looks under states/raw.
  - For warmup runs it looks under states.
  - If --save_tag is omitted, the newest matching state is fetched.
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

cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

run_id="diffusive_1d_pmlr_L512_rho100_gamma1_V16_prod_ns1000000_nr600_20260420-032120"
mode="production"
save_tag=""
state_glob="*.jld2"
remote_user="${REMOTE_USER:-${CLUSTER_REMOTE_USER:-}}"
remote_host="${REMOTE_HOST:-${CLUSTER_REMOTE_HOST:-}}"
remote_root="${REMOTE_ROOT:-${CLUSTER_CODE_ROOT:-}}"
secondary_remote_root="${SECONDARY_REMOTE_ROOT:-${CLUSTER_DATA_ROOT:-}}"
local_root="${LOCAL_RESULTS_ROOT:-${REPO_ROOT}/cluster_results}"
julia_setup="${JULIA_SETUP_SCRIPT:-${CLUSTER_JULIA_SETUP_SCRIPT:-}}"
julia_bin="${JULIA_BIN:-julia}"
plot_after_fetch="true"

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
        --save_tag)
            save_tag="${2:-}"
            shift 2
            ;;
        --state_glob)
            state_glob="${2:-}"
            shift 2
            ;;
        --remote_user)
            remote_user="${2:-}"
            shift 2
            ;;
        --remote_host)
            remote_host="${2:-}"
            shift 2
            ;;
        --remote_root)
            remote_root="${2:-}"
            shift 2
            ;;
        --secondary_remote_root)
            secondary_remote_root="${2:-}"
            shift 2
            ;;
        --local_root)
            local_root="${2:-}"
            shift 2
            ;;
        --julia_bin)
            julia_bin="${2:-}"
            shift 2
            ;;
        --no_plot)
            plot_after_fetch="false"
            shift 1
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

case "${mode}" in
    production|warmup)
        ;;
    *)
        echo "--mode must be production or warmup. Got '${mode}'."
        exit 1
        ;;
esac

if [[ -z "${remote_user}" || -z "${remote_host}" ]]; then
    echo "Missing cluster SSH settings. Set cluster_scripts/cluster_env.sh or pass --remote_user/--remote_host."
    exit 1
fi
if [[ -z "${remote_root}" && -z "${secondary_remote_root}" ]]; then
    echo "Need at least one remote root. Pass --remote_root and/or --secondary_remote_root."
    exit 1
fi
if [[ -z "${state_glob}" ]]; then
    echo "--state_glob cannot be empty."
    exit 1
fi

ssh_remote() {
    ssh "${remote_user}@${remote_host}" "$@"
}

scp_remote() {
    scp "$@"
}

remote_run_rel="runs/diffusive_1d_pmlr/${mode}/${run_id}"
state_subpath="states"
if [[ "${mode}" == "production" ]]; then
    state_subpath="states/raw"
fi

remote_run_dir=""
for candidate_root in "${remote_root}" "${secondary_remote_root}"; do
    [[ -n "${candidate_root}" ]] || continue
    candidate_run_dir="${candidate_root}/${remote_run_rel}"
    if ssh_remote "test -d $(printf '%q' "${candidate_run_dir}")"; then
        remote_run_dir="${candidate_run_dir}"
        break
    fi
done

if [[ -z "${remote_run_dir}" ]]; then
    echo "Could not resolve remote run root for ${run_id}."
    echo "Tried:"
    [[ -n "${remote_root}" ]] && echo "  ${remote_root}/${remote_run_rel}"
    [[ -n "${secondary_remote_root}" ]] && echo "  ${secondary_remote_root}/${remote_run_rel}"
    exit 1
fi

remote_state_dir="${remote_run_dir}/${state_subpath}"
if ! ssh_remote "test -d $(printf '%q' "${remote_state_dir}")"; then
    echo "Remote state directory not found: ${remote_state_dir}"
    exit 1
fi

local_run_dir="${local_root}/runs/diffusive_1d_pmlr/${mode}/${run_id}"
local_state_dir="${local_run_dir}/${state_subpath}"
mkdir -p "${local_state_dir}"

if ssh_remote "test -f $(printf '%q' "${remote_run_dir}/run_info.txt")"; then
    scp_remote "${remote_user}@${remote_host}:${remote_run_dir}/run_info.txt" "${local_run_dir}/run_info.txt"
fi

if [[ -n "${save_tag}" ]]; then
    remote_select_cmd=$(
        cat <<EOF
cd $(printf '%q' "${remote_state_dir}") && find . -maxdepth 1 -type f -name $(printf '%q' "*_id-${save_tag}.jld2") -printf '%T@ %f\n' | sort -nr | head -n 1
EOF
    )
else
    remote_select_cmd=$(
        cat <<EOF
cd $(printf '%q' "${remote_state_dir}") && find . -maxdepth 1 -type f -name $(printf '%q' "${state_glob}") -printf '%T@ %f\n' | sort -nr | head -n 1
EOF
    )
fi

selected_remote_file="$(
    ssh_remote "${remote_select_cmd}" | awk '{ $1=""; sub(/^ /,""); print }' | tail -n 1
)"

if [[ -z "${selected_remote_file}" ]]; then
    if [[ -n "${save_tag}" ]]; then
        echo "No remote state matched save_tag=${save_tag} under ${remote_state_dir}"
    else
        echo "No remote state matched pattern '${state_glob}' under ${remote_state_dir}"
    fi
    exit 1
fi

selected_basename="$(basename "${selected_remote_file}")"
local_state_path="${local_state_dir}/${selected_basename}"

echo "Fetching single state:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  remote_run_dir=${remote_run_dir}"
echo "  remote_state_dir=${remote_state_dir}"
echo "  selected_file=${selected_basename}"
echo "  local_state_path=${local_state_path}"

scp_remote "${remote_user}@${remote_host}:${remote_state_dir}/${selected_basename}" "${local_state_path}"

if [[ "${plot_after_fetch}" != "true" ]]; then
    echo "Fetched state to: ${local_state_path}"
    exit 0
fi

if [[ -n "${julia_setup}" && -f "${julia_setup}" ]]; then
    # shellcheck disable=SC1090
    source "${julia_setup}"
fi

if ! command -v "${julia_bin}" >/dev/null 2>&1; then
    echo "Julia executable '${julia_bin}' not found."
    exit 1
fi

plot_out_dir="${local_run_dir}/reports/single_state_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${plot_out_dir}"

echo "Running load_and_plot.jl on the fetched state..."
"${julia_bin}" "${REPO_ROOT}/load_and_plot.jl" "${local_state_path}" --mode single --out_dir "${plot_out_dir}"

echo "Plots saved to: ${plot_out_dir}"
echo "Fetched state: ${local_state_path}"
