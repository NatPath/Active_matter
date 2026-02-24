#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash copy_data_from_cluster.sh [options]

Core options:
  --mode <warmup|production>      Run mode for --list/--latest (for --run_id, registry mode is used)
  --run_id <id>                   Exact run_id to fetch
  --latest                        Fetch latest run_id in selected mode (can combine with filters)
  --list                          List run registry entries and exit
  --tail <N>                      Number of listed entries (default: 30)

Run filters for --list / --latest:
  --L <int>                       Match L column
  --rho <value>                   Match rho0 column
  --n_sweeps <int>                Match n_sweeps column
  --d_min <int>                   Match d_min column
  --d_max <int>                   Match d_max column
  --d_step <int>                  Match d_step column

Paths / connection:
  --remote_user <name>            SSH user (default: nativmr)
  --remote_host <host>            SSH host (default: tech-ui02.hep.technion.ac.il)
  --remote_root <path>            Cluster repo root (default: /storage/ph_kafri/nativmr/run_and_tumble)
  --local_root <path>             Local root for downloaded data (default: <repo>/cluster_results)

Post-processing:
  --plot                          Run load_and_plot.jl after sync
  --skip_per_state_sweep          Pass --skip_per_state_sweep to load_and_plot.jl
  --baseline_j2 <value>           Baseline for two_force_d analysis (default: 89000)
  --mode_plot <single|two_force_d>
                                  load_and_plot mode (default: two_force_d)

Examples:
  bash copy_data_from_cluster.sh --list --mode warmup
  bash copy_data_from_cluster.sh --latest --mode warmup --L 128 --rho 1000 --n_sweeps 100000
  bash copy_data_from_cluster.sh --run_id warmup_L128_rho1000_ns100000_d2-32-s2_20260223-120101 --mode warmup --plot --skip_per_state_sweep
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

mode="warmup"
mode_explicit="false"
run_id=""
latest="false"
list_only="false"
tail_n="30"

filter_L=""
filter_rho=""
filter_n_sweeps=""
filter_d_min=""
filter_d_max=""
filter_d_step=""

remote_user="${REMOTE_USER:-nativmr}"
remote_host="${REMOTE_HOST:-tech-ui02.hep.technion.ac.il}"
remote_root="${REMOTE_ROOT:-/storage/ph_kafri/nativmr/run_and_tumble}"
local_root="${LOCAL_RESULTS_ROOT:-${REPO_ROOT}/cluster_results}"

plot_after_sync="false"
skip_per_state="false"
baseline_j2="89000"
plot_mode="two_force_d"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            mode="${2:-}"
            mode_explicit="true"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --latest)
            latest="true"
            shift 1
            ;;
        --list)
            list_only="true"
            shift 1
            ;;
        --tail)
            tail_n="${2:-}"
            shift 2
            ;;
        --L)
            filter_L="${2:-}"
            shift 2
            ;;
        --rho)
            filter_rho="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            filter_n_sweeps="${2:-}"
            shift 2
            ;;
        --d_min)
            filter_d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            filter_d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            filter_d_step="${2:-}"
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
        --local_root)
            local_root="${2:-}"
            shift 2
            ;;
        --plot)
            plot_after_sync="true"
            shift 1
            ;;
        --skip_per_state_sweep)
            skip_per_state="true"
            shift 1
            ;;
        --baseline_j2)
            baseline_j2="${2:-}"
            shift 2
            ;;
        --mode_plot)
            plot_mode="${2:-}"
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

if [[ "${mode}" != "warmup" && "${mode}" != "production" ]]; then
    echo "--mode must be warmup or production. Got '${mode}'."
    exit 1
fi
if ! [[ "${tail_n}" =~ ^[0-9]+$ ]] || (( tail_n <= 0 )); then
    echo "--tail must be a positive integer. Got '${tail_n}'."
    exit 1
fi
if [[ "${plot_mode}" != "single" && "${plot_mode}" != "two_force_d" ]]; then
    echo "--mode_plot must be single or two_force_d. Got '${plot_mode}'."
    exit 1
fi

registry_rel="runs/two_force_d/run_registry.csv"
local_registry="${local_root}/${registry_rel}"
mkdir -p "$(dirname "${local_registry}")"

echo "Syncing run registry from cluster..."
scp "${remote_user}@${remote_host}:${remote_root}/${registry_rel}" "${local_registry}"

if [[ ! -f "${local_registry}" ]]; then
    echo "Registry download failed: ${local_registry}"
    exit 1
fi

awk_filter='
NR==1 {print; next}
{
    keep = 1
    if (mode != "" && $3 != mode) keep = 0
    if (fL != "" && $4 != fL) keep = 0
    if (fRho != "" && $5 != fRho) keep = 0
    if (fSweeps != "" && $6 != fSweeps) keep = 0
    if (fDmin != "" && $7 != fDmin) keep = 0
    if (fDmax != "" && $8 != fDmax) keep = 0
    if (fDstep != "" && $9 != fDstep) keep = 0
    if (keep) print
}'

if [[ "${list_only}" == "true" ]]; then
    filtered="$(mktemp)"
    awk -F, \
        -v mode="${mode}" \
        -v fL="${filter_L}" \
        -v fRho="${filter_rho}" \
        -v fSweeps="${filter_n_sweeps}" \
        -v fDmin="${filter_d_min}" \
        -v fDmax="${filter_d_max}" \
        -v fDstep="${filter_d_step}" \
        "${awk_filter}" "${local_registry}" > "${filtered}"

    echo "Registry: ${local_registry}"
    echo "Showing last ${tail_n} matching entries:"
    if command -v column >/dev/null 2>&1; then
        tail -n "${tail_n}" "${filtered}" | column -s, -t
    else
        tail -n "${tail_n}" "${filtered}"
    fi
    rm -f "${filtered}"
    exit 0
fi

if [[ -z "${run_id}" ]]; then
    if [[ "${latest}" != "true" ]]; then
        echo "Provide --run_id <id> or use --latest."
        exit 1
    fi
    run_id="$(
        awk -F, \
            -v mode="${mode}" \
            -v fL="${filter_L}" \
            -v fRho="${filter_rho}" \
            -v fSweeps="${filter_n_sweeps}" \
            -v fDmin="${filter_d_min}" \
            -v fDmax="${filter_d_max}" \
            -v fDstep="${filter_d_step}" '
            NR==1 {next}
            {
                keep = 1
                if (mode != "" && $3 != mode) keep = 0
                if (fL != "" && $4 != fL) keep = 0
                if (fRho != "" && $5 != fRho) keep = 0
                if (fSweeps != "" && $6 != fSweeps) keep = 0
                if (fDmin != "" && $7 != fDmin) keep = 0
                if (fDmax != "" && $8 != fDmax) keep = 0
                if (fDstep != "" && $9 != fDstep) keep = 0
                if (keep) chosen = $2
            }
            END {print chosen}
        ' "${local_registry}"
    )"
    if [[ -z "${run_id}" ]]; then
        echo "No run matches the requested filters."
        exit 1
    fi
fi

registry_row="$(
    awk -F, -v rid="${run_id}" '
        NR == 1 {next}
        $2 == rid {row = $0}
        END {print row}
    ' "${local_registry}"
)"

if [[ -z "${registry_row}" ]]; then
    echo "run_id '${run_id}' not found in registry: ${local_registry}"
    exit 1
fi

IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_dmin reg_dmax reg_dstep reg_cpus reg_mem reg_run_root reg_log_dir reg_state_dir reg_warmup_state_dir <<< "${registry_row}"

if [[ "${mode_explicit}" == "true" && "${mode}" != "${reg_mode}" ]]; then
    echo "WARNING: --mode='${mode}' does not match registry mode='${reg_mode}' for run_id='${run_id}'. Using registry mode."
fi
mode="${reg_mode}"

if [[ -n "${reg_run_root}" ]]; then
    remote_run_dir="${reg_run_root}"
else
    remote_run_dir="${remote_root}/runs/two_force_d/${mode}/${run_id}"
fi

local_run_dir="${local_root}/runs/two_force_d/${mode}/${run_id}"
mkdir -p "${local_run_dir}"

echo "Fetching run:"
echo "  run_id=${run_id}"
echo "  from=${remote_user}@${remote_host}:${remote_run_dir}"
echo "  to=${local_run_dir}"
rsync -av --progress "${remote_user}@${remote_host}:${remote_run_dir}/" "${local_run_dir}/"

if [[ "${plot_after_sync}" == "true" ]]; then
    states_dir="${local_run_dir}/states"
    if [[ ! -d "${states_dir}" ]]; then
        echo "No states directory found at ${states_dir}; skipping plot."
        exit 0
    fi
    mapfile -t state_files < <(find "${states_dir}" -maxdepth 1 -type f -name '*.jld2' | sort)
    if (( ${#state_files[@]} == 0 )); then
        echo "No JLD2 states found in ${states_dir}; skipping plot."
        exit 0
    fi

    julia_setup="${JULIA_SETUP_SCRIPT:-/Local/ph_kafri/julia-1.7.2/bin/setup.sh}"
    if [[ -f "${julia_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${julia_setup}"
    fi
    julia_bin="${JULIA_BIN:-julia}"
    if ! command -v "${julia_bin}" >/dev/null 2>&1; then
        echo "Julia executable '${julia_bin}' not found. Skipping plot."
        exit 0
    fi

    out_dir="${local_run_dir}/reports/load_and_plot_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${out_dir}"

    cmd=("${julia_bin}" "${REPO_ROOT}/load_and_plot.jl")
    cmd+=("${state_files[@]}")
    cmd+=("--mode" "${plot_mode}" "--out_dir" "${out_dir}" "--baseline_j2" "${baseline_j2}")
    if [[ "${skip_per_state}" == "true" ]]; then
        cmd+=("--skip_per_state_sweep")
    fi

    echo "Running load_and_plot on ${#state_files[@]} state(s)..."
    "${cmd[@]}"
    echo "Plots saved to: ${out_dir}"
fi

echo "Done. Local run folder: ${local_run_dir}"
