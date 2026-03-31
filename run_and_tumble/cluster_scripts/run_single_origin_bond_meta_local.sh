#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_single_origin_bond_meta_local.sh --mode <warmup|production> --L <int> --rho <value> --n_sweeps <int> [options]

Required:
  --mode              warmup or production
  --L                 system size (even integer)
  --rho               density value for ρ₀
  --n_sweeps          number of sweeps for selected mode

Optional:
  --warmup_state_dir  warmup state directory for production mode
  --num_replicas      number of independent replicas to run and aggregate (default: 1)
  --status_interval   progress snapshot period in seconds (default: 20)
  --plot_sweeps       enable plot_sweep display during local runs
  --ffr               fluctuation rate at origin bond (default: 1.0)
  --force_strength    forcing magnitude at origin bond (default: 1.0)
  --run_label         optional custom run label prefix
  -h, --help          show this help

Behavior:
  - warmup: runs without --initial_state
  - production: starts from latest warmup state
  - with --num_replicas > 1: runs replicated simulations in parallel workers and saves an aggregated state
  - creates a local run folder under runs/single_origin_bond/local/<mode>/<run_id>
  - runs load_and_plot.jl on the saved state into the run reports folder
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

GENERATE_SCRIPT="${SCRIPT_DIR}/generate_single_origin_bond_configs.sh"
RUNNER_SCRIPT="${SCRIPT_DIR}/run_diffusive_no_activity_from_config.sh"
if [[ ! -f "${GENERATE_SCRIPT}" || ! -f "${RUNNER_SCRIPT}" ]]; then
    echo "Could not find required scripts in ${SCRIPT_DIR}"
    exit 1
fi

mode=""
L_val=""
rho_val=""
n_sweeps_val=""
warmup_state_dir=""
status_interval="20"
plot_sweeps="false"
num_replicas="1"
ffr_val="1.0"
force_strength_val="1.0"
run_label=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            mode="${2:-}"
            shift 2
            ;;
        --L)
            L_val="${2:-}"
            shift 2
            ;;
        --rho)
            rho_val="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps_val="${2:-}"
            shift 2
            ;;
        --warmup_state_dir)
            warmup_state_dir="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --status_interval)
            status_interval="${2:-}"
            shift 2
            ;;
        --plot_sweeps)
            plot_sweeps="true"
            shift 1
            ;;
        --ffr)
            ffr_val="${2:-}"
            shift 2
            ;;
        --force_strength)
            force_strength_val="${2:-}"
            shift 2
            ;;
        --run_label)
            run_label="${2:-}"
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

is_number() {
    local value="$1"
    [[ "${value}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]
}

if [[ -z "${mode}" || -z "${L_val}" || -z "${rho_val}" || -z "${n_sweeps_val}" ]]; then
    echo "Missing required arguments."
    usage
    exit 1
fi

if [[ "${mode}" != "warmup" && "${mode}" != "production" ]]; then
    echo "--mode must be 'warmup' or 'production'."
    exit 1
fi

if ! [[ "${L_val}" =~ ^[0-9]+$ ]] || (( L_val <= 0 )) || (( L_val % 2 != 0 )); then
    echo "--L must be a positive even integer. Got '${L_val}'."
    exit 1
fi

if ! [[ "${n_sweeps_val}" =~ ^[0-9]+$ ]] || (( n_sweeps_val <= 0 )); then
    echo "--n_sweeps must be a positive integer. Got '${n_sweeps_val}'."
    exit 1
fi

if ! [[ "${status_interval}" =~ ^[0-9]+$ ]] || (( status_interval <= 0 )); then
    echo "--status_interval must be a positive integer. Got '${status_interval}'."
    exit 1
fi

if ! is_number "${ffr_val}"; then
    echo "--ffr must be numeric. Got '${ffr_val}'."
    exit 1
fi

if ! is_number "${force_strength_val}"; then
    echo "--force_strength must be numeric. Got '${force_strength_val}'."
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

build_show_times_csv() {
    local max_sweeps="$1"
    local -a times=()
    local scale=1
    while (( scale <= max_sweeps )); do
        for m in {1..9}; do
            local t=$((m * scale))
            if (( t <= max_sweeps )); then
                times+=("${t}")
            else
                break
            fi
        done
        scale=$((scale * 10))
    done
    local IFS=,
    echo "${times[*]}"
}

write_runtime_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local save_dir="$3"
    local show_csv="$4"
    local override_show="$5"
    local save_dir_line="save_dir: \"${save_dir}\""
    local show_line="show_times: [${show_csv}]"

    awk -v save_dir_line="${save_dir_line}" -v show_line="${show_line}" -v override_show="${override_show}" '
    BEGIN {seen_performance=0; seen_cluster=0; seen_save=0; seen_show=0}
    {
        if ($0 ~ /^performance_mode:[[:space:]]*/) {
            print "performance_mode: false"
            seen_performance=1
            next
        }
        if ($0 ~ /^cluster_mode:[[:space:]]*/) {
            print "cluster_mode: false"
            seen_cluster=1
            next
        }
        if ($0 ~ /^save_dir:[[:space:]]*/) {
            print save_dir_line
            seen_save=1
            next
        }
        if (override_show == "true" && $0 ~ /^show_times:[[:space:]]*/) {
            print show_line
            seen_show=1
            next
        }
        print
    }
    END {
        if (!seen_performance) print "performance_mode: false"
        if (!seen_cluster) print "cluster_mode: false"
        if (!seen_save) print save_dir_line
        if (override_show == "true" && !seen_show) print show_line
    }' "${src_cfg}" > "${dst_cfg}"
}

extract_progress_percent() {
    local log_file="$1"
    if [[ ! -s "${log_file}" ]]; then
        echo 0
        return
    fi
    local pct
    pct="$(
        tr '\r' '\n' < "${log_file}" \
            | grep -Eo '[0-9]{1,3}%' \
            | tail -n 1 \
            | tr -d '%' \
            || true
    )"
    if [[ -n "${pct}" && "${pct}" =~ ^[0-9]+$ ]]; then
        if (( pct > 100 )); then
            pct=100
        fi
        echo "${pct}"
        return
    fi
    if grep -q "Simulation complete" "${log_file}" 2>/dev/null; then
        echo 100
        return
    fi
    echo 0
}

extract_saved_state_from_log() {
    local log_file="$1"
    local raw
    raw="$(
        grep -E 'Final state saved to:|Saved a state to ' "${log_file}" 2>/dev/null \
            | grep -Eo '[^[:space:]]+\.jld2' \
            | tail -n 1 \
            || true
    )"
    if [[ -z "${raw}" ]]; then
        echo ""
        return
    fi
    if [[ "${raw}" = /* ]]; then
        echo "${raw}"
    else
        echo "${REPO_ROOT}/${raw}"
    fi
}

export L="${L_val}"
export RHO0="${rho_val}"
export FFR="${ffr_val}"
export FORCE_STRENGTH="${force_strength_val}"
if [[ "${mode}" == "warmup" ]]; then
    export WARMUP_SWEEPS="${n_sweeps_val}"
else
    export PRODUCTION_SWEEPS="${n_sweeps_val}"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
rho_tag="$(slugify "${rho_val}")"
if [[ -z "${run_label}" ]]; then
    run_label="${mode}_L${L_val}_rho${rho_tag}_ns${n_sweeps_val}_f${force_strength_val}_ffr${ffr_val}_local"
    if (( num_replicas > 1 )); then
        run_label="${run_label}_nr${num_replicas}"
    fi
fi
run_label="$(slugify "${run_label}")"
run_id="${run_label}_${timestamp}"
run_root="${REPO_ROOT}/runs/single_origin_bond/local/${mode}/${run_id}"
run_config_dir="${run_root}/configs"
run_log_dir="${run_root}/logs"
run_state_dir="${run_root}/states"
run_report_dir="${run_root}/reports"
run_manifest="${run_root}/manifest.csv"
run_info="${run_root}/run_info.txt"
local_registry="${REPO_ROOT}/runs/single_origin_bond/local/run_registry.csv"

mkdir -p "${run_config_dir}" "${run_log_dir}" "${run_state_dir}" "${run_report_dir}"

echo "Preparing local single-origin-bond run:"
echo "  run_id=${run_id}"
echo "  mode=${mode}"
echo "  L=${L}"
echo "  rho0=${RHO0}"
echo "  n_sweeps=${n_sweeps_val}"
echo "  num_replicas=${num_replicas}"
echo "  force_strength=${FORCE_STRENGTH}"
echo "  ffr=${FFR}"
echo "  status_interval=${status_interval}s"
echo "  plot_sweeps=${plot_sweeps}"
echo "  run_root=${run_root}"

bash "${GENERATE_SCRIPT}"

if [[ "${mode}" == "production" && -z "${warmup_state_dir}" ]]; then
    if [[ -f "${local_registry}" ]]; then
        warmup_state_dir="$(
            awk -F, -v L="${L}" -v rho="${RHO0}" -v ffr="${FFR}" -v f="${FORCE_STRENGTH}" '
                $3=="warmup" && $4==L && $5==rho {
                    if (NF >= 8 && ($7 != ffr || $8 != f)) next
                    state_dir=$11
                }
                END {if (state_dir != "") print state_dir}
            ' "${local_registry}"
        )"
    fi
    if [[ -z "${warmup_state_dir}" ]]; then
        cluster_registry="${REPO_ROOT}/runs/single_origin_bond/run_registry.csv"
        if [[ -f "${cluster_registry}" ]]; then
            warmup_state_dir="$(
                awk -F, -v L="${L}" -v rho="${RHO0}" -v ffr="${FFR}" -v f="${FORCE_STRENGTH}" '
                    $3=="warmup" && $4==L && $5==rho {
                        if (NF >= 14 && ($13 != ffr || $14 != f)) next
                        state_dir=$11
                    }
                    END {if (state_dir != "") print state_dir}
                ' "${cluster_registry}"
            )"
        fi
    fi
    if [[ -z "${warmup_state_dir}" ]]; then
        warmup_state_dir="${REPO_ROOT}/saved_states/single_origin_bond/warmup"
    fi
fi

source_config="${REPO_ROOT}/configuration_files/single_origin_bond/${mode}/params.yaml"
if [[ ! -f "${source_config}" ]]; then
    echo "Missing generated config: ${source_config}"
    exit 1
fi

runtime_config="${run_config_dir}/single_origin_bond_${mode}.yaml"
show_csv=""
if [[ "${plot_sweeps}" == "true" ]]; then
    show_csv="$(build_show_times_csv "${n_sweeps_val}")"
fi
write_runtime_config "${source_config}" "${runtime_config}" "${run_state_dir}" "${show_csv}" "${plot_sweeps}"

initial_state=""
if [[ "${mode}" == "production" ]]; then
    initial_state="$(ls -1t "${warmup_state_dir}"/single_origin_bond_warmup_*.jld2 2>/dev/null | head -n 1 || true)"
    if [[ -z "${initial_state}" ]]; then
        initial_state="$(ls -1t "${warmup_state_dir}"/*.jld2 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -z "${initial_state}" ]]; then
        echo "ERROR: no warmup state found under ${warmup_state_dir}"
        exit 1
    fi
fi

run_log="${run_log_dir}/single_origin_bond_${mode}.log"
runner_extra_args=()
if (( num_replicas > 1 )); then
    runner_extra_args+=(--num_runs "${num_replicas}" --save_tag "aggregated_${run_id}")
fi
if [[ "${mode}" == "warmup" ]]; then
    bash "${RUNNER_SCRIPT}" "${runtime_config}" "${runner_extra_args[@]}" > "${run_log}" 2>&1 &
else
    bash "${RUNNER_SCRIPT}" "${runtime_config}" --initial_state "${initial_state}" "${runner_extra_args[@]}" > "${run_log}" 2>&1 &
fi
run_pid="$!"

next_status_at="$(date +%s)"
while kill -0 "${run_pid}" 2>/dev/null; do
    now_epoch="$(date +%s)"
    if (( now_epoch >= next_status_at )); then
        pct="$(extract_progress_percent "${run_log}")"
        printf "[%s] %s RUN %3d%% log=%s\n" "$(date +%H:%M:%S)" "${mode}" "${pct}" "${run_log}"
        next_status_at=$((now_epoch + status_interval))
    fi
    sleep 1
done

if ! wait "${run_pid}"; then
    echo "Local ${mode} run failed. Log: ${run_log}"
    exit 1
fi

saved_state="$(extract_saved_state_from_log "${run_log}")"
if [[ -z "${saved_state}" || ! -f "${saved_state}" ]]; then
    saved_state="$(ls -1t "${run_state_dir}"/*.jld2 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${saved_state}" || ! -f "${saved_state}" ]]; then
    echo "WARNING: run completed but no saved state found in ${run_state_dir}"
else
    echo "Saved state: ${saved_state}"
fi

plot_status="SKIPPED"
if [[ -n "${saved_state}" && -f "${saved_state}" ]]; then
    julia_setup="${JULIA_SETUP_SCRIPT:-/Local/ph_kafri/julia-1.7.2/bin/setup.sh}"
    if [[ -f "${julia_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${julia_setup}"
    fi
    julia_bin="${JULIA_BIN:-julia}"
    if command -v "${julia_bin}" >/dev/null 2>&1; then
        mkdir -p "${run_report_dir}"
        echo "Running load_and_plot.jl on saved state..."
        if "${julia_bin}" "${REPO_ROOT}/load_and_plot.jl" "${saved_state}" --mode single --out_dir "${run_report_dir}"; then
            plot_status="DONE"
        else
            plot_status="FAILED"
        fi
    else
        echo "WARNING: Julia executable '${julia_bin}' not found. Skipping post-plot."
    fi
fi

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=${mode}
L=${L}
rho0=${RHO0}
n_sweeps=${n_sweeps_val}
num_replicas=${num_replicas}
ffr=${FFR}
force_strength=${FORCE_STRENGTH}
run_root=${run_root}
config=${runtime_config}
log=${run_log}
state_dir=${run_state_dir}
saved_state=${saved_state}
reports_dir=${run_report_dir}
plot_status=${plot_status}
warmup_state_dir=${warmup_state_dir}
initial_state=${initial_state}
EOF

echo "mode,status,config_path,log_file,state_file,plot_status,reports_dir,initial_state" > "${run_manifest}"
printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${mode}" "DONE" "${runtime_config}" "${run_log}" "${saved_state}" "${plot_status}" "${run_report_dir}" "${initial_state}" \
    >> "${run_manifest}"

mkdir -p "$(dirname "${local_registry}")"
if [[ ! -f "${local_registry}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,ffr,force_strength,run_root,log_file,state_dir,reports_dir,warmup_state_dir,initial_state" > "${local_registry}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "${mode}" "${L}" "${RHO0}" "${n_sweeps_val}" "${FFR}" "${FORCE_STRENGTH}" \
    "${run_root}" "${run_log}" "${run_state_dir}" "${run_report_dir}" "${warmup_state_dir}" "${initial_state}" >> "${local_registry}"

echo "Local run completed."
echo "Manifest: ${run_manifest}"
echo "Run info: ${run_info}"
echo "Reports: ${run_report_dir}"
