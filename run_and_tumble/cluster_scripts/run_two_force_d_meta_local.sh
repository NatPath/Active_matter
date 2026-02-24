#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_two_force_d_meta_local.sh --mode <warmup|production> --L <int> --rho <value> --n_sweeps <int> [options]

Required:
  --mode              warmup or production
  --L                 system size (even integer)
  --rho               density value for ρ₀
  --n_sweeps          number of sweeps for selected mode

Optional:
  --warmup_state_dir  warmup state directory for production mode
  --d_min             minimum d (default: 2)
  --d_max             maximum d (default: L/4)
  --d_step            d step (default: 2)
  --status_interval   progress snapshot period in seconds (default: 20)
  --plot_sweeps       enable plot_sweep display during local runs
  -h, --help          show this help

Behavior:
  - warmup: runs configs without --initial_state
  - production: finds matching warmup state for each d and errors if any is missing
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

GENERATE_SCRIPT="${SCRIPT_DIR}/generate_two_force_d_sweep_configs.sh"
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
d_min="2"
d_max=""
d_step="2"
status_interval="20"
plot_sweeps="false"

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
        --d_min)
            d_min="${2:-}"
            shift 2
            ;;
        --d_max)
            d_max="${2:-}"
            shift 2
            ;;
        --d_step)
            d_step="${2:-}"
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

if ! [[ "${d_min}" =~ ^[0-9]+$ ]] || ! [[ "${d_step}" =~ ^[0-9]+$ ]]; then
    echo "--d_min and --d_step must be positive integers."
    exit 1
fi
if ! [[ "${status_interval}" =~ ^[0-9]+$ ]] || (( status_interval <= 0 )); then
    echo "--status_interval must be a positive integer. Got '${status_interval}'."
    exit 1
fi

if [[ -z "${d_max}" ]]; then
    d_max="$((L_val / 4))"
fi
if ! [[ "${d_max}" =~ ^[0-9]+$ ]]; then
    echo "--d_max must be an integer. Got '${d_max}'."
    exit 1
fi
if (( d_step <= 0 || d_max < d_min )); then
    echo "Invalid d range: d_min=${d_min}, d_max=${d_max}, d_step=${d_step}."
    exit 1
fi

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
    local show_csv="$3"
    local show_line="show_times: [${show_csv}]"

    awk -v show_line="${show_line}" '
    BEGIN {seen_cluster=0; seen_show=0}
    {
        if ($0 ~ /^cluster_mode:[[:space:]]*/) {
            print "cluster_mode: false"
            seen_cluster=1
            next
        }
        if ($0 ~ /^show_times:[[:space:]]*/) {
            print show_line
            seen_show=1
            next
        }
        print
    }
    END {
        if (!seen_cluster) print "cluster_mode: false"
        if (!seen_show) print show_line
    }' "${src_cfg}" > "${dst_cfg}"
}

export L="${L_val}"
export RHO0="${rho_val}"
export D_MIN="${d_min}"
export D_MAX="${d_max}"
export D_STEP="${d_step}"

if [[ "${mode}" == "warmup" ]]; then
    export WARMUP_SWEEPS="${n_sweeps_val}"
else
    export PRODUCTION_SWEEPS="${n_sweeps_val}"
fi

echo "Preparing local two-force d sweep:"
echo "  mode=${mode}"
echo "  L=${L}"
echo "  rho0=${RHO0}"
echo "  n_sweeps=${n_sweeps_val}"
echo "  d range: ${D_MIN}:${D_STEP}:${D_MAX}"
echo "  status_interval=${status_interval}s"
echo "  plot_sweeps=${plot_sweeps}"

bash "${GENERATE_SCRIPT}"

if [[ "${mode}" == "production" ]]; then
    if [[ -z "${warmup_state_dir}" ]]; then
        warmup_state_dir="${REPO_ROOT}/saved_states/two_force_d_sweep/warmup"
    fi
fi

tmp_cfg_dir=""
show_csv=""
if [[ "${plot_sweeps}" == "true" ]]; then
    show_csv="$(build_show_times_csv "${n_sweeps_val}")"
    tmp_cfg_dir="$(mktemp -d)"
    trap '[[ -n "${tmp_cfg_dir}" ]] && rm -rf "${tmp_cfg_dir}"' EXIT
fi

render_bar() {
    local percent="$1"
    local width=24
    local filled=$(( percent * width / 100 ))
    local empty=$(( width - filled ))
    local bar=""
    local i
    for ((i=0; i<filled; i++)); do
        bar="${bar}#"
    done
    for ((i=0; i<empty; i++)); do
        bar="${bar}-"
    done
    printf "%s" "${bar}"
}

print_status_snapshot() {
    local mode_local="$1"
    local logs_dir_local="$2"
    local -n ds_ref=$3
    local -n statuses_ref=$4
    local -n logs_ref=$5
    local timestamp
    timestamp="$(date +%H:%M:%S)"
    printf "[%s] Local %s jobs: %d\n" "${timestamp}" "${mode_local}" "${#ds_ref[@]}"
    printf "Logs: %s\n" "${logs_dir_local}"
    for idx in "${!ds_ref[@]}"; do
        local d pct bar
        d="${ds_ref[idx]}"
        pct="$(extract_progress_percent "${logs_ref[idx]}")"
        if [[ "${statuses_ref[idx]}" == "DONE" ]]; then
            pct=100
        fi
        bar="$(render_bar "${pct}")"
        printf "d=%-4s %-5s [%s] %3d%%\n" "${d}" "${statuses_ref[idx]}" "${bar}" "${pct}"
    done
    printf -- "---------------------------------------------------------------\n"
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

logs_root="${REPO_ROOT}/condor_logs/local_two_force_d"
run_stamp="$(date +%Y%m%d-%H%M%S)"
logs_dir="${logs_root}/${mode}_${run_stamp}"
mkdir -p "${logs_dir}"
plot_out_dir="${REPO_ROOT}/results_figures/fitting/local_two_force_d/${mode}_${run_stamp}"

declare -a job_ds=()
declare -a job_configs=()
declare -a job_init_states=()
declare -a job_logs=()
declare -a job_pids=()
declare -a job_statuses=()

for d in $(seq "${D_MIN}" "${D_STEP}" "${D_MAX}"); do
    if (( d % 2 != 0 )); then
        continue
    fi

    config_path="${REPO_ROOT}/configuration_files/two_force_d_sweep/${mode}/d_${d}.yaml"
    if [[ ! -f "${config_path}" ]]; then
        echo "ERROR: missing config ${config_path}"
        exit 1
    fi

    runtime_config="${config_path}"
    if [[ "${plot_sweeps}" == "true" ]]; then
        runtime_config="${tmp_cfg_dir}/${mode}_d_${d}.yaml"
        write_runtime_config "${config_path}" "${runtime_config}" "${show_csv}"
    fi

    init_state=""
    if [[ "${mode}" == "warmup" ]]; then
        :
    else
        init_state="$(ls -1t "${warmup_state_dir}"/two_force_d${d}_warmup_* 2>/dev/null | head -n 1 || true)"
        if [[ -z "${init_state}" ]]; then
            echo "ERROR: no warmup state matching ${warmup_state_dir}/two_force_d${d}_warmup_* for d=${d}"
            exit 1
        fi
    fi

    job_ds+=("${d}")
    job_configs+=("${runtime_config}")
    job_init_states+=("${init_state}")
    job_logs+=("${logs_dir}/d_${d}.log")
    job_statuses+=("PENDING")
done

if (( ${#job_ds[@]} == 0 )); then
    echo "No jobs to run for d range ${D_MIN}:${D_STEP}:${D_MAX}."
    exit 0
fi

echo "Launching ${#job_ds[@]} local ${mode} jobs in parallel."
echo "Logs: ${logs_dir}"

cleanup_children() {
    for pid in "${job_pids[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap 'cleanup_children; echo; echo "Interrupted. Child processes terminated."; exit 130' INT TERM

for idx in "${!job_ds[@]}"; do
    d="${job_ds[idx]}"
    cfg="${job_configs[idx]}"
    log="${job_logs[idx]}"
    init_state="${job_init_states[idx]}"
    if [[ "${mode}" == "warmup" ]]; then
        bash "${RUNNER_SCRIPT}" "${cfg}" > "${log}" 2>&1 &
    else
        bash "${RUNNER_SCRIPT}" "${cfg}" --initial_state "${init_state}" > "${log}" 2>&1 &
    fi
    job_pids+=("$!")
    job_statuses[idx]="RUN"
done

next_status_at="$(date +%s)"
while true; do
    all_done="true"
    for idx in "${!job_ds[@]}"; do
        d="${job_ds[idx]}"
        pid="${job_pids[idx]}"
        status="${job_statuses[idx]}"

        if [[ "${status}" == "RUN" ]]; then
            if kill -0 "${pid}" 2>/dev/null; then
                all_done="false"
            else
                if wait "${pid}"; then
                    status="DONE"
                else
                    status="FAIL"
                fi
                job_statuses[idx]="${status}"
            fi
        fi
    done

    now_epoch="$(date +%s)"
    if (( now_epoch >= next_status_at )) || [[ "${all_done}" == "true" ]]; then
        print_status_snapshot "${mode}" "${logs_dir}" job_ds job_statuses job_logs
        next_status_at=$((now_epoch + status_interval))
    fi

    if [[ "${all_done}" == "true" ]]; then
        break
    fi
    sleep 1
done

fail_count=0
for status in "${job_statuses[@]}"; do
    if [[ "${status}" != "DONE" ]]; then
        fail_count=$((fail_count + 1))
    fi
done

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

run_load_and_plot_batch() {
    local -a files=()
    local d log recovered
    for idx in "${!job_ds[@]}"; do
        d="${job_ds[idx]}"
        log="${job_logs[idx]}"
        recovered="$(extract_saved_state_from_log "${log}")"
        if [[ -z "${recovered}" || ! -f "${recovered}" ]]; then
            local mode_tag
            if [[ "${mode}" == "warmup" ]]; then
                mode_tag="warmup"
            else
                mode_tag="prod"
            fi
            recovered="$(ls -1t "${REPO_ROOT}/saved_states/two_force_d_sweep/${mode}"/two_force_d${d}_${mode_tag}_*.jld2 2>/dev/null | head -n 1 || true)"
        fi
        if [[ -n "${recovered}" && -f "${recovered}" ]]; then
            files+=("${recovered}")
        else
            echo "WARNING: could not resolve saved state for d=${d}; skipping in post-plot."
        fi
    done

    if (( ${#files[@]} == 0 )); then
        echo "WARNING: no saved states found for post-plot."
        return 0
    fi

    local julia_setup="${JULIA_SETUP_SCRIPT:-/Local/ph_kafri/julia-1.7.2/bin/setup.sh}"
    if [[ -f "${julia_setup}" ]]; then
        # shellcheck disable=SC1090
        source "${julia_setup}"
    fi
    local julia_bin="${JULIA_BIN:-julia}"
    if ! command -v "${julia_bin}" >/dev/null 2>&1; then
        echo "WARNING: Julia executable '${julia_bin}' not found. Skipping post-plot."
        return 0
    fi

    mkdir -p "${plot_out_dir}"
    echo "Running load_and_plot.jl (mode=two_force_d) on ${#files[@]} states..."
    "${julia_bin}" "${REPO_ROOT}/load_and_plot.jl" "${files[@]}" --mode two_force_d --out_dir "${plot_out_dir}"
    echo "Post-plot outputs saved under: ${plot_out_dir}"
}

echo
if (( fail_count == 0 )); then
    echo "Local ${mode} run completed successfully for d values in ${D_MIN}:${D_STEP}:${D_MAX}."
    run_load_and_plot_batch
else
    echo "Local ${mode} run completed with ${fail_count} failed job(s)."
    echo "Inspect logs in: ${logs_dir}"
    exit 1
fi
