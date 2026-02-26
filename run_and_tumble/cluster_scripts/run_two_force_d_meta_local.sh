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
  --num_replicas      number of independent replicas to run and aggregate per d (default: 1)
  --d_min             minimum d (default: 2)
  --d_max             maximum d (default: L/4)
  --d_step            d step (default: 2)
  --status_interval   progress snapshot period in seconds (default: 20)
  --plot_sweeps       enable plot_sweep display during local runs
  --run_label         optional custom run label prefix
  -h, --help          show this help

Behavior:
  - warmup: runs configs without --initial_state
  - production: finds matching warmup state for each d and errors if any is missing
  - with --num_replicas > 1: each d-run executes replicated simulations in parallel workers and saves an aggregated state
  - creates a local run folder under runs/two_force_d/local/<mode>/<run_id>
  - runs load_and_plot.jl (mode=two_force_d) and saves analysis under run reports
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
num_replicas="1"
status_interval="20"
plot_sweeps="false"
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
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
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

slugify() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

write_runtime_config() {
    local src_cfg="$1"
    local dst_cfg="$2"
    local show_csv="$3"
    local save_dir="$4"
    local show_line="show_times: [${show_csv}]"
    local save_dir_line="save_dir: \"${save_dir}\""

    awk -v show_line="${show_line}" -v save_dir_line="${save_dir_line}" '
    BEGIN {seen_cluster=0; seen_show=0; seen_save=0}
    {
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
        if ($0 ~ /^show_times:[[:space:]]*/) {
            print show_line
            seen_show=1
            next
        }
        print
    }
    END {
        if (!seen_cluster) print "cluster_mode: false"
        if (!seen_save) print save_dir_line
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
echo "  num_replicas=${num_replicas}"
echo "  d range: ${D_MIN}:${D_STEP}:${D_MAX}"
echo "  status_interval=${status_interval}s"
echo "  plot_sweeps=${plot_sweeps}"

bash "${GENERATE_SCRIPT}"

if [[ "${mode}" == "production" ]]; then
    if [[ -z "${warmup_state_dir}" ]]; then
        warmup_state_dir="${REPO_ROOT}/saved_states/two_force_d_sweep/warmup"
    fi
fi

show_csv=""
if [[ "${plot_sweeps}" == "true" ]]; then
    show_csv="$(build_show_times_csv "${n_sweeps_val}")"
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
timestamp="$(date +%Y%m%d-%H%M%S)"
rho_tag="$(slugify "${rho_val}")"
if [[ -z "${run_label}" ]]; then
    run_label="${mode}_L${L_val}_rho${rho_tag}_ns${n_sweeps_val}_d${D_MIN}-${D_MAX}-s${D_STEP}_local"
    if (( num_replicas > 1 )); then
        run_label="${run_label}_nr${num_replicas}"
    fi
fi
run_label="$(slugify "${run_label}")"
run_id="${run_label}_${timestamp}"

logs_dir="${logs_root}/${run_id}"
mkdir -p "${logs_dir}"
run_root_local="${REPO_ROOT}/runs/two_force_d/local/${mode}/${run_id}"
state_root_local="${run_root_local}/states"
config_root_local="${run_root_local}/configs"
plot_out_dir="${run_root_local}/reports/load_and_plot"
run_info="${run_root_local}/run_info.txt"
run_manifest="${run_root_local}/manifest.csv"
local_registry="${REPO_ROOT}/runs/two_force_d/local/run_registry.csv"
mkdir -p "${state_root_local}" "${config_root_local}" "${plot_out_dir}"

declare -a job_ds=()
declare -a job_configs=()
declare -a job_init_states=()
declare -a job_logs=()
declare -a job_pids=()
declare -a job_statuses=()

latest_matching_two_force_state() {
    local search_root="$1"
    local d="$2"
    local newest_path=""
    local newest_mtime=0
    local candidate mtime

    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= newest_mtime )); then
            newest_mtime="${mtime}"
            newest_path="${candidate}"
        fi
    done < <(find "${search_root}" -type f \( -name "two_force_d${d}_warmup_*.jld2" -o -name "two_force_d${d}_*.jld2" \) -print0 2>/dev/null)

    printf "%s" "${newest_path}"
}

for d in $(seq "${D_MIN}" "${D_STEP}" "${D_MAX}"); do
    if (( d % 2 != 0 )); then
        continue
    fi

    config_path="${REPO_ROOT}/configuration_files/two_force_d_sweep/${mode}/d_${d}.yaml"
    if [[ ! -f "${config_path}" ]]; then
        echo "ERROR: missing config ${config_path}"
        exit 1
    fi

    runtime_config="${config_root_local}/${mode}_d_${d}.yaml"
    save_dir_for_d="${state_root_local}/d_${d}"
    mkdir -p "${save_dir_for_d}"
    write_runtime_config "${config_path}" "${runtime_config}" "${show_csv}" "${save_dir_for_d}"

    init_state=""
    if [[ "${mode}" == "warmup" ]]; then
        :
    else
        init_state="$(latest_matching_two_force_state "${warmup_state_dir}" "${d}")"
        if [[ -z "${init_state}" ]]; then
            echo "ERROR: no warmup state matching two_force_d${d}_warmup_* under ${warmup_state_dir} for d=${d}"
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
    runner_args=()
    if (( num_replicas > 1 )); then
        runner_args+=(--num_runs "${num_replicas}" --save_tag "aggregated_${run_id}_d${d}")
    fi
    if [[ "${mode}" == "warmup" ]]; then
        bash "${RUNNER_SCRIPT}" "${cfg}" "${runner_args[@]}" > "${log}" 2>&1 &
    else
        bash "${RUNNER_SCRIPT}" "${cfg}" --initial_state "${init_state}" "${runner_args[@]}" > "${log}" 2>&1 &
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
            recovered="$(ls -1t "${state_root_local}/d_${d}"/*.jld2 2>/dev/null | head -n 1 || true)"
        fi
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
run_status="DONE"
plot_status="SKIPPED"
if (( fail_count == 0 )); then
    echo "Local ${mode} run completed successfully for d values in ${D_MIN}:${D_STEP}:${D_MAX}."
    if run_load_and_plot_batch; then
        plot_status="DONE"
    else
        plot_status="FAILED"
    fi
else
    run_status="FAILED"
    echo "Local ${mode} run completed with ${fail_count} failed job(s)."
    echo "Inspect logs in: ${logs_dir}"
fi

saved_state_sample="$(find "${state_root_local}" -type f -name '*.jld2' | sort | tail -n 1 || true)"

cat > "${run_info}" <<EOF
run_id=${run_id}
timestamp=${timestamp}
mode=${mode}
L=${L}
rho0=${RHO0}
n_sweeps=${n_sweeps_val}
d_min=${D_MIN}
d_max=${D_MAX}
d_step=${D_STEP}
num_replicas=${num_replicas}
status_interval=${status_interval}
plot_sweeps=${plot_sweeps}
run_status=${run_status}
plot_status=${plot_status}
run_root=${run_root_local}
logs_dir=${logs_dir}
state_dir=${state_root_local}
saved_state_sample=${saved_state_sample}
reports_dir=${plot_out_dir}
warmup_state_dir=${warmup_state_dir}
EOF

echo "d,status,config_path,log_file,state_file,initial_state" > "${run_manifest}"
for idx in "${!job_ds[@]}"; do
    d="${job_ds[idx]}"
    status="${job_statuses[idx]}"
    cfg="${job_configs[idx]}"
    log="${job_logs[idx]}"
    init_state="${job_init_states[idx]}"
    state_file="$(extract_saved_state_from_log "${log}")"
    if [[ -z "${state_file}" || ! -f "${state_file}" ]]; then
        state_file="$(ls -1t "${state_root_local}/d_${d}"/*.jld2 2>/dev/null | head -n 1 || true)"
    fi
    printf "%s,%s,%s,%s,%s,%s\n" \
        "${d}" "${status}" "${cfg}" "${log}" "${state_file}" "${init_state}" >> "${run_manifest}"
done

mkdir -p "$(dirname "${local_registry}")"
if [[ ! -f "${local_registry}" ]]; then
    echo "timestamp,run_id,mode,L,rho0,n_sweeps,d_min,d_max,d_step,num_replicas,run_status,plot_status,run_root,logs_dir,state_dir,reports_dir,warmup_state_dir" > "${local_registry}"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${run_id}" "${mode}" "${L}" "${RHO0}" "${n_sweeps_val}" \
    "${D_MIN}" "${D_MAX}" "${D_STEP}" "${num_replicas}" "${run_status}" "${plot_status}" \
    "${run_root_local}" "${logs_dir}" "${state_root_local}" "${plot_out_dir}" "${warmup_state_dir}" >> "${local_registry}"

echo "Manifest: ${run_manifest}"
echo "Run info: ${run_info}"
echo "Reports: ${plot_out_dir}"

if [[ "${run_status}" != "DONE" ]]; then
    exit 1
fi
