#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_coupled_sde_fixed_signal_7h.sh [options]

Options:
  --run_id <token>             Run id. Default: coupled_sde_fixed_signal_<timestamp>
  --target_hours <float>       Target wall time per replica job. Default: 7
  --num_replicas <int>         Replicas per separation. Default: 24
  --no_submit                  Prepare configs/DAG but do not submit.
  -h, --help                   Show this help.

Environment overrides:
  L, RHO0, D_VALUES_CSV, DT, D0, MU_BATH, MU_OBJ, F0, SIGMA_F
  SAMPLE_INTERVAL, WARMUP_FRACTION, TARGET_UTILIZATION
  PARTICLE_UPDATES_PER_SECOND
  REQUEST_CPUS, REQUEST_MEMORY, AGGREGATE_REQUEST_CPUS
  FIT_MIN, FIT_MAX, PERIODIC_FIT, PLOT_AGGREGATE

Notes:
  The target is per Condor replica job once the job starts running. Queueing and
  DAG throttling are not included. The default throughput uses the local probe
  measurement from this repo; set PARTICLE_UPDATES_PER_SECOND after a cluster
  pilot if you want tighter timing.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_coupled_sde_fixed_separation_dag.sh"

if [[ ! -f "${SUBMIT_SCRIPT}" ]]; then
    echo "Missing submit helper: ${SUBMIT_SCRIPT}"
    exit 1
fi

run_id=""
target_hours="${TARGET_HOURS:-7}"
num_replicas="${NUM_REPLICAS:-24}"
no_submit="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --target_hours)
            target_hours="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
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

if [[ -z "${run_id}" ]]; then
    run_id="coupled_sde_fixed_signal_$(date +%Y%m%d-%H%M%S)"
fi
if ! [[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run_id must match [A-Za-z0-9._-]+. Got '${run_id}'."
    exit 1
fi
if ! [[ "${num_replicas}" =~ ^[0-9]+$ ]] || (( num_replicas <= 0 )); then
    echo "--num_replicas must be a positive integer. Got '${num_replicas}'."
    exit 1
fi

L="${L:-256}"
RHO0="${RHO0:-10}"
D0="${D0:-1.0}"
DT="${DT:-0.001}"
MU_BATH="${MU_BATH:-1.0}"
MU_OBJ="${MU_OBJ:-0.0002}"
F0="${F0:-1.5}"
SIGMA_F="${SIGMA_F:-1.5}"
D_VALUES_CSV="${D_VALUES_CSV:-6,8,10,12,16,20,24,32,40,48,56,64}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-50}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.12}"
TARGET_UTILIZATION="${TARGET_UTILIZATION:-0.85}"
PARTICLE_UPDATES_PER_SECOND="${PARTICLE_UPDATES_PER_SECOND:-1170000}"
REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-6 GB}"
AGGREGATE_REQUEST_CPUS="${AGGREGATE_REQUEST_CPUS:-1}"
FIT_MIN="${FIT_MIN:-8}"
FIT_MAX="${FIT_MAX:-64}"
PERIODIC_FIT="${PERIODIC_FIT:-false}"
PLOT_AGGREGATE="${PLOT_AGGREGATE:-false}"

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )); then
    echo "L must be a positive integer. Got '${L}'."
    exit 1
fi

N_PARTICLES="$(awk -v L="${L}" -v rho="${RHO0}" 'BEGIN { printf "%d", int(L * rho + 0.5) }')"
if ! [[ "${N_PARTICLES}" =~ ^[0-9]+$ ]] || (( N_PARTICLES <= 0 )); then
    echo "Computed invalid N=${N_PARTICLES} from L=${L}, RHO0=${RHO0}."
    exit 1
fi

computed_values="$(awk \
    -v target_hours="${target_hours}" \
    -v util="${TARGET_UTILIZATION}" \
    -v updates_per_second="${PARTICLE_UPDATES_PER_SECOND}" \
    -v n_particles="${N_PARTICLES}" \
    -v warmup_fraction="${WARMUP_FRACTION}" '
    BEGIN {
        target_seconds = target_hours * 3600.0 * util
        total_steps = int(target_seconds * updates_per_second / n_particles)
        if (total_steps < 1000) total_steps = 1000
        warmup_steps = int(total_steps * warmup_fraction)
        if (warmup_steps < 1000) warmup_steps = 1000
        production_steps = total_steps - warmup_steps
        if (production_steps < 1000) production_steps = 1000
        printf "%d %d %d %.1f", total_steps, warmup_steps, production_steps, target_seconds
    }')"
read -r TOTAL_STEPS COMPUTED_WARMUP_STEPS COMPUTED_PRODUCTION_STEPS EFFECTIVE_SECONDS <<< "${computed_values}"

WARMUP_STEPS="${WARMUP_STEPS:-${COMPUTED_WARMUP_STEPS}}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-${COMPUTED_PRODUCTION_STEPS}}"

CONFIG_DIR="${REPO_ROOT}/configuration_files/coupled_sde_active_objects/fixed_separation/${run_id}"

echo "Preparing coupled-SDE fixed-separation signal run"
echo "  run_id=${run_id}"
echo "  L=${L}, rho0=${RHO0}, N=${N_PARTICLES}"
echo "  separations=${D_VALUES_CSV}"
echo "  replicas_per_separation=${num_replicas}"
echo "  target_hours=${target_hours}, target_utilization=${TARGET_UTILIZATION}"
echo "  particle_updates_per_second=${PARTICLE_UPDATES_PER_SECOND}"
echo "  computed_total_steps=${TOTAL_STEPS}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  production_steps=${PRODUCTION_STEPS}"
echo "  sample_interval=${SAMPLE_INTERVAL}"
echo "  fit_window=[${FIT_MIN}, ${FIT_MAX}]"

export L RHO0 D0 DT MU_BATH MU_OBJ F0 SIGMA_F D_VALUES_CSV SAMPLE_INTERVAL
export WARMUP_STEPS PRODUCTION_STEPS

submit_args=(
    --config_dir "${CONFIG_DIR}"
    --num_replicas "${num_replicas}"
    --run_id "${run_id}"
    --request_cpus "${REQUEST_CPUS}"
    --request_memory "${REQUEST_MEMORY}"
    --aggregate_request_cpus "${AGGREGATE_REQUEST_CPUS}"
    --fit_min "${FIT_MIN}"
    --fit_max "${FIT_MAX}"
    --generate_configs
)
if [[ "${PERIODIC_FIT}" == "true" ]]; then
    submit_args+=(--periodic_fit)
fi
if [[ "${PLOT_AGGREGATE}" == "true" ]]; then
    submit_args+=(--plot_aggregate)
fi
if [[ "${no_submit}" == "true" ]]; then
    submit_args+=(--no_submit)
fi

"${SUBMIT_SCRIPT}" "${submit_args[@]}"

run_root="${REPO_ROOT}/runs/coupled_sde_active_objects/fixed_separation/${run_id}"
cat > "${run_root}/local_followup_commands.txt" <<EOF
# Re-run aggregate with plots after jobs complete. This wrapper sources
# cluster_scripts/cluster_env.sh and CLUSTER_JULIA_SETUP_SCRIPT before Julia.
RUN_TAG="${run_id}"
STATE_DIR="runs/coupled_sde_active_objects/fixed_separation/\${RUN_TAG}/states"
ANALYSIS_DIR="runs/coupled_sde_active_objects/fixed_separation/\${RUN_TAG}/analysis_local_plot"

bash cluster_scripts/analyze_coupled_sde_fixed_separation.sh \\
  --state_dir "\${STATE_DIR}" \\
  --output_dir "\${ANALYSIS_DIR}" \\
  --save_tag "\${RUN_TAG}" \\
  --fit_min ${FIT_MIN} \\
  --fit_max ${FIT_MAX}
EOF

echo "Follow-up commands written to ${run_root}/local_followup_commands.txt"
