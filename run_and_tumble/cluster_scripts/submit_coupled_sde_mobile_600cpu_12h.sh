#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_coupled_sde_mobile_600cpu_12h.sh [options]

Options:
  --run_id <token>       Run id. Default: coupled_sde_mobile_600cpu_12h_<timestamp>
  --target_hours <float> Target wall time per replica job. Default: 12
  --total_cpus <int>     Number of single-core replica jobs. Default: 600
  --no_submit            Prepare configs/DAG but do not submit.
  -h, --help             Show this help.

Environment overrides:
  L, RHO0, DT, D0, MU_BATH, MU_OBJ, F0, SIGMA_F
  SAMPLE_INTERVAL, WARMUP_FRACTION, TARGET_UTILIZATION
  PARTICLE_UPDATES_PER_SECOND
  N_BINS, HISTORY_INTERVAL, MAX_HISTORY_RECORDS, SAVE_RAW_HISTORY
  CHECKPOINTS_PER_REPLICA, CHECKPOINT_INTERVAL_STEPS
  INITIAL_SEPARATION, RANDOM_INITIAL_OBJECTS, INITIAL_MIN_SEPARATION, INITIAL_MAX_SEPARATION
  REQUEST_CPUS, REQUEST_MEMORY, AGGREGATE_REQUEST_CPUS

Outputs:
  - One Condor job per replica, one CPU each.
  - Per-replica completed JLD2 results under runs/.../<run_id>/states.
  - One latest checkpoint per replica under runs/.../<run_id>/checkpoints.
  - Final aggregate CSV/JLD2/summary under runs/.../<run_id>/analysis.
  - Distance CSV contains P_ss(distance) and conditional diffusion diagnostics.
  - Location CSV contains P_A(x), P_B(x), and pair-center P(x).
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_coupled_sde_mobile_configs.sh"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_coupled_sde_mobile_objects_dag.sh"

for required in "${GENERATE_SCRIPT}" "${SUBMIT_SCRIPT}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Missing helper script: ${required}"
        exit 1
    fi
done

run_id=""
target_hours="${TARGET_HOURS:-12}"
total_cpus="${TOTAL_CPUS:-600}"
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
        --total_cpus)
            total_cpus="${2:-}"
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
    run_id="coupled_sde_mobile_600cpu_12h_$(date +%Y%m%d-%H%M%S)"
fi
if ! [[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run_id must match [A-Za-z0-9._-]+. Got '${run_id}'."
    exit 1
fi
if ! [[ "${total_cpus}" =~ ^[0-9]+$ ]] || (( total_cpus <= 0 )); then
    echo "--total_cpus must be a positive integer. Got '${total_cpus}'."
    exit 1
fi

L="${L:-256}"
RHO0="${RHO0:-10}"
D0="${D0:-1.0}"
DT="${DT:-0.001}"
MU_BATH="${MU_BATH:-1.0}"
MU_OBJ="${MU_OBJ:-0.0015}"
F0="${F0:-1.5}"
SIGMA_F="${SIGMA_F:-1.5}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-20}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.25}"
TARGET_UTILIZATION="${TARGET_UTILIZATION:-0.88}"
PARTICLE_UPDATES_PER_SECOND="${PARTICLE_UPDATES_PER_SECOND:-1170000}"
N_BINS="${N_BINS:-256}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-20000}"
MAX_HISTORY_RECORDS="${MAX_HISTORY_RECORDS:-2000}"
SAVE_RAW_HISTORY="${SAVE_RAW_HISTORY:-false}"
CHECKPOINTS_PER_REPLICA="${CHECKPOINTS_PER_REPLICA:-8}"
INITIAL_SEPARATION="${INITIAL_SEPARATION:-64}"
RANDOM_INITIAL_OBJECTS="${RANDOM_INITIAL_OBJECTS:-true}"
INITIAL_MIN_SEPARATION="${INITIAL_MIN_SEPARATION:-4}"
DEFAULT_INITIAL_MAX_SEPARATION=$((L / 2))
INITIAL_MAX_SEPARATION="${INITIAL_MAX_SEPARATION:-${DEFAULT_INITIAL_MAX_SEPARATION}}"
REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-6 GB}"
AGGREGATE_REQUEST_CPUS="${AGGREGATE_REQUEST_CPUS:-1}"

for numeric_name in L total_cpus REQUEST_CPUS AGGREGATE_REQUEST_CPUS N_BINS SAMPLE_INTERVAL CHECKPOINTS_PER_REPLICA; do
    numeric_value="${!numeric_name}"
    if ! [[ "${numeric_value}" =~ ^[0-9]+$ ]] || (( numeric_value <= 0 )); then
        echo "${numeric_name} must be a positive integer. Got '${numeric_value}'."
        exit 1
    fi
done

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
        if (total_steps < 200000) total_steps = 200000
        warmup_steps = int(total_steps * warmup_fraction)
        if (warmup_steps < 100000) warmup_steps = 100000
        production_steps = total_steps - warmup_steps
        if (production_steps < 100000) production_steps = 100000
        printf "%d %d %d %.1f", total_steps, warmup_steps, production_steps, target_seconds
    }')"
read -r TOTAL_STEPS COMPUTED_WARMUP_STEPS COMPUTED_PRODUCTION_STEPS EFFECTIVE_SECONDS <<< "${computed_values}"

WARMUP_STEPS="${WARMUP_STEPS:-${COMPUTED_WARMUP_STEPS}}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-${COMPUTED_PRODUCTION_STEPS}}"

if [[ -z "${CHECKPOINT_INTERVAL_STEPS:-}" ]]; then
    CHECKPOINT_INTERVAL_STEPS="$(awk -v total_steps="${TOTAL_STEPS}" -v n="${CHECKPOINTS_PER_REPLICA}" 'BEGIN {
        interval = int(total_steps / n)
        if (interval < 50000) interval = 50000
        printf "%d", interval
    }')"
fi
if ! [[ "${CHECKPOINT_INTERVAL_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "CHECKPOINT_INTERVAL_STEPS must be a nonnegative integer. Got '${CHECKPOINT_INTERVAL_STEPS}'."
    exit 1
fi

CONFIG_DIR="${REPO_ROOT}/configuration_files/coupled_sde_active_objects/mobile_objects/${run_id}"

echo "Preparing coupled-SDE two-mobile-object 600-CPU run"
echo "  run_id=${run_id}"
echo "  L=${L}, rho0=${RHO0}, N=${N_PARTICLES}"
echo "  replicas=${total_cpus}, request_cpus=${REQUEST_CPUS}"
echo "  mu_obj=${MU_OBJ}, f0=${F0}, sigma_f=${SIGMA_F}"
echo "  random_initial_objects=${RANDOM_INITIAL_OBJECTS}, initial_min=${INITIAL_MIN_SEPARATION}, initial_max=${INITIAL_MAX_SEPARATION}"
echo "  target_hours=${target_hours}, target_utilization=${TARGET_UTILIZATION}"
echo "  particle_updates_per_second=${PARTICLE_UPDATES_PER_SECOND}"
echo "  computed_total_steps=${TOTAL_STEPS}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  production_steps=${PRODUCTION_STEPS}"
echo "  sample_interval=${SAMPLE_INTERVAL}, expected_samples_per_replica=$((PRODUCTION_STEPS / SAMPLE_INTERVAL))"
echo "  n_bins=${N_BINS}"
echo "  save_raw_history=${SAVE_RAW_HISTORY}, checkpoint_interval_steps=${CHECKPOINT_INTERVAL_STEPS}"

export L RHO0 D0 DT MU_BATH F0 SIGMA_F SAMPLE_INTERVAL
export WARMUP_STEPS PRODUCTION_STEPS N_BINS HISTORY_INTERVAL MAX_HISTORY_RECORDS SAVE_RAW_HISTORY
export INITIAL_SEPARATION RANDOM_INITIAL_OBJECTS INITIAL_MIN_SEPARATION INITIAL_MAX_SEPARATION
export MU_OBJ_VALUES_CSV="${MU_OBJ}"

CONFIG_ROOT="${CONFIG_DIR}" "${GENERATE_SCRIPT}"
mapfile -t configs < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
if (( ${#configs[@]} != 1 )); then
    echo "Expected exactly one generated mobile config under ${CONFIG_DIR}, found ${#configs[@]}."
    exit 1
fi
config_path="${configs[0]}"

submit_args=(
    --config "${config_path}"
    --num_replicas "${total_cpus}"
    --run_id "${run_id}"
    --request_cpus "${REQUEST_CPUS}"
    --request_memory "${REQUEST_MEMORY}"
    --aggregate_request_cpus "${AGGREGATE_REQUEST_CPUS}"
    --checkpoint_interval_steps "${CHECKPOINT_INTERVAL_STEPS}"
)
if [[ "${no_submit}" == "true" ]]; then
    submit_args+=(--no_submit)
fi

"${SUBMIT_SCRIPT}" "${submit_args[@]}"

run_root="${REPO_ROOT}/runs/coupled_sde_active_objects/mobile_objects/${run_id}"
cat > "${run_root}/continuation_notes.txt" <<EOF
This run writes one completed result per replica under:
  ${run_root}/states

It also keeps one latest checkpoint per replica under:
  ${run_root}/checkpoints

Resume one interrupted replica from its latest checkpoint by rerunning the same
submit-file command with:
  --resume_checkpoint <checkpoint_file> --checkpoint_path <checkpoint_file> --checkpoint_interval_steps ${CHECKPOINT_INTERVAL_STEPS}

Start a new production chunk from a completed result by using:
  --initial_state <completed_result.jld2>

For a continuation chunk, use a config with warmup_steps: 0 and keep the same
physical parameters and n_bins so the old and new chunks remain aggregate-compatible.
EOF

cat > "${run_root}/analysis_commands.txt" <<EOF
# On the cluster filesystem:
find "${run_root}/states" -maxdepth 1 -type f -name '*.jld2' | wc -l
find "${run_root}/checkpoints" -maxdepth 1 -type f -name '*.checkpoint.jld2' | wc -l
bash cluster_scripts/analyze_coupled_sde_mobile_objects.sh --state_dir "${run_root}/states" --output_dir "${run_root}/analysis" --save_tag "partial_${run_id}" --no_plot
EOF

cat > "${run_root}/fetch_aggregates_and_plot_command.txt" <<EOF
bash cluster_scripts/copy_data_from_cluster.sh \\
  --run_id ${run_id} \\
  --run_family coupled_sde \\
  --sync_scope aggregation \\
  --plot
EOF

echo "Run preparation complete:"
echo "  run_root=${run_root}"
echo "  analysis commands: ${run_root}/analysis_commands.txt"
echo "  continuation notes: ${run_root}/continuation_notes.txt"
echo "  fetch/plot command: ${run_root}/fetch_aggregates_and_plot_command.txt"
