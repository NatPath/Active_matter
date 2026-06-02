#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_coupled_sde_mobile_signal_7h.sh [options]

Options:
  --run_id <token>             Run id. Default: coupled_sde_mobile_signal_<timestamp>
  --target_hours <float>       Target wall time per replica job. Default: 7
  --num_replicas <int>         Mobile-object replicas. Default: 48
  --no_submit                  Prepare configs/DAG but do not submit.
  -h, --help                   Show this help.

Environment overrides:
  L, RHO0, DT, D0, MU_BATH, MU_OBJ, F0, SIGMA_F, PROFILE_TYPE
  HARD_MIN_SEPARATION, HARD_MIN_SEPARATION_SIGMA
  SAMPLE_INTERVAL, WARMUP_FRACTION, TARGET_UTILIZATION
  PARTICLE_UPDATES_PER_SECOND
  N_BINS, HISTORY_INTERVAL, MAX_HISTORY_RECORDS, SAVE_RAW_HISTORY
  INITIAL_SEPARATION, RANDOM_INITIAL_OBJECTS, INITIAL_MIN_SEPARATION, INITIAL_MAX_SEPARATION
  REQUEST_CPUS, REQUEST_MEMORY, AGGREGATE_REQUEST_CPUS

Statistics collected:
  Each replica stores online binned accumulators over production only:
    count(delta_bin)
    sum[(delta Delta)^2 | delta_bin]
    sum[S_A^2 + S_B^2 | delta_bin]
    histogram counts for P_ss(delta)
  The aggregate computes:
    D_rel_traj(delta) = <(delta Delta)^2 | delta> / (2 dt)
    D_rel_proxy(delta) = (mu_obj^2 / 2) <S_A^2 + S_B^2 | delta>
    P_ss(delta), normalized 1/D_rel_proxy, and P_ss * D_rel_proxy
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
target_hours="${TARGET_HOURS:-7}"
num_replicas="${NUM_REPLICAS:-48}"
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
    run_id="coupled_sde_mobile_signal_$(date +%Y%m%d-%H%M%S)"
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
MU_OBJ="${MU_OBJ:-0.0015}"
F0="${F0:-1.5}"
SIGMA_F="${SIGMA_F:-1.5}"
PROFILE_TYPE="${PROFILE_TYPE:-gaussian}"
HARD_MIN_SEPARATION="${HARD_MIN_SEPARATION:-}"
HARD_MIN_SEPARATION_SIGMA="${HARD_MIN_SEPARATION_SIGMA:-}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-20}"
WARMUP_FRACTION="${WARMUP_FRACTION:-0.20}"
TARGET_UTILIZATION="${TARGET_UTILIZATION:-0.85}"
PARTICLE_UPDATES_PER_SECOND="${PARTICLE_UPDATES_PER_SECOND:-1170000}"
N_BINS="${N_BINS:-128}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-1000}"
MAX_HISTORY_RECORDS="${MAX_HISTORY_RECORDS:-20000}"
SAVE_RAW_HISTORY="${SAVE_RAW_HISTORY:-true}"
INITIAL_SEPARATION="${INITIAL_SEPARATION:-64}"
RANDOM_INITIAL_OBJECTS="${RANDOM_INITIAL_OBJECTS:-true}"
INITIAL_MIN_SEPARATION="${INITIAL_MIN_SEPARATION:-4}"
DEFAULT_INITIAL_MAX_SEPARATION=$((L / 2))
INITIAL_MAX_SEPARATION="${INITIAL_MAX_SEPARATION:-${DEFAULT_INITIAL_MAX_SEPARATION}}"
REQUEST_CPUS="${REQUEST_CPUS:-1}"
REQUEST_MEMORY="${REQUEST_MEMORY:-6 GB}"
AGGREGATE_REQUEST_CPUS="${AGGREGATE_REQUEST_CPUS:-1}"

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

CONFIG_DIR="${REPO_ROOT}/configuration_files/coupled_sde_active_objects/mobile_objects/${run_id}"

echo "Preparing coupled-SDE mobile-object signal run"
echo "  run_id=${run_id}"
echo "  L=${L}, rho0=${RHO0}, N=${N_PARTICLES}"
echo "  replicas=${num_replicas}"
echo "  mu_obj=${MU_OBJ}, f0=${F0}, sigma_f=${SIGMA_F}, profile_type=${PROFILE_TYPE}"
if [[ -n "${HARD_MIN_SEPARATION}" ]]; then
    echo "  hard_min_separation=${HARD_MIN_SEPARATION}"
fi
if [[ -n "${HARD_MIN_SEPARATION_SIGMA}" ]]; then
    echo "  hard_min_separation_sigma=${HARD_MIN_SEPARATION_SIGMA}"
fi
echo "  random_initial_objects=${RANDOM_INITIAL_OBJECTS}, initial_min=${INITIAL_MIN_SEPARATION}, initial_max=${INITIAL_MAX_SEPARATION}"
echo "  target_hours=${target_hours}, target_utilization=${TARGET_UTILIZATION}"
echo "  particle_updates_per_second=${PARTICLE_UPDATES_PER_SECOND}"
echo "  computed_total_steps=${TOTAL_STEPS}"
echo "  warmup_steps=${WARMUP_STEPS}"
echo "  production_steps=${PRODUCTION_STEPS}"
echo "  sample_interval=${SAMPLE_INTERVAL}, n_bins=${N_BINS}"

export L RHO0 D0 DT MU_BATH F0 SIGMA_F PROFILE_TYPE SAMPLE_INTERVAL
export HARD_MIN_SEPARATION HARD_MIN_SEPARATION_SIGMA
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
    --num_replicas "${num_replicas}"
    --run_id "${run_id}"
    --request_cpus "${REQUEST_CPUS}"
    --request_memory "${REQUEST_MEMORY}"
    --aggregate_request_cpus "${AGGREGATE_REQUEST_CPUS}"
)
if [[ "${no_submit}" == "true" ]]; then
    submit_args+=(--no_submit)
fi

"${SUBMIT_SCRIPT}" "${submit_args[@]}"

run_root="${REPO_ROOT}/runs/coupled_sde_active_objects/mobile_objects/${run_id}"
cat > "${run_root}/fetch_and_plot_command.txt" <<EOF
bash cluster_scripts/copy_data_from_cluster.sh \\
  --run_id ${run_id} \\
  --sync_scope aggregation \\
  --plot
EOF

echo "Fetch command written to ${run_root}/fetch_and_plot_command.txt"
echo "After completion, fetch and analyze with:"
cat "${run_root}/fetch_and_plot_command.txt"
