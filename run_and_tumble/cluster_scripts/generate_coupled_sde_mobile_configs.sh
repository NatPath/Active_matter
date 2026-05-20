#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configuration_files/coupled_sde_active_objects/mobile_objects}"
mkdir -p "${CONFIG_ROOT}"

L="${L:-256}"
RHO0="${RHO0:-10}"
N_OVERRIDE="${N_OVERRIDE:-}"
D0="${D0:-1.0}"
DT="${DT:-0.001}"
MU_BATH="${MU_BATH:-1.0}"
MU_OBJ_VALUES_CSV="${MU_OBJ_VALUES_CSV:-0.00005}"
F0="${F0:-1.0}"
SIGMA_F="${SIGMA_F:-1.5}"
INITIAL_SEPARATION="${INITIAL_SEPARATION:-64}"
WARMUP_STEPS="${WARMUP_STEPS:-1000000}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-5000000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
N_BINS="${N_BINS:-96}"
HISTORY_INTERVAL="${HISTORY_INTERVAL:-1000}"
MAX_HISTORY_RECORDS="${MAX_HISTORY_RECORDS:-50000}"
SAVE_RAW_HISTORY="${SAVE_RAW_HISTORY:-true}"
PROFILE_TYPE="${PROFILE_TYPE:-gaussian}"
SAVE_DIR="${SAVE_DIR:-saved_states/coupled_sde_active_objects/mobile_objects}"

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )); then
    echo "L must be a positive integer. Got '${L}'."
    exit 1
fi

IFS=',' read -r -a MU_VALUES <<< "${MU_OBJ_VALUES_CSV}"
if (( ${#MU_VALUES[@]} == 0 )); then
    echo "MU_OBJ_VALUES_CSV produced no values."
    exit 1
fi

sanitize_token() {
    printf "%s" "$1" | sed -E 's/[^A-Za-z0-9._-]+/-/g; s/[.]/p/g; s/-+/-/g; s/^-//; s/-$//'
}

for mu_obj in "${MU_VALUES[@]}"; do
    token="$(sanitize_token "${mu_obj}")"
    cfg="${CONFIG_ROOT}/mu_${token}.yaml"
    {
        echo "mode: \"mobile_objects\""
        echo "description: \"coupled_sde_mobile_mu${token}\""
        echo ""
        echo "L: ${L}"
        echo "rho0: ${RHO0}"
        if [[ -n "${N_OVERRIDE}" ]]; then
            echo "N: ${N_OVERRIDE}"
        fi
        echo ""
        echo "D0: ${D0}"
        echo "dt: ${DT}"
        echo "mu_bath: ${MU_BATH}"
        echo "mu_obj: ${mu_obj}"
        echo ""
        echo "profile_type: \"${PROFILE_TYPE}\""
        echo "f0: ${F0}"
        echo "sigma_f: ${SIGMA_F}"
        echo ""
        echo "separation: ${INITIAL_SEPARATION}"
        echo ""
        echo "warmup_steps: ${WARMUP_STEPS}"
        echo "n_steps: ${PRODUCTION_STEPS}"
        echo "sample_interval: ${SAMPLE_INTERVAL}"
        echo ""
        echo "n_bins: ${N_BINS}"
        echo "history_interval: ${HISTORY_INTERVAL}"
        echo "max_history_records: ${MAX_HISTORY_RECORDS}"
        echo "save_raw_history: ${SAVE_RAW_HISTORY}"
        echo ""
        echo "seed: 0"
        echo "performance_mode: true"
        echo "cluster_mode: false"
        echo "save_dir: \"${SAVE_DIR}\""
    } > "${cfg}"
    echo "Generated ${cfg}"
done

echo "Done. Mobile-object configs under ${CONFIG_ROOT}"
