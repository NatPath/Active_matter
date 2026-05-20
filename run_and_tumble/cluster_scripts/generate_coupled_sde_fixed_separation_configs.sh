#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Missing spacing utility: ${SPACING_UTILS}"
    exit 1
fi
# shellcheck disable=SC1090
source "${SPACING_UTILS}"

CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/configuration_files/coupled_sde_active_objects/fixed_separation}"
mkdir -p "${CONFIG_ROOT}"

L="${L:-256}"
RHO0="${RHO0:-10}"
N_OVERRIDE="${N_OVERRIDE:-}"
D0="${D0:-1.0}"
DT="${DT:-0.001}"
MU_BATH="${MU_BATH:-1.0}"
MU_OBJ="${MU_OBJ:-0.0001}"
F0="${F0:-1.0}"
SIGMA_F="${SIGMA_F:-1.5}"
WARMUP_STEPS="${WARMUP_STEPS:-100000}"
PRODUCTION_STEPS="${PRODUCTION_STEPS:-1000000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"
PROFILE_TYPE="${PROFILE_TYPE:-gaussian}"
SAVE_DIR="${SAVE_DIR:-saved_states/coupled_sde_active_objects/fixed_separation}"

D_MIN="${D_MIN:-4}"
D_STEP="${D_STEP:-2}"
D_SPACING="${D_SPACING:-linear}"
DEFAULT_D_MAX=$((L / 4))
D_MAX="${D_MAX:-${DEFAULT_D_MAX}}"

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )); then
    echo "L must be a positive integer. Got '${L}'."
    exit 1
fi
if ! [[ "${D_MIN}" =~ ^[0-9]+$ && "${D_MAX}" =~ ^[0-9]+$ && "${D_STEP}" =~ ^[0-9]+$ ]]; then
    echo "D_MIN, D_MAX, and D_STEP must be nonnegative integers."
    exit 1
fi
if (( D_STEP <= 0 || D_MAX < D_MIN )); then
    echo "Invalid separation range: D_MIN=${D_MIN}, D_MAX=${D_MAX}, D_STEP=${D_STEP}"
    exit 1
fi

D_SPACING="$(two_force_d_normalize_spacing_mode "${D_SPACING}")" || {
    echo "Invalid D_SPACING='${D_SPACING}'."
    exit 1
}
if [[ -n "${D_VALUES_CSV:-}" ]]; then
    D_VALUES=()
    if ! two_force_d_csv_to_array "${D_VALUES_CSV}" D_VALUES; then
        echo "Invalid D_VALUES_CSV='${D_VALUES_CSV}'."
        exit 1
    fi
else
    mapfile -t D_VALUES < <(two_force_d_generate_d_values "${D_SPACING}" "${D_MIN}" "${D_MAX}" "${D_STEP}")
fi
if (( ${#D_VALUES[@]} == 0 )); then
    echo "No separations generated."
    exit 1
fi

for d in "${D_VALUES[@]}"; do
    cfg="${CONFIG_ROOT}/d_${d}.yaml"
    {
        echo "mode: \"fixed_separation\""
        echo "description: \"coupled_sde_fixed_d${d}\""
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
        echo "mu_obj: ${MU_OBJ}"
        echo ""
        echo "profile_type: \"${PROFILE_TYPE}\""
        echo "f0: ${F0}"
        echo "sigma_f: ${SIGMA_F}"
        echo ""
        echo "separation: ${d}"
        echo ""
        echo "warmup_steps: ${WARMUP_STEPS}"
        echo "n_steps: ${PRODUCTION_STEPS}"
        echo "sample_interval: ${SAMPLE_INTERVAL}"
        echo ""
        echo "seed: 0"
        echo "performance_mode: true"
        echo "cluster_mode: false"
        echo "save_dir: \"${SAVE_DIR}\""
    } > "${cfg}"
    echo "Generated ${cfg}"
done

echo "Done. Fixed-separation configs under ${CONFIG_ROOT}"
