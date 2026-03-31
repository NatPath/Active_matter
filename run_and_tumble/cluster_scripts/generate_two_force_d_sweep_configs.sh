#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../run_diffusive_no_activity.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi
SPACING_UTILS="${SCRIPT_DIR}/two_force_d_spacing_utils.sh"
if [[ ! -f "${SPACING_UTILS}" ]]; then
    echo "Could not find spacing utils script: ${SPACING_UTILS}"
    exit 1
fi
source "${SPACING_UTILS}"

CONFIG_ROOT="${REPO_ROOT}/configuration_files/two_force_d_sweep"
WARMUP_DIR="${CONFIG_ROOT}/warmup"
PRODUCTION_DIR="${CONFIG_ROOT}/production"
mkdir -p "${WARMUP_DIR}" "${PRODUCTION_DIR}"

L="${L:-512}"
RHO0="${RHO0:-1000}"

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )); then
    echo "Invalid L='${L}'. L must be a positive integer."
    exit 1
fi
if (( L % 2 != 0 )); then
    echo "Invalid L='${L}'. L must be even for symmetric bond placement."
    exit 1
fi
CENTER_SITE=$((L / 2))

WARMUP_SWEEPS="${WARMUP_SWEEPS:-100000}"
PRODUCTION_SWEEPS="${PRODUCTION_SWEEPS:-1000000}"

D_MIN="${D_MIN:-2}"
D_STEP="${D_STEP:-2}"
D_SPACING="${D_SPACING:-linear}"
DEFAULT_D_MAX=$((L / 4))
D_MAX="${D_MAX:-${DEFAULT_D_MAX}}"

if ! [[ "${D_MIN}" =~ ^[0-9]+$ ]] || ! [[ "${D_MAX}" =~ ^[0-9]+$ ]] || ! [[ "${D_STEP}" =~ ^[0-9]+$ ]]; then
    echo "Invalid d-range values: D_MIN='${D_MIN}', D_MAX='${D_MAX}', D_STEP='${D_STEP}'."
    exit 1
fi
if (( D_STEP <= 0 )); then
    echo "D_STEP must be positive. Got ${D_STEP}."
    exit 1
fi
if (( D_MAX < D_MIN )); then
    echo "D_MAX (${D_MAX}) must be >= D_MIN (${D_MIN})."
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
    echo "No d values generated for spacing='${D_SPACING}' and range ${D_MIN}:${D_STEP}:${D_MAX}."
    exit 1
fi

periodic_site() {
    local raw="$1"
    local size="$2"
    echo $(( (raw - 1 + size) % size + 1 ))
}

for d in "${D_VALUES[@]}"; do
    left_offset=$((d / 2))
    right_offset=$(((d + 1) / 2))
    left_bond_right="$(periodic_site "$((CENTER_SITE - left_offset))" "${L}")"
    left_bond_left="$(periodic_site "$((left_bond_right - 1))" "${L}")"
    right_bond_left="$(periodic_site "$((CENTER_SITE + right_offset))" "${L}")"
    right_bond_right="$(periodic_site "$((right_bond_left + 1))" "${L}")"

    warmup_cfg="${WARMUP_DIR}/d_${d}.yaml"
    production_cfg="${PRODUCTION_DIR}/d_${d}.yaml"

    cat > "${warmup_cfg}" <<EOF
dim_num: 1
L: ${L}
ρ₀: ${RHO0}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${WARMUP_SWEEPS}
warmup_sweeps: 0
performance_mode: true
description: "two_force_d${d}_warmup"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [${left_bond_left}, ${left_bond_right}]
  - [${right_bond_left}, ${right_bond_right}]

forcing_magnitudes: [1.0, 1.0]
ffrs: [1.0, 1.0]
forcing_direction_flags: [true, true]
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "saved_states/two_force_d_sweep/warmup"
EOF

    cat > "${production_cfg}" <<EOF
dim_num: 1
L: ${L}
ρ₀: ${RHO0}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${PRODUCTION_SWEEPS}
warmup_sweeps: 0
performance_mode: true
description: "two_force_d${d}_prod"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [${left_bond_left}, ${left_bond_right}]
  - [${right_bond_left}, ${right_bond_right}]

forcing_magnitudes: [1.0, 1.0]
ffrs: [1.0, 1.0]
forcing_direction_flags: [true, true]
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "saved_states/two_force_d_sweep/production"
EOF

    echo "Generated d=${d}: ${warmup_cfg} and ${production_cfg}"
done

echo "Done. Configs under ${CONFIG_ROOT}"
