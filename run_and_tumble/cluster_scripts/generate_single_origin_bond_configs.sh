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

CONFIG_ROOT="${REPO_ROOT}/configuration_files/single_origin_bond"
WARMUP_DIR="${CONFIG_ROOT}/warmup"
PRODUCTION_DIR="${CONFIG_ROOT}/production"
mkdir -p "${WARMUP_DIR}" "${PRODUCTION_DIR}"

L="${L:-512}"
RHO0="${RHO0:-1000}"
WARMUP_SWEEPS="${WARMUP_SWEEPS:-100000}"
PRODUCTION_SWEEPS="${PRODUCTION_SWEEPS:-1000000}"
FFR="${FFR:-1.0}"
FORCE_STRENGTH="${FORCE_STRENGTH:-1.0}"

is_number() {
    local value="$1"
    [[ "${value}" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]
}

if ! [[ "${L}" =~ ^[0-9]+$ ]] || (( L <= 0 )) || (( L % 2 != 0 )); then
    echo "Invalid L='${L}'. L must be a positive even integer."
    exit 1
fi
if ! [[ "${WARMUP_SWEEPS}" =~ ^[0-9]+$ ]] || (( WARMUP_SWEEPS <= 0 )); then
    echo "Invalid WARMUP_SWEEPS='${WARMUP_SWEEPS}'."
    exit 1
fi
if ! [[ "${PRODUCTION_SWEEPS}" =~ ^[0-9]+$ ]] || (( PRODUCTION_SWEEPS <= 0 )); then
    echo "Invalid PRODUCTION_SWEEPS='${PRODUCTION_SWEEPS}'."
    exit 1
fi
if ! is_number "${FFR}"; then
    echo "Invalid FFR='${FFR}'."
    exit 1
fi
if ! is_number "${FORCE_STRENGTH}"; then
    echo "Invalid FORCE_STRENGTH='${FORCE_STRENGTH}'."
    exit 1
fi

origin_left=$((L / 2))
origin_right=$((origin_left + 1))
if (( origin_right > L )); then
    origin_right=1
fi

WARMUP_CFG="${WARMUP_DIR}/params.yaml"
PRODUCTION_CFG="${PRODUCTION_DIR}/params.yaml"

cat > "${WARMUP_CFG}" <<EOF
dim_num: 1
L: ${L}
ρ₀: ${RHO0}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${WARMUP_SWEEPS}
warmup_sweeps: 0
performance_mode: true
description: "single_origin_bond_warmup"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [${origin_left}, ${origin_right}]

forcing_magnitudes: [${FORCE_STRENGTH}]
ffrs: [${FFR}]
forcing_direction_flags: [true]
forcing_fluctuation_type: "alternating_direction"
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "saved_states/single_origin_bond/warmup"
EOF

cat > "${PRODUCTION_CFG}" <<EOF
dim_num: 1
L: ${L}
ρ₀: ${RHO0}
D: 1.0
T: 1.0
γ: 0.0
n_sweeps: ${PRODUCTION_SWEEPS}
warmup_sweeps: 0
performance_mode: true
description: "single_origin_bond_prod"

potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"

forcing_bond_pairs:
  - [${origin_left}, ${origin_right}]

forcing_magnitudes: [${FORCE_STRENGTH}]
ffrs: [${FFR}]
forcing_direction_flags: [true]
forcing_fluctuation_type: "alternating_direction"
bond_pass_count_mode: "all_forcing_bonds"

show_times: []
save_times: []
save_dir: "saved_states/single_origin_bond/production"
EOF

echo "Generated single-origin-bond configs:"
echo "  warmup: ${WARMUP_CFG}"
echo "  production: ${PRODUCTION_CFG}"
echo "  center periodic origin bond: [${origin_left}, ${origin_right}]"
echo "  forcing strength: ${FORCE_STRENGTH}"
echo "  ffr: ${FFR}"
