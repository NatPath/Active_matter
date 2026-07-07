#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JULIA_BIN="${JULIA_BIN:-julia}"

# Tail-focused local run.
#
# A = core_radius^2 gives a useful compute/statistics compromise:
#   b = sqrt(core_radius^2 + A) = sqrt(2)
#   dt * max(D) / core_radius^2 = 0.02
#   3.5 <= d <= 35 spans one decade for a fixed d^-2 amplitude fit.
#
# Stationary initialization removes the slow L^2/D equilibration transient.
# It uses the known stationary density, so this mode validates the estimator,
# tail fit, and Euler time-step bias; it is not evidence of relaxation from an
# arbitrary initial condition. Override with --initial_distribution uniform
# and a suitable --sample_start_step to test relaxation dynamically.
#
# Supply --display to open the live plot. Later options override defaults.
exec "${JULIA_BIN}" --startup-file=no --project=. run_radial_diffusion_live.jl \
    --L 80 \
    --boundary periodic \
    --interpretation ito \
    --D_inf 1.0 \
    --A 1.0 \
    --core_radius 1.0 \
    --initial_distribution stationary \
    --dt 0.01 \
    --steps 200000 \
    --walkers 50000 \
    --bins 640 \
    --sample_start_step 0 \
    --sample_interval 100 \
    --tail_min 3.5 \
    --tail_max 35 \
    --tail_log_bins 16 \
    --plot_interval_steps 20000 \
    --checkpoint_interval_steps 50000 \
    "$@"
