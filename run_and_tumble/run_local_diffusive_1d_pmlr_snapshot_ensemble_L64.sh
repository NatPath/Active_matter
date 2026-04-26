#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

julia --startup-file=no utility_scripts/run_diffusive_1d_pmlr_snapshot_ensemble.jl \
  --L 64 \
  --rho 100 \
  --gamma 1 \
  --potential_strength 16 \
  --burnin_sweeps 2000 \
  --n_snapshots 16 \
  --collapse_indices 6:1:14 \
  "$@"
