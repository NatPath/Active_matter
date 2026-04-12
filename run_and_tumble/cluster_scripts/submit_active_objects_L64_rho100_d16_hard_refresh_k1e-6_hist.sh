#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/configuration_files/active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k1e-6.yaml"
WRAPPER="${SCRIPT_DIR}/submit_active_objects_histogram_dag.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash cluster_scripts/submit_active_objects_L64_rho100_d16_hard_refresh_k1e-6_hist.sh \
      --num_replicas <int> \
      --n_sweeps <int> \
      --tr <int> \
      [--run_id <token>] \
      [--request_cpus <int>] \
      [--request_memory <value>] \
      [--aggregate_request_cpus <int>] \
      [--max_sweep <int>] \
      [--no_submit]

Behavior:
  - Uses configuration_files/active_objects_1d_two_objects_L64_rho100_d16_hard_refresh_k1e-6.yaml
  - Delegates to cluster_scripts/submit_active_objects_histogram_dag.sh

Example:
  bash cluster_scripts/submit_active_objects_L64_rho100_d16_hard_refresh_k1e-6_hist.sh \
      --num_replicas 600 \
      --n_sweeps 10000000 \
      --tr 1000000
EOF
    exit 0
fi

exec bash "${WRAPPER}" --config "${CONFIG_PATH}" "$@"
