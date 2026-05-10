#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/managed_diffusive.sh <command> [options]

Commands:
  init              Initialize a managed run
  submit            Submit/plan the next managed batch
  status            Print managed run status
  aggregate         Submit/plan an aggregate job
  import-warmups    Register existing warmup states
  reclaim           Clear stale running claims after checking jobs are gone

All commands take --case <diffusive_1d_pmlr|single_origin_bond> where needed.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command_name="$1"
shift

case "${command_name}" in
    init)
        exec bash "${SCRIPT_DIR}/init_managed_diffusive.sh" "$@"
        ;;
    submit|batch)
        exec bash "${SCRIPT_DIR}/submit_managed_diffusive_batch.sh" "$@"
        ;;
    status)
        exec bash "${SCRIPT_DIR}/status_managed_diffusive.sh" "$@"
        ;;
    aggregate|submit-aggregate)
        exec bash "${SCRIPT_DIR}/submit_managed_diffusive_aggregate.sh" "$@"
        ;;
    import-warmups|import)
        exec bash "${SCRIPT_DIR}/import_managed_diffusive_warmups.sh" "$@"
        ;;
    reclaim)
        exec bash "${SCRIPT_DIR}/reclaim_managed_diffusive_claims.sh" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown managed diffusive command: ${command_name}" >&2
        usage
        exit 1
        ;;
esac
