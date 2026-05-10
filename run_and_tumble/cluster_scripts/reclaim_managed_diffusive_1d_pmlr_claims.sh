#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/reclaim_managed_diffusive_claims.sh" --case diffusive_1d_pmlr "$@"
