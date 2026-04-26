#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/aggregate_diffusive_origin_batch.sh"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
    echo "Missing helper script: ${TARGET_SCRIPT}"
    exit 1
fi

exec bash "${TARGET_SCRIPT}" "$@"
