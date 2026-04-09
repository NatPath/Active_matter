#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/run_notification_probe.sh \
      --trace_file <path> \
      [--sleep_seconds <int>] \
      [--run_id <token>]

Behavior:
  - writes a small trace file so the test job leaves an artifact
  - sleeps briefly
  - exits successfully
EOF
}

trace_file=""
sleep_seconds="10"
run_id=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trace_file)
            trace_file="${2:-}"
            shift 2
            ;;
        --sleep_seconds)
            sleep_seconds="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${trace_file}" ]]; then
    echo "--trace_file is required."
    usage
    exit 1
fi
if ! [[ "${sleep_seconds}" =~ ^[0-9]+$ ]]; then
    echo "--sleep_seconds must be a non-negative integer. Got '${sleep_seconds}'."
    exit 1
fi

mkdir -p "$(dirname "${trace_file}")"

{
    echo "run_id=${run_id}"
    echo "started_at=$(date +%Y-%m-%dT%H:%M:%S)"
    echo "hostname=$(hostname)"
    echo "user=${USER:-unknown}"
    echo "pwd=$(pwd)"
    echo "sleep_seconds=${sleep_seconds}"
} > "${trace_file}"

sleep "${sleep_seconds}"

{
    echo "finished_at=$(date +%Y-%m-%dT%H:%M:%S)"
    echo "status=success"
} >> "${trace_file}"
