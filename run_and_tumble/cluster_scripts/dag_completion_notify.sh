#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

status_log=""
run_id=""
dag_label=""
dag_status=""
failed_count=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status_log)
            status_log="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --dag_label)
            dag_label="${2:-}"
            shift 2
            ;;
        --dag_status)
            dag_status="${2:-}"
            shift 2
            ;;
        --failed_count)
            failed_count="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${status_log}" || -z "${run_id}" || -z "${dag_label}" || -z "${dag_status}" || -z "${failed_count}" ]]; then
    echo "Missing required arguments." >&2
    exit 2
fi

mkdir -p "$(dirname "${status_log}")"

timestamp="$(date +%Y-%m-%dT%H:%M:%S)"
host_name="$(hostname 2>/dev/null || echo unknown)"
body_file="$(mktemp)"
notify_output_file="$(mktemp)"
trap 'rm -f "${body_file}" "${notify_output_file}"' EXIT

dag_result="success"
ntfy_priority="${NOTIFY_NTFY_PRIORITY:-default}"
ntfy_tags="${NOTIFY_NTFY_TAGS:-white_check_mark}"
subject="[condor] ${run_id} ${dag_label} completed"
if [[ "${dag_status}" != "0" ]]; then
    dag_result="failure"
    ntfy_priority="${NOTIFY_NTFY_PRIORITY_FAILURE:-high}"
    ntfy_tags="${NOTIFY_NTFY_TAGS_FAILURE:-warning}"
    subject="[condor] ${run_id} ${dag_label} failed"
fi

{
    echo "run_id=${run_id}"
    echo "dag_label=${dag_label}"
    echo "dag_result=${dag_result}"
    echo "dag_status=${dag_status}"
    echo "failed_count=${failed_count}"
    echo "timestamp=${timestamp}"
    echo "host=${host_name}"
    echo "pwd=$(pwd)"
} > "${body_file}"

curl_candidate="${NOTIFY_CURL_BIN:-curl}"
curl_resolved="$(command -v "${curl_candidate}" 2>/dev/null || true)"
ntfy_server="${NOTIFY_NTFY_SERVER:-https://ntfy.sh}"
ntfy_topic="${NOTIFY_NTFY_TOPIC:-}"
ntfy_token="${NOTIFY_NTFY_TOKEN:-}"
ntfy_configured="false"
if [[ -n "${ntfy_topic}" ]]; then
    ntfy_configured="true"
fi

notify_status="not_attempted"
notify_rc=0
notify_backend="none"

{
    echo "timestamp=${timestamp}"
    echo "run_id=${run_id}"
    echo "dag_label=${dag_label}"
    echo "dag_result=${dag_result}"
    echo "dag_status=${dag_status}"
    echo "failed_count=${failed_count}"
    echo "host=${host_name}"
    echo "cluster_env_path=${cluster_env_path}"
    echo "ntfy_configured=${ntfy_configured}"
    echo "ntfy_server=${ntfy_server}"
    echo "ntfy_topic=${ntfy_topic}"
    echo "ntfy_priority=${ntfy_priority}"
    echo "ntfy_tags=${ntfy_tags}"
    echo "curl_candidate=${curl_candidate}"
    echo "curl_resolved=${curl_resolved}"
} >> "${status_log}"

if [[ "${ntfy_configured}" == "true" ]]; then
    if [[ -z "${curl_resolved}" ]]; then
        notify_status="curl_missing"
        notify_rc=127
        printf '[skip] curl not found: %s\n' "${curl_candidate}" > "${notify_output_file}"
    elif [[ ! -x "${curl_resolved}" ]]; then
        notify_status="curl_not_executable"
        notify_rc=127
        printf '[skip] curl not executable: %s\n' "${curl_resolved}" > "${notify_output_file}"
    else
        cmd=(
            "${curl_resolved}"
            --silent
            --show-error
            -H "Title: ${subject}"
            -H "Priority: ${ntfy_priority}"
            -H "Tags: ${ntfy_tags}"
            --data-binary "@${body_file}"
        )
        if [[ -n "${ntfy_token}" ]]; then
            cmd+=(-H "Authorization: Bearer ${ntfy_token}")
        fi
        cmd+=("${ntfy_server%/}/${ntfy_topic}")
        if "${cmd[@]}" > "${notify_output_file}" 2>&1; then
            notify_status="sent_ok"
            notify_backend="${curl_resolved} ntfy"
            notify_rc=0
        else
            notify_status="notify_command_failed"
            notify_backend="${curl_resolved} ntfy"
            notify_rc=$?
            printf '[fail] backend=%s rc=%s\n' "${notify_backend}" "${notify_rc}" >> "${notify_output_file}"
        fi
    fi
else
    notify_status="ntfy_not_configured"
    notify_rc=0
    printf '[skip] NOTIFY_NTFY_TOPIC is not configured\n' > "${notify_output_file}"
fi

{
    echo "notify_status=${notify_status}"
    echo "notify_rc=${notify_rc}"
    echo "notify_backend=${notify_backend}"
    echo "notify_output_begin"
    cat "${notify_output_file}"
    echo "notify_output_end"
    echo
} >> "${status_log}"

if [[ "${dag_status}" == "0" ]]; then
    exit 0
fi
exit 1
