#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cluster_env_path="${CLUSTER_ENV_PATH:-${REPO_ROOT}/cluster_scripts/cluster_env.sh}"
if [[ -f "${cluster_env_path}" ]]; then
    # shellcheck disable=SC1090
    source "${cluster_env_path}"
fi

notify_email=""
status_log=""
run_id=""
node_name=""
return_code=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --notify_email)
            notify_email="${2:-}"
            shift 2
            ;;
        --status_log)
            status_log="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --node_name)
            node_name="${2:-}"
            shift 2
            ;;
        --return_code)
            return_code="${2:-}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${status_log}" ]]; then
    echo "Missing required arguments." >&2
    exit 2
fi

mkdir -p "$(dirname "${status_log}")"

timestamp="$(date +%Y-%m-%dT%H:%M:%S)"
host_name="$(hostname 2>/dev/null || echo unknown)"
body_file="$(mktemp)"
smtp_payload_file="$(mktemp)"
mail_output_file="$(mktemp)"
trap 'rm -f "${body_file}" "${smtp_payload_file}" "${mail_output_file}"' EXIT

{
    echo "run_id=${run_id}"
    echo "node_name=${node_name}"
    echo "return_code=${return_code}"
    echo "timestamp=${timestamp}"
    echo "host=${host_name}"
    echo "pwd=$(pwd)"
} > "${body_file}"

subject="[condor] ${run_id} node ${node_name} finished with return ${return_code}"
mail_rc=0
mail_status="not_attempted"
mail_backend="none"

send_via_mail_like() {
    local candidate="$1"
    "${candidate}" -s "${subject}" "${notify_email}" < "${body_file}"
}

send_via_sendmail() {
    local candidate="$1"
    {
        printf 'To: %s\n' "${notify_email}"
        printf 'Subject: %s\n' "${subject}"
        printf '\n'
        cat "${body_file}"
    } | "${candidate}" -t
}

send_via_curl_smtp() {
    local candidate="$1"
    local smtp_url="$2"
    local smtp_user="$3"
    local smtp_pass="$4"
    local from_email="$5"

    {
        printf 'From: %s\n' "${from_email}"
        printf 'To: %s\n' "${notify_email}"
        printf 'Subject: %s\n' "${subject}"
        printf 'Date: %s\n' "$(date -R)"
        printf '\n'
        cat "${body_file}"
    } > "${smtp_payload_file}"

    cmd=(
        "${candidate}"
        --silent
        --show-error
        --url "${smtp_url}"
        --mail-from "${from_email}"
        --mail-rcpt "${notify_email}"
        --upload-file "${smtp_payload_file}"
    )
    if [[ "${smtp_url}" == smtp://* ]]; then
        cmd+=(--ssl-reqd)
    fi
    if [[ -n "${smtp_user}" || -n "${smtp_pass}" ]]; then
        cmd+=(--user "${smtp_user}:${smtp_pass}")
    fi
    "${cmd[@]}"
}

send_via_ntfy() {
    local candidate="$1"
    local ntfy_server="$2"
    local ntfy_topic="$3"
    local ntfy_priority="$4"
    local ntfy_tags="$5"
    local ntfy_email="$6"
    local ntfy_token="$7"

    cmd=(
        "${candidate}"
        --silent
        --show-error
        -H "Title: ${subject}"
        -H "Priority: ${ntfy_priority}"
        -H "Tags: ${ntfy_tags}"
        --data-binary "@${body_file}"
    )
    if [[ -n "${ntfy_email}" ]]; then
        cmd+=(-H "Email: ${ntfy_email}")
    fi
    if [[ -n "${ntfy_token}" ]]; then
        cmd+=(-H "Authorization: Bearer ${ntfy_token}")
    fi
    cmd+=("${ntfy_server%/}/${ntfy_topic}")
    "${cmd[@]}"
}

mail_candidates=()
if [[ -n "${notify_email}" ]]; then
    if [[ -n "${MAIL:-}" ]]; then
        mail_candidates+=("${MAIL}")
    fi
    mail_candidates+=(
        "/usr/bin/mail"
        "mail"
        "/usr/bin/mailx"
        "mailx"
        "/usr/bin/s-nail"
        "s-nail"
        "/usr/sbin/sendmail"
        "/usr/lib/sendmail"
        "sendmail"
    )
fi
curl_candidate="${NOTIFY_CURL_BIN:-curl}"
curl_resolved="$(command -v "${curl_candidate}" 2>/dev/null || true)"
ntfy_server="${NOTIFY_NTFY_SERVER:-https://ntfy.sh}"
ntfy_topic="${NOTIFY_NTFY_TOPIC:-}"
ntfy_priority="${NOTIFY_NTFY_PRIORITY:-default}"
ntfy_tags="${NOTIFY_NTFY_TAGS:-white_check_mark}"
ntfy_email="${NOTIFY_NTFY_EMAIL:-}"
ntfy_token="${NOTIFY_NTFY_TOKEN:-}"
ntfy_configured="false"
if [[ -n "${ntfy_topic}" ]]; then
    ntfy_configured="true"
fi
smtp_url="${NOTIFY_SMTP_URL:-}"
smtp_user="${NOTIFY_SMTP_USER:-}"
smtp_pass="${NOTIFY_SMTP_PASS:-}"
smtp_from_email="${NOTIFY_FROM_EMAIL:-}"
smtp_configured="false"
if [[ -n "${notify_email}" && -n "${smtp_url}" && -n "${smtp_from_email}" ]]; then
    smtp_configured="true"
fi

{
    echo "timestamp=${timestamp}"
    echo "notify_email=${notify_email}"
    echo "node_name=${node_name}"
    echo "return_code=${return_code}"
    echo "host=${host_name}"
    echo "cluster_env_path=${cluster_env_path}"
    echo "ntfy_configured=${ntfy_configured}"
    echo "ntfy_server=${ntfy_server}"
    echo "ntfy_topic=${ntfy_topic}"
    echo "ntfy_priority=${ntfy_priority}"
    echo "ntfy_tags=${ntfy_tags}"
    echo "ntfy_email=${ntfy_email}"
    echo "smtp_configured=${smtp_configured}"
    echo "smtp_url=${smtp_url}"
    echo "smtp_from_email=${smtp_from_email}"
    echo "curl_candidate=${curl_candidate}"
    echo "curl_resolved=${curl_resolved}"
    echo "mail_candidates=${mail_candidates[*]}"
} >> "${status_log}"

if [[ "${ntfy_configured}" == "true" ]]; then
    if [[ -z "${curl_resolved}" ]]; then
        printf '[skip] curl not found: %s\n' "${curl_candidate}" >> "${mail_output_file}"
    elif [[ ! -x "${curl_resolved}" ]]; then
        printf '[skip] curl not executable: %s\n' "${curl_resolved}" >> "${mail_output_file}"
    else
        if send_via_ntfy "${curl_resolved}" "${ntfy_server}" "${ntfy_topic}" "${ntfy_priority}" "${ntfy_tags}" "${ntfy_email}" "${ntfy_token}" >> "${mail_output_file}" 2>&1; then
            mail_status="sent_ok"
            mail_backend="${curl_resolved} ntfy"
            mail_rc=0
        else
            mail_status="mail_command_failed"
            mail_backend="${curl_resolved} ntfy"
            mail_rc=$?
            printf '[fail] backend=%s rc=%s\n' "${mail_backend}" "${mail_rc}" >> "${mail_output_file}"
        fi
    fi
fi

if [[ "${mail_status}" == "not_attempted" && "${smtp_configured}" == "true" ]]; then
    if [[ -z "${curl_resolved}" ]]; then
        printf '[skip] curl not found: %s\n' "${curl_candidate}" >> "${mail_output_file}"
    elif [[ ! -x "${curl_resolved}" ]]; then
        printf '[skip] curl not executable: %s\n' "${curl_resolved}" >> "${mail_output_file}"
    else
        if send_via_curl_smtp "${curl_resolved}" "${smtp_url}" "${smtp_user}" "${smtp_pass}" "${smtp_from_email}" >> "${mail_output_file}" 2>&1; then
            mail_status="sent_ok"
            mail_backend="${curl_resolved} smtp"
            mail_rc=0
        else
            mail_status="mail_command_failed"
            mail_backend="${curl_resolved} smtp"
            mail_rc=$?
            printf '[fail] backend=%s rc=%s\n' "${mail_backend}" "${mail_rc}" >> "${mail_output_file}"
        fi
    fi
fi

if [[ "${mail_status}" == "not_attempted" || "${mail_status}" == "mail_command_failed" ]]; then
    for candidate in "${mail_candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        if [[ "${candidate}" == */* ]]; then
            resolved_candidate="${candidate}"
        else
            resolved_candidate="$(command -v "${candidate}" 2>/dev/null || true)"
        fi
        if [[ -z "${resolved_candidate}" ]]; then
            {
                printf '[skip] candidate not found: %s\n' "${candidate}"
            } >> "${mail_output_file}"
            continue
        fi
        if [[ ! -x "${resolved_candidate}" ]]; then
            {
                printf '[skip] candidate not executable: %s\n' "${resolved_candidate}"
            } >> "${mail_output_file}"
            continue
        fi

        if [[ "${resolved_candidate##*/}" == "sendmail" ]]; then
            if send_via_sendmail "${resolved_candidate}" >> "${mail_output_file}" 2>&1; then
                mail_status="sent_ok"
                mail_backend="${resolved_candidate}"
                mail_rc=0
                break
            else
                mail_status="mail_command_failed"
                mail_backend="${resolved_candidate}"
                mail_rc=$?
                {
                    printf '[fail] backend=%s rc=%s\n' "${resolved_candidate}" "${mail_rc}"
                } >> "${mail_output_file}"
            fi
        else
            if send_via_mail_like "${resolved_candidate}" >> "${mail_output_file}" 2>&1; then
                mail_status="sent_ok"
                mail_backend="${resolved_candidate}"
                mail_rc=0
                break
            else
                mail_status="mail_command_failed"
                mail_backend="${resolved_candidate}"
                mail_rc=$?
                {
                    printf '[fail] backend=%s rc=%s\n' "${resolved_candidate}" "${mail_rc}"
                } >> "${mail_output_file}"
            fi
        fi
    done
fi

if [[ "${mail_status}" == "not_attempted" ]]; then
    mail_status="no_mail_backend_found"
    mail_rc=127
fi

{
    echo "mail_status=${mail_status}"
    echo "mail_rc=${mail_rc}"
    echo "mail_backend=${mail_backend}"
    echo "mail_output_begin"
    cat "${mail_output_file}"
    echo "mail_output_end"
    echo
} >> "${status_log}"

# This notifier should never flip a successful workflow into failure.
exit 0
