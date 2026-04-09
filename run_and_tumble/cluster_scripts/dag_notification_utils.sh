#!/usr/bin/env bash

dag_notification_is_disabled() {
    local raw="${NO_DAG_NOTIFICATION:-false}"
    raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
    [[ "${raw}" == "1" || "${raw}" == "true" || "${raw}" == "yes" || "${raw}" == "on" ]]
}

dag_notification_prepare() {
    local submit_dir="$1"
    local log_dir="$2"
    local run_root="$3"
    local dag_label="$4"

    DAG_NOTIFICATION_STATUS_LOG=""
    DAG_NOTIFICATION_FINAL_SUBMIT_FILE=""
    DAG_NOTIFICATION_NOTIFY_SCRIPT=""

    if dag_notification_is_disabled; then
        return 0
    fi

    local utils_dir
    utils_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local notify_script="${utils_dir}/dag_completion_notify.sh"
    if [[ ! -f "${notify_script}" ]]; then
        echo "Missing DAG completion notifier: ${notify_script}" >&2
        return 1
    fi

    mkdir -p "${submit_dir}" "${log_dir}" "${run_root}/notification"

    DAG_NOTIFICATION_STATUS_LOG="${run_root}/notification/${dag_label}_completion_status.txt"
    DAG_NOTIFICATION_NOTIFY_SCRIPT="${notify_script}"
}

dag_append_post_notification_script() {
    local dag_file="$1"
    local node_name="$2"
    local submit_dir="$3"
    local log_dir="$4"
    local run_root="$5"
    local run_id="$6"
    local dag_label="$7"
    local _repo_root="$8"

    dag_notification_prepare "${submit_dir}" "${log_dir}" "${run_root}" "${dag_label}" || return 1
    [[ -n "${DAG_NOTIFICATION_STATUS_LOG}" && -n "${DAG_NOTIFICATION_NOTIFY_SCRIPT}" ]] || return 0

    printf "SCRIPT POST %s %s --status_log %s --run_id %s --dag_label %s --dag_status \$DAG_STATUS --failed_count \$FAILED_COUNT\n" \
        "${node_name}" "${DAG_NOTIFICATION_NOTIFY_SCRIPT}" "${DAG_NOTIFICATION_STATUS_LOG}" "${run_id}" "${dag_label}" >> "${dag_file}"
}

dag_append_final_notification_node() {
    local dag_file="$1"
    local submit_dir="$2"
    local log_dir="$3"
    local run_root="$4"
    local run_id="$5"
    local dag_label="$6"
    local repo_root="$7"

    dag_notification_prepare "${submit_dir}" "${log_dir}" "${run_root}" "${dag_label}" || return 1
    [[ -n "${DAG_NOTIFICATION_STATUS_LOG}" && -n "${DAG_NOTIFICATION_NOTIFY_SCRIPT}" ]] || return 0

    local notify_submit_file="${submit_dir}/${dag_label}_completion_notify.sub"
    local notify_output_file="${log_dir}/${dag_label}_completion_notify.out"
    local notify_error_file="${log_dir}/${dag_label}_completion_notify.err"
    local notify_log_file="${log_dir}/${dag_label}_completion_notify.log"

    cat > "${notify_submit_file}" <<EOF
Universe   = vanilla
Executable = /bin/true
initialdir = ${repo_root}
should_transfer_files = NO
output     = ${notify_output_file}
error      = ${notify_error_file}
log        = ${notify_log_file}
queue
EOF

    printf "FINAL DAG_NOTIFY %s NOOP\n" "${notify_submit_file}" >> "${dag_file}"
    printf "SCRIPT PRE DAG_NOTIFY %s --status_log %s --run_id %s --dag_label %s --dag_status \$DAG_STATUS --failed_count \$FAILED_COUNT\n" \
        "${DAG_NOTIFICATION_NOTIFY_SCRIPT}" "${DAG_NOTIFICATION_STATUS_LOG}" "${run_id}" "${dag_label}" >> "${dag_file}"

    DAG_NOTIFICATION_FINAL_SUBMIT_FILE="${notify_submit_file}"
}
