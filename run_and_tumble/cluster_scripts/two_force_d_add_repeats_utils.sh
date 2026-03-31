#!/usr/bin/env bash

# Shared helpers for inspecting two_force_d add-repeats batches.

two_force_repo_root_from_script_dir() {
    local script_dir="$1"
    if [[ -f "${script_dir}/../run_diffusive_no_activity.jl" ]]; then
        (cd "${script_dir}/.." && pwd)
        return 0
    fi
    if [[ -f "${script_dir}/run_diffusive_no_activity.jl" ]]; then
        (cd "${script_dir}" && pwd)
        return 0
    fi
    echo "Could not locate repo root from script location: ${script_dir}" >&2
    return 1
}

two_force_sanitize_token() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

two_force_read_key_value() {
    local file_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${file_path}" | tail -n 1
}

two_force_run_hash() {
    local run_id="$1"
    printf "%s" "${run_id}" | cksum | awk '{print $1}'
}

two_force_compact_timestamp() {
    local timestamp="$1"
    printf "%s" "${timestamp}" | tr -cd '0-9'
}

two_force_trim_spaces() {
    local raw="$1"
    printf "%s" "${raw}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

two_force_parse_csv_ints() {
    local raw_csv="$1"
    local -n out_ref="$2"
    local seen=" "
    local raw value
    local -a values=()

    out_ref=()
    IFS=',' read -r -a values <<< "${raw_csv}"
    for raw in "${values[@]}"; do
        value="$(two_force_trim_spaces "${raw}")"
        [[ -z "${value}" ]] && continue
        if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
            echo "Invalid integer '${value}' in csv list '${raw_csv}'." >&2
            return 1
        fi
        if [[ "${seen}" != *" ${value} "* ]]; then
            out_ref+=("${value}")
            seen="${seen}${value} "
        fi
    done
}

two_force_find_run_info_by_run_id() {
    local repo_root="$1"
    local lookup_run_id="$2"
    local mode_hint="$3"
    local candidate=""
    local registry_file="${repo_root}/runs/two_force_d/run_registry.csv"

    if [[ "${mode_hint}" == "warmup" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/warmup/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "production" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi
    if [[ "${mode_hint}" == "warmup_production" || "${mode_hint}" == "auto" ]]; then
        candidate="${repo_root}/runs/two_force_d/warmup_production/${lookup_run_id}/run_info.txt"
        [[ -f "${candidate}" ]] && { echo "${candidate}"; return 0; }
    fi

    if [[ -f "${registry_file}" ]]; then
        local registry_row reg_run_root
        registry_row="$(awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${registry_file}")"
        if [[ -n "${registry_row}" ]]; then
            IFS=',' read -r _ts _rid _mode _L _rho _ns _dmin _dmax _dstep _cpus _mem reg_run_root _log_dir _state_dir _warmup_state_dir <<< "${registry_row}"
            if [[ -n "${reg_run_root}" && -f "${reg_run_root}/run_info.txt" ]]; then
                echo "${reg_run_root}/run_info.txt"
                return 0
            fi
        fi
    fi

    return 1
}

two_force_infer_spacing_from_run_id() {
    local rid="$1"
    if [[ "${rid}" =~ -lm(_|$) ]]; then
        echo "log_midpoints"
    else
        echo "linear"
    fi
}

two_force_resolve_target_production_run_info() {
    local repo_root="$1"
    local lookup_run_id="$2"
    local mode_hint="$3"
    local resolved_info resolved_mode production_run_info production_run_id

    resolved_info="$(two_force_find_run_info_by_run_id "${repo_root}" "${lookup_run_id}" "${mode_hint}" || true)"
    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve run_info for run_id='${lookup_run_id}' (mode=${mode_hint})." >&2
        return 1
    fi

    resolved_mode="$(two_force_read_key_value "${resolved_info}" "mode")"
    if [[ "${resolved_mode}" == "warmup_production" ]]; then
        production_run_info="$(two_force_read_key_value "${resolved_info}" "production_run_info")"
        production_run_id="$(two_force_read_key_value "${resolved_info}" "production_run_id")"
        if [[ -n "${production_run_info}" && -f "${production_run_info}" ]]; then
            resolved_info="${production_run_info}"
        elif [[ -n "${production_run_id}" ]]; then
            resolved_info="$(two_force_find_run_info_by_run_id "${repo_root}" "${production_run_id}" "production" || true)"
        fi
    fi

    if [[ -z "${resolved_info}" || ! -f "${resolved_info}" ]]; then
        echo "Could not resolve production run_info for run_id='${lookup_run_id}'." >&2
        return 1
    fi
    if [[ "$(two_force_read_key_value "${resolved_info}" "mode")" != "production" ]]; then
        echo "run_id='${lookup_run_id}' does not resolve to a production run." >&2
        return 1
    fi

    echo "${resolved_info}"
}

two_force_lookup_latest_matching_warmup_state_dir() {
    local repo_root="$1"
    local prod_run_info="$2"
    local registry_file="${repo_root}/runs/two_force_d/run_registry.csv"
    local L_val rho_val d_min d_max d_step d_spacing result

    [[ -f "${registry_file}" ]] || return 1

    L_val="$(two_force_read_key_value "${prod_run_info}" "L")"
    rho_val="$(two_force_read_key_value "${prod_run_info}" "rho0")"
    d_min="$(two_force_read_key_value "${prod_run_info}" "d_min")"
    d_max="$(two_force_read_key_value "${prod_run_info}" "d_max")"
    d_step="$(two_force_read_key_value "${prod_run_info}" "d_step")"
    d_spacing="$(two_force_read_key_value "${prod_run_info}" "d_spacing")"
    if [[ -z "${d_spacing}" ]]; then
        d_spacing="$(two_force_infer_spacing_from_run_id "$(two_force_read_key_value "${prod_run_info}" "run_id")")"
    fi

    result="$(
        awk -F, -v L="${L_val}" -v rho="${rho_val}" -v dmin="${d_min}" -v dmax="${d_max}" -v dstep="${d_step}" -v spacing="${d_spacing}" '
            NR == 1 {next}
            $3 == "warmup" && $4 == L && $5 == rho && $7 == dmin && $8 == dmax && $9 == dstep {
                if (spacing == "log_midpoints" && $2 !~ /-lm(_|$)/) next
                if (spacing != "log_midpoints" && $2 ~ /-lm(_|$)/) next
                state_dir = $14
            }
            END {print state_dir}
        ' "${registry_file}"
    )"
    if [[ -n "${result}" && -d "${result}" ]]; then
        echo "${result}"
        return 0
    fi
    return 1
}

two_force_resolve_warmup_state_dir_for_run() {
    local repo_root="$1"
    local prod_run_info="$2"
    local current_info="$2"
    local depth=0
    local candidate warmup_run_id continue_run_id warmup_info

    while (( depth < 12 )); do
        candidate="$(two_force_read_key_value "${current_info}" "warmup_state_dir")"
        if [[ -n "${candidate}" && -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi

        warmup_run_id="$(two_force_read_key_value "${current_info}" "warmup_run_id")"
        if [[ -n "${warmup_run_id}" ]]; then
            warmup_info="$(two_force_find_run_info_by_run_id "${repo_root}" "${warmup_run_id}" "warmup" || true)"
            if [[ -n "${warmup_info}" && -f "${warmup_info}" ]]; then
                candidate="$(two_force_read_key_value "${warmup_info}" "state_dir")"
                if [[ -n "${candidate}" && -d "${candidate}" ]]; then
                    echo "${candidate}"
                    return 0
                fi
            fi
        fi

        continue_run_id="$(two_force_read_key_value "${current_info}" "continue_run_id")"
        if [[ -z "${continue_run_id}" ]]; then
            break
        fi
        current_info="$(two_force_find_run_info_by_run_id "${repo_root}" "${continue_run_id}" "production" || true)"
        if [[ -z "${current_info}" || ! -f "${current_info}" ]]; then
            break
        fi
        depth=$((depth + 1))
    done

    two_force_lookup_latest_matching_warmup_state_dir "${repo_root}" "${prod_run_info}" || return 1
}

two_force_list_add_repeats_job_roots() {
    local repo_root="$1"
    local target_run_id="$2"
    local job_label="${3:-}"
    local job_token="${4:-}"
    local safe_target_run_id safe_job_label safe_job_token
    local -a candidates=()
    local root base

    safe_target_run_id="$(two_force_sanitize_token "${target_run_id}")"
    [[ -d "${repo_root}/runs/two_force_d/add_repeats_jobs" ]] || return 0

    mapfile -t candidates < <(
        find "${repo_root}/runs/two_force_d/add_repeats_jobs" -mindepth 1 -maxdepth 1 -type d \
            -name "${safe_target_run_id}_*" -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | awk '{ $1=""; sub(/^ /,""); print }'
    )

    safe_job_label=""
    safe_job_token=""
    [[ -n "${job_label}" ]] && safe_job_label="$(two_force_sanitize_token "${job_label}")"
    [[ -n "${job_token}" ]] && safe_job_token="$(two_force_sanitize_token "${job_token}")"

    for root in "${candidates[@]}"; do
        base="$(basename "${root}")"
        if [[ -n "${safe_job_label}" && "${base}" != "${safe_target_run_id}_${safe_job_label}"* ]]; then
            continue
        fi
        if [[ -n "${safe_job_token}" && "${base}" != "${safe_target_run_id}_${safe_job_token}" && "${base}" != *"_${safe_job_token}" && "${base}" != *"_${safe_job_token}_"* ]]; then
            continue
        fi
        printf "%s\n" "${root}"
    done
}

two_force_pick_add_repeats_job_root() {
    local repo_root="$1"
    local target_run_id="$2"
    local job_label="${3:-}"
    local job_token="${4:-}"
    local -a matches=()

    mapfile -t matches < <(two_force_list_add_repeats_job_roots "${repo_root}" "${target_run_id}" "${job_label}" "${job_token}")
    (( ${#matches[@]} > 0 )) || return 1
    echo "${matches[0]}"
}

two_force_extract_submit_path_value() {
    local submit_file="$1"
    local key="$2"
    awk -F= -v wanted="${key}" '
        $1 ~ "^[[:space:]]*" wanted "[[:space:]]*$" {
            val = substr($0, index($0, "=") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
            print val
            exit
        }
    ' "${submit_file}"
}

two_force_extract_save_tag_from_submit() {
    local submit_file="$1"
    awk '
        match($0, /--save_tag[[:space:]]+([^[:space:]]+)/, m) {
            print m[1]
            exit
        }
    ' "${submit_file}"
}

two_force_replica_save_tag() {
    local run_id="$1"
    local timestamp="$2"
    local d_val="$3"
    local replica_idx="$4"
    local run_hash timestamp_compact

    run_hash="$(two_force_run_hash "${run_id}")"
    timestamp_compact="$(two_force_compact_timestamp "${timestamp}")"
    printf "replica_addrep_%s_%s_d%s_r%s" "${run_hash}" "${timestamp_compact}" "${d_val}" "${replica_idx}"
}

two_force_resolve_replica_node_metadata() {
    local submit_dir="$1"
    local log_dir="$2"
    local run_id="$3"
    local timestamp="$4"
    local d_val="$5"
    local replica_idx="$6"
    local -n save_tag_ref="$7"
    local -n out_ref="$8"
    local -n err_ref="$9"
    local -n log_ref="${10}"
    local -n submit_ref="${11}"

    submit_ref="${submit_dir}/d_${d_val}_replica_${replica_idx}.sub"
    if [[ -f "${submit_ref}" ]]; then
        save_tag_ref="$(two_force_extract_save_tag_from_submit "${submit_ref}")"
        out_ref="$(two_force_extract_submit_path_value "${submit_ref}" "output")"
        err_ref="$(two_force_extract_submit_path_value "${submit_ref}" "error")"
        log_ref="$(two_force_extract_submit_path_value "${submit_ref}" "log")"
        return 0
    fi

    save_tag_ref="$(two_force_replica_save_tag "${run_id}" "${timestamp}" "${d_val}" "${replica_idx}")"
    out_ref="${log_dir}/d_${d_val}_r${replica_idx}.out"
    err_ref="${log_dir}/d_${d_val}_r${replica_idx}.err"
    log_ref="${log_dir}/d_${d_val}_r${replica_idx}.log"
}

two_force_latest_state_for_id_tag_top_level() {
    local dir="$1"
    local id_tag="$2"
    local best_path=""
    local best_mtime=0
    local candidate mtime

    [[ -d "${dir}" ]] || { printf ""; return 0; }
    while IFS= read -r -d '' candidate; do
        mtime="$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
        if [[ "${mtime}" =~ ^[0-9]+$ ]] && (( mtime >= best_mtime )); then
            best_mtime="${mtime}"
            best_path="${candidate}"
        fi
    done < <(find "${dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" ! -size 0 -print0 2>/dev/null)
    printf "%s" "${best_path}"
}

two_force_any_state_for_id_tag_top_level() {
    local dir="$1"
    local id_tag="$2"
    local candidate=""

    [[ -d "${dir}" ]] || { printf ""; return 0; }
    candidate="$(find "${dir}" -maxdepth 1 -type f -name "*_id-${id_tag}.jld2" -print 2>/dev/null | head -n 1 || true)"
    printf "%s" "${candidate}"
}

two_force_file_mtime_epoch() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        stat -c %Y "${path}" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

two_force_timestamp_to_epoch() {
    local stamp="$1"
    local year month day hour minute second formatted=""

    if [[ "${stamp}" =~ ^([0-9]{4})([0-9]{2})([0-9]{2})-([0-9]{2})([0-9]{2})([0-9]{2})$ ]]; then
        year="${BASH_REMATCH[1]}"
        month="${BASH_REMATCH[2]}"
        day="${BASH_REMATCH[3]}"
        hour="${BASH_REMATCH[4]}"
        minute="${BASH_REMATCH[5]}"
        second="${BASH_REMATCH[6]}"
        formatted="${year}-${month}-${day} ${hour}:${minute}:${second}"
        date -d "${formatted}" +%s 2>/dev/null || echo 0
        return 0
    fi
    echo 0
}

two_force_condor_log_state() {
    local log_path="$1"
    local return_value=""

    if [[ ! -f "${log_path}" ]]; then
        echo "missing"
        return 0
    fi
    if grep -q "Job was aborted" "${log_path}" 2>/dev/null; then
        echo "aborted"
        return 0
    fi
    if grep -q "Job was held" "${log_path}" 2>/dev/null; then
        echo "held"
        return 0
    fi
    return_value="$(sed -nE 's/.*return value ([0-9]+).*/\1/p' "${log_path}" | tail -n 1)"
    if [[ -n "${return_value}" ]]; then
        if [[ "${return_value}" == "0" ]]; then
            echo "terminated_0"
        else
            echo "terminated_${return_value}"
        fi
        return 0
    fi
    if grep -q "Job terminated" "${log_path}" 2>/dev/null; then
        echo "terminated_unknown"
        return 0
    fi
    if grep -q "executing on host" "${log_path}" 2>/dev/null; then
        echo "running"
        return 0
    fi
    if grep -q "submitted from host" "${log_path}" 2>/dev/null; then
        echo "submitted"
        return 0
    fi
    echo "unknown"
}

two_force_classify_node_state() {
    local saved_path="$1"
    local zero_path="$2"
    local condor_state="$3"
    local min_saved_epoch="${4:-0}"
    local saved_mtime=0

    if [[ -n "${saved_path}" ]]; then
        if [[ "${min_saved_epoch}" =~ ^[0-9]+$ ]] && (( min_saved_epoch > 0 )); then
            saved_mtime="$(two_force_file_mtime_epoch "${saved_path}")"
            if [[ "${saved_mtime}" =~ ^[0-9]+$ ]] && (( saved_mtime < min_saved_epoch )); then
                echo "preexisting"
                return 0
            fi
        fi
        echo "done"
        return 0
    fi
    if [[ -n "${zero_path}" ]]; then
        echo "zero_size"
        return 0
    fi
    case "${condor_state}" in
        terminated_0)
            echo "missing_output"
            ;;
        terminated_*|held|aborted|terminated_unknown)
            echo "failed"
            ;;
        running)
            echo "running"
            ;;
        submitted)
            echo "submitted"
            ;;
        missing)
            echo "missing"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

two_force_should_show_problem_state() {
    local only_mode="$1"
    local node_state="$2"

    case "${only_mode}" in
        all)
            return 0
            ;;
        done)
            [[ "${node_state}" == "done" ]]
            return
            ;;
        failed)
            [[ "${node_state}" == "failed" || "${node_state}" == "zero_size" || "${node_state}" == "missing_output" ]]
            return
            ;;
        problems)
            [[ "${node_state}" != "done" ]]
            return
            ;;
        *)
            return 1
            ;;
    esac
}

two_force_state_rank() {
    local node_state="$1"
    case "${node_state}" in
        done)
            echo 70
            ;;
        preexisting)
            echo 65
            ;;
        running)
            echo 60
            ;;
        submitted)
            echo 50
            ;;
        failed|missing_output|zero_size)
            echo 40
            ;;
        unknown)
            echo 30
            ;;
        missing)
            echo 20
            ;;
        *)
            echo 10
            ;;
    esac
}

two_force_pick_better_state() {
    local current_state="$1"
    local candidate_state="$2"
    local current_rank candidate_rank

    current_rank="$(two_force_state_rank "${current_state}")"
    candidate_rank="$(two_force_state_rank "${candidate_state}")"
    if (( candidate_rank > current_rank )); then
        echo "${candidate_state}"
    else
        echo "${current_state}"
    fi
}

two_force_print_file_tail() {
    local label="$1"
    local path="$2"
    local tail_lines="$3"

    if [[ ! -f "${path}" ]]; then
        echo "  ${label}: missing"
        return 0
    fi

    echo "  ${label}: ${path}"
    tail -n "${tail_lines}" "${path}" | sed 's/^/    /'
}

two_force_last_nonempty_line() {
    local path="$1"
    [[ -f "${path}" ]] || return 0
    tac "${path}" 2>/dev/null | awk 'NF {print; exit}'
}

two_force_last_matching_line() {
    local path="$1"
    local pattern="$2"
    [[ -f "${path}" ]] || return 0
    awk -v pat="${pattern}" '$0 ~ pat {line=$0} END {print line}' "${path}"
}

two_force_failure_signature() {
    local err_path="$1"
    local out_path="$2"
    local log_path="$3"
    local line=""

    line="$(two_force_last_matching_line "${err_path}" "(ERROR|Error|error|Exception|exception|Traceback|traceback|Killed|killed|OutOfMemory|oom|OOM|signal|terminated)")"
    [[ -n "${line}" ]] || line="$(two_force_last_matching_line "${out_path}" "(ERROR|Error|error|Exception|exception|Traceback|traceback|Killed|killed|OutOfMemory|oom|OOM|signal|terminated)")"
    [[ -n "${line}" ]] || line="$(two_force_last_nonempty_line "${err_path}")"
    [[ -n "${line}" ]] || line="$(two_force_last_nonempty_line "${out_path}")"
    [[ -n "${line}" ]] || line="$(two_force_last_matching_line "${log_path}" "(return value|Job was held|Job was aborted|Job terminated)")"
    [[ -n "${line}" ]] || line="(no signature found)"
    printf "%s" "${line}"
}
