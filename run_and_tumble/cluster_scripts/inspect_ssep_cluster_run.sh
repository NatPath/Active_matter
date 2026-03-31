#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash inspect_ssep_cluster_run.sh --submit_command "<original submit command>"

Direct modes:
  bash inspect_ssep_cluster_run.sh --run_id <production_run_id>
  bash inspect_ssep_cluster_run.sh --run_id <production_run_id> --topup_ns <int> --topup_nr <int>
  bash inspect_ssep_cluster_run.sh --rho <float> --num_replicas <int> [--n_sweeps <int>]

Examples:
  bash cluster_scripts/inspect_ssep_cluster_run.sh \
      --submit_command "bash cluster_scripts/submit_ssep_add_states_to_aggregate.sh --run_id ssep_ctmc_single_center_bond_L256_rho05_ns500000000_nr600_dag_20260327-212942 --ns 1000000000 --nr 600"

  bash cluster_scripts/inspect_ssep_cluster_run.sh \
      --submit_command "bash cluster_scripts/submit_ssep_single_center_bond_L256_dag.sh --num_replicas 600 --rho 0.75"

  bash cluster_scripts/inspect_ssep_cluster_run.sh \
      --submit_command "bash cluster_scripts/submit_ssep_saved_states_into_latest_aggregate.sh --run_id ssep_ctmc_single_center_bond_L256_rho075_ns500000000_nr600_dag_20260329-021733"

Options:
  --submit_command <string>      Original submit command. Parsed locally.
  --run_id <id>                  Production SSEP run_id.
  --rho <float>                  Resolve the latest matching production run when --run_id is omitted.
  --num_replicas <int>           Replica count filter / resolver.
  --n_sweeps <int>               Sweep count filter for production runs (default: 500000000).
  --topup_ns <int>               Inspect the latest matching top-up batch under the production run.
  --topup_nr <int>               Top-up replica count used together with --topup_ns.
  --job_token_glob <glob>        Override the default top-up dir pattern.
  --repo_root <path>             Override repo root (default: inferred from script location).
  -h, --help                     Show help.

Behavior:
  - Runs on the cluster filesystem directly. No SSH, no copying.
  - Resolves the target production run from runs/ssep/single_center_bond/run_registry.csv.
  - For top-up jobs, resolves the latest matching add_to_aggregate_jobs/<job_token>.
  - Reads run_info/manifest/logs/submit files in place.
  - Captures local condor_q / condor_history snapshots when cluster_id is known.
  - Writes summary.txt under the inspected run directory.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../run_ssep.jl" ]]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/run_ssep.jl" ]]; then
    REPO_ROOT="${SCRIPT_DIR}"
else
    echo "Could not locate repo root from script location: ${SCRIPT_DIR}"
    exit 1
fi

submit_command=""
run_id=""
rho=""
num_replicas=""
n_sweeps="500000000"
topup_ns=""
topup_nr=""
job_token_glob=""
repo_root_override=""

parse_submit_command() {
    local raw="$1"
    local -a tokens
    read -r -a tokens <<< "${raw}"
    local script_name=""
    local token next_token

    for token in "${tokens[@]}"; do
        case "$(basename "${token}")" in
            submit_ssep_single_center_bond_L256_dag.sh|submit_ssep_add_states_to_aggregate.sh|submit_ssep_saved_states_into_latest_aggregate.sh)
                script_name="$(basename "${token}")"
                break
                ;;
        esac
    done

    [[ -n "${script_name}" ]] || {
        echo "Could not identify an SSEP submit script inside --submit_command."
        exit 1
    }

    for ((i = 0; i < ${#tokens[@]}; i++)); do
        token="${tokens[$i]}"
        next_token="${tokens[$((i + 1))]:-}"
        case "${token}" in
            --run_id)
                run_id="${next_token}"
                ;;
            --rho)
                rho="${next_token}"
                ;;
            --num_replicas|--nr)
                num_replicas="${next_token}"
                ;;
            --n_sweeps|--ns)
                if [[ "${script_name}" == "submit_ssep_add_states_to_aggregate.sh" ]]; then
                    topup_ns="${next_token}"
                else
                    n_sweeps="${next_token}"
                fi
                ;;
        esac
    done

    if [[ "${script_name}" == "submit_ssep_add_states_to_aggregate.sh" ]]; then
        topup_nr="${num_replicas}"
    fi
}

read_run_info_value() {
    local run_info_path="$1"
    local key="$2"
    awk -F= -v k="${key}" '$1 == k {print substr($0, index($0, "=") + 1)}' "${run_info_path}" | tail -n 1
}

find_registry_row_by_run_id() {
    local registry_path="$1"
    local lookup_run_id="$2"
    awk -F, -v rid="${lookup_run_id}" 'NR > 1 && $2 == rid {row = $0} END {print row}' "${registry_path}"
}

find_latest_matching_registry_row() {
    local registry_path="$1"
    local target_rho="$2"
    local target_num_replicas="$3"
    local target_n_sweeps="$4"
    awk -F, \
        -v rho="${target_rho}" \
        -v numrep="${target_num_replicas}" \
        -v ns="${target_n_sweeps}" '
        NR == 1 { next }
        $3 != "production" { next }
        rho != "" && (($5 + 0.0) != (rho + 0.0)) { next }
        numrep != "" && $8 != numrep { next }
        ns != "" && $6 != ns { next }
        { row = $0 }
        END { print row }
    ' "${registry_path}"
}

latest_local_dir_matching() {
    local parent_dir="$1"
    local pattern="$2"
    [[ -d "${parent_dir}" ]] || return 0
    find "${parent_dir}" -maxdepth 1 -mindepth 1 -type d -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | awk '{ $1=""; sub(/^ /,""); print }'
}

capture_state_listing() {
    local state_dir="$1"
    local output_file="$2"
    mkdir -p "$(dirname "${output_file}")"
    if [[ -d "${state_dir}" ]]; then
        find "${state_dir}" -maxdepth 1 \( -type f -o -type l \) -name '*.jld2' -printf '%f\n' | sort > "${output_file}"
    else
        : > "${output_file}"
    fi
}

capture_condor_snapshots() {
    local cluster_id="$1"
    local output_dir="$2"
    mkdir -p "${output_dir}"

    if ! command -v condor_q >/dev/null 2>&1 || ! command -v condor_history >/dev/null 2>&1; then
        printf "condor_q/condor_history not found in PATH\n" > "${output_dir}/condor_q.txt"
        printf "condor_q/condor_history not found in PATH\n" > "${output_dir}/condor_history.txt"
        return 0
    fi

    if [[ ! "${cluster_id}" =~ ^[0-9]+$ ]]; then
        printf "cluster_id not available\n" > "${output_dir}/condor_q.txt"
        printf "cluster_id not available\n" > "${output_dir}/condor_history.txt"
        return 0
    fi

    condor_q -constraint "ClusterId == ${cluster_id} || DAGManJobId == ${cluster_id}" -nobatch \
        -af:jr ClusterId ProcId JobStatus HoldReason Cmd Args Out Err \
        > "${output_dir}/condor_q.txt" 2>/dev/null || true

    condor_history -match 5000 -constraint "ClusterId == ${cluster_id} || DAGManJobId == ${cluster_id}" \
        -af:jr ClusterId ProcId JobStatus ExitCode HoldReason RemoveReason Cmd Args Out Err \
        > "${output_dir}/condor_history.txt" 2>/dev/null || true
}

summarize_state_coverage() {
    local manifest_path="$1"
    local state_listing="$2"
    local output_path="$3"
    [[ -f "${manifest_path}" && -f "${state_listing}" ]] || return 0

    local expected_tags_file present_tags_file missing_tags_file
    expected_tags_file="$(mktemp)"
    present_tags_file="$(mktemp)"
    missing_tags_file="$(mktemp)"

    awk -F, 'NR > 1 && $1 == "replica" {print $7}' "${manifest_path}" > "${expected_tags_file}"
    : > "${present_tags_file}"
    : > "${missing_tags_file}"

    while IFS= read -r save_tag; do
        [[ -n "${save_tag}" ]] || continue
        if grep -Fq "_id-${save_tag}.jld2" "${state_listing}"; then
            echo "${save_tag}" >> "${present_tags_file}"
        else
            echo "${save_tag}" >> "${missing_tags_file}"
        fi
    done < "${expected_tags_file}"

    {
        echo "Expected replica states: $(wc -l < "${expected_tags_file}")"
        echo "Present replica states:  $(wc -l < "${present_tags_file}")"
        echo "Missing replica states:  $(wc -l < "${missing_tags_file}")"
        if [[ -s "${missing_tags_file}" ]]; then
            echo
            echo "First missing replica tags:"
            head -n 25 "${missing_tags_file}"
        fi
    } >> "${output_path}"

    rm -f "${expected_tags_file}" "${present_tags_file}" "${missing_tags_file}"
}

append_log_signature_summary() {
    local log_dir="$1"
    local extension="$2"
    local output_path="$3"
    [[ -d "${log_dir}" ]] || return 0

    local total_count nonempty_count
    total_count="$(find "${log_dir}" -maxdepth 1 -type f -name "*.${extension}" | wc -l | tr -d ' ')"
    nonempty_count="$(find "${log_dir}" -maxdepth 1 -type f -name "*.${extension}" -size +0c | wc -l | tr -d ' ')"

    {
        echo
        echo "${extension^^} files: total=${total_count}, nonempty=${nonempty_count}"
    } >> "${output_path}"

    if [[ "${nonempty_count}" == "0" ]]; then
        return 0
    fi

    {
        echo "Largest ${extension} files:"
        find "${log_dir}" -maxdepth 1 -type f -name "*.${extension}" -size +0c -printf '%s %f\n' \
            | sort -nr | head -n 15
        echo
        echo "Top ${extension} signatures:"
        find "${log_dir}" -maxdepth 1 -type f -name "*.${extension}" -size +0c -print0 \
            | xargs -0 rg -h -i 'error|exception|permission denied|not found|killed|out of memory|oom|hold|held|evict|abnormal|failed|signal|stacktrace|terminated' 2>/dev/null \
            | sed -E 's/[[:space:]]+/ /g' \
            | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//' \
            | awk 'length($0) > 0' \
            | sort | uniq -c | sort -nr | head -n 20
        echo
        echo "Tails of largest ${extension} files:"
        while read -r _size file_name; do
            [[ -n "${file_name}" ]] || continue
            echo "----- ${file_name} -----"
            tail -n 40 "${log_dir}/${file_name}" 2>/dev/null || true
            echo
        done < <(find "${log_dir}" -maxdepth 1 -type f -name "*.${extension}" -size +0c -printf '%s %f\n' | sort -nr | head -n 3)
    } >> "${output_path}" || true
}

append_condor_summary() {
    local condor_q_file="$1"
    local condor_history_file="$2"
    local output_path="$3"

    {
        echo
        echo "Condor q snapshot: ${condor_q_file}"
        if [[ -s "${condor_q_file}" ]]; then
            awk '
                function status_field() { return ($1 ~ /^[0-9]+\.[0-9]+$/ ? 4 : 3) }
                function status_name(code) {
                    return code == 1 ? "Idle" :
                           code == 2 ? "Running" :
                           code == 3 ? "Removed" :
                           code == 4 ? "Completed" :
                           code == 5 ? "Held" :
                           code == 6 ? "TransferringOutput" :
                           code == 7 ? "Suspended" : "Unknown"
                }
                { counts[$status_field()]++ }
                END {
                    for (code in counts) {
                        printf("  %s: %d\n", status_name(code), counts[code])
                    }
                }
            ' "${condor_q_file}"
            echo
            echo "Held jobs:"
            awk '
                {
                    status_col = ($1 ~ /^[0-9]+\.[0-9]+$/ ? 4 : 3)
                    if ($status_col == 5) print
                }
            ' "${condor_q_file}" | head -n 50
        else
            echo "  No active jobs found for the captured query."
        fi

        echo
        echo "Condor history snapshot: ${condor_history_file}"
        if [[ -s "${condor_history_file}" ]]; then
            awk '
                function status_field() { return ($1 ~ /^[0-9]+\.[0-9]+$/ ? 4 : 3) }
                function status_name(code) {
                    return code == 1 ? "Idle" :
                           code == 2 ? "Running" :
                           code == 3 ? "Removed" :
                           code == 4 ? "Completed" :
                           code == 5 ? "Held" :
                           code == 6 ? "TransferringOutput" :
                           code == 7 ? "Suspended" : "Unknown"
                }
                { counts[$status_field()]++ }
                END {
                    for (code in counts) {
                        printf("  %s: %d\n", status_name(code), counts[code])
                    }
                }
            ' "${condor_history_file}"
            echo
            echo "Non-zero exit / removal lines:"
            awk '
                {
                    status_col = ($1 ~ /^[0-9]+\.[0-9]+$/ ? 4 : 3)
                    exit_col = status_col + 1
                    hold_col = status_col + 2
                    remove_col = status_col + 3
                    if (($exit_col <= NF && $exit_col != "" && $exit_col != "0" && $exit_col != "undefined") ||
                        ($hold_col <= NF && $hold_col != "" && $hold_col != "undefined") ||
                        ($remove_col <= NF && $remove_col != "" && $remove_col != "undefined")) {
                        print
                    }
                }
            ' "${condor_history_file}" | head -n 50
        else
            echo "  No condor_history rows captured."
        fi
    } >> "${output_path}"
}

write_summary() {
    local job_root="$1"
    local label="$2"
    local state_dir="$3"
    local summary_file="${job_root}/summary.txt"
    local manifest_file="${job_root}/manifest.csv"
    local log_dir="${job_root}/logs"
    local cluster_id state_listing_file condor_dir

    cluster_id=""
    if [[ -f "${job_root}/run_info.txt" ]]; then
        cluster_id="$(read_run_info_value "${job_root}/run_info.txt" "cluster_id")"
    fi
    condor_dir="${job_root}/debug"
    state_listing_file="${condor_dir}/state_listing.txt"

    mkdir -p "${condor_dir}"
    capture_state_listing "${state_dir}" "${state_listing_file}"
    capture_condor_snapshots "${cluster_id}" "${condor_dir}"

    {
        echo "Inspection label: ${label}"
        echo "Job root: ${job_root}"
        echo "State dir: ${state_dir}"
        echo "Cluster ID: ${cluster_id:-unknown}"
    } > "${summary_file}"

    summarize_state_coverage "${manifest_file}" "${state_listing_file}" "${summary_file}"
    append_log_signature_summary "${log_dir}" "err" "${summary_file}"
    append_log_signature_summary "${log_dir}" "out" "${summary_file}"
    append_condor_summary "${condor_dir}/condor_q.txt" "${condor_dir}/condor_history.txt" "${summary_file}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit_command)
            submit_command="${2:-}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-}"
            shift 2
            ;;
        --rho)
            rho="${2:-}"
            shift 2
            ;;
        --num_replicas)
            num_replicas="${2:-}"
            shift 2
            ;;
        --n_sweeps)
            n_sweeps="${2:-}"
            shift 2
            ;;
        --topup_ns)
            topup_ns="${2:-}"
            shift 2
            ;;
        --topup_nr)
            topup_nr="${2:-}"
            shift 2
            ;;
        --job_token_glob)
            job_token_glob="${2:-}"
            shift 2
            ;;
        --repo_root)
            repo_root_override="${2:-}"
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

if [[ -n "${repo_root_override}" ]]; then
    REPO_ROOT="$(cd "${repo_root_override}" && pwd)"
fi

if [[ -n "${submit_command}" ]]; then
    parse_submit_command "${submit_command}"
fi

if [[ -z "${run_id}" && ( -z "${rho}" || -z "${num_replicas}" ) ]]; then
    echo "Provide either --run_id, or --submit_command, or --rho plus --num_replicas."
    usage
    exit 1
fi

if [[ -n "${topup_ns}" && -z "${topup_nr}" ]]; then
    echo "--topup_ns requires --topup_nr."
    exit 1
fi

registry_file="${REPO_ROOT}/runs/ssep/single_center_bond/run_registry.csv"
[[ -f "${registry_file}" ]] || {
    echo "Registry not found: ${registry_file}"
    exit 1
}

registry_row=""
if [[ -n "${run_id}" ]]; then
    registry_row="$(find_registry_row_by_run_id "${registry_file}" "${run_id}")"
    if [[ -z "${registry_row}" ]]; then
        echo "run_id '${run_id}' was not found in ${registry_file}"
        exit 1
    fi
else
    registry_row="$(find_latest_matching_registry_row "${registry_file}" "${rho}" "${num_replicas}" "${n_sweeps}")"
    if [[ -z "${registry_row}" ]]; then
        echo "Could not resolve a production SSEP run for rho='${rho}', num_replicas='${num_replicas}', n_sweeps='${n_sweeps}'."
        exit 1
    fi
fi

IFS=',' read -r reg_ts reg_run_id reg_mode reg_L reg_rho reg_ns reg_warmup reg_numrep reg_cpus reg_mem reg_run_root reg_submit_dir reg_log_dir reg_state_dir reg_config_path reg_aggregate_run_id <<< "${registry_row}"
run_id="${reg_run_id}"
production_root="${reg_run_root}"

[[ -d "${production_root}" ]] || {
    echo "Production run directory not found: ${production_root}"
    exit 1
}

write_summary "${production_root}" "production" "${reg_state_dir}"
echo "Wrote production summary: ${production_root}/summary.txt"

if [[ -n "${topup_ns}" && -n "${topup_nr}" ]]; then
    if [[ -z "${job_token_glob}" ]]; then
        job_token_glob="*topup_ns${topup_ns}_nr${topup_nr}_*"
    fi

    topup_parent="${production_root}/add_to_aggregate_jobs"
    topup_root="$(latest_local_dir_matching "${topup_parent}" "${job_token_glob}")"
    [[ -n "${topup_root}" ]] || {
        echo "Could not resolve a top-up job under ${topup_parent} matching ${job_token_glob}"
        exit 1
    }

    raw_state_dir="$(read_run_info_value "${topup_root}/run_info.txt" "raw_state_dir")"
    if [[ -z "${raw_state_dir}" ]]; then
        topup_token="$(basename "${topup_root}")"
        raw_state_dir="${reg_state_dir}/topup_batches/${topup_token}"
    fi

    write_summary "${topup_root}" "topup" "${raw_state_dir}"
    echo "Wrote top-up summary: ${topup_root}/summary.txt"
fi
