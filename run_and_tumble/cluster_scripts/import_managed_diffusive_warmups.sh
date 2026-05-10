#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/import_managed_diffusive_warmups.sh \
      --case <diffusive_1d_pmlr|single_origin_bond> --run_id <managed_run_id> \
      --source_state_dir <warmup_states_dir> [options]

Options:
  --warmup_sweeps <int>      Sweeps represented by imported states.
                             Defaults to warmup_threshold_sweeps from run_spec.yaml.
  --link_mode <mode>         register, auto, hardlink, copy, or symlink (default: register)
  --limit <int>              Import at most this many new states
  --dry_run                  Print planned imports without writing files
  -h, --help                 Show this help

The script does not rename or modify source files. By default it only records
the existing state paths in the managed ledger, which keeps the file count low.
Use --link_mode hardlink/copy/symlink/auto only when you explicitly want
canonical current.jld2 files materialized under the managed run.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"

case_name=""
run_id=""
source_state_dir=""
warmup_sweeps=""
link_mode="register"
limit="0"
dry_run="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case) case_name="${2:-}"; shift 2 ;;
        --run_id) run_id="${2:-}"; shift 2 ;;
        --source_state_dir) source_state_dir="${2:-}"; shift 2 ;;
        --warmup_sweeps) warmup_sweeps="${2:-}"; shift 2 ;;
        --link_mode) link_mode="${2:-}"; shift 2 ;;
        --limit) limit="${2:-}"; shift 2 ;;
        --dry_run) dry_run="true"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "${case_name}" || -z "${run_id}" || -z "${source_state_dir}" ]]; then
    echo "Missing --case, --run_id, or --source_state_dir." >&2
    usage
    exit 1
fi

case_name="$(managed_normalize_case "${case_name}")"
run_id="$(managed_slugify "${run_id}")"
if [[ "${source_state_dir}" != /* ]]; then
    source_state_dir="${REPO_ROOT}/${source_state_dir}"
fi
if [[ ! -d "${source_state_dir}" ]]; then
    echo "Source state directory not found: ${source_state_dir}" >&2
    exit 1
fi

case "${link_mode}" in
    register|auto|hardlink|copy|symlink) ;;
    *) echo "--link_mode must be register, auto, hardlink, copy, or symlink. Got '${link_mode}'." >&2; exit 1 ;;
esac

managed_require_positive_int "limit" "${limit:-0}" 2>/dev/null || {
    if [[ "${limit}" != "0" ]]; then
        echo "--limit must be a non-negative integer. Got '${limit}'." >&2
        exit 1
    fi
}

run_root="$(managed_run_root "${REPO_ROOT}" "${case_name}" "${run_id}")"
spec_path="${run_root}/run_spec.yaml"
replicas_csv="${run_root}/replicas.csv"
replica_root="${run_root}/replicas"
lock_file="${run_root}/manager.lock"
if [[ ! -f "${spec_path}" || ! -f "${replicas_csv}" ]]; then
    echo "Managed run is not initialized: ${run_root}" >&2
    echo "Run init_managed_diffusive.sh --case ${case_name} first." >&2
    exit 1
fi

if [[ -z "${warmup_sweeps}" ]]; then
    warmup_sweeps="$(managed_yaml_value "${spec_path}" "warmup_threshold_sweeps")"
fi
managed_require_positive_int "warmup_sweeps" "${warmup_sweeps}"

already_registered_source() {
    local source_path="$1"
    awk -F, -v source_path="${source_path}" 'NR > 1 && $6 == source_path { found=1 } END { exit found ? 0 : 1 }' "${replicas_csv}"
}

source_tag_from_file() {
    local path="$1"
    basename "${path}" | sed -nE 's/.*_id-([^.]*)\.jld2$/\1/p'
}

install_state_file() {
    local source_path="$1"
    local target_path="$2"
    local tmp_path="${target_path}.tmp.$$"
    rm -f "${tmp_path}"

    case "${link_mode}" in
        register)
            return 0
            ;;
        hardlink)
            ln "${source_path}" "${tmp_path}"
            ;;
        copy)
            cp -p "${source_path}" "${tmp_path}"
            ;;
        symlink)
            ln -s "${source_path}" "${tmp_path}"
            ;;
        auto)
            if ! ln "${source_path}" "${tmp_path}" 2>/dev/null; then
                cp -p "${source_path}" "${tmp_path}"
            fi
            ;;
    esac
    mv -f "${tmp_path}" "${target_path}"
}

imported=0
skipped=0
timestamp="$(managed_timestamp)"

{
    flock 9
    next_idx="$(managed_next_replica_index "${replicas_csv}")"

    while IFS= read -r source_path; do
        [[ -n "${source_path}" ]] || continue
        if already_registered_source "${source_path}"; then
            skipped=$((skipped + 1))
            continue
        fi
        if (( limit > 0 && imported >= limit )); then
            break
        fi

        replica_id="$(managed_replica_id "${next_idx}")"
        replica_dir="${replica_root}/${replica_id}"
        if [[ "${link_mode}" == "register" ]]; then
            target_state="${source_path}"
        else
            target_state="${replica_dir}/current.jld2"
        fi
        source_tag="$(source_tag_from_file "${source_path}")"
        [[ -n "${source_tag}" ]] || source_tag="unknown"

        echo "Import ${source_path} -> ${replica_id} (link_mode=${link_mode})"
        if [[ "${dry_run}" != "true" ]]; then
            if [[ "${link_mode}" != "register" ]]; then
                mkdir -p "${replica_dir}"
                install_state_file "${source_path}" "${target_state}"
                cat > "${replica_dir}/current.meta" <<EOF
replica_id=${replica_id}
phase=ready
elapsed_sweeps=${warmup_sweeps}
statistics_sweeps=0
latest_state=${target_state}
source_path=${source_path}
source_tag=${source_tag}
status=idle
updated_at=${timestamp}
EOF
            fi
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
                "${replica_id}" "ready" "${warmup_sweeps}" "0" "${target_state}" "${source_path}" "${source_tag}" "" "idle" "${timestamp}" \
                >> "${replicas_csv}"
        fi

        imported=$((imported + 1))
        next_idx=$((next_idx + 1))
    done < <(find "${source_state_dir}" -maxdepth 1 -type f -name "*.jld2" ! -size 0 -print | sort -V)
} 9>"${lock_file}"

echo "Import summary:"
echo "  run_id=${run_id}"
echo "  case=${case_name}"
echo "  source_state_dir=${source_state_dir}"
echo "  imported=${imported}"
echo "  skipped_existing=${skipped}"
echo "  dry_run=${dry_run}"
