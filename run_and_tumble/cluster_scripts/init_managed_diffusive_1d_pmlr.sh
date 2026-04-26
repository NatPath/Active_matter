#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/init_managed_diffusive_1d_pmlr.sh [options]

Options:
  --run_id <id>                 Stable managed run id
  --L <int>                     System size (default: 2048)
  --rho <float>                 Density (default: 100)
  --gamma <float>               Potential switch rate (default: 1)
  --potential_strength <float>  PmLr potential strength (default: 16)
  --warmup_threshold <int>      Sweeps before statistics start (default: 1000000)
  --segment_sweeps <int>        Default sweeps per submitted job (default: 1000000)
  --checkpoint_interval <int>   Checkpoint interval in sweeps (default: 100000)
  --target_replicas <int>       Desired managed replica pool size (default: 600)
  --force_spec_update           Rewrite run_spec.yaml if it already exists
  -h, --help                    Show this help

This script only creates ledgers and directories. It does not submit jobs.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/managed_diffusive_1d_pmlr_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"

L="2048"
rho="100"
gamma="1"
potential_strength="16"
warmup_threshold="1000000"
segment_sweeps="1000000"
checkpoint_interval="100000"
target_replicas="600"
run_id=""
force_spec_update="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_id) run_id="${2:-}"; shift 2 ;;
        --L) L="${2:-}"; shift 2 ;;
        --rho) rho="${2:-}"; shift 2 ;;
        --gamma) gamma="${2:-}"; shift 2 ;;
        --potential_strength) potential_strength="${2:-}"; shift 2 ;;
        --warmup_threshold) warmup_threshold="${2:-}"; shift 2 ;;
        --segment_sweeps) segment_sweeps="${2:-}"; shift 2 ;;
        --checkpoint_interval) checkpoint_interval="${2:-}"; shift 2 ;;
        --target_replicas) target_replicas="${2:-}"; shift 2 ;;
        --force_spec_update) force_spec_update="true"; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

managed_require_positive_int "L" "${L}"
managed_require_positive_int "warmup_threshold" "${warmup_threshold}"
managed_require_positive_int "segment_sweeps" "${segment_sweeps}"
managed_require_positive_int "checkpoint_interval" "${checkpoint_interval}"
managed_require_positive_int "target_replicas" "${target_replicas}"
managed_require_positive_float "rho" "${rho}"
managed_require_positive_float "gamma" "${gamma}"
managed_require_positive_float "potential_strength" "${potential_strength}"

if [[ -z "${run_id}" ]]; then
    run_id="$(managed_default_run_id "${L}" "${rho}" "${gamma}" "${potential_strength}")"
else
    run_id="$(managed_slugify "${run_id}")"
fi

run_root="$(managed_run_root "${REPO_ROOT}" "${run_id}")"
replica_root="${run_root}/replicas"
batch_root="${run_root}/batches"
aggregate_root="${run_root}/aggregates"
claims_root="${run_root}/claims"
spec_path="${run_root}/run_spec.yaml"
run_info="${run_root}/run_info.txt"
replicas_csv="${run_root}/replicas.csv"
segments_csv="${run_root}/segments.csv"
registry_file="${REPO_ROOT}/runs/diffusive_1d_pmlr/run_registry.csv"

mkdir -p "${replica_root}" "${batch_root}" "${aggregate_root}/current" "${aggregate_root}/history" "${claims_root}"

if [[ ! -f "${spec_path}" || "${force_spec_update}" == "true" ]]; then
    cat > "${spec_path}" <<EOF
schema_version: 1
run_id: "${run_id}"
simulation: "diffusive_1d_pmlr"
dim_num: 1
L: ${L}
rho0: $(managed_format_float "${rho}")
D: 1.0
T: 1.0
gamma: $(managed_format_float "${gamma}")
potential_type: "ratchet_PmLr"
fluctuation_type: "profile_switch"
potential_strength: $(managed_format_float "${potential_strength}")
forcing_rate_scheme: "symmetric_normalized"
bond_pass_count_mode: "none"
warmup_threshold_sweeps: ${warmup_threshold}
default_segment_sweeps: ${segment_sweeps}
checkpoint_interval_sweeps: ${checkpoint_interval}
target_replica_count: ${target_replicas}
aggregation_policy: "replace_current_on_demand"
aggregation_default: "stable_only"
created_or_updated_at: "$(managed_timestamp)"
EOF
fi

if [[ ! -f "${replicas_csv}" ]]; then
    echo "replica_id,phase,elapsed_sweeps,statistics_sweeps,latest_state,source_path,source_tag,claim_id,status,updated_at" > "${replicas_csv}"
fi

if [[ ! -f "${segments_csv}" ]]; then
    echo "segment_id,replica_id,batch_id,kind,requested_sweeps,start_state,output_state,checkpoint_state,status,statistics_sweeps,submitted_at,completed_at" > "${segments_csv}"
fi

cat > "${run_info}" <<EOF
run_id=${run_id}
mode=managed
simulation=diffusive_1d_pmlr
run_root=${run_root}
run_spec=${spec_path}
replicas_csv=${replicas_csv}
segments_csv=${segments_csv}
replica_root=${replica_root}
batch_root=${batch_root}
aggregate_root=${aggregate_root}
claims_root=${claims_root}
EOF

mkdir -p "$(dirname "${registry_file}")"
registry_header="timestamp,run_id,mode,L,rho0,gamma,potential_strength,n_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,config_path,save_tag_prefix,aggregate_root,cumulative_tag"
registry_row="$(printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
    "$(managed_timestamp)" "${run_id}" "managed" "${L}" "$(managed_format_float "${rho}")" "$(managed_format_float "${gamma}")" "$(managed_format_float "${potential_strength}")" \
    "${segment_sweeps}" "${target_replicas}" "1" "managed" "${run_root}" "${batch_root}" "${batch_root}" "${replica_root}" "${spec_path}" \
    "managed" "${aggregate_root}" "managed_current")"
if [[ ! -f "${registry_file}" ]]; then
    echo "${registry_header}" > "${registry_file}"
    echo "${registry_row}" >> "${registry_file}"
else
    tmp_registry="${registry_file}.tmp.$$"
    awk -F, -v OFS=',' -v rid="${run_id}" -v replacement="${registry_row}" '
        NR == 1 { print; next }
        $2 == rid {
            if (!replaced) {
                print replacement
                replaced=1
            }
            next
        }
        { print }
        END {
            if (!replaced) print replacement
        }
    ' "${registry_file}" > "${tmp_registry}"
    mv -f "${tmp_registry}" "${registry_file}"
fi

echo "Initialized managed diffusive 1D PmLr run."
echo "  run_id=${run_id}"
echo "  run_root=${run_root}"
echo "  run_spec=${spec_path}"
echo "  replicas_csv=${replicas_csv}"
