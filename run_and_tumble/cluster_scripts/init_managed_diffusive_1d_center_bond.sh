#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash cluster_scripts/init_managed_diffusive_1d_center_bond.sh [options]

Options:
  --run_id <id>                 Stable managed run id
  --L <int>                     System size (default: 2048)
  --rho <float>                 Density (default: 100)
  --force_strength <float>      Center-bond forcing magnitude f (default: 1)
  --ffr <float>                 Center-bond flip rate (default: 1)
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
source "${SCRIPT_DIR}/managed_diffusive_1d_center_bond_common.sh"
REPO_ROOT="$(managed_repo_root "${SCRIPT_DIR}")"

default_center_bond_run_id() {
    local L="$1"
    local rho="$2"
    local force_strength="$3"
    local ffr="$4"
    local rho_slug force_slug ffr_slug
    rho_slug="$(managed_slugify "$(managed_format_float "${rho}")")"
    force_slug="$(managed_slugify "$(managed_format_float "${force_strength}")")"
    ffr_slug="$(managed_slugify "$(managed_format_float "${ffr}")")"
    printf "diffusive_1d_center_bond_L%s_rho%s_f%s_ffr%s" "${L}" "${rho_slug}" "${force_slug}" "${ffr_slug}"
}

L="2048"
rho="100"
force_strength="1"
ffr="1"
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
        --force_strength) force_strength="${2:-}"; shift 2 ;;
        --ffr) ffr="${2:-}"; shift 2 ;;
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
managed_require_nonnegative_float "force_strength" "${force_strength}"
managed_require_nonnegative_float "ffr" "${ffr}"
if (( L % 2 != 0 )); then
    echo "--L must be even so center_bond_x is unambiguous. Got L=${L}." >&2
    exit 1
fi

if [[ -z "${run_id}" ]]; then
    run_id="$(default_center_bond_run_id "${L}" "${rho}" "${force_strength}" "${ffr}")"
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
registry_file="$(managed_registry_file "${REPO_ROOT}")"

mkdir -p "${replica_root}" "${batch_root}" "${aggregate_root}/current" "${aggregate_root}/history" "${claims_root}"

rho_display="$(managed_format_float "${rho}")"
force_display="$(managed_format_float "${force_strength}")"
ffr_display="$(managed_format_float "${ffr}")"

if [[ ! -f "${spec_path}" || "${force_spec_update}" == "true" ]]; then
    cat > "${spec_path}" <<EOF
schema_version: 1
run_id: "${run_id}"
simulation: "diffusive_1d_center_bond"
dim_num: 1
L: ${L}
rho0: ${rho_display}
D: 1.0
T: 1.0
gamma: 0.0
potential_type: "zero"
fluctuation_type: "no-fluctuation"
potential_magnitude: 0.0
ic: "random"
forcing_type: "center_bond_x"
forcing_magnitude: ${force_display}
ffr: ${ffr_display}
forcing_direction_flags: true
forcing_rate_scheme: "symmetric_normalized"
bond_pass_count_mode: "all_forcing_bonds"
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
simulation=diffusive_1d_center_bond
run_root=${run_root}
run_spec=${spec_path}
replicas_csv=${replicas_csv}
segments_csv=${segments_csv}
replica_root=${replica_root}
batch_root=${batch_root}
aggregate_root=${aggregate_root}
claims_root=${claims_root}
managed_aggregate_glob=*id-aggregated_*.jld2
EOF

mkdir -p "$(dirname "${registry_file}")"
registry_header="timestamp,run_id,mode,L,rho0,forcing_magnitude,ffr,n_sweeps,num_replicas,request_cpus,request_memory,run_root,submit_dir,log_dir,state_dir,config_path,save_tag_prefix,aggregate_root,cumulative_tag"
registry_row="$(printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
    "$(managed_timestamp)" "${run_id}" "managed" "${L}" "${rho_display}" "${force_display}" "${ffr_display}" \
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

echo "Initialized managed diffusive 1D center-bond run."
echo "  run_id=${run_id}"
echo "  run_root=${run_root}"
echo "  run_spec=${spec_path}"
echo "  replicas_csv=${replicas_csv}"
