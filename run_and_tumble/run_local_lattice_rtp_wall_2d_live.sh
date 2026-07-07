#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_local_lattice_rtp_wall_2d_live.sh [quick|signal|strong] [extra runner args...]

Local 2D lattice RTP wall run with multi-occupancy soft crowding and live panels.
The diagnostic measures density accumulation near a vertical wall and the
connected density covariance along the wall, subtracting a far-from-wall row.

Presets:
  quick   Fast local smoke/preview.
  signal  Recommended local run for seeing the wall-induced along-wall signal.
  strong  Larger persistence and density; more likely to show MIPS/jamming.

Main environment overrides:
  LX, LY, RHO0, D, ALPHA, EPSILON, WALL_MAGNITUDE, WARMUP_SWEEPS, N_SWEEPS,
  STATISTICS_INTERVAL, LIVE_PANEL_INTERVAL, CORR_WALL_DISTANCE,
  OCCUPANCY_HOP_MODEL, OCCUPANCY_HOP_SCOPE, OCCUPANCY_HOP_SCALE,
  OCCUPANCY_HOP_POWER, OUT_DIR, LIVE_PANEL_KEEP_HISTORY=true,
  LIVE_PANEL_DISPLAY=true
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

MODE="${MODE:-quick}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE="$1"
    shift
fi

case "${MODE}" in
    quick)
        DEFAULT_LX=64
        DEFAULT_LY=96
        DEFAULT_RHO0=0.7
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.03
        DEFAULT_EPSILON=1.5
        DEFAULT_WALL_MAGNITUDE=8
        DEFAULT_WARMUP_SWEEPS=500
        DEFAULT_N_SWEEPS=4000
        DEFAULT_STATISTICS_INTERVAL=10
        DEFAULT_LIVE_PANEL_INTERVAL=500
        DEFAULT_OCCUPANCY_HOP_SCALE=6
        DEFAULT_OCCUPANCY_HOP_POWER=1
        DEFAULT_CORR_WALL_DISTANCE=1
        ;;
    signal)
        DEFAULT_LX=128
        DEFAULT_LY=192
        DEFAULT_RHO0=0.8
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.015
        DEFAULT_EPSILON=1.8
        DEFAULT_WALL_MAGNITUDE=9
        DEFAULT_WARMUP_SWEEPS=3000
        DEFAULT_N_SWEEPS=30000
        DEFAULT_STATISTICS_INTERVAL=20
        DEFAULT_LIVE_PANEL_INTERVAL=1000
        DEFAULT_OCCUPANCY_HOP_SCALE=5
        DEFAULT_OCCUPANCY_HOP_POWER=1
        DEFAULT_CORR_WALL_DISTANCE=1
        ;;
    strong)
        DEFAULT_LX=160
        DEFAULT_LY=256
        DEFAULT_RHO0=1.1
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.01
        DEFAULT_EPSILON=2.0
        DEFAULT_WALL_MAGNITUDE=10
        DEFAULT_WARMUP_SWEEPS=5000
        DEFAULT_N_SWEEPS=50000
        DEFAULT_STATISTICS_INTERVAL=25
        DEFAULT_LIVE_PANEL_INTERVAL=1000
        DEFAULT_OCCUPANCY_HOP_SCALE=3
        DEFAULT_OCCUPANCY_HOP_POWER=1
        DEFAULT_CORR_WALL_DISTANCE=1
        ;;
    *)
        echo "Unknown preset '${MODE}'. Use quick, signal, or strong." >&2
        usage >&2
        exit 2
        ;;
esac

JULIA_BIN="${JULIA_BIN:-julia}"
RUN_ID="${RUN_ID:-local_rtp_wall_2d_${MODE}_$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/analysis_outputs/lattice_rtp_wall_2d/local_live/${RUN_ID}}"

LX="${LX:-${DEFAULT_LX}}"
LY="${LY:-${DEFAULT_LY}}"
RHO0="${RHO0:-${DEFAULT_RHO0}}"
D="${D:-${DEFAULT_D}}"
ALPHA="${ALPHA:-${DEFAULT_ALPHA}}"
EPSILON="${EPSILON:-${DEFAULT_EPSILON}}"
WALL_MAGNITUDE="${WALL_MAGNITUDE:-${DEFAULT_WALL_MAGNITUDE}}"
WARMUP_SWEEPS="${WARMUP_SWEEPS:-${DEFAULT_WARMUP_SWEEPS}}"
N_SWEEPS="${N_SWEEPS:-${DEFAULT_N_SWEEPS}}"
STATISTICS_INTERVAL="${STATISTICS_INTERVAL:-${DEFAULT_STATISTICS_INTERVAL}}"
LIVE_PANEL_INTERVAL="${LIVE_PANEL_INTERVAL:-${DEFAULT_LIVE_PANEL_INTERVAL}}"
OCCUPANCY_HOP_MODEL="${OCCUPANCY_HOP_MODEL:-inverse}"
OCCUPANCY_HOP_SCOPE="${OCCUPANCY_HOP_SCOPE:-source}"
OCCUPANCY_HOP_SCALE="${OCCUPANCY_HOP_SCALE:-${DEFAULT_OCCUPANCY_HOP_SCALE}}"
OCCUPANCY_HOP_POWER="${OCCUPANCY_HOP_POWER:-${DEFAULT_OCCUPANCY_HOP_POWER}}"
CORR_WALL_DISTANCE="${CORR_WALL_DISTANCE:-${DEFAULT_CORR_WALL_DISTANCE}}"
LIVE_PANEL_KEEP_HISTORY="${LIVE_PANEL_KEEP_HISTORY:-false}"
LIVE_PANEL_DISPLAY="${LIVE_PANEL_DISPLAY:-false}"

mkdir -p "${OUT_DIR}"

extra_args=()
if [[ "${LIVE_PANEL_KEEP_HISTORY}" == "true" ]]; then
    extra_args+=(--live_panel_keep_history)
fi
if [[ "${LIVE_PANEL_DISPLAY}" == "true" ]]; then
    extra_args+=(--live_panel_display)
fi

cat <<EOF
Starting local 2D RTP wall run
  preset=${MODE}
  output_dir=${OUT_DIR}
  live_panel=${OUT_DIR}/live_panels/live_panel_latest.png
  final_panel=${OUT_DIR}/lattice_rtp_wall_2d_panel.png
  target_scaling=along-wall excess covariance compared with r^-3

To watch the live panel from another terminal:
  xdg-open "${OUT_DIR}/live_panels/live_panel_latest.png"

EOF

exec "${JULIA_BIN}" --startup-file=no --project=. \
    utility_scripts/run_lattice_rtp_wall_2d_panel.jl \
    --Lx "${LX}" \
    --Ly "${LY}" \
    --rho0 "${RHO0}" \
    --D "${D}" \
    --alpha "${ALPHA}" \
    --epsilon "${EPSILON}" \
    --T 1.0 \
    --wall_magnitude "${WALL_MAGNITUDE}" \
    --warmup_sweeps "${WARMUP_SWEEPS}" \
    --n_sweeps "${N_SWEEPS}" \
    --statistics_interval "${STATISTICS_INTERVAL}" \
    --live_panel_interval "${LIVE_PANEL_INTERVAL}" \
    --save_dir "${OUT_DIR}" \
    --corr_wall_distance "${CORR_WALL_DISTANCE}" \
    --occupancy_hop_model "${OCCUPANCY_HOP_MODEL}" \
    --occupancy_hop_scope "${OCCUPANCY_HOP_SCOPE}" \
    --occupancy_hop_scale "${OCCUPANCY_HOP_SCALE}" \
    --occupancy_hop_power "${OCCUPANCY_HOP_POWER}" \
    "${extra_args[@]}" \
    "$@"
