#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash run_local_lattice_rtp_wall_softcrowding_live.sh [quick|signal|strong] [extra runner args...]

Local exploratory RTP wall run with multi-occupancy soft crowding and live panels.

Presets:
  quick   Small sanity/preview run. Default.
  signal  Longer local run tuned to show a wall layer without immediately entering
          a severe MIPS regime.
  strong  Higher persistence/density run. More likely to jam or phase separate.

Override any preset value with environment variables, for example:
  MODE=signal N_SWEEPS=100000 LIVE_PANEL_INTERVAL=1000 bash run_local_lattice_rtp_wall_softcrowding_live.sh

Main environment overrides:
  L, RHO0, D, ALPHA, EPSILON, WALL_MAGNITUDE, WARMUP_SWEEPS, N_SWEEPS,
  STATISTICS_INTERVAL, LIVE_PANEL_INTERVAL, OCCUPANCY_HOP_MODEL,
  OCCUPANCY_HOP_SCOPE, OCCUPANCY_HOP_SCALE, OCCUPANCY_HOP_POWER, OUT_DIR,
  LIVE_PANEL_KEEP_HISTORY=true, LIVE_PANEL_DISPLAY=true
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
        DEFAULT_L=128
        DEFAULT_RHO0=1.2
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.02
        DEFAULT_EPSILON=1.5
        DEFAULT_WALL_MAGNITUDE=8
        DEFAULT_WARMUP_SWEEPS=1000
        DEFAULT_N_SWEEPS=8000
        DEFAULT_STATISTICS_INTERVAL=10
        DEFAULT_LIVE_PANEL_INTERVAL=500
        DEFAULT_OCCUPANCY_HOP_SCALE=6
        DEFAULT_OCCUPANCY_HOP_POWER=1
        ;;
    signal)
        DEFAULT_L=256
        DEFAULT_RHO0=1.5
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.01
        DEFAULT_EPSILON=2.0
        DEFAULT_WALL_MAGNITUDE=10
        DEFAULT_WARMUP_SWEEPS=5000
        DEFAULT_N_SWEEPS=50000
        DEFAULT_STATISTICS_INTERVAL=25
        DEFAULT_LIVE_PANEL_INTERVAL=1000
        DEFAULT_OCCUPANCY_HOP_SCALE=4
        DEFAULT_OCCUPANCY_HOP_POWER=1
        ;;
    strong)
        DEFAULT_L=256
        DEFAULT_RHO0=2.0
        DEFAULT_D=0.5
        DEFAULT_ALPHA=0.005
        DEFAULT_EPSILON=2.5
        DEFAULT_WALL_MAGNITUDE=10
        DEFAULT_WARMUP_SWEEPS=10000
        DEFAULT_N_SWEEPS=80000
        DEFAULT_STATISTICS_INTERVAL=25
        DEFAULT_LIVE_PANEL_INTERVAL=1000
        DEFAULT_OCCUPANCY_HOP_SCALE=2
        DEFAULT_OCCUPANCY_HOP_POWER=1
        ;;
    *)
        echo "Unknown preset '${MODE}'. Use quick, signal, or strong." >&2
        usage >&2
        exit 2
        ;;
esac

JULIA_BIN="${JULIA_BIN:-julia}"
RUN_ID="${RUN_ID:-local_rtp_wall_${MODE}_$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/analysis_outputs/lattice_rtp_wall/local_live/${RUN_ID}}"

L="${L:-${DEFAULT_L}}"
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
Starting local RTP wall soft-crowding run
  preset=${MODE}
  output_dir=${OUT_DIR}
  live_panel=${OUT_DIR}/live_panels/live_panel_latest.png
  final_panel=${OUT_DIR}/lattice_rtp_wall_density_correlation_panel.png

To watch the live panel from another terminal:
  xdg-open "${OUT_DIR}/live_panels/live_panel_latest.png"

EOF

exec "${JULIA_BIN}" --startup-file=no --project=. \
    utility_scripts/run_lattice_rtp_wall_panel.jl \
    --L "${L}" \
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
    --dynamics discrete \
    --occupancy_hop_model "${OCCUPANCY_HOP_MODEL}" \
    --occupancy_hop_scope "${OCCUPANCY_HOP_SCOPE}" \
    --occupancy_hop_scale "${OCCUPANCY_HOP_SCALE}" \
    --occupancy_hop_power "${OCCUPANCY_HOP_POWER}" \
    "${extra_args[@]}" \
    "$@"
