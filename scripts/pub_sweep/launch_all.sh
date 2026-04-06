#!/usr/bin/env bash
# ============================================================
# Publication sweep — master launcher
#
# 10 seeds × (4 n-back levels × 3 neuron sizes + 3 robot sizes)
# = 150 experiment dirs, each running all 4 methods = 600 training runs
#
# Usage:
#   LOCAL (sequential):    bash scripts/pub_sweep/launch_all.sh local
#   LOCAL (parallel):      bash scripts/pub_sweep/launch_all.sh local-parallel
#   SLURM (HPC):           bash scripts/pub_sweep/launch_all.sh slurm
# ============================================================
set -euo pipefail

MODE="${1:-local}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "PUBLICATION SWEEP — 10 seeds, 150 experiments, 600 runs"
echo "Mode: $MODE"
echo "============================================================"

case "$MODE" in

  slurm)
    echo "Submitting SLURM jobs..."
    jid1=$(sbatch --parsable "$DIR/sweep_nback_32n.sh")
    jid2=$(sbatch --parsable "$DIR/sweep_robot_32n.sh")
    jid3=$(sbatch --parsable "$DIR/sweep_nback_64n.sh")
    jid4=$(sbatch --parsable "$DIR/sweep_robot_64n.sh")
    jid5=$(sbatch --parsable "$DIR/sweep_nback_128n.sh")
    jid6=$(sbatch --parsable "$DIR/sweep_robot_128n.sh")
    echo "  32n nback=$jid1  robot=$jid2"
    echo "  64n nback=$jid3  robot=$jid4"
    echo "  128n nback=$jid5  robot=$jid6"

    sbatch --dependency=afterok:${jid1}:${jid2}:${jid3}:${jid4}:${jid5}:${jid6} \
           "$DIR/run_analysis.sh"
    echo "Analysis job queued after all experiments."
    echo "Monitor: squeue -u \$USER"
    ;;

  local-parallel)
    echo "Launching 6 background processes..."
    mkdir -p results/pub/logs
    nohup bash "$DIR/sweep_nback_32n.sh"  > results/pub/logs/nback_32n.log  2>&1 &
    nohup bash "$DIR/sweep_robot_32n.sh"  > results/pub/logs/robot_32n.log  2>&1 &
    nohup bash "$DIR/sweep_nback_64n.sh"  > results/pub/logs/nback_64n.log  2>&1 &
    nohup bash "$DIR/sweep_robot_64n.sh"  > results/pub/logs/robot_64n.log  2>&1 &
    nohup bash "$DIR/sweep_nback_128n.sh" > results/pub/logs/nback_128n.log 2>&1 &
    nohup bash "$DIR/sweep_robot_128n.sh" > results/pub/logs/robot_128n.log 2>&1 &
    echo "All launched. Tail: tail -f results/pub/logs/nback_32n.log"
    echo "When all done: bash $DIR/run_analysis.sh"
    ;;

  local)
    echo "Running sequentially..."
    bash "$DIR/sweep_nback_32n.sh"
    bash "$DIR/sweep_robot_32n.sh"
    bash "$DIR/sweep_nback_64n.sh"
    bash "$DIR/sweep_robot_64n.sh"
    bash "$DIR/sweep_nback_128n.sh"
    bash "$DIR/sweep_robot_128n.sh"
    bash "$DIR/run_analysis.sh"
    ;;

  *)
    echo "Usage: $0 {local|local-parallel|slurm}"; exit 1 ;;
esac
