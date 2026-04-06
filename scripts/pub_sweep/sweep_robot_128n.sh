#!/usr/bin/env bash
#SBATCH --job-name=pub_robot_128n
#SBATCH --output=results/pub/logs/robot_128n_%j.out
#SBATCH --error=results/pub/logs/robot_128n_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64; ITERS=500; NN=128
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"

mkdir -p "$RESULTS"

for SEED in $SEEDS; do
    OUT="$RESULTS/robot_T20_neurons${NN}_seed${SEED}"
    [ -d "$OUT" ] && echo "[skip] $OUT" && continue
    echo ">>> robot neurons=${NN} seed=${SEED}"
    python3 "$SCRIPT" \
        --task robot --neurons "$NN" --seed "$SEED" \
        --method all --ea-gens "$GENS" --ea-pop "$POP" --bptt-iters "$ITERS" \
        --scale-sigma --scale-pop \
        --save --output "$OUT"
done
echo "=== robot 128n done ==="
