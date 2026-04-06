#!/usr/bin/env bash
#SBATCH --job-name=pub_nback_64n
#SBATCH --output=results/pub/logs/nback_64n_%j.out
#SBATCH --error=results/pub/logs/nback_64n_%j.err
#SBATCH --time=16:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64; ITERS=500; NN=64
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

mkdir -p "$RESULTS"

for NB in $NBACKS; do
    for SEED in $SEEDS; do
        OUT="$RESULTS/nback${NB}_neurons${NN}_seed${SEED}"
        [ -d "$OUT" ] && echo "[skip] $OUT" && continue
        echo ">>> nback=${NB} neurons=${NN} seed=${SEED}"
        python3 "$SCRIPT" \
            --task nback --n-back "$NB" --neurons "$NN" --seed "$SEED" \
            --method all --ea-gens "$GENS" --ea-pop "$POP" --bptt-iters "$ITERS" \
            --save --output "$OUT"
    done
done
echo "=== nback 64n done ==="
