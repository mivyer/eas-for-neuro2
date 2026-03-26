#!/usr/bin/env bash
# Run n-back 0-6 × 128 neurons × seeds 42,123,456 with dimension scaling.
# Uses --scale-sigma and --scale-pop (recommended for 128n+).
# Skips runs whose output directory already exists.
set -e

SCRIPT="scripts/run_experiment.py"
GENS=500
POP=64
ITERS=500

run_one() {
    local NB=$1 SEED=$2
    local OUT="results/nback/nback${NB}_neurons128_seed${SEED}"
    if [ -d "$OUT" ]; then
        echo "[skip] $OUT already exists"
        return
    fi
    echo ">>> nback=${NB}  neurons=128  seed=${SEED}"
    python3 "$SCRIPT" \
        --task nback \
        --n-back "$NB" \
        --neurons 128 \
        --seed "$SEED" \
        --method all \
        --ea-gens "$GENS" \
        --ea-pop "$POP" \
        --bptt-iters "$ITERS" \
        --scale-sigma \
        --scale-pop \
        --save \
        --output "$OUT"
    echo "<<< done $OUT"
}

echo "=== 128n nback scaled: nback 0-6 x 128 neurons x seeds 42,123,456 ==="
for NB in 0 1 2 3 4 5 6; do
    for SEED in 42 123 456; do
        run_one "$NB" "$SEED"
    done
done

echo "=== ALL DONE ==="
