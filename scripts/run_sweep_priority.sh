#!/usr/bin/env bash
# Priority sweep:
#   1. nback 4–6 × 64 neurons × seeds 42,123,456
#   2. nback 4–6 × 128,256 neurons × seeds 123,456 (seed 42 already done)
#   3. nback 0–3 × 256 neurons × seeds 42,123,456
# Skips runs whose output directory already exists.
set -e

SCRIPT="scripts/run_experiment.py"
GENS=500
POP=64
ITERS=500
LOG="results/sweep_priority.log"

run_one() {
    local NB=$1 NN=$2 SEED=$3
    local OUT="results/nback/nback${NB}_neurons${NN}_seed${SEED}"
    if [ -d "$OUT" ]; then
        echo "[skip] $OUT already exists"
        return
    fi
    echo ">>> nback=${NB}  neurons=${NN}  seed=${SEED}"
    python3 "$SCRIPT" \
        --task nback \
        --n-back "$NB" \
        --neurons "$NN" \
        --seed "$SEED" \
        --method all \
        --ea-gens "$GENS" \
        --ea-pop "$POP" \
        --bptt-iters "$ITERS" \
        --save \
        --output "$OUT"
    echo "<<< done $OUT"
}

echo "=== GROUP 1: nback 4-6 x 64 neurons x all seeds ==="
for NB in 4 5 6; do
    for SEED in 42 123 456; do
        run_one "$NB" 64 "$SEED"
    done
done

echo "=== GROUP 2: nback 4-6 x 128,256 neurons x seeds 123,456 ==="
for NB in 4 5 6; do
    for NN in 128 256; do
        for SEED in 123 456; do
            run_one "$NB" "$NN" "$SEED"
        done
    done
done

echo "=== GROUP 3: nback 0-3 x 256 neurons x all seeds ==="
for NB in 0 1 2 3; do
    for SEED in 42 123 456; do
        run_one "$NB" 256 "$SEED"
    done
done

echo "=== ALL DONE ==="
