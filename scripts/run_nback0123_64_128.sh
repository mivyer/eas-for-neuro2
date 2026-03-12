#!/usr/bin/env bash
# Run n-back 0,1,2,3 × neurons 64,128 × seeds 42,123,456
# Skips runs whose output directory already exists.
set -e

SCRIPT="scripts/run_experiment.py"
GENS=500
POP=64
ITERS=500

run_one() {
    local NB=$1 NN=$2 SEED=$3
    local OUT="results/nback${NB}_neurons${NN}_seed${SEED}"
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

for NB in 0 1 2 3; do
    for NN in 64 128; do
        for SEED in 42 123 456; do
            run_one "$NB" "$NN" "$SEED"
        done
    done
done

echo "=== ALL DONE ==="
