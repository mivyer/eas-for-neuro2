#!/usr/bin/env bash
# Re-run stale results that predate ES/GA 1/5-success-rule + dimension-scaling.
# Stale: nback 0-3 × 64/128n, robot T20 × 64/128/256n.
# Archives existing dirs before overwriting.
set -e

SCRIPT="scripts/run_experiment.py"
GENS=500
POP=64
ITERS=500
ARCHIVE="results/archive_pre_sigma_adapt"
LOG="results/sweep_requeue.log"

mkdir -p "$ARCHIVE"

archive_and_run_nback() {
    local NB=$1 NN=$2 SEED=$3
    local OUT="results/nback/nback${NB}_neurons${NN}_seed${SEED}"
    if [ -d "$OUT" ]; then
        echo "[archive] $OUT"
        mv "$OUT" "$ARCHIVE/"
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

archive_and_run_robot() {
    local NN=$1 SEED=$2
    local OUT="results/robot/robot_T20_neurons${NN}_seed${SEED}"
    if [ -d "$OUT" ]; then
        echo "[archive] $OUT"
        mv "$OUT" "$ARCHIVE/"
    fi
    echo ">>> robot T20  neurons=${NN}  seed=${SEED}"
    python3 "$SCRIPT" \
        --task robot \
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

echo "=== GROUP 1: nback 0-3 x 64 neurons x all seeds ==="
for NB in 0 1 2 3; do
    for SEED in 42 123 456; do
        archive_and_run_nback "$NB" 64 "$SEED"
    done
done

echo "=== GROUP 2: nback 0-3 x 128 neurons x all seeds ==="
for NB in 0 1 2 3; do
    for SEED in 42 123 456; do
        archive_and_run_nback "$NB" 128 "$SEED"
    done
done

echo "=== GROUP 3: robot T20 x 64,128,256 neurons x all seeds ==="
for NN in 64 128 256; do
    for SEED in 42 123 456; do
        archive_and_run_robot "$NN" "$SEED"
    done
done

echo "=== ALL DONE ==="
