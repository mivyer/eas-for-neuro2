#!/usr/bin/env bash
# Rerun N=64 EA methods with dimension-aware hyperparameter scaling.
#
# Changes vs original sweep:
#   ES:      --scale-sigma (sigma 0.02→0.011) + --scale-pop (pop 64→268)
#   GA:      --scale-pop only (pop 64→268; mut_std already hardcoded scaled)
#   GA+Oja:  --scale-pop only (pop 64→268; mut_std already hardcoded scaled)
#
# bptt/ subdirs are never modified.
# N=32 and N=128 are not touched.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64; NN=64
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

mkdir -p "$RESULTS/logs"

run_method() {
    local M=$1 TASK=$2 NB=$3 SEED=$4 FLAGS=$5
    local OUT
    if [ "$TASK" = "nback" ]; then
        OUT="$RESULTS/nback${NB}_neurons${NN}_seed${SEED}"
    else
        OUT="$RESULTS/robot_T20_neurons${NN}_seed${SEED}"
    fi
    mkdir -p "$OUT"
    rm -rf "$OUT/$M"
    echo "  >>> $M $TASK nb=$NB seed=$SEED"
    if [ "$TASK" = "nback" ]; then
        python3 "$SCRIPT" \
            --task nback --n-back "$NB" --neurons "$NN" --seed "$SEED" \
            --method "$M" --ea-gens "$GENS" --ea-pop "$POP" \
            $FLAGS --save --output "$OUT"
    else
        python3 "$SCRIPT" \
            --task robot --neurons "$NN" --seed "$SEED" \
            --method "$M" --ea-gens "$GENS" --ea-pop "$POP" \
            $FLAGS --save --output "$OUT"
    fi
}

echo "========================================================"
echo "N=64 scaled rerun — EA methods only"
echo "  ES:     sigma 0.02→0.011, pop 64→268"
echo "  GA:     pop 64→268"
echo "  GA+Oja: pop 64→268"
echo "========================================================"

# N-back
for NB in $NBACKS; do
    for SEED in $SEEDS; do
        echo "--- nback${NB} N=64 seed=${SEED} ---"
        run_method es     nback "$NB" "$SEED" "--scale-sigma --scale-pop"
        run_method ga     nback "$NB" "$SEED" "--scale-pop"
        run_method ga_oja nback "$NB" "$SEED" "--scale-pop"
    done
done

# Robot arm
for SEED in $SEEDS; do
    echo "--- robot N=64 seed=${SEED} ---"
    run_method es     robot "" "$SEED" "--scale-sigma --scale-pop"
    run_method ga     robot "" "$SEED" "--scale-pop"
    run_method ga_oja robot "" "$SEED" "--scale-pop"
done

echo "========================================================"
echo "DONE. Verify settings used:"
echo "  python3 -c \\"
echo "    \"import json; d=json.load(open('results/pub/nback1_neurons64_seed42/config.json')); print(d)\""
echo ""
echo "Then rerun stats:"
echo "  python3 scripts/stats_analysis_10seed.py \\"
echo "      --results-dir results/pub/ --out results/"
echo "========================================================"
