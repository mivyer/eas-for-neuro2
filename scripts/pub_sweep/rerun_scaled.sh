#!/usr/bin/env bash
# Rerun EA methods with correct dimension-aware hyperparameter scaling.
#
# AUDIT FINDINGS (2026-04-15)
# ─────────────────────────────────────────────────────────────────────────────
# All N=32 and N=64 runs used: ea_pop=64, ea_sigma_scaling=False, ea_auto_pop=False
#
# GA mut_std is hardcoded in train_ga.py as 0.3 * sqrt(32/N):
#   N=32 → 0.300  N=64 → 0.212  N=128 → 0.150   (already correct, no action needed)
#
# ES sigma scaling (--scale-sigma):
#   N=32: sigma = 0.02 / sqrt(1344/1344) = 0.02  → NO-OP, skip
#   N=64: sigma = 0.02 / sqrt(4480/1344) = 0.011 → MISSING, rerun ES only
#
# Population scaling (--scale-pop):
#   N=32: pop = max(64, 4*sqrt(1344)) = 147       → MISSING for all EA methods
#   N=64: pop = max(64, 4*sqrt(4480)) = 268       → MISSING for all EA methods
#
# N=128: both flags were already applied in original sweep → skip
# BPTT:  not affected by any of these flags → skip
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64; ITERS=500
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

mkdir -p "$RESULTS/logs"

# Helper: run one EA method, removing old result dir for that method only.
# bptt/ subdirectory is never touched.
run_ea() {
    local METHOD=$1 TASK=$2 NB=$3 NN=$4 SEED=$5 EXTRA_FLAGS=$6
    local OUT
    if [ "$TASK" = "nback" ]; then
        OUT="$RESULTS/nback${NB}_neurons${NN}_seed${SEED}"
    else
        OUT="$RESULTS/robot_T20_neurons${NN}_seed${SEED}"
    fi
    mkdir -p "$OUT"
    rm -rf "$OUT/$METHOD"
    echo "  >>> $METHOD task=$TASK nb=$NB N=$NN seed=$SEED [$EXTRA_FLAGS]"
    if [ "$TASK" = "nback" ]; then
        python3 "$SCRIPT" \
            --task nback --n-back "$NB" --neurons "$NN" --seed "$SEED" \
            --method "$METHOD" --ea-gens "$GENS" --ea-pop "$POP" \
            $EXTRA_FLAGS \
            --save --output "$OUT"
    else
        python3 "$SCRIPT" \
            --task robot --neurons "$NN" --seed "$SEED" \
            --method "$METHOD" --ea-gens "$GENS" --ea-pop "$POP" \
            $EXTRA_FLAGS \
            --save --output "$OUT"
    fi
}

# ── N=32: --scale-pop only (--scale-sigma is a no-op at N=32) ────────────────
echo ""
echo "=== N=32: scale-pop rerun (all EA methods, both tasks) ==="
for NB in $NBACKS; do
    for SEED in $SEEDS; do
        for M in es ga ga_oja; do
            run_ea "$M" nback "$NB" 32 "$SEED" "--scale-pop"
        done
    done
done
for SEED in $SEEDS; do
    for M in es ga ga_oja; do
        run_ea "$M" robot "" 32 "$SEED" "--scale-pop"
    done
done

# ── N=64: --scale-sigma + --scale-pop (ES); --scale-pop only (GA, GA+Oja) ───
echo ""
echo "=== N=64: scale-sigma + scale-pop rerun (all EA methods, both tasks) ==="
for NB in $NBACKS; do
    for SEED in $SEEDS; do
        run_ea es     nback "$NB" 64 "$SEED" "--scale-sigma --scale-pop"
        run_ea ga     nback "$NB" 64 "$SEED" "--scale-pop"
        run_ea ga_oja nback "$NB" 64 "$SEED" "--scale-pop"
    done
done
for SEED in $SEEDS; do
    run_ea es     robot "" 64 "$SEED" "--scale-sigma --scale-pop"
    run_ea ga     robot "" 64 "$SEED" "--scale-pop"
    run_ea ga_oja robot "" 64 "$SEED" "--scale-pop"
done

echo ""
echo "=== All scaled reruns complete ==="
