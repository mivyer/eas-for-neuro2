#!/usr/bin/env bash
# Rerun ES at N=32 with --scale-pop (pop 64→147).
# ES only — fast enough for local background execution (~3.5hrs).
# GA and GA+Oja N=32 must go to HPC — see rerun_32n_slurm.sh.
#
# SAFE MODE: skips any run where es/history.json already exists.
# To force a rerun of a specific seed, delete its es/ subdir manually first.
# bptt/ subdirs never touched.
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64; NN=32
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

mkdir -p "$RESULTS/logs"

echo "========================================================"
echo "ES N=32 scaled rerun (local background)"
echo "  --scale-pop: pop 64→147"
echo "  --scale-sigma: NOT applied (no-op at N=32, baseline=1344)"
echo "  Skipping any run where es/history.json already exists."
echo "  Estimated time for missing runs: ~250s each"
echo "========================================================"

# N-back
for NB in $NBACKS; do
    for SEED in $SEEDS; do
        OUT="$RESULTS/nback${NB}_neurons${NN}_seed${SEED}"
        if [ -f "$OUT/es/history.json" ]; then
            echo "[skip] es nback${NB} N=${NN} seed=${SEED} (exists)"
            continue
        fi
        mkdir -p "$OUT"
        echo ">>> es nback${NB} N=${NN} seed=${SEED}"
        python3 "$SCRIPT" \
            --task nback --n-back "$NB" --neurons "$NN" --seed "$SEED" \
            --method es --ea-gens "$GENS" --ea-pop "$POP" \
            --scale-pop \
            --save --output "$OUT"
    done
done

# Robot arm
for SEED in $SEEDS; do
    OUT="$RESULTS/robot_T20_neurons${NN}_seed${SEED}"
    if [ -f "$OUT/es/history.json" ]; then
        echo "[skip] es robot N=${NN} seed=${SEED} (exists)"
        continue
    fi
    mkdir -p "$OUT"
    echo ">>> es robot N=${NN} seed=${SEED}"
    python3 "$SCRIPT" \
        --task robot --neurons "$NN" --seed "$SEED" \
        --method es --ea-gens "$GENS" --ea-pop "$POP" \
        --scale-pop \
        --save --output "$OUT"
done

echo "========================================================"
echo "ES N=32 rerun complete."
echo "Submit GA and GA+Oja N=32 to sagehen:"
echo "  bash scripts/pub_sweep/rerun_32n_slurm.sh"
echo "========================================================"
