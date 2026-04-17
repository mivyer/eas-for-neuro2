#!/usr/bin/env bash
# Submit GA and GA+Oja N=32 scaled reruns to sagehen HPC.
# ES N=32 runs locally — see rerun_32n_es_local.sh.
#
# SAFE MODE: each job skips runs where {method}/history.json already exists.
# bptt/ subdirs never touched.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"
RESULTS="$ROOT/results/pub"
mkdir -p "$RESULTS/logs"

submit_job() {
    local M=$1 TIME=$2 MEM=$3
    local JOBNAME="pub32_${M}_scaled"
    local JOBSCRIPT="$ROOT/scripts/pub_sweep/_rerun_32n_${M}.sh"

    cat > "$JOBSCRIPT" << SLURM
#!/usr/bin/env bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${RESULTS}/logs/${JOBNAME}_%j.out
#SBATCH --error=${RESULTS}/logs/${JOBNAME}_%j.err
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=4
set -euo pipefail

if command -v module &>/dev/null; then
    module load miniconda3/py312_24.9.2-0 2>/dev/null || true
fi

cd "${ROOT}"

SCRIPT="scripts/run_experiment.py"
RESULTS="${RESULTS}"
GENS=500; POP=64; NN=32
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

echo "=== ${M} N=32 scaled rerun: --scale-pop (pop 64→147) ==="
echo "=== --scale-sigma NOT applied (no-op at N=32) ==="
echo "=== Skipping runs where ${M}/history.json already exists ==="

# N-back
for NB in \$NBACKS; do
    for SEED in \$SEEDS; do
        OUT="\$RESULTS/nback\${NB}_neurons\${NN}_seed\${SEED}"
        if [ -f "\$OUT/${M}/history.json" ]; then
            echo "[skip] ${M} nback\${NB} N=\${NN} seed=\${SEED} (exists)"
            continue
        fi
        mkdir -p "\$OUT"
        echo ">>> ${M} nback\${NB} N=\${NN} seed=\${SEED}"
        python3 "\$SCRIPT" \
            --task nback --n-back "\$NB" --neurons "\$NN" --seed "\$SEED" \
            --method ${M} --ea-gens "\$GENS" --ea-pop "\$POP" \
            --scale-pop \
            --save --output "\$OUT"
    done
done

# Robot arm
for SEED in \$SEEDS; do
    OUT="\$RESULTS/robot_T20_neurons\${NN}_seed\${SEED}"
    if [ -f "\$OUT/${M}/history.json" ]; then
        echo "[skip] ${M} robot N=\${NN} seed=\${SEED} (exists)"
        continue
    fi
    mkdir -p "\$OUT"
    echo ">>> ${M} robot N=\${NN} seed=\${SEED}"
    python3 "\$SCRIPT" \
        --task robot --neurons "\$NN" --seed "\$SEED" \
        --method ${M} --ea-gens "\$GENS" --ea-pop "\$POP" \
        --scale-pop \
        --save --output "\$OUT"
done

echo "=== ${M} N=32 done ==="
SLURM

    chmod +x "$JOBSCRIPT"
    JID=$(sbatch --parsable "$JOBSCRIPT")
    echo "Submitted ${JOBNAME}: job ${JID}  (time=${TIME} mem=${MEM})"
}

echo "Submitting GA and GA+Oja N=32 to sagehen..."
submit_job ga     "48:00:00" "16G"
submit_job ga_oja "48:00:00" "16G"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Run ES locally in parallel:"
echo "  nohup bash scripts/pub_sweep/rerun_32n_es_local.sh \\"
echo "      > results/pub/logs/es_32n_local.log 2>&1 &"
echo "  echo \$! > results/pub/logs/es_32n_local.pid"
