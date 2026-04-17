#!/usr/bin/env bash
# Submit 3 parallel SLURM jobs for N=64 scaled rerun, one per EA method.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"
RESULTS="$ROOT/results/pub"
mkdir -p "$RESULTS/logs"

submit_job() {
    local M=$1 FLAGS=$2 TIME=$3 MEM=$4
    local JOBNAME="pub64_${M}_scaled"
    local JOBSCRIPT="$ROOT/scripts/pub_sweep/_rerun_64n_${M}.sh"

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
GENS=500; POP=64; NN=64
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

run_one() {
    local TASK=\$1 NB=\$2 SEED=\$3
    local OUT
    if [ "\$TASK" = "nback" ]; then
        OUT="\$RESULTS/nback\${NB}_neurons\${NN}_seed\${SEED}"
    else
        OUT="\$RESULTS/robot_T20_neurons\${NN}_seed\${SEED}"
    fi
    mkdir -p "\$OUT"
    rm -rf "\$OUT/${M}"
    if [ "\$TASK" = "nback" ]; then
        python3 "\$SCRIPT" \
            --task nback --n-back "\$NB" --neurons "\$NN" --seed "\$SEED" \
            --method ${M} --ea-gens "\$GENS" --ea-pop "\$POP" \
            ${FLAGS} --save --output "\$OUT"
    else
        python3 "\$SCRIPT" \
            --task robot --neurons "\$NN" --seed "\$SEED" \
            --method ${M} --ea-gens "\$GENS" --ea-pop "\$POP" \
            ${FLAGS} --save --output "\$OUT"
    fi
}

echo "=== ${M} N=64 scaled rerun (flags: ${FLAGS}) ==="
for NB in \$NBACKS; do
    for SEED in \$SEEDS; do
        echo "nback\${NB} seed\${SEED}"
        run_one nback "\$NB" "\$SEED"
    done
done
for SEED in \$SEEDS; do
    echo "robot seed\${SEED}"
    run_one robot "" "\$SEED"
done
echo "=== ${M} N=64 done ==="
SLURM

    chmod +x "$JOBSCRIPT"
    JID=$(sbatch --parsable "$JOBSCRIPT")
    echo "Submitted ${JOBNAME}: job ${JID}  (time=${TIME} mem=${MEM})"
}

echo "Submitting 3 parallel jobs for N=64 scaled rerun..."
submit_job es     "--scale-sigma --scale-pop"  "16:00:00" "16G"
submit_job ga     "--scale-pop"                "48:00:00" "32G"
submit_job ga_oja "--scale-pop"                "48:00:00" "32G"

echo ""
echo "Monitor: squeue -u \$USER"
echo "After all complete:"
echo "  python3 scripts/stats_analysis_10seed.py \\"
echo "      --results-dir results/pub/ --out results/"
