#!/usr/bin/env bash
# SLURM launcher for dimension-aware EA reruns.
# Submits 6 independent jobs, one per (method, N) pair.
# Each job handles all seeds × n-back levels + robot arm for that combination.
#
# Usage (from repo root on sagehen):
#   bash scripts/pub_sweep/rerun_scaled_slurm.sh
#
# Jobs:
#   es_32n      N=32 ES        --scale-pop                  ~4h  8G  2cpu
#   ga_32n      N=32 GA        --scale-pop                  ~4h  8G  2cpu
#   gaoja_32n   N=32 GA+Oja    --scale-pop                  ~4h  8G  2cpu
#   es_64n      N=64 ES        --scale-sigma --scale-pop    ~16h 16G 4cpu
#   ga_64n      N=64 GA        --scale-pop                  ~48h 16G 4cpu
#   gaoja_64n   N=64 GA+Oja    --scale-pop                  ~48h 16G 4cpu
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"
LOGS="$REPO/results/pub/logs"
mkdir -p "$LOGS"

# ── common job header / body emitted into each per-job script ─────────────────
common_header() {
    local JOBNAME=$1 TIME=$2 MEM=$3 CPUS=$4
    cat <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${LOGS}/${JOBNAME}_%j.out
#SBATCH --error=${LOGS}/${JOBNAME}_%j.err
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
module load miniconda3/py312_24.9.2-0
set -euo pipefail
cd "${REPO}"

SCRIPT="scripts/run_experiment.py"
RESULTS="results/pub"
GENS=500; POP=64
SEEDS="42 123 456 789 1011 1213 1415 1617 1819 2021"
NBACKS="1 2 3 4"

run_ea() {
    local METHOD=\$1 TASK=\$2 NB=\$3 NN=\$4 SEED=\$5; shift 5
    local OUT
    if [ "\$TASK" = "nback" ]; then OUT="\$RESULTS/nback\${NB}_neurons\${NN}_seed\${SEED}"
    else                            OUT="\$RESULTS/robot_T20_neurons\${NN}_seed\${SEED}"; fi
    mkdir -p "\$OUT"
    rm -rf "\$OUT/\$METHOD"
    echo "  >>> \$METHOD task=\$TASK nb=\$NB N=\$NN seed=\$SEED [\$*]"
    if [ "\$TASK" = "nback" ]; then
        python3 "\$SCRIPT" --task nback --n-back "\$NB" --neurons "\$NN" --seed "\$SEED" \
            --method "\$METHOD" --ea-gens "\$GENS" --ea-pop "\$POP" "\$@" \
            --save --output "\$OUT"
    else
        python3 "\$SCRIPT" --task robot --neurons "\$NN" --seed "\$SEED" \
            --method "\$METHOD" --ea-gens "\$GENS" --ea-pop "\$POP" "\$@" \
            --save --output "\$OUT"
    fi
}
EOF
}

# ── N=32 jobs ─────────────────────────────────────────────────────────────────

for METHOD in es ga ga_oja; do
    SAFE="${METHOD//_/}"   # ga_oja → gaoja
    JOBNAME="${SAFE}_32n"
    TMPSCRIPT="$LOGS/${JOBNAME}.sh"
    {
        common_header "$JOBNAME" "04:00:00" "8G" "2"
        cat <<EOF

echo "=== ${JOBNAME}: N=32 --scale-pop ==="
for NB in \$NBACKS; do
    for SEED in \$SEEDS; do
        run_ea ${METHOD} nback "\$NB" 32 "\$SEED" --scale-pop
    done
done
for SEED in \$SEEDS; do
    run_ea ${METHOD} robot "" 32 "\$SEED" --scale-pop
done
echo "=== ${JOBNAME} done ==="
EOF
    } > "$TMPSCRIPT"
    sbatch "$TMPSCRIPT"
    echo "Submitted: $JOBNAME  (04:00:00  8G  2cpu)"
done

# ── N=64 jobs ─────────────────────────────────────────────────────────────────

# ES: needs both --scale-sigma and --scale-pop
JOBNAME="es_64n"
TMPSCRIPT="$LOGS/${JOBNAME}.sh"
{
    common_header "$JOBNAME" "16:00:00" "16G" "4"
    cat <<EOF

echo "=== ${JOBNAME}: N=64 --scale-sigma --scale-pop ==="
for NB in \$NBACKS; do
    for SEED in \$SEEDS; do
        run_ea es nback "\$NB" 64 "\$SEED" --scale-sigma --scale-pop
    done
done
for SEED in \$SEEDS; do
    run_ea es robot "" 64 "\$SEED" --scale-sigma --scale-pop
done
echo "=== ${JOBNAME} done ==="
EOF
} > "$TMPSCRIPT"
sbatch "$TMPSCRIPT"
echo "Submitted: $JOBNAME  (16:00:00  16G  4cpu)"

# GA / GA+Oja: --scale-pop only (mut_std already hardcoded correctly)
for METHOD in ga ga_oja; do
    SAFE="${METHOD//_/}"
    JOBNAME="${SAFE}_64n"
    TMPSCRIPT="$LOGS/${JOBNAME}.sh"
    {
        common_header "$JOBNAME" "48:00:00" "16G" "4"
        cat <<EOF

echo "=== ${JOBNAME}: N=64 --scale-pop ==="
for NB in \$NBACKS; do
    for SEED in \$SEEDS; do
        run_ea ${METHOD} nback "\$NB" 64 "\$SEED" --scale-pop
    done
done
for SEED in \$SEEDS; do
    run_ea ${METHOD} robot "" 64 "\$SEED" --scale-pop
done
echo "=== ${JOBNAME} done ==="
EOF
    } > "$TMPSCRIPT"
    sbatch "$TMPSCRIPT"
    echo "Submitted: $JOBNAME  (48:00:00  16G  4cpu)"
done

echo ""
echo "All 6 jobs submitted. Monitor with: squeue -u \$USER"
