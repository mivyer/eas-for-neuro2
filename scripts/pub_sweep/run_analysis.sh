#!/usr/bin/env bash
#SBATCH --job-name=pub_analysis
#SBATCH --output=results/pub/logs/analysis_%j.out
#SBATCH --error=results/pub/logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
# ============================================================
# Post-sweep analysis: cross-seed figures, summary JSON, stats
# Run after all experiment jobs have completed.
# ============================================================
set -euo pipefail

if command -v module &>/dev/null; then
    module load miniconda3/py312_24.9.2-0 2>/dev/null || true
fi

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/eas-for-neuro-main 2")"

RESULTS="results/pub"

echo "============================================================"
echo "POST-SWEEP ANALYSIS"
echo "Results dir: $RESULTS"
echo "Start: $(date)"
echo "============================================================"

# 1. Check completeness
echo ""
echo "--- Checking experiment completeness ---"
EXPECTED_NBACK=$((4 * 3 * 10))   # 4 n-back × 3 sizes × 10 seeds
EXPECTED_ROBOT=$((3 * 10))        # 3 sizes × 10 seeds
FOUND_NBACK=$(find "$RESULTS" -maxdepth 1 -name "nback*" -type d | wc -l)
FOUND_ROBOT=$(find "$RESULTS" -maxdepth 1 -name "robot*" -type d | wc -l)
echo "  N-back: $FOUND_NBACK / $EXPECTED_NBACK"
echo "  Robot:  $FOUND_ROBOT / $EXPECTED_ROBOT"

# Count completed (have all 4 method subdirs with history.json)
COMPLETE=0
INCOMPLETE=""
for DIR in "$RESULTS"/nback* "$RESULTS"/robot*; do
    [ -d "$DIR" ] || continue
    METHODS_DONE=0
    for M in bptt es ga ga_oja; do
        [ -f "$DIR/$M/history.json" ] && METHODS_DONE=$((METHODS_DONE + 1))
    done
    if [ "$METHODS_DONE" -eq 4 ]; then
        COMPLETE=$((COMPLETE + 1))
    else
        INCOMPLETE="$INCOMPLETE\n  $DIR ($METHODS_DONE/4 methods)"
    fi
done
echo "  Complete (4/4 methods): $COMPLETE"
if [ -n "$INCOMPLETE" ]; then
    echo "  INCOMPLETE:"
    echo -e "$INCOMPLETE"
fi

# 2. Cross-seed analysis (n-back)
echo ""
echo "--- Cross-seed analysis (n-back) ---"
python3 scripts/analyze_cross_seed.py \
    --results-dir "$RESULTS/" \
    --out-dir "$RESULTS/cross_seed_analysis/"

# 3. Robot arm analysis
echo ""
echo "--- Robot arm analysis ---"
python3 scripts/analyze_robot_t20.py \
    --results-dir "$RESULTS/" \
    --out-dir "$RESULTS/robot_analysis/"

# 4. Summary JSON
echo ""
echo "--- Summary JSON ---"
python3 scripts/make_summary.py \
    --results-dir "$RESULTS/" \
    --out "$RESULTS/results_summary"

# 5. Statistical tests (if script exists)
echo ""
echo "--- Statistical tests ---"
if [ -f "scripts/statistical_tests.py" ]; then
    python3 scripts/statistical_tests.py \
        --results-dir "$RESULTS/" \
        --neurons 32 64 128 \
        --seeds 42 123 456 789 1011 1213 1415 1617 1819 2021 \
        > "$RESULTS/statistical_tests_10seeds.txt" 2>&1
    echo "  Saved: $RESULTS/statistical_tests_10seeds.txt"
else
    echo "  scripts/statistical_tests.py not found — skipping"
fi

# 6. CNS composite figure
echo ""
echo "--- CNS composite figure ---"
if [ -f "scripts/make_cns_figure.py" ]; then
    python3 scripts/make_cns_figure.py \
        --results-dir "$RESULTS/" \
        --out "$RESULTS/cns_figure_10seeds.png"
    echo "  Saved: $RESULTS/cns_figure_10seeds.png"
fi

echo ""
echo "============================================================"
echo "ANALYSIS COMPLETE"
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Key outputs:"
echo "  $RESULTS/cross_seed_analysis/   — n-back figures"
echo "  $RESULTS/robot_analysis/        — robot arm figures"
echo "  $RESULTS/results_summary.json   — compact data for thesis"
echo "  $RESULTS/statistical_tests_10seeds.txt"
