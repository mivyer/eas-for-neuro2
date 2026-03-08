#!/bin/bash
# ============================================================
# Full thesis experiment sweep
# 4 methods × 4 n-back × 3 seeds × 3 network sizes = 144 runs
# 
# Usage:
#   chmod +x scripts/run_full_sweep.sh
#   nohup bash scripts/run_full_sweep.sh > sweep_log.txt 2>&1 &
#
# Or submit individual blocks to HPC as needed.
# ============================================================

set -e

EA_GENS=500
BPTT_ITERS=2000
SEEDS="42 123 456"
# NBACKS="1 2 3 4"
NEURONS="32 64 128"


#do for nb in $NBACKS; 
TOTAL=0
for n in $NEURONS; do for s in $SEEDS; do TOTAL=$((TOTAL+1)); done; done
echo "============================================================"
echo "FULL THESIS SWEEP: $TOTAL experiments"
echo "Neurons: $NEURONS"
echo "Task: robot"
# echo "N-back: $NBACKS"
echo "Seeds: $SEEDS"
echo "EA gens: $EA_GENS | BPTT iters: $BPTT_ITERS"
echo "============================================================"
echo ""

COUNT=0
for NEUR in $NEURONS; do
    echo "============================================================"
    echo "=== STARTING $NEUR NEURON BLOCK ==="
    echo "============================================================"
    
    # for NB in $NBACKS; do
        for SEED in $SEEDS; do
            COUNT=$((COUNT+1))
            echo ""
            echo "[$COUNT/$TOTAL] neurons=$NEUR seed=$SEED"
            echo "------------------------------------------------------------"
            
            python3 scripts/run_experiment.py \
                --method all \
                --task robot \
                --neurons $NEUR \
                --ea-gens $EA_GENS \
                --bptt-iters $BPTT_ITERS \
                --save \
                --seed $SEED
            
            echo "[$COUNT/$TOTAL] DONE"
        done
    done
    
    echo ""
    echo "=== COMPLETED $NEUR NEURON BLOCK ==="
    echo ""
done

echo "============================================================"
echo "ALL $TOTAL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Run analysis:"
echo "  python3 scripts/analyze_cross_seed.py --results-dir results/ --out-dir results/cross_seed_analysis/"
