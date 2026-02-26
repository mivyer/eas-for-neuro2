# sweep_ea_hparams.py
"""
EA Hyperparameter Sweep on 1-back

Tests combinations of sigma, pop_size, lr, and generations
to determine whether any EA configuration can solve 1-back recall.

If the best config still can't beat ~40%, the EA's failure is fundamental.
If it cracks 1-back, run the winner on 2-back to find the real ceiling.
"""

import numpy as np
import time
import os
import json
import itertools

from run_full_analysis import Config, train_ea
from envs.letter_nback import LetterNBackTask, SYMBOL_VALUES, SYMBOL_LABELS, N_SYMBOLS
from models.rsnn_policy import RSNNPolicy

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ============================================================================
# Sweep config
# ============================================================================

SWEEP_GRID = {
    'ea_sigma':      [0.01, 0.05, 0.1, 0.2],
    'ea_pop_size':   [128, 256, 512],
    'ea_lr':         [0.01, 0.03, 0.1],
    'ea_generations': [300, 500],
}

# For a quick first pass, use a smaller grid:
SWEEP_GRID_QUICK = {
    'ea_sigma':      [0.02, 0.05, 0.1, 0.2],
    'ea_pop_size':   [128, 256],
    'ea_lr':         [0.01, 0.03, 0.1],
    'ea_generations': [300],
}


def run_sweep(n_back=1, n_neurons=32, seed=42, output_dir="results/ea_sweep",
              quick=False):
    """Run the hyperparameter sweep."""

    grid = SWEEP_GRID_QUICK if quick else SWEEP_GRID
    os.makedirs(output_dir, exist_ok=True)

    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    n_combos = len(combos)

    print("=" * 65)
    print(f"EA HYPERPARAMETER SWEEP | {n_back}-back | {n_neurons}N")
    print(f"Grid: {' x '.join(f'{k}={len(v)}' for k, v in grid.items())} = {n_combos} configs")
    print("=" * 65)

    results = []

    for idx, combo in enumerate(combos):
        hparams = dict(zip(keys, combo))

        print(f"\n[{idx+1}/{n_combos}] sigma={hparams['ea_sigma']} "
              f"pop={hparams['ea_pop_size']} lr={hparams['ea_lr']} "
              f"gens={hparams['ea_generations']}")

        conf = Config(
            task="nback",
            n_neurons=n_neurons,
            n_back=n_back,
            seq_length=20,
            seed=seed,
            output_dir=os.path.join(output_dir, f"run_{idx}"),
            print_every=999,  # suppress per-gen output
            **hparams,
        )

        t0 = time.time()
        ea = train_ea(conf)
        elapsed = time.time() - t0

        # Test on 100 trials
        task = LetterNBackTask(n_back=n_back, seq_length=conf.seq_length)
        rng = np.random.default_rng(seed + 999)
        policy = RSNNPolicy(ea['W_rec_final'], ea['W_in_final'], ea['W_out_final'])
        test = task.evaluate_policy(policy, n_trials=100, rng=rng)

        entry = {
            **hparams,
            'test_accuracy': test['accuracy'],
            'test_fitness': test['fitness'],
            'best_train_fitness': ea['best_fitness'],
            'final_train_accuracy': ea['history']['accuracy'][-1],
            'time': elapsed,
        }
        results.append(entry)

        print(f"  → acc={test['accuracy']:.1%}  fit={test['fitness']:+.4f}  "
              f"time={elapsed:.0f}s")

    # Sort by test accuracy
    results.sort(key=lambda r: r['test_accuracy'], reverse=True)

    # Save
    with open(os.path.join(output_dir, 'sweep_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================================
# Analysis & Plotting
# ============================================================================

def print_results_table(results, top_n=15):
    print(f"\n{'='*80}")
    print(f"{'Rank':>4} | {'sigma':>6} {'pop':>5} {'lr':>6} {'gens':>5} | "
          f"{'Test Acc':>8} {'Test Fit':>9} {'Time':>6}")
    print("-" * 80)
    for i, r in enumerate(results[:top_n]):
        print(f"{i+1:>4} | {r['ea_sigma']:>6.3f} {r['ea_pop_size']:>5d} "
              f"{r['ea_lr']:>6.3f} {r['ea_generations']:>5d} | "
              f"{r['test_accuracy']:>7.1%} {r['test_fitness']:>+9.4f} "
              f"{r['time']:>5.0f}s")
    print("=" * 80)

    best = results[0]
    worst = results[-1]
    print(f"\nBest:  acc={best['test_accuracy']:.1%}  "
          f"(sigma={best['ea_sigma']}, pop={best['ea_pop_size']}, "
          f"lr={best['ea_lr']}, gens={best['ea_generations']})")
    print(f"Worst: acc={worst['test_accuracy']:.1%}")
    print(f"Chance: {1/N_SYMBOLS:.0%}")

    if best['test_accuracy'] > 0.5:
        print("\n★ EA CAN solve this level — run the best config on higher n-back")
    elif best['test_accuracy'] > 0.35:
        print("\n~ EA shows some learning — might need more compute")
    else:
        print("\n✗ EA cannot solve even this level — the failure is fundamental")


def plot_results(results, output_dir="results/ea_sweep"):
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Accuracy by sigma ---
    ax = axes[0, 0]
    for sigma in sorted(set(r['ea_sigma'] for r in results)):
        subset = [r for r in results if r['ea_sigma'] == sigma]
        accs = [r['test_accuracy'] for r in subset]
        ax.scatter([sigma] * len(accs), accs, s=40, alpha=0.6)
    ax.axhline(1/N_SYMBOLS, color='gray', ls='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Sigma (noise scale)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Sigma', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Accuracy by pop size ---
    ax = axes[0, 1]
    for pop in sorted(set(r['ea_pop_size'] for r in results)):
        subset = [r for r in results if r['ea_pop_size'] == pop]
        accs = [r['test_accuracy'] for r in subset]
        ax.scatter([pop] * len(accs), accs, s=40, alpha=0.6)
    ax.axhline(1/N_SYMBOLS, color='gray', ls='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Population Size')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Pop Size', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Accuracy by lr ---
    ax = axes[1, 0]
    for lr in sorted(set(r['ea_lr'] for r in results)):
        subset = [r for r in results if r['ea_lr'] == lr]
        accs = [r['test_accuracy'] for r in subset]
        ax.scatter([lr] * len(accs), accs, s=40, alpha=0.6)
    ax.axhline(1/N_SYMBOLS, color='gray', ls='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs LR', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # --- Accuracy vs compute (pop * gens) ---
    ax = axes[1, 1]
    compute = [r['ea_pop_size'] * r['ea_generations'] for r in results]
    accs = [r['test_accuracy'] for r in results]
    ax.scatter(compute, accs, s=40, alpha=0.6, c=[r['ea_sigma'] for r in results],
               cmap='viridis')
    ax.axhline(1/N_SYMBOLS, color='gray', ls='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Total Evaluations (pop × gens)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Accuracy vs Compute', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle('EA Hyperparameter Sweep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_hparams.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # --- Heatmaps: sigma x lr for each pop size ---
    sigmas = sorted(set(r['ea_sigma'] for r in results))
    lrs = sorted(set(r['ea_lr'] for r in results))
    pops = sorted(set(r['ea_pop_size'] for r in results))

    fig, axes = plt.subplots(1, len(pops), figsize=(5 * len(pops), 4))
    if len(pops) == 1:
        axes = [axes]

    for ax, pop in zip(axes, pops):
        grid = np.full((len(sigmas), len(lrs)), np.nan)
        for r in results:
            if r['ea_pop_size'] == pop:
                si = sigmas.index(r['ea_sigma'])
                li = lrs.index(r['ea_lr'])
                # Take max if multiple gens
                if np.isnan(grid[si, li]) or r['test_accuracy'] > grid[si, li]:
                    grid[si, li] = r['test_accuracy']

        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto',
                        vmin=1/N_SYMBOLS, vmax=max(0.5, np.nanmax(grid)))
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs])
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f'{s}' for s in sigmas])
        ax.set_xlabel('LR')
        ax.set_ylabel('Sigma')
        ax.set_title(f'Pop={pop}', fontweight='bold')

        # Annotate cells
        for si in range(len(sigmas)):
            for li in range(len(lrs)):
                if not np.isnan(grid[si, li]):
                    ax.text(li, si, f'{grid[si, li]:.0%}',
                            ha='center', va='center', fontsize=9,
                            color='white' if grid[si, li] > 0.35 else 'black')

    fig.colorbar(im, ax=axes, shrink=0.8, label='Accuracy')
    plt.suptitle('Accuracy Heatmap: Sigma × LR', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="EA hyperparameter sweep")
    p.add_argument('--n-back', type=int, default=1,
                   help='N-back level to test (default: 1, the easiest)')
    p.add_argument('--neurons', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='results/ea_sweep')
    p.add_argument('--quick', action='store_true',
                   help='Use smaller grid for fast iteration')
    args = p.parse_args()

    results = run_sweep(
        n_back=args.n_back,
        n_neurons=args.neurons,
        seed=args.seed,
        output_dir=args.output,
        quick=args.quick,
    )

    print_results_table(results)

    if PLOT_AVAILABLE:
        plot_results(results, output_dir=args.output)

    print(f"\nResults saved to {args.output}/")
