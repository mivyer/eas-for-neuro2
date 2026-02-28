#!/usr/bin/env python3
# scripts/sweep_ga_vs_ga_stdp.py
"""
Experimental sweep: GA vs GA+STDP vs BPTT

Produces the key thesis figure: accuracy vs n-back level for all methods.

Also sweeps GA hyperparameters:
  - mutation_rate: [0.01, 0.05, 0.1]
  - mutation_std: [0.1, 0.3, 0.5]
  - pop_size: [64, 128, 256]

Usage:
    # Quick: 1-back only, small sweep
    python scripts/sweep_ga_vs_ga_stdp.py --quick

    # Full: all n-back levels, all methods
    python scripts/sweep_ga_vs_ga_stdp.py --full

    # Just n-back sweep with best GA config
    python scripts/sweep_ga_vs_ga_stdp.py --nback-sweep
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from itertools import product

from config import Config

try:
    import matplotlib.pyplot as plt
    PLOT = True
except ImportError:
    PLOT = False


# ============================================================================
# GA hyperparameter sweep (1-back only)
# ============================================================================

def sweep_ga_hparams(quick=True):
    """Find best GA config on 1-back."""
    from trainers.train_ga import train_ga

    if quick:
        mut_rates = [0.03, 0.05, 0.1]
        mut_stds = [0.2, 0.3, 0.5]
        pop_sizes = [128]
        gens = 150
    else:
        mut_rates = [0.01, 0.03, 0.05, 0.1, 0.15]
        mut_stds = [0.1, 0.2, 0.3, 0.5]
        pop_sizes = [64, 128, 256]
        gens = 300

    configs = list(product(mut_rates, mut_stds, pop_sizes))
    print(f"\nGA Hyperparameter Sweep: {len(configs)} configs, {gens} gens each")
    print("=" * 70)

    results = []
    for i, (mr, ms, ps) in enumerate(configs):
        conf = Config(
            task='nback', n_back=1, n_neurons=32,
            ea_pop_size=ps, ea_generations=gens,
            ga_mutation_rate=mr, ga_mutation_std=ms,
            ea_n_eval_trials=20, seed=42,
            print_every=999,  # silent
            output_dir='results/ga_sweep_tmp',
        )
        t0 = time.time()
        r = train_ga(conf)
        elapsed = time.time() - t0
        acc = r['history']['accuracy'][-1]
        fit = r['best_fitness']
        results.append({
            'mut_rate': mr, 'mut_std': ms, 'pop_size': ps,
            'accuracy': acc, 'fitness': fit, 'time': elapsed,
        })
        print(f"[{i+1}/{len(configs)}] mr={mr} ms={ms} pop={ps} "
              f"→ acc={acc:.1%} fit={fit:+.4f} ({elapsed:.0f}s)")

    # Rank
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    print("\n" + "=" * 70)
    print(f"{'Rank':>4} | {'mut_rate':>8} {'mut_std':>8} {'pop':>5} | "
          f"{'Accuracy':>8} {'Fitness':>8} {'Time':>6}")
    print("-" * 70)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:4d} | {r['mut_rate']:8.3f} {r['mut_std']:8.2f} "
              f"{r['pop_size']:5d} | {r['accuracy']:7.1%} "
              f"{r['fitness']:+8.4f} {r['time']:5.0f}s")

    return results


# ============================================================================
# GA+STDP hyperparameter sweep
# ============================================================================

def sweep_ga_stdp_hparams(quick=True):
    """Find best GA+STDP config on 1-back."""
    from trainers.train_ga_stdp import train_ga_stdp

    # GA+STDP is slower, so fewer configs
    if quick:
        pop_sizes = [64]
        gens_list = [100]
    else:
        pop_sizes = [64, 128]
        gens_list = [150, 300]

    configs = list(product(pop_sizes, gens_list))
    print(f"\nGA+STDP Sweep: {len(configs)} configs")
    print("=" * 70)

    results = []
    for i, (ps, gens) in enumerate(configs):
        conf = Config(
            task='nback', n_back=1, n_neurons=32,
            ea_pop_size=ps, ea_generations=gens,
            ea_n_eval_trials=10, seed=42,
            print_every=999,
            output_dir='results/ga_stdp_sweep_tmp',
        )
        t0 = time.time()
        r = train_ga_stdp(conf)
        elapsed = time.time() - t0
        acc = r['history']['accuracy'][-1]
        fit = r['best_fitness']
        stdp_p = r.get('stdp_params', None)

        results.append({
            'pop_size': ps, 'gens': gens,
            'accuracy': acc, 'fitness': fit, 'time': elapsed,
            'stdp_params': stdp_p.tolist() if stdp_p is not None else None,
        })
        stdp_str = ""
        if stdp_p is not None:
            stdp_str = f" A+={stdp_p[0]:.4f} A-={stdp_p[1]:.4f}"
        print(f"[{i+1}/{len(configs)}] pop={ps} gens={gens} "
              f"→ acc={acc:.1%} fit={fit:+.4f} ({elapsed:.0f}s){stdp_str}")

    return results


# ============================================================================
# N-back sweep: GA vs GA+STDP vs ES vs BPTT
# ============================================================================

def sweep_nback_all_methods(n_values=[1, 2, 3, 4], gens=300):
    """Run all methods at each n-back level."""
    from trainers.train_ga import train_ga
    from trainers.train_ga_stdp import train_ga_stdp
    from trainers.train_es import train_es
    from trainers.train_bptt import train_bptt

    all_results = {}

    for n in n_values:
        print(f"\n{'='*60}")
        print(f"  {n}-BACK")
        print(f"{'='*60}")

        level_results = {}

        # GA
        print(f"\n  Training GA...")
        conf_ga = Config(task='nback', n_back=n, n_neurons=32,
                         ea_pop_size=128, ea_generations=gens,
                         ea_n_eval_trials=20, seed=42, print_every=100,
                         output_dir=f'results/sweep_all/{n}back')
        t0 = time.time()
        r = train_ga(conf_ga)
        level_results['ga'] = {
            'accuracy': r['history']['accuracy'][-1],
            'fitness': r['best_fitness'],
            'time': time.time() - t0,
        }

        # GA+STDP
        print(f"\n  Training GA+STDP...")
        conf_stdp = Config(task='nback', n_back=n, n_neurons=32,
                           ea_pop_size=64, ea_generations=gens,
                           ea_n_eval_trials=10, seed=42, print_every=100,
                           output_dir=f'results/sweep_all/{n}back')
        t0 = time.time()
        r = train_ga_stdp(conf_stdp)
        level_results['ga_stdp'] = {
            'accuracy': r['history']['accuracy'][-1],
            'fitness': r['best_fitness'],
            'time': time.time() - t0,
            'stdp_params': r.get('stdp_params', np.zeros(6)).tolist(),
        }

        # ES (tuned)
        print(f"\n  Training ES...")
        conf_es = Config(task='nback', n_back=n, n_neurons=32,
                         ea_pop_size=256, ea_generations=gens,
                         ea_lr=0.03, ea_sigma=0.02,
                         ea_n_eval_trials=20, seed=42, print_every=100,
                         output_dir=f'results/sweep_all/{n}back')
        t0 = time.time()
        r = train_es(conf_es)
        level_results['es'] = {
            'accuracy': r['history']['accuracy'][-1],
            'fitness': r['best_fitness'],
            'time': time.time() - t0,
        }

        # BPTT
        print(f"\n  Training BPTT...")
        conf_bptt = Config(task='nback', n_back=n, n_neurons=32,
                           bptt_iterations=1000, seed=42, print_every=100,
                           output_dir=f'results/sweep_all/{n}back')
        t0 = time.time()
        r = train_bptt(conf_bptt)
        if r:
            level_results['bptt'] = {
                'accuracy': r['history']['accuracy'][-1],
                'fitness': r['history']['fitness'][-1],
                'time': time.time() - t0,
            }

        all_results[n] = level_results

        # Print summary for this level
        print(f"\n  --- {n}-back results ---")
        for method, res in level_results.items():
            print(f"  {method:>8s}: acc={res['accuracy']:.1%}  "
                  f"time={res['time']:.0f}s")

    # Save
    save_dir = 'results/sweep_all'
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'sweep_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    if PLOT:
        plot_nback_comparison(all_results, save_dir)

    return all_results


def plot_nback_comparison(all_results, save_dir):
    """The key thesis figure: accuracy vs n-back for all methods."""
    methods = ['bptt', 'es', 'ga_stdp', 'ga']
    colors = {'bptt': '#2196F3', 'es': '#FF9800', 'ga_stdp': '#4CAF50', 'ga': '#F44336'}
    labels = {'bptt': 'BPTT', 'es': 'ES (tuned)', 'ga_stdp': 'GA+STDP', 'ga': 'GA'}
    markers = {'bptt': 'o', 'es': 's', 'ga_stdp': 'D', 'ga': '^'}

    n_values = sorted(all_results.keys(), key=int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy vs n-back
    for method in methods:
        accs = []
        ns = []
        for n in n_values:
            if method in all_results[n]:
                accs.append(all_results[n][method]['accuracy'] * 100)
                ns.append(int(n))
        if accs:
            ax1.plot(ns, accs, f'-{markers[method]}', color=colors[method],
                     label=labels[method], linewidth=2, markersize=8)

    ax1.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Chance (20%)')
    ax1.set_xlabel('N-back level', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Working Memory Performance', fontsize=14)
    ax1.set_xticks([int(n) for n in n_values])
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Time vs n-back
    for method in methods:
        times = []
        ns = []
        for n in n_values:
            if method in all_results[n]:
                times.append(all_results[n][method]['time'])
                ns.append(int(n))
        if times:
            ax2.plot(ns, times, f'-{markers[method]}', color=colors[method],
                     label=labels[method], linewidth=2, markersize=8)

    ax2.set_xlabel('N-back level', fontsize=12)
    ax2.set_ylabel('Training time (s)', fontsize=12)
    ax2.set_title('Compute Cost', fontsize=14)
    ax2.set_xticks([int(n) for n in n_values])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'nback_all_methods.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true',
                   help='Quick GA hparam sweep on 1-back only')
    p.add_argument('--full', action='store_true',
                   help='Full sweep: GA hparams + all methods on all n-back')
    p.add_argument('--nback-sweep', action='store_true',
                   help='N-back sweep with all methods (skip hparam tuning)')
    p.add_argument('--ga-hparams', action='store_true',
                   help='GA hyperparameter sweep only')
    p.add_argument('--gens', type=int, default=300)
    args = p.parse_args()

    if args.ga_hparams or args.quick:
        sweep_ga_hparams(quick=True)

    if args.full:
        print("\n\n=== Phase 1: GA Hyperparameter Sweep ===")
        sweep_ga_hparams(quick=False)
        print("\n\n=== Phase 2: GA+STDP Sweep ===")
        sweep_ga_stdp_hparams(quick=False)
        print("\n\n=== Phase 3: N-back Sweep (all methods) ===")
        sweep_nback_all_methods(gens=args.gens)

    if args.nback_sweep:
        sweep_nback_all_methods(gens=args.gens)

    if not any([args.quick, args.full, args.nback_sweep, args.ga_hparams]):
        print("Usage:")
        print("  --quick        Quick GA hparam sweep (1-back, ~30 min)")
        print("  --ga-hparams   GA hparam sweep only")
        print("  --nback-sweep  All methods across n-back levels (~2 hours)")
        print("  --full         Everything (~4+ hours)")
