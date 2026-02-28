#!/usr/bin/env python3
# scripts/run_experiment.py
"""
Main entry point for all thesis experiments.

Usage:
    python scripts/run_experiment.py --method ga --n-back 1
    python scripts/run_experiment.py --method ga_stdp --n-back 1
    python scripts/run_experiment.py --method es --n-back 2
    python scripts/run_experiment.py --method all --n-back 1
"""

import sys
import os
import time
import json

# Add project root to path so imports work from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


def run(conf, method="ga", run_bptt=True):
    """Run experiment with specified method(s)."""

    os.makedirs(conf.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"EXPERIMENT: {conf.task.upper()} | {conf.n_neurons} neurons")
    print(f"Method: {method}" + (" + BPTT" if run_bptt else ""))
    print("=" * 60)
    if conf.task == "nback":
        print(f"  {conf.n_back}-back recall, seq_len={conf.seq_length}, 5 symbols")

    results = {}
    timings = {}

    # --- Evolutionary methods ---
    if method in ("es", "all"):
        print("\n" + "-" * 60)
        print("Training ES (OpenAI Evolution Strategy)...")
        print("-" * 60)
        from trainers.train_es import train_es
        t0 = time.time()
        results['es'] = train_es(conf)
        timings['es'] = time.time() - t0
        print(f"ES time: {timings['es']:.1f}s\n")

    if method in ("ga", "all"):
        print("-" * 60)
        print("Training GA (Genetic Algorithm)...")
        print("-" * 60)
        from trainers.train_ga import train_ga
        t0 = time.time()
        results['ga'] = train_ga(conf)
        timings['ga'] = time.time() - t0
        print(f"GA time: {timings['ga']:.1f}s\n")

    if method in ("ga_stdp", "all"):
        print("-" * 60)
        print("Training GA+STDP (Baldwin Effect)...")
        print("-" * 60)
        from trainers.train_ga_stdp import train_ga_stdp
        t0 = time.time()
        results['ga_stdp'] = train_ga_stdp(conf)
        timings['ga_stdp'] = time.time() - t0
        print(f"GA+STDP time: {timings['ga_stdp']:.1f}s\n")

    # --- BPTT (rate-coded baseline) ---
    if run_bptt:
        print("-" * 60)
        print("Training BPTT (rate-coded)...")
        print("-" * 60)
        from trainers.train_bptt import train_bptt
        t0 = time.time()
        results['bptt'] = train_bptt(conf, use_lif=False)
        timings['bptt'] = time.time() - t0
        if results['bptt']:
            print(f"BPTT time: {timings['bptt']:.1f}s\n")

    # --- BPTT-LIF (surrogate gradient on spiking neurons) ---
    if method in ("bptt_lif", "all"):
        print("-" * 60)
        print("Training BPTT-LIF (surrogate gradient)...")
        print("-" * 60)
        from trainers.train_bptt import train_bptt
        t0 = time.time()
        results['bptt_lif'] = train_bptt(conf, use_lif=True)
        timings['bptt_lif'] = time.time() - t0
        if results['bptt_lif']:
            print(f"BPTT-LIF time: {timings['bptt_lif']:.1f}s\n")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, r in results.items():
        if r is None:
            continue
        if 'best_fitness' in r:
            acc = r['history']['accuracy'][-1]
            fit = r['best_fitness']
        else:
            acc = r['history']['accuracy'][-1]
            fit = r['history']['fitness'][-1]
        t = timings.get(name, 0)
        print(f"  {name:>8s}: acc={acc:.1%}  fit={fit:+.4f}  time={t:.0f}s")

    print(f"\nOutput: {conf.output_dir}/")

    # --- Save summary ---
    summary = {'config': conf.to_dict(), 'timings': timings}
    for name, r in results.items():
        if r and 'history' in r:
            summary[f'{name}_final_accuracy'] = r['history']['accuracy'][-1]
    with open(os.path.join(conf.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run thesis experiments")
    p.add_argument('--task', choices=['nback', 'wm'], default='nback')
    p.add_argument('--neurons', type=int, default=32)
    p.add_argument('--n-back', type=int, default=2)
    p.add_argument('--method', choices=['es', 'ga', 'ga_stdp', 'bptt_lif', 'all'], default='ga')
    p.add_argument('--no-bptt', action='store_true',
                   help='Skip rate-coded BPTT baseline')
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--ea-gens', type=int, default=300)
    p.add_argument('--ea-pop', type=int, default=128)
    p.add_argument('--ea-trials', type=int, default=20)
    p.add_argument('--bptt-iters', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    output_dir = args.output or f"results/{args.task}_{args.neurons}n"

    conf = Config(
        task=args.task,
        n_neurons=args.neurons,
        n_back=args.n_back,
        output_dir=output_dir,
        ea_generations=args.ea_gens,
        ea_pop_size=args.ea_pop,
        ea_n_eval_trials=args.ea_trials,
        bptt_iterations=args.bptt_iters,
        seed=args.seed,
    )

    run(conf, method=args.method, run_bptt=not args.no_bptt)