#!/usr/bin/env python3
"""
Run thesis experiments. Output auto-named results/nback{N}_neurons{M}_seed{S}/.

    python scripts/run_experiment.py --method all --n-back 2 --save
    python scripts/run_experiment.py --method all --n-back 1 --ea-gens 50 --bptt-iters 100
"""

import sys
import os
import time
import json
import datetime
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def _save_config(conf: Config, exp_dir: str) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    meta = conf.to_dict()
    meta['timestamp'] = datetime.datetime.now().isoformat()
    meta['git_commit'] = _git_hash()
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def _save_method(result: dict, method: str, exp_dir: str) -> None:
    method_dir = os.path.join(exp_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    np.savez(os.path.join(method_dir, 'weights_init.npz'),
             W_rec=result['W_rec_init'],
             W_in=result['W_in_init'],
             W_out=result['W_out_init'])

    np.savez(os.path.join(method_dir, 'weights_final.npz'),
             W_rec=result['W_rec_final'],
             W_in=result['W_in_final'],
             W_out=result['W_out_final'])

    if result.get('W_rec_post_oja') is not None:
        np.savez(os.path.join(method_dir, 'weights_post_oja.npz'),
                 W_rec=result['W_rec_post_oja'],
                 W_in=result['W_in_post_oja'],
                 W_out=result['W_out_post_oja'])

    with open(os.path.join(method_dir, 'history.json'), 'w') as f:
        json.dump(result['history'], f, indent=2, default=float)

    if result.get('model') is not None:
        import torch
        torch.save(result['model'].state_dict(),
                   os.path.join(method_dir, 'model.pt'))

    if result.get('best_gene') is not None:
        np.save(os.path.join(method_dir, 'best_gene.npy'), result['best_gene'])


def _auto_exp_name(conf: Config) -> str:
    if conf.task == "evidence":
        return (f"evidence_s{conf.evidence_strength}_n{conf.noise_std}"
                f"_neurons{conf.n_neurons}_seed{conf.seed}")
    if conf.task == "robot":
        return f"robot_T{conf.seq_length}_neurons{conf.n_neurons}_seed{conf.seed}"
    return f"nback{conf.n_back}_neurons{conf.n_neurons}_seed{conf.seed}"


def run(conf: Config, method: str = "ga", run_bptt: bool = True,
        save: bool = False, run_analysis: bool = False) -> dict:
    if save:
        os.makedirs(conf.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"EXPERIMENT: {conf.task.upper()} | {conf.n_neurons} neurons")
    print(f"Method: {method}" + (" + BPTT" if run_bptt else ""))
    if save:
        print(f"Saving → {conf.output_dir}/")
    print("=" * 60)
    if conf.task == "nback":
        print(f"  {conf.n_back}-back recall, seq_len={conf.seq_length}, 5 symbols")
    elif conf.task == "evidence":
        print(f"  evidence accumulation: strength={conf.evidence_strength}, "
              f"noise={conf.noise_std}, T={conf.trial_length}, resp={conf.response_length}")
    elif conf.task == "robot":
        print(f"  robot arm endpoint prediction: T={conf.seq_length}, "
              f"obs_dim={conf.obs_dim}, action_dim={conf.action_dim}")

    results = {}
    timings = {}

    if method in ("es", "all"):
        from trainers.train_es import train_es
        t0 = time.time()
        results['es'] = train_es(conf)
        timings['es'] = time.time() - t0
        print(f"ES time: {timings['es']:.1f}s\n")

    if method in ("ga", "all"):
        from trainers.train_ga import train_ga
        t0 = time.time()
        results['ga'] = train_ga(conf)
        timings['ga'] = time.time() - t0
        print(f"GA time: {timings['ga']:.1f}s\n")

    if method in ("ga_oja", "all"):
        from trainers.train_ga_oja import train_ga_oja
        t0 = time.time()
        results['ga_oja'] = train_ga_oja(conf)
        timings['ga_oja'] = time.time() - t0
        print(f"GA-Oja time: {timings['ga_oja']:.1f}s\n")

    if method == "ga_stdp":
        from trainers.train_ga_stdp import train_ga_stdp
        t0 = time.time()
        results['ga_stdp'] = train_ga_stdp(conf)
        timings['ga_stdp'] = time.time() - t0
        print(f"GA+STDP time: {timings['ga_stdp']:.1f}s\n")

    if run_bptt:
        from trainers.train_bptt import train_bptt
        t0 = time.time()
        results['bptt'] = train_bptt(conf, use_lif=False)
        timings['bptt'] = time.time() - t0
        if results['bptt']:
            print(f"BPTT time: {timings['bptt']:.1f}s\n")

    if method == "bptt_lif":
        from trainers.train_bptt import train_bptt
        t0 = time.time()
        results['bptt_lif'] = train_bptt(conf, use_lif=True)
        timings['bptt_lif'] = time.time() - t0
        if results['bptt_lif']:
            print(f"BPTT-LIF time: {timings['bptt_lif']:.1f}s\n")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, r in results.items():
        if r is None:
            continue
        fit = r.get('best_fitness', r['history']['fitness'][-1])
        t   = timings.get(name, 0)
        if 'best_accuracy' in r:
            best_acc = r['best_accuracy']
            mean_acc = r['history']['accuracy'][-1]
            print(f"  {name:>8s}: best_acc={best_acc:.1%}  mean_acc={mean_acc:.1%}"
                  f"  fit={fit:+.4f}  time={t:.0f}s")
        else:
            acc = r['history']['accuracy'][-1]
            print(f"  {name:>8s}: acc={acc:.1%}  fit={fit:+.4f}  time={t:.0f}s")

    if save:
        _save_config(conf, conf.output_dir)
        for name, r in results.items():
            if r is not None:
                _save_method(r, name, conf.output_dir)
                print(f"  Saved {conf.output_dir}/{name}/")

        if run_analysis:
            from scripts.analyze_connectivity import analyze
            analyze(results, conf.output_dir, ei_ratio=conf.ei_ratio)

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--task',    choices=['nback', 'wm', 'evidence', 'robot'], default='nback')
    p.add_argument('--neurons', type=int, default=32)
    p.add_argument('--n-back',  type=int, default=2)
    p.add_argument('--method',  choices=['es', 'ga', 'ga_oja', 'ga_stdp', 'bptt_lif', 'all'],
                   default='ga')
    p.add_argument('--no-bptt',       action='store_true')
    p.add_argument('--save',          action='store_true')
    p.add_argument('--with-analysis', action='store_true',
                   help='Generate per-run connectivity figures')
    p.add_argument('--output',     type=str,   default=None)
    p.add_argument('--ea-gens',    type=int,   default=300)
    p.add_argument('--ea-pop',     type=int,   default=128)
    p.add_argument('--ea-trials',  type=int,   default=20)
    p.add_argument('--patience',   type=int,   default=999_999)
    p.add_argument('--bptt-iters', type=int,   default=1000)
    p.add_argument('--bptt-lr',    type=float, default=1e-3)
    p.add_argument('--mut-std',    type=float, default=None)
    p.add_argument('--scale-sigma', action='store_true',
                   help='Scale sigma by 1/sqrt(n_params/baseline); recommended for 128n+')
    p.add_argument('--scale-pop',   action='store_true',
                   help='Scale pop by sqrt(n_params/baseline)')
    p.add_argument('--l2-coef',    type=float, default=None)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--seq-length', type=int,   default=20)
    p.add_argument('--evidence-strength', type=float, default=0.1)
    p.add_argument('--noise-std',         type=float, default=0.5)
    p.add_argument('--trial-length',      type=int,   default=50)
    p.add_argument('--response-length',   type=int,   default=5)
    args = p.parse_args()

    obs_dim    = 2 if args.task == "robot" else 5
    action_dim = 2 if args.task == "robot" else 5

    conf = Config(
        task=args.task,
        n_neurons=args.neurons,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_back=args.n_back,
        seq_length=args.seq_length,
        ea_generations=args.ea_gens,
        ea_pop_size=args.ea_pop,
        ea_n_eval_trials=args.ea_trials,
        ea_patience=args.patience,
        bptt_iterations=args.bptt_iters,
        bptt_lr=args.bptt_lr,
        seed=args.seed,
        evidence_strength=args.evidence_strength,
        noise_std=args.noise_std,
        trial_length=args.trial_length,
        response_length=args.response_length,
        output_dir='',
    )
    if args.mut_std is not None:
        conf.ga_mutation_std = args.mut_std
    if args.scale_sigma:
        conf.ea_sigma_scaling = True
    if args.scale_pop:
        conf.ea_auto_pop = True
    if args.l2_coef is not None:
        conf.ea_l2_coef = args.l2_coef
    conf.output_dir = args.output or f"results/{_auto_exp_name(conf)}"

    run(conf, method=args.method, run_bptt=not args.no_bptt,
        save=args.save, run_analysis=args.with_analysis)
