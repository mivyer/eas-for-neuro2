#!/usr/bin/env python3
# scripts/load_results.py
"""
Loader utility for saved experiment directories.

Directory layout expected:
    {exp_dir}/
    ├── config.json
    ├── bptt/
    │   ├── weights_init.npz
    │   ├── weights_final.npz
    │   ├── history.json
    │   ├── model.pt          (optional)
    │   └── best_gene.npy     (not present for BPTT)
    ├── es/
    │   ├── weights_init.npz
    │   ├── weights_final.npz
    │   ├── history.json
    │   └── best_gene.npy
    ├── ga/
    │   └── ...
    └── ga_oja/
        ├── weights_init.npz      — population centroid at gen 0
        ├── weights_final.npz     — evolved genotype weights (pre-Oja)
        ├── weights_post_oja.npz  — post-Oja weights after within-trial plasticity
        ├── history.json
        └── best_gene.npy

Usage:
    from scripts.load_results import load_experiment, list_experiments

    exp = load_experiment('results/nback2_neurons32_seed42')
    print(exp['config']['n_back'])          # 2
    print(list(exp['methods'].keys()))      # ['bptt', 'es', 'ga']

    bptt = exp['methods']['bptt']
    bptt['W_rec_init']    # (N, N) initial recurrent weights
    bptt['W_rec_final']   # (N, N) trained recurrent weights
    bptt['history']       # dict of lists: loss, accuracy, fitness, ...
    bptt['model_path']    # path to model.pt (load with torch.load)

    # Feed directly into analyze_connectivity.analyze():
    from scripts.analyze_connectivity import analyze
    results = to_analyze_format(exp)       # same keys as trainer return dicts
    analyze(results, 'results/nback2_neurons32_seed42')
"""

import os
import json
import numpy as np

KNOWN_METHODS = ['bptt', 'bptt_lif', 'es', 'ga', 'ga_oja', 'ga_stdp']


def load_experiment(path: str) -> dict:
    """
    Load a complete saved experiment directory.

    Returns:
        {
          'config':  dict or None,
          'methods': {
              'bptt': {
                  'W_rec_init':   np.ndarray,
                  'W_in_init':    np.ndarray,
                  'W_out_init':   np.ndarray,
                  'W_rec_final':  np.ndarray,
                  'W_in_final':   np.ndarray,
                  'W_out_final':  np.ndarray,
                  'history':      dict of lists,
                  'model_path':   str | None,   # path to model.pt
                  'best_gene':    np.ndarray | None,
              },
              'es': { ... },
              'ga': { ... },
          },
        }
    """
    exp = {'config': None, 'path': os.path.abspath(path), 'methods': {}}

    # Config
    config_path = os.path.join(path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            exp['config'] = json.load(f)

    # Per-method data
    for method in KNOWN_METHODS:
        method_dir = os.path.join(path, method)
        if not os.path.isdir(method_dir):
            continue

        m = {}

        # Weight arrays
        for label in ('init', 'final'):
            npz_path = os.path.join(method_dir, f'weights_{label}.npz')
            if os.path.exists(npz_path):
                d = np.load(npz_path)
                for k in d.files:              # 'W_rec', 'W_in', 'W_out'
                    m[f'{k}_{label}'] = d[k]  # → 'W_rec_init', etc.

        # Post-Oja weights (GA-Oja only)
        post_oja_path = os.path.join(method_dir, 'weights_post_oja.npz')
        if os.path.exists(post_oja_path):
            d = np.load(post_oja_path)
            for k in d.files:                        # 'W_rec', 'W_in', 'W_out'
                m[f'{k}_post_oja'] = d[k]            # → 'W_rec_post_oja', etc.

        # Learning-curve history
        hist_path = os.path.join(method_dir, 'history.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                m['history'] = json.load(f)

        # PyTorch model (path only — avoids mandatory torch import)
        model_pt = os.path.join(method_dir, 'model.pt')
        m['model_path'] = model_pt if os.path.exists(model_pt) else None

        # Best gene vector (ES / GA)
        gene_path = os.path.join(method_dir, 'best_gene.npy')
        m['best_gene'] = np.load(gene_path) if os.path.exists(gene_path) else None

        exp['methods'][method] = m

    return exp


def load_model(exp: dict, method: str = 'bptt'):
    """
    Load the saved PyTorch model for a BPTT result.

    Requires torch. Returns the loaded state_dict.
    """
    import torch
    m = exp['methods'].get(method, {})
    path = m.get('model_path')
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"No model.pt found for method '{method}'")
    return torch.load(path, map_location='cpu')


def to_analyze_format(exp: dict) -> dict:
    """
    Convert a load_experiment() result into the format expected by
    analyze_connectivity.analyze().

    The returned dict is keyed by method name and each value has the same
    keys as trainer return dicts (W_rec_init, W_rec_final, ...).
    """
    return {method: data for method, data in exp['methods'].items()}


def list_experiments(results_dir: str = 'results') -> list:
    """
    List all experiment directories under results_dir that contain config.json.

    Returns list of absolute paths, sorted by modification time (newest first).
    """
    if not os.path.isdir(results_dir):
        return []
    entries = []
    for name in os.listdir(results_dir):
        path = os.path.join(results_dir, name)
        config_json = os.path.join(path, 'config.json')
        if os.path.isdir(path) and os.path.exists(config_json):
            entries.append((os.path.getmtime(config_json), path))
    entries.sort(reverse=True)
    return [p for _, p in entries]


def summarize(exp: dict) -> None:
    """Print a quick summary of an experiment to stdout."""
    cfg = exp.get('config') or {}
    print(f"Experiment: {exp.get('path', '?')}")
    if cfg:
        print(f"  Task:    {cfg.get('task')} | n_back={cfg.get('n_back')} | "
              f"neurons={cfg.get('n_neurons')} | seed={cfg.get('seed')}")
        print(f"  Saved:   {cfg.get('timestamp', '?')[:19]}  "
              f"git={cfg.get('git_commit', '?')}")
    for method, m in exp['methods'].items():
        acc = None
        if 'history' in m and 'accuracy' in m['history']:
            acc = m['history']['accuracy'][-1]
        has_model = m.get('model_path') is not None
        has_gene  = m.get('best_gene') is not None
        extras = []
        if has_model: extras.append('model.pt')
        if has_gene:  extras.append('best_gene')
        extras_str = f"  [{', '.join(extras)}]" if extras else ''
        acc_str = f"acc={acc:.1%}" if acc is not None else 'acc=?'
        print(f"  {method:>8s}:  {acc_str}{extras_str}")


# ── CLI (quick inspect) ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        # List all experiments
        exps = list_experiments()
        if not exps:
            print("No saved experiments found in results/")
        else:
            print(f"Found {len(exps)} experiment(s):\n")
            for path in exps:
                exp = load_experiment(path)
                summarize(exp)
                print()
    else:
        # Summarize a specific experiment
        path = sys.argv[1]
        exp = load_experiment(path)
        summarize(exp)
