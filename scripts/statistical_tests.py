#!/usr/bin/env python3
"""
Statistical tests for thesis results.

Tests:
  1. Method accuracy comparisons — Wilcoxon rank-sum (non-parametric, 3 seeds).
  2. Effective rank: BPTT vs EA methods, per n-back and neuron size.
  3. BPTT W_out fraction trend — Spearman correlation with n-back difficulty.
  4. BPTT vs EA W_out fraction — consistent directional difference test.
  5. Scaling: 32n vs 64n effective rank ratios (rank / N).

Usage:
    python3 scripts/statistical_tests.py --results-dir results/nback/ \\
        [--neurons 32 64] [--alpha 0.05] [--out results/statistical_tests_summary.txt]
"""

import os
import sys
import json
import argparse
import itertools
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS  = ['bptt', 'es', 'ga', 'ga_oja']
MLABELS  = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA+Oja'}
NBACKS   = [1, 2, 3, 4]
SEEDS    = [42, 123, 456]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(results_dir, n_neurons):
    """Return data[nb][seed][method] = {history, weights_init, weights_final, ...}"""
    data = {}
    for nb in NBACKS:
        data[nb] = {}
        for seed in SEEDS:
            run_dir = os.path.join(results_dir, f"nback{nb}_neurons{n_neurons}_seed{seed}")
            if not os.path.isdir(run_dir):
                continue
            data[nb][seed] = {}
            for method in METHODS:
                mdir = os.path.join(run_dir, method)
                if not os.path.isdir(mdir):
                    continue
                entry = {}
                hp = os.path.join(mdir, "history.json")
                if os.path.exists(hp):
                    with open(hp) as f:
                        entry['history'] = json.load(f)
                for tag in ('weights_init', 'weights_final', 'weights_post_oja'):
                    wp = os.path.join(mdir, f"{tag}.npz")
                    if os.path.exists(wp):
                        entry[tag] = dict(np.load(wp))
                data[nb][seed][method] = entry
    return data


def extract_metrics(data):
    """
    Returns:
        acc[nb][method]      list of best-accuracy floats (one per seed, 0–100)
        rank[nb][method]     list of effective-rank ints (W_rec, 90% var)
        frac_out[nb][method] list of W_out fraction floats (0–1)
        frac_in[nb][method]  list of W_in fraction floats
        frac_rec[nb][method] list of W_rec fraction floats
    """
    acc      = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    rank     = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    frac_out = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    frac_in  = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    frac_rec = {nb: {m: [] for m in METHODS} for nb in NBACKS}

    for nb in NBACKS:
        for seed in SEEDS:
            if seed not in data.get(nb, {}):
                continue
            for method in METHODS:
                if method not in data[nb][seed]:
                    continue
                d = data[nb][seed][method]
                h = d.get('history', {})

                # Accuracy
                if method == 'bptt':
                    a = h.get('accuracy', [0])[-1] * 100
                else:
                    bf = h.get('best_fitness', h.get('accuracy', [0]))
                    a = max(bf) * 100
                acc[nb][method].append(a)

                # Weight change metrics
                if 'weights_init' in d and 'weights_final' in d:
                    wi, wf = d['weights_init'], d['weights_final']
                    dWi  = np.linalg.norm(wf['W_in']  - wi['W_in'])
                    dWr  = np.linalg.norm(wf['W_rec'] - wi['W_rec'])
                    dWo  = np.linalg.norm(wf['W_out'] - wi['W_out'])
                    tot  = dWi + dWr + dWo
                    if tot > 1e-8:
                        frac_in[nb][method].append(dWi / tot)
                        frac_rec[nb][method].append(dWr / tot)
                        frac_out[nb][method].append(dWo / tot)

                    U, S, _ = np.linalg.svd(wf['W_rec'])
                    cumvar  = np.cumsum(S**2) / np.sum(S**2)
                    rank[nb][method].append(int(np.searchsorted(cumvar, 0.9)) + 1)

    return acc, rank, frac_out, frac_in, frac_rec


# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

def mannwhitney(a, b, alternative='two-sided'):
    """Mann-Whitney U with fallback for degenerate cases."""
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    try:
        u, p = stats.mannwhitneyu(a, b, alternative=alternative)
        return float(u), float(p)
    except Exception:
        return float('nan'), float('nan')


def spearman(x, y):
    if len(x) < 3:
        return float('nan'), float('nan')
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


def stars(p, alpha):
    if np.isnan(p):   return '(n/a)'
    if p < alpha/10:  return '***'
    if p < alpha:     return '**'
    if p < alpha*2:   return '*'
    return 'ns'


# ─────────────────────────────────────────────────────────────────────────────
# Test sections
# ─────────────────────────────────────────────────────────────────────────────

def test_accuracy_comparisons(acc, alpha, lines):
    lines.append("=" * 70)
    lines.append("TEST 1 — Method accuracy comparisons (Mann-Whitney U)")
    lines.append("H0: two methods have equal accuracy distributions across seeds")
    lines.append("=" * 70)
    lines.append(f"{'N-back':>7}  {'Comparison':>20}  {'U':>8}  {'p':>10}  {'sig':>5}  {'direction'}")
    lines.append("-" * 70)

    pairs = [('bptt', 'es'), ('bptt', 'ga'), ('bptt', 'ga_oja'),
             ('es', 'ga'), ('es', 'ga_oja'), ('ga', 'ga_oja')]
    for nb in NBACKS:
        for m1, m2 in pairs:
            a, b = acc[nb][m1], acc[nb][m2]
            u, p = mannwhitney(a, b)
            if a and b:
                direction = f"{MLABELS[m1]}>{MLABELS[m2]}" if np.mean(a) > np.mean(b) \
                            else f"{MLABELS[m2]}>{MLABELS[m1]}"
            else:
                direction = '—'
            label = f"{MLABELS[m1]} vs {MLABELS[m2]}"
            lines.append(f"{nb:>7}  {label:>20}  {u:>8.1f}  {p:>10.4f}  {stars(p,alpha):>5}  {direction}")
        lines.append("")


def test_effective_rank(rank, n_neurons, alpha, lines):
    lines.append("=" * 70)
    lines.append(f"TEST 2 — Effective rank: BPTT vs EA methods ({n_neurons}n)")
    lines.append("H1 (directional): BPTT rank < EA rank")
    lines.append("=" * 70)
    lines.append(f"{'N-back':>7}  {'EA method':>12}  {'BPTT mean':>10}  {'EA mean':>10}  {'p (less)':>10}  {'sig':>5}")
    lines.append("-" * 70)

    ea_methods = ['es', 'ga', 'ga_oja']
    for nb in NBACKS:
        bptt_vals = rank[nb]['bptt']
        for m in ea_methods:
            ea_vals = rank[nb][m]
            _, p = mannwhitney(bptt_vals, ea_vals, alternative='less')
            bm = f"{np.mean(bptt_vals):.1f}" if bptt_vals else "—"
            em = f"{np.mean(ea_vals):.1f}"   if ea_vals   else "—"
            lines.append(f"{nb:>7}  {MLABELS[m]:>12}  {bm:>10}  {em:>10}  {p:>10.4f}  {stars(p,alpha):>5}")
        lines.append("")


def test_wout_trend(frac_out, alpha, lines):
    lines.append("=" * 70)
    lines.append("TEST 3 — BPTT W_out fraction: Spearman correlation with n-back")
    lines.append("H1: W_out fraction increases monotonically with difficulty")
    lines.append("=" * 70)

    # One data point per (nb, seed) pair
    x = []   # n-back level
    y = []   # W_out fraction
    for nb in NBACKS:
        for v in frac_out[nb]['bptt']:
            x.append(nb)
            y.append(v)

    r, p = spearman(x, y)
    lines.append(f"  BPTT W_out ~ n-back:  ρ={r:+.3f}  p={p:.4f}  {stars(p, alpha)}")
    lines.append(f"  n={len(x)} data points (up to {len(NBACKS)*len(SEEDS)} n-back×seed)")
    lines.append("")

    # Also test each EA method (expecting flat / no trend)
    lines.append("  EA methods (expecting ρ ≈ 0, no systematic trend):")
    for m in ['es', 'ga', 'ga_oja']:
        xm, ym = [], []
        for nb in NBACKS:
            for v in frac_out[nb][m]:
                xm.append(nb)
                ym.append(v)
        r2, p2 = spearman(xm, ym)
        lines.append(f"  {MLABELS[m]:>8} W_out ~ n-back:  ρ={r2:+.3f}  p={p2:.4f}  {stars(p2, alpha)}")
    lines.append("")


def test_wout_bptt_vs_ea(frac_out, alpha, lines):
    lines.append("=" * 70)
    lines.append("TEST 4 — BPTT vs EA W_out fraction (directional, per n-back)")
    lines.append("H1: BPTT W_out > EA W_out at hard n-back levels")
    lines.append("=" * 70)
    lines.append(f"{'N-back':>7}  {'EA method':>12}  {'BPTT mean%':>11}  {'EA mean%':>9}  {'p (greater)':>12}  {'sig':>5}")
    lines.append("-" * 70)

    for nb in NBACKS:
        bptt_vals = frac_out[nb]['bptt']
        for m in ['es', 'ga', 'ga_oja']:
            ea_vals = frac_out[nb][m]
            _, p = mannwhitney(bptt_vals, ea_vals, alternative='greater')
            bm = f"{np.mean(bptt_vals)*100:.1f}%" if bptt_vals else "—"
            em = f"{np.mean(ea_vals)*100:.1f}%"   if ea_vals   else "—"
            lines.append(f"{nb:>7}  {MLABELS[m]:>12}  {bm:>11}  {em:>9}  {p:>12.4f}  {stars(p, alpha):>5}")
        lines.append("")


def test_rank_scaling(rank_32, rank_64, alpha, lines):
    lines.append("=" * 70)
    lines.append("TEST 5 — Effective rank scaling: rank/N ratio 32n vs 64n")
    lines.append("H0: rank/N ratio is the same at 32n and 64n (no change in dimensionality use)")
    lines.append("=" * 70)
    lines.append(f"{'Method':>10}  {'32n rank/N mean':>16}  {'64n rank/N mean':>16}  {'p (two-sided)':>14}  {'sig':>5}")
    lines.append("-" * 70)

    for m in METHODS:
        # Pool across all n-backs for power
        vals_32, vals_64 = [], []
        for nb in NBACKS:
            r32 = rank_32[nb][m]
            r64 = rank_64[nb][m]
            vals_32.extend([v / 32 for v in r32])
            vals_64.extend([v / 64 for v in r64])
        if not vals_32 or not vals_64:
            continue
        _, p = mannwhitney(vals_32, vals_64)
        m32 = f"{np.mean(vals_32):.3f}"
        m64 = f"{np.mean(vals_64):.3f}"
        lines.append(f"{MLABELS[m]:>10}  {m32:>16}  {m64:>16}  {p:>14.4f}  {stars(p, alpha):>5}")
    lines.append("")


def print_summary_table(acc, rank, frac_out, n_neurons, lines):
    lines.append("=" * 70)
    lines.append(f"SUMMARY TABLE — {n_neurons} neurons (mean ± std, {len(SEEDS)} seeds)")
    lines.append("=" * 70)
    lines.append(f"{'':>12} | {'Accuracy%':>13} | {'Eff. rank':>10} | {'W_out%':>8}")
    lines.append("-" * 55)
    for nb in NBACKS:
        lines.append(f"  {nb}-back:")
        for m in METHODS:
            a  = acc[nb][m]
            r  = rank[nb][m]
            fo = frac_out[nb][m]
            as_ = f"{np.mean(a):.1f}±{np.std(a):.1f}"  if a  else "—"
            rs_ = f"{np.mean(r):.1f}±{np.std(r):.1f}"  if r  else "—"
            fs_ = f"{np.mean(fo)*100:.1f}±{np.std(fo)*100:.1f}" if fo else "—"
            lines.append(f"  {MLABELS[m]:>10} | {as_:>13} | {rs_:>10} | {fs_:>8}")
        lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical tests for thesis EA vs BPTT results")
    parser.add_argument('--results-dir', default='results/nback/',
                        help='Root results directory containing nback{N}_neurons{M}_seed{S}/ dirs')
    parser.add_argument('--neurons', type=int, nargs='+', default=[32, 64],
                        help='Neuron counts to analyse (default: 32 64)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance threshold (default 0.05)')
    parser.add_argument('--out', type=str, default='results/statistical_tests_summary.txt',
                        help='Path to write text summary')
    args = parser.parse_args()

    alpha = args.alpha
    lines = []
    lines.append("STATISTICAL TESTS — EA vs BPTT on N-back Task")
    lines.append(f"Results dir: {args.results_dir}")
    lines.append(f"Neuron sizes: {args.neurons}  |  Alpha: {alpha}")
    lines.append(f"Seeds: {SEEDS}  |  N-back levels: {NBACKS}")
    lines.append(f"Tests: Mann-Whitney U (non-parametric, two-sample), Spearman ρ")
    lines.append(f"Significance: *** p<{alpha/10:.4f}  ** p<{alpha:.4f}  * p<{alpha*2:.4f}  ns otherwise")
    lines.append("")

    all_data  = {}
    all_metrics = {}
    for N in args.neurons:
        print(f"Loading {N}n data...")
        all_data[N] = load_data(args.results_dir, N)
        all_metrics[N] = extract_metrics(all_data[N])

    for N in args.neurons:
        acc, rank, frac_out, frac_in, frac_rec = all_metrics[N]
        lines.append("")
        lines.append("█" * 70)
        lines.append(f"  {N} NEURONS")
        lines.append("█" * 70)
        print_summary_table(acc, rank, frac_out, N, lines)
        test_accuracy_comparisons(acc, alpha, lines)
        test_effective_rank(rank, N, alpha, lines)
        test_wout_trend(frac_out, alpha, lines)
        test_wout_bptt_vs_ea(frac_out, alpha, lines)

    # Cross-size scaling test if both 32 and 64 available
    if 32 in all_metrics and 64 in all_metrics:
        lines.append("")
        lines.append("█" * 70)
        lines.append("  CROSS-SIZE SCALING (32n vs 64n)")
        lines.append("█" * 70)
        test_rank_scaling(all_metrics[32][1], all_metrics[64][1], alpha, lines)

    # Print and write
    text = "\n".join(lines)
    print(text)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(text + "\n")
    print(f"\nWritten to {args.out}")


if __name__ == '__main__':
    main()
