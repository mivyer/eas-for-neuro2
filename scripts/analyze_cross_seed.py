#!/usr/bin/env python3
"""
Cross-seed analysis with standard deviation error bars.
Run from your project root:
    python3 scripts/analyze_cross_seed.py --results-dir results/
"""

import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
METHODS = ['bptt', 'es', 'ga', 'ga_oja']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA+Oja'}
METHOD_COLORS = {'bptt': '#2196F3', 'es': '#FF9800', 'ga': '#4CAF50', 'ga_oja': '#E91E63'}
NBACKS = [1, 2, 3, 4]
SEEDS = [42, 123, 456]
LAYER_COLORS = {'W_in': '#2196F3', 'W_rec': '#FF9800', 'W_out': '#4CAF50'}

def load_all(results_dir, n_neurons=32):
    """Load all experimental data."""
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
                hpath = os.path.join(mdir, "history.json")
                if os.path.exists(hpath):
                    with open(hpath) as f:
                        entry['history'] = json.load(f)
                for wt in ['weights_init', 'weights_final']:
                    wp = os.path.join(mdir, f"{wt}.npz")
                    if os.path.exists(wp):
                        entry[wt] = dict(np.load(wp))
                # Post-oja weights
                wp = os.path.join(mdir, "weights_post_oja.npz")
                if os.path.exists(wp):
                    entry['weights_post_oja'] = dict(np.load(wp))
                data[nb][seed][method] = entry
    return data


def compute_metrics(data):
    """Compute performance and connectivity metrics."""
    perf = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    conn = {nb: {m: {'frac_in': [], 'frac_rec': [], 'frac_out': [],
                      'norm_total': [], 'eff_rank': []}
                 for m in METHODS} for nb in NBACKS}

    for nb in NBACKS:
        for seed in SEEDS:
            if seed not in data.get(nb, {}):
                continue
            for method in METHODS:
                if method not in data[nb][seed]:
                    continue
                d = data[nb][seed][method]

                # Performance
                h = d.get('history', {})
                if method == 'bptt':
                    acc = h.get('accuracy', [0])[-1] * 100
                else:
                    bf = h.get('best_fitness', h.get('accuracy', [0]))
                    acc = max(bf) * 100
                perf[nb][method].append(acc)

                # Connectivity
                if 'weights_init' in d and 'weights_final' in d:
                    wi, wf = d['weights_init'], d['weights_final']
                    dW_in = np.linalg.norm(wf['W_in'] - wi['W_in'])
                    dW_rec = np.linalg.norm(wf['W_rec'] - wi['W_rec'])
                    dW_out = np.linalg.norm(wf['W_out'] - wi['W_out'])
                    total = dW_in + dW_rec + dW_out
                    if total > 1e-8:
                        conn[nb][method]['frac_in'].append(dW_in / total)
                        conn[nb][method]['frac_rec'].append(dW_rec / total)
                        conn[nb][method]['frac_out'].append(dW_out / total)
                    conn[nb][method]['norm_total'].append(total)

                    U, S, Vt = np.linalg.svd(wf['W_rec'])
                    cumvar = np.cumsum(S**2) / np.sum(S**2)
                    conn[nb][method]['eff_rank'].append(int(np.searchsorted(cumvar, 0.9)) + 1)

    return perf, conn


def fig1_accuracy_vs_nback(perf, out_dir, n_neurons=32):
    """Line plot: accuracy vs n-back with std error bars."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in METHODS:
        means, stds, nbs = [], [], []
        for nb in NBACKS:
            vals = perf[nb][method]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                nbs.append(nb)
        if means:
            ax.errorbar(nbs, means, yerr=stds, marker='o', linewidth=2.5,
                        markersize=9, capsize=6, capthick=2,
                        label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.axhline(20, color='gray', linestyle='--', alpha=0.4, label='Chance')
    ax.set_xlabel('N-back Level', fontsize=13)
    ax.set_ylabel('Best Individual Accuracy (%)', fontsize=13)
    ax.set_title(f'Performance vs Task Difficulty\n{n_neurons} neurons, 3 seeds, mean ± std', fontsize=14)
    ax.set_xticks(NBACKS)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_accuracy_vs_nback.png'), dpi=300)
    plt.close()
    print("  Saved fig1_accuracy_vs_nback.png")


def fig2_layer_fractions(conn, out_dir, n_neurons=32):
    """Stacked bar: per-layer ΔW fractions across n-back, with std bars."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)
    for idx, nb in enumerate(NBACKS):
        ax = axes[idx]
        x_positions = np.arange(len(METHODS))
        for mi, method in enumerate(METHODS):
            c = conn[nb][method]
            if not c['frac_in']:
                continue
            fi_m, fr_m, fo_m = np.mean(c['frac_in']), np.mean(c['frac_rec']), np.mean(c['frac_out'])
            fi_s, fr_s, fo_s = np.std(c['frac_in']), np.std(c['frac_rec']), np.std(c['frac_out'])

            ax.bar(mi, fi_m, 0.7, color=LAYER_COLORS['W_in'], alpha=0.85)
            ax.bar(mi, fr_m, 0.7, bottom=fi_m, color=LAYER_COLORS['W_rec'], alpha=0.85)
            ax.bar(mi, fo_m, 0.7, bottom=fi_m + fr_m, color=LAYER_COLORS['W_out'], alpha=0.85)

            # Std error bars on each segment
            ax.errorbar(mi, fi_m / 2, yerr=fi_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.5)
            ax.errorbar(mi, fi_m + fr_m / 2, yerr=fr_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.5)
            ax.errorbar(mi, fi_m + fr_m + fo_m / 2, yerr=fo_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.5)

            # Labels
            ax.text(mi, fi_m / 2, f'{fi_m*100:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(mi, fi_m + fr_m / 2, f'{fr_m*100:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            ax.text(mi, fi_m + fr_m + fo_m / 2, f'{fo_m*100:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, fontsize=10)
        ax.set_title(f'{nb}-back', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Fraction of total ||ΔW||', fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=LAYER_COLORS['W_in'], label='W_in'),
                       Patch(facecolor=LAYER_COLORS['W_rec'], label='W_rec'),
                       Patch(facecolor=LAYER_COLORS['W_out'], label='W_out')]
    axes[0].legend(handles=legend_elements, fontsize=10, loc='upper left')
    fig.suptitle(f'Per-Layer Weight Change Fractions (mean ± std, 3 seeds, {n_neurons}n)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_layer_fractions.png'), dpi=300)
    plt.close()
    print("  Saved fig2_layer_fractions.png")


def fig3_effective_rank(conn, out_dir, n_neurons=32):
    """Grouped bar: effective rank with std."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(NBACKS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    for i, method in enumerate(METHODS):
        means, stds = [], []
        for nb in NBACKS:
            vals = conn[nb][method]['eff_rank']
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.bar(x + offsets[i] * width, means, width, yerr=stds,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method],
               alpha=0.85, capsize=4, ecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{nb}-back' for nb in NBACKS], fontsize=12)
    ax.set_ylabel('Effective Rank of W_rec (90% variance)', fontsize=12)
    ax.set_title(f'Effective Rank by Method and Task Difficulty\n{n_neurons} neurons, mean ± std', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0, max(25, n_neurons // 2))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_effective_rank.png'), dpi=300)
    plt.close()
    print("  Saved fig3_effective_rank.png")


def fig4_bptt_layer_shift(conn, out_dir):
    """BPTT-specific: how layer fractions shift with n-back difficulty."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for layer, key in [('W_in', 'frac_in'), ('W_rec', 'frac_rec'), ('W_out', 'frac_out')]:
        means, stds = [], []
        for nb in NBACKS:
            vals = [v * 100 for v in conn[nb]['bptt'][key]]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(NBACKS, means, yerr=stds, marker='s', linewidth=2.5,
                    markersize=8, capsize=6, capthick=2,
                    label=layer, color=LAYER_COLORS[layer])
    ax.set_xlabel('N-back Level', fontsize=13)
    ax.set_ylabel('Fraction of Total ΔW (%)', fontsize=13)
    ax.set_title('BPTT: Layer-Specific Weight Changes vs Difficulty\nW_out increases, W_in decreases with harder tasks', fontsize=13)
    ax.set_xticks(NBACKS)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 55)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_bptt_layer_shift.png'), dpi=300)
    plt.close()
    print("  Saved fig4_bptt_layer_shift.png")


def fig5_total_norm(conn, out_dir, n_neurons=32):
    """Total ΔW norm by method across n-back."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in METHODS:
        means, stds = [], []
        for nb in NBACKS:
            vals = conn[nb][method]['norm_total']
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(NBACKS, means, yerr=stds, marker='s', linewidth=2.5,
                    markersize=8, capsize=6, capthick=2,
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xlabel('N-back Level', fontsize=13)
    ax.set_ylabel('Total ||ΔW|| (Frobenius norm)', fontsize=13)
    ax.set_title(f'Total Weight Change Magnitude\n{n_neurons} neurons, mean ± std', fontsize=14)
    ax.set_xticks(NBACKS)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_total_norm.png'), dpi=300)
    plt.close()
    print("  Saved fig5_total_norm.png")


def fig6_learning_curves(data, out_dir, n_neurons=32):
    """Learning curves with std shading."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for idx, nb in enumerate(NBACKS):
        ax = axes[idx]
        for method in METHODS:
            curves = []
            for seed in SEEDS:
                if seed not in data.get(nb, {}) or method not in data[nb][seed]:
                    continue
                h = data[nb][seed][method].get('history', {})
                if method == 'bptt':
                    c = np.array(h.get('accuracy', [])) * 100
                else:
                    bf = np.array(h.get('best_fitness', h.get('accuracy', []))) * 100
                    c = np.maximum.accumulate(bf) if len(bf) > 0 else np.array([])
                if len(c) > 0:
                    curves.append(c)
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            aligned = np.array([c[:min_len] for c in curves])
            mean_c = aligned.mean(axis=0)
            std_c = aligned.std(axis=0)
            xvals = np.arange(min_len)
            ax.plot(xvals, mean_c, linewidth=2, label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method])
            ax.fill_between(xvals, mean_c - std_c, np.minimum(mean_c + std_c, 105),
                            alpha=0.15, color=METHOD_COLORS[method])
        ax.axhline(20, color='gray', linestyle='--', alpha=0.4)
        ax.set_title(f'{nb}-back', fontsize=13)
        ax.set_xlabel('Iteration / Generation', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Best Accuracy (%)', fontsize=12)
        ax.set_ylim(0, 108)
        ax.grid(True, alpha=0.2)
    axes[0].legend(fontsize=9, loc='lower right')
    fig.suptitle(f'Learning Curves (mean ± std, 3 seeds, {n_neurons}n)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig6_learning_curves.png'), dpi=300)
    plt.close()
    print("  Saved fig6_learning_curves.png")


def print_tables(perf, conn):
    """Print summary tables."""
    print("\n" + "=" * 70)
    print("PERFORMANCE TABLE (best individual accuracy, mean ± std)")
    print("=" * 70)
    print(f"{'N-back':>7} | {'BPTT':>12} | {'ES':>12} | {'GA':>12} | {'GA+Oja':>12}")
    print("-" * 63)
    for nb in NBACKS:
        parts = []
        for m in METHODS:
            v = perf[nb][m]
            if v:
                parts.append(f"{np.mean(v):>5.1f} ± {np.std(v):>4.1f}")
            else:
                parts.append(f"{'—':>12}")
        print(f"{nb:>7} | {'  |  '.join(parts)}")

    print("\n" + "=" * 70)
    print("CONNECTIVITY TABLE (mean ± std across seeds)")
    print("=" * 70)
    for method in METHODS:
        print(f"\n  {METHOD_LABELS[method]}:")
        print(f"  {'N-back':>7} | {'W_in%':>10} | {'W_rec%':>10} | {'W_out%':>10} | {'Rank':>10}")
        print(f"  {'-'*57}")
        for nb in NBACKS:
            c = conn[nb][method]
            if c['frac_in']:
                fi = f"{np.mean(c['frac_in'])*100:.1f}±{np.std(c['frac_in'])*100:.1f}"
                fr = f"{np.mean(c['frac_rec'])*100:.1f}±{np.std(c['frac_rec'])*100:.1f}"
                fo = f"{np.mean(c['frac_out'])*100:.1f}±{np.std(c['frac_out'])*100:.1f}"
                er = f"{np.mean(c['eff_rank']):.1f}±{np.std(c['eff_rank']):.1f}"
                print(f"  {nb:>7} | {fi:>10} | {fr:>10} | {fo:>10} | {er:>10}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/', help='Path to results directory')
    parser.add_argument('--out-dir', default='results/cross_seed_analysis/', help='Output directory for figures')
    parser.add_argument('--neurons', type=int, default=32, help='Network size (default 32)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    N = args.neurons
    print(f"Loading data ({N} neurons)...")
    data = load_all(args.results_dir, n_neurons=N)

    print("Computing metrics...")
    perf, conn = compute_metrics(data)

    print_tables(perf, conn)

    print("\nGenerating figures...")
    fig1_accuracy_vs_nback(perf, args.out_dir, n_neurons=N)
    fig2_layer_fractions(conn, args.out_dir, n_neurons=N)
    fig3_effective_rank(conn, args.out_dir, n_neurons=N)
    fig4_bptt_layer_shift(conn, args.out_dir)
    fig5_total_norm(conn, args.out_dir, n_neurons=N)
    fig6_learning_curves(data, args.out_dir, n_neurons=N)

    print(f"\nAll figures saved to {args.out_dir}")
