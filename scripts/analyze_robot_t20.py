#!/usr/bin/env python3
"""
Analysis of robot arm T20 experiments: 3 neuron sizes × 3 seeds = 9 experiments.

Usage (from project root):
    python3 scripts/analyze_robot_t20.py --results-dir results/ --out-dir results/robot_T20_analysis/
"""

import json, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
NEURON_SIZES = [32, 64, 128]
SEEDS        = [42, 123, 456]
METHODS      = ['bptt', 'es', 'ga', 'ga_oja']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA+Oja'}
METHOD_COLORS = {'bptt': '#2196F3', 'es': '#FF9800', 'ga': '#4CAF50', 'ga_oja': '#E91E63'}
LAYER_COLORS  = {'W_in': '#2196F3', 'W_rec': '#FF9800', 'W_out': '#4CAF50'}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_all(results_dir):
    """Return nested dict: data[neurons][seed][method] = {history, weights_init, weights_final, ...}"""
    data = {}
    for n in NEURON_SIZES:
        data[n] = {}
        for seed in SEEDS:
            run_dir = os.path.join(results_dir, f"robot_T20_neurons{n}_seed{seed}")
            if not os.path.isdir(run_dir):
                print(f"  [warn] missing: {run_dir}")
                continue
            data[n][seed] = {}
            for method in METHODS:
                mdir = os.path.join(run_dir, method)
                if not os.path.isdir(mdir):
                    continue
                entry = {}
                hpath = os.path.join(mdir, "history.json")
                if os.path.exists(hpath):
                    with open(hpath) as f:
                        entry['history'] = json.load(f)
                for tag in ['weights_init', 'weights_final', 'weights_post_oja']:
                    wp = os.path.join(mdir, f"{tag}.npz")
                    if os.path.exists(wp):
                        entry[tag] = dict(np.load(wp))
                data[n][seed][method] = entry
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Metric extraction helpers
# ──────────────────────────────────────────────────────────────────────────────
def best_accuracy(entry, method):
    """Return best accuracy as a percentage.
    - BPTT: uses 'accuracy' (single-model accuracy per iter, fraction 0-1)
    - ES/GA/GA-Oja: uses 'accuracy' (mean population accuracy per gen, fraction 0-1)
    Note: 'best_fitness' in EA methods is neg-CE loss, not usable as accuracy directly.
    """
    h = entry.get('history', {})
    acc = h.get('accuracy', [0])
    return max(acc) * 100


def weight_change_metrics(entry):
    """Returns (frac_in, frac_rec, frac_out, norm_total, eff_rank) or None."""
    if 'weights_init' not in entry or 'weights_final' not in entry:
        return None
    wi, wf = entry['weights_init'], entry['weights_final']
    dW_in  = np.linalg.norm(wf['W_in']  - wi['W_in'])
    dW_rec = np.linalg.norm(wf['W_rec'] - wi['W_rec'])
    dW_out = np.linalg.norm(wf['W_out'] - wi['W_out'])
    total  = dW_in + dW_rec + dW_out
    frac_in  = dW_in  / total if total > 1e-8 else 0
    frac_rec = dW_rec / total if total > 1e-8 else 0
    frac_out = dW_out / total if total > 1e-8 else 0
    U, S, Vt = np.linalg.svd(wf['W_rec'])
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    eff_rank = int(np.searchsorted(cumvar, 0.9)) + 1
    return frac_in, frac_rec, frac_out, total, eff_rank


def compute_metrics(data):
    """Aggregate performance and connectivity metrics across seeds."""
    perf = {n: {m: [] for m in METHODS} for n in NEURON_SIZES}
    conn = {n: {m: {'frac_in': [], 'frac_rec': [], 'frac_out': [],
                    'norm_total': [], 'eff_rank': []}
                for m in METHODS} for n in NEURON_SIZES}
    for n in NEURON_SIZES:
        for seed in SEEDS:
            if seed not in data.get(n, {}):
                continue
            for method in METHODS:
                if method not in data[n][seed]:
                    continue
                entry = data[n][seed][method]
                perf[n][method].append(best_accuracy(entry, method))
                m = weight_change_metrics(entry)
                if m:
                    fi, fr, fo, nt, er = m
                    conn[n][method]['frac_in'].append(fi)
                    conn[n][method]['frac_rec'].append(fr)
                    conn[n][method]['frac_out'].append(fo)
                    conn[n][method]['norm_total'].append(nt)
                    conn[n][method]['eff_rank'].append(er)
                    # Also store absolute norms per layer
                    conn[n][method].setdefault('norm_in',  []).append(fi * nt)
                    conn[n][method].setdefault('norm_rec', []).append(fr * nt)
                    conn[n][method].setdefault('norm_out', []).append(fo * nt)
    return perf, conn


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — Performance bar chart (mean ± std, grouped by neuron count)
# ──────────────────────────────────────────────────────────────────────────────
def fig1_performance_bars(perf, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_methods = len(METHODS)
    x = np.arange(len(NEURON_SIZES))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, n_methods)

    for i, method in enumerate(METHODS):
        means = [np.mean(perf[n][method]) if perf[n][method] else 0 for n in NEURON_SIZES]
        stds  = [np.std(perf[n][method])  if perf[n][method] else 0 for n in NEURON_SIZES]
        bars = ax.bar(x + offsets[i] * width, means, width, yerr=stds,
                      label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                      alpha=0.87, capsize=5, ecolor='black', error_kw={'elinewidth': 1.5})
        # Annotate bar tops
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f'{mean:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} neurons' for n in NEURON_SIZES], fontsize=12)
    ax.set_ylabel('Best Accuracy (%)', fontsize=13)
    ax.set_title('Robot Arm T20 — Performance by Neuron Count\n(mean ± std, 3 seeds)', fontsize=14)
    ax.set_ylim(0, 110)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.35, label='50% baseline')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_performance_bars.png'), dpi=300)
    plt.close()
    print("  Saved fig1_performance_bars.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — Learning curves (3 panels × neuron count, mean ± std shading)
# ──────────────────────────────────────────────────────────────────────────────
def fig2_learning_curves(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for idx, n in enumerate(NEURON_SIZES):
        ax = axes[idx]
        for method in METHODS:
            curves = []
            for seed in SEEDS:
                if seed not in data.get(n, {}) or method not in data[n][seed]:
                    continue
                h = data[n][seed][method].get('history', {})
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
            std_c  = aligned.std(axis=0)
            xvals  = np.arange(min_len)
            ax.plot(xvals, mean_c, linewidth=2.2, label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method])
            ax.fill_between(xvals, np.maximum(mean_c - std_c, 0),
                            np.minimum(mean_c + std_c, 105),
                            alpha=0.15, color=METHOD_COLORS[method])

        ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
        ax.set_title(f'{n} neurons', fontsize=13)
        ax.set_xlabel('Iteration / Generation', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Best Accuracy (%)', fontsize=12)
        ax.set_ylim(0, 108)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=10, loc='lower right')
    fig.suptitle('Robot Arm T20 — Learning Curves (mean ± std, 3 seeds)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_learning_curves.png'), dpi=300)
    plt.close()
    print("  Saved fig2_learning_curves.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Per-seed performance heatmap
# ──────────────────────────────────────────────────────────────────────────────
def fig3_seed_heatmap(data, out_dir):
    n_methods = len(METHODS)
    n_neurons  = len(NEURON_SIZES)
    n_seeds    = len(SEEDS)

    fig, axes = plt.subplots(1, n_methods, figsize=(4.5 * n_methods, 4.5))
    for mi, method in enumerate(METHODS):
        ax = axes[mi]
        mat = np.full((n_neurons, n_seeds), np.nan)
        for ni, n in enumerate(NEURON_SIZES):
            for si, seed in enumerate(SEEDS):
                if seed in data.get(n, {}) and method in data[n][seed]:
                    mat[ni, si] = best_accuracy(data[n][seed][method], method)
        im = ax.imshow(mat, vmin=50, vmax=100, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(n_seeds))
        ax.set_xticklabels([f'seed\n{s}' for s in SEEDS], fontsize=10)
        ax.set_yticks(range(n_neurons))
        ax.set_yticklabels([f'{n}N' for n in NEURON_SIZES], fontsize=10)
        ax.set_title(METHOD_LABELS[method], fontsize=13, fontweight='bold')
        for ni in range(n_neurons):
            for si in range(n_seeds):
                if not np.isnan(mat[ni, si]):
                    ax.text(si, ni, f'{mat[ni, si]:.1f}', ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='black' if mat[ni, si] < 80 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Accuracy (%)')

    fig.suptitle('Robot Arm T20 — Per-Seed Best Accuracy (%)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_seed_heatmap.png'), dpi=300)
    plt.close()
    print("  Saved fig3_seed_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Weight change fractions (stacked bars by neuron count)
# ──────────────────────────────────────────────────────────────────────────────
def fig4_weight_fractions(conn, out_dir):
    fig, axes = plt.subplots(1, len(NEURON_SIZES), figsize=(16, 5.5), sharey=True)
    for idx, n in enumerate(NEURON_SIZES):
        ax = axes[idx]
        for mi, method in enumerate(METHODS):
            c = conn[n][method]
            if not c['frac_in']:
                continue
            fi_m = np.mean(c['frac_in'])
            fr_m = np.mean(c['frac_rec'])
            fo_m = np.mean(c['frac_out'])
            fi_s = np.std(c['frac_in'])
            fr_s = np.std(c['frac_rec'])
            fo_s = np.std(c['frac_out'])
            ax.bar(mi, fi_m, 0.7, color=LAYER_COLORS['W_in'], alpha=0.85)
            ax.bar(mi, fr_m, 0.7, bottom=fi_m, color=LAYER_COLORS['W_rec'], alpha=0.85)
            ax.bar(mi, fo_m, 0.7, bottom=fi_m + fr_m, color=LAYER_COLORS['W_out'], alpha=0.85)
            ax.errorbar(mi, fi_m / 2, yerr=fi_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.2)
            ax.errorbar(mi, fi_m + fr_m / 2, yerr=fr_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.2)
            ax.errorbar(mi, fi_m + fr_m + fo_m / 2, yerr=fo_s, fmt='none', ecolor='black', capsize=3, elinewidth=1.2)
            ax.text(mi, fi_m / 2, f'{fi_m*100:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
            ax.text(mi, fi_m + fr_m / 2, f'{fr_m*100:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
            ax.text(mi, fi_m + fr_m + fo_m / 2, f'{fo_m*100:.0f}%', ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, fontsize=10)
        ax.set_title(f'{n} neurons', fontsize=13)
        if idx == 0:
            ax.set_ylabel('Fraction of total ||ΔW||', fontsize=12)

    legend_elements = [Patch(facecolor=LAYER_COLORS['W_in'],  label='W_in'),
                       Patch(facecolor=LAYER_COLORS['W_rec'], label='W_rec'),
                       Patch(facecolor=LAYER_COLORS['W_out'], label='W_out')]
    axes[0].legend(handles=legend_elements, fontsize=10, loc='upper left')
    fig.suptitle('Robot Arm T20 — Per-Layer Weight Change Fractions (mean ± std, 3 seeds)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_weight_fractions.png'), dpi=300)
    plt.close()
    print("  Saved fig4_weight_fractions.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Effective rank of W_rec by neuron count
# ──────────────────────────────────────────────────────────────────────────────
def fig5_effective_rank(conn, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(NEURON_SIZES))
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(METHODS))
    for i, method in enumerate(METHODS):
        means = [np.mean(conn[n][method]['eff_rank']) if conn[n][method]['eff_rank'] else 0
                 for n in NEURON_SIZES]
        stds  = [np.std(conn[n][method]['eff_rank'])  if conn[n][method]['eff_rank'] else 0
                 for n in NEURON_SIZES]
        ax.bar(x + offsets[i] * width, means, width, yerr=stds,
               label=METHOD_LABELS[method], color=METHOD_COLORS[method],
               alpha=0.87, capsize=5, ecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} neurons' for n in NEURON_SIZES], fontsize=12)
    ax.set_ylabel('Effective Rank of W_rec (90% variance)', fontsize=12)
    ax.set_title('Robot Arm T20 — Effective Rank of W_rec\n(mean ± std, 3 seeds)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig5_effective_rank.png'), dpi=300)
    plt.close()
    print("  Saved fig5_effective_rank.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Total ΔW norm vs neuron count
# ──────────────────────────────────────────────────────────────────────────────
def fig6_total_norm(conn, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHODS:
        means = [np.mean(conn[n][method]['norm_total']) if conn[n][method]['norm_total'] else 0
                 for n in NEURON_SIZES]
        stds  = [np.std(conn[n][method]['norm_total'])  if conn[n][method]['norm_total'] else 0
                 for n in NEURON_SIZES]
        ax.errorbar(NEURON_SIZES, means, yerr=stds, marker='o', linewidth=2.5,
                    markersize=9, capsize=6, capthick=2,
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xlabel('Network Size (neurons)', fontsize=13)
    ax.set_xscale('log', base=2)
    ax.set_xticks(NEURON_SIZES)
    ax.set_xticklabels(NEURON_SIZES)
    ax.set_ylabel('Total ||ΔW|| (Frobenius)', fontsize=13)
    ax.set_title('Robot Arm T20 — Total Weight Change Magnitude\n(mean ± std, 3 seeds)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig6_total_norm.png'), dpi=300)
    plt.close()
    print("  Saved fig6_total_norm.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7 — GA mean-sigma over generations (32N only, all seeds)
# ──────────────────────────────────────────────────────────────────────────────
def fig7_ga_sigma_evolution(data, out_dir):
    fig, axes = plt.subplots(1, len(NEURON_SIZES), figsize=(18, 5), sharey=False)
    sigma_methods = ['ga', 'ga_oja']
    sigma_labels  = {'ga': 'GA', 'ga_oja': 'GA+Oja'}
    sigma_colors  = {'ga': METHOD_COLORS['ga'], 'ga_oja': METHOD_COLORS['ga_oja']}
    linestyles    = {'ga': '-', 'ga_oja': '--'}

    for idx, n in enumerate(NEURON_SIZES):
        ax = axes[idx]
        for method in sigma_methods:
            curves = []
            for seed in SEEDS:
                if seed not in data.get(n, {}) or method not in data[n][seed]:
                    continue
                h = data[n][seed][method].get('history', {})
                sig = h.get('mean_sigma', [])
                if sig:
                    curves.append(np.array(sig))
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            aligned = np.array([c[:min_len] for c in curves])
            mean_c = aligned.mean(axis=0)
            std_c  = aligned.std(axis=0)
            xvals  = np.arange(min_len)
            ax.plot(xvals, mean_c, linewidth=2.2, linestyle=linestyles[method],
                    label=sigma_labels[method], color=sigma_colors[method])
            ax.fill_between(xvals, mean_c - std_c, mean_c + std_c,
                            alpha=0.15, color=sigma_colors[method])

        ax.set_title(f'{n} neurons', fontsize=13)
        ax.set_xlabel('Generation', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Mean Sigma (mutation std)', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10)

    fig.suptitle('Robot Arm T20 — GA Mutation Sigma Evolution (mean ± std, 3 seeds)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig7_ga_sigma_evolution.png'), dpi=300)
    plt.close()
    print("  Saved fig7_ga_sigma_evolution.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8 — Accuracy vs neurons (scaling plot)
# ──────────────────────────────────────────────────────────────────────────────
def fig8_scaling(perf, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHODS:
        means = [np.mean(perf[n][method]) if perf[n][method] else np.nan for n in NEURON_SIZES]
        stds  = [np.std(perf[n][method])  if perf[n][method] else 0       for n in NEURON_SIZES]
        ax.errorbar(NEURON_SIZES, means, yerr=stds, marker='o', linewidth=2.5,
                    markersize=9, capsize=6, capthick=2,
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_xscale('log', base=2)
    ax.set_xticks(NEURON_SIZES)
    ax.set_xticklabels(NEURON_SIZES)
    ax.set_xlabel('Network Size (neurons)', fontsize=13)
    ax.set_ylabel('Best Accuracy (%)', fontsize=13)
    ax.set_title('Robot Arm T20 — Accuracy Scaling with Network Size\n(mean ± std, 3 seeds)', fontsize=14)
    ax.set_ylim(50, 103)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.35)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig8_scaling.png'), dpi=300)
    plt.close()
    print("  Saved fig8_scaling.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 9 — Oja delta weight heatmap (GA_oja final vs post_oja for each seed)
# ──────────────────────────────────────────────────────────────────────────────
def fig9_oja_delta_heatmaps(data, out_dir):
    # Show W_rec delta introduced by Oja for each (neuron_size, seed) combination
    fig, axes = plt.subplots(len(NEURON_SIZES), len(SEEDS),
                              figsize=(4.5 * len(SEEDS), 4.0 * len(NEURON_SIZES)))
    vmax_global = 0
    deltas = {}
    for ni, n in enumerate(NEURON_SIZES):
        deltas[n] = {}
        for si, seed in enumerate(SEEDS):
            if seed not in data.get(n, {}) or 'ga_oja' not in data[n][seed]:
                deltas[n][seed] = None
                continue
            entry = data[n][seed]['ga_oja']
            if 'weights_post_oja' not in entry or 'weights_final' not in entry:
                deltas[n][seed] = None
                continue
            d = entry['weights_post_oja']['W_rec'] - entry['weights_final']['W_rec']
            deltas[n][seed] = d
            vmax_global = max(vmax_global, np.abs(d).max())

    vmax = vmax_global if vmax_global > 0 else 1.0
    for ni, n in enumerate(NEURON_SIZES):
        for si, seed in enumerate(SEEDS):
            ax = axes[ni, si]
            d = deltas[n][seed]
            if d is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14,
                        transform=ax.transAxes)
                ax.set_axis_off()
            else:
                im = ax.imshow(d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if ni == 0:
                ax.set_title(f'seed {seed}', fontsize=12)
            if si == 0:
                ax.set_ylabel(f'{n} neurons', fontsize=11)

    fig.suptitle('Robot Arm T20 — Oja Rule ΔW_rec (post_oja − final_weights)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig9_oja_delta_heatmaps.png'), dpi=300)
    plt.close()
    print("  Saved fig9_oja_delta_heatmaps.png")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 10 — Absolute ΔW norm per layer (W_in / W_rec / W_out separately)
# ──────────────────────────────────────────────────────────────────────────────
def fig10_per_layer_absolute_norms(conn, out_dir):
    """3×1 panel: one row per layer, x-axis = neuron count, lines per method."""
    layers = [('norm_in', 'W_in'), ('norm_rec', 'W_rec'), ('norm_out', 'W_out')]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, (key, label) in zip(axes, layers):
        for method in METHODS:
            means, stds = [], []
            for n in NEURON_SIZES:
                vals = conn[n][method].get(key, [])
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals)  if vals else 0)
            ax.errorbar(NEURON_SIZES, means, yerr=stds, marker='o', linewidth=2.5,
                        markersize=9, capsize=6, capthick=2,
                        label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_xscale('log', base=2)
        ax.set_xticks(NEURON_SIZES)
        ax.set_xticklabels(NEURON_SIZES)
        ax.set_xlabel('Network Size (neurons)', fontsize=12)
        ax.set_ylabel(f'||Δ{label}|| (Frobenius)', fontsize=12)
        ax.set_title(f'{label} absolute weight change', fontsize=13)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10)

    fig.suptitle('Robot Arm T20 — Per-Layer Absolute Weight Change Magnitude\n(mean ± std, 3 seeds)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig10_per_layer_absolute_norms.png'), dpi=300)
    plt.close()
    print("  Saved fig10_per_layer_absolute_norms.png")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
def print_summary_table(perf, conn):
    print("\n" + "=" * 80)
    print("PERFORMANCE TABLE — Robot Arm T20 (best individual accuracy, mean ± std, 3 seeds)")
    print("=" * 80)
    header = f"{'Neurons':>9} | {'BPTT':>12} | {'ES':>12} | {'GA':>12} | {'GA+Oja':>12}"
    print(header)
    print("-" * 65)
    for n in NEURON_SIZES:
        parts = []
        for m in METHODS:
            v = perf[n][m]
            if v:
                parts.append(f"{np.mean(v):>5.1f} ± {np.std(v):>4.1f}")
            else:
                parts.append(f"{'—':>12}")
        print(f"{n:>9} | {'  |  '.join(parts)}")

    print("\n" + "=" * 80)
    print("CONNECTIVITY TABLE — W_rec Effective Rank (mean ± std)")
    print("=" * 80)
    print(f"{'Neurons':>9} | {'BPTT':>12} | {'ES':>12} | {'GA':>12} | {'GA+Oja':>12}")
    print("-" * 65)
    for n in NEURON_SIZES:
        parts = []
        for m in METHODS:
            v = conn[n][m]['eff_rank']
            if v:
                parts.append(f"{np.mean(v):>5.1f} ± {np.std(v):>4.1f}")
            else:
                parts.append(f"{'—':>12}")
        print(f"{n:>9} | {'  |  '.join(parts)}")

    print("\n" + "=" * 80)
    print("WEIGHT CHANGE FRACTIONS — W_rec fraction (mean ± std)")
    print("=" * 80)
    print(f"{'Neurons':>9} | {'BPTT':>12} | {'ES':>12} | {'GA':>12} | {'GA+Oja':>12}")
    print("-" * 65)
    for n in NEURON_SIZES:
        parts = []
        for m in METHODS:
            v = conn[n][m]['frac_rec']
            if v:
                parts.append(f"{np.mean(v)*100:>5.1f} ± {np.std(v)*100:>4.1f}")
            else:
                parts.append(f"{'—':>12}")
        print(f"{n:>9} | {'  |  '.join(parts)}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/', help='Root results directory')
    parser.add_argument('--out-dir', default='results/robot_T20_analysis/',
                        help='Output directory for figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data from", args.results_dir)
    data = load_all(args.results_dir)

    print("Computing metrics...")
    perf, conn = compute_metrics(data)

    print_summary_table(perf, conn)

    print("\nGenerating figures...")
    fig1_performance_bars(perf, args.out_dir)
    fig2_learning_curves(data, args.out_dir)
    fig3_seed_heatmap(data, args.out_dir)
    fig4_weight_fractions(conn, args.out_dir)
    fig5_effective_rank(conn, args.out_dir)
    fig6_total_norm(conn, args.out_dir)
    fig7_ga_sigma_evolution(data, args.out_dir)
    fig8_scaling(perf, args.out_dir)
    fig9_oja_delta_heatmaps(data, args.out_dir)
    fig10_per_layer_absolute_norms(conn, args.out_dir)

    print(f"\nAll figures saved to {args.out_dir}")
