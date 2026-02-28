#!/usr/bin/env python3
# scripts/analyze_connectivity.py
"""
Connectivity analysis and publication-quality figures for trained RNN models.

Computes per-layer weight-change metrics and generates five figures:
  Fig 1 — Per-layer weight change fractions (stacked bar)
  Fig 2 — ΔW distribution violins (per layer)
  Fig 3 — ΔW_rec heatmaps (per method, with E/I boundary)
  Fig 4 — Final weight distribution violins (per layer)
  Fig 5 — Per-layer sparsity (grouped bar)

Usage:
    # From a saved results directory:
    python scripts/analyze_connectivity.py --from-results results/nback_32n/

    # Called automatically from run_experiment.py after training.
"""

import os
import sys
import json
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import seaborn as sns
    _SEABORN = True
except ImportError:
    _SEABORN = False

# ── Global style ──────────────────────────────────────────────────────────────

METHOD_ORDER  = ['bptt', 'es', 'ga']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA'}
METHOD_COLORS = {'bptt': '#2166ac', 'es': '#d6604d', 'ga': '#4dac26'}
LAYER_NAMES   = ['W_in', 'W_rec', 'W_out']
LAYER_COLORS  = ['#4393c3', '#f4a582', '#74c476']   # blue / orange / green
SPARSITY_THR  = 0.01
MAX_VIOLIN_PTS = 30_000   # subsample large matrices for violin plots

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         100,
    'savefig.dpi':        300,
})


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    """
    Compute weight-change statistics for one trained model.

    Returns a dict with:
      norm_dW_{in,rec,out}   — Frobenius norm of each ΔW
      frac_{in,rec,out}      — fraction of total Frobenius norm
      sparsity_{in,rec,out}  — fraction of final weights with |w| < SPARSITY_THR
      eff_rank_rec           — singular values needed for 90% variance in W_rec_final
      dW_{in,rec,out}        — the ΔW arrays themselves (for plotting)
    """
    dW_in  = result['W_in_final']  - result['W_in_init']
    dW_rec = result['W_rec_final'] - result['W_rec_init']
    dW_out = result['W_out_final'] - result['W_out_init']

    norm_in  = float(np.linalg.norm(dW_in))
    norm_rec = float(np.linalg.norm(dW_rec))
    norm_out = float(np.linalg.norm(dW_out))
    total    = norm_in + norm_rec + norm_out + 1e-12

    sp_in  = float((np.abs(result['W_in_final'])  < SPARSITY_THR).mean())
    sp_rec = float((np.abs(result['W_rec_final']) < SPARSITY_THR).mean())
    sp_out = float((np.abs(result['W_out_final']) < SPARSITY_THR).mean())

    sv = np.linalg.svd(result['W_rec_final'], compute_uv=False)
    cumvar = np.cumsum(sv ** 2) / (sv ** 2).sum()
    eff_rank = int(np.searchsorted(cumvar, 0.90)) + 1

    return {
        'norm_dW_in':  norm_in,
        'norm_dW_rec': norm_rec,
        'norm_dW_out': norm_out,
        'frac_in':     norm_in  / total,
        'frac_rec':    norm_rec / total,
        'frac_out':    norm_out / total,
        'sparsity_in':  sp_in,
        'sparsity_rec': sp_rec,
        'sparsity_out': sp_out,
        'eff_rank_rec': eff_rank,
        'dW_in':  dW_in,
        'dW_rec': dW_rec,
        'dW_out': dW_out,
    }


# ── Text summary ──────────────────────────────────────────────────────────────

def print_summary(metrics: dict, methods: list):
    """Print per-method, per-layer statistics to stdout."""
    sep = "-" * 68
    header = (f"{'Method':>6} | {'Layer':>5} | {'||ΔW||_F':>9} | "
              f"{'Frac%':>6} | {'Sparsity':>8} | {'EffRank':>7}")
    print(f"\n{'=' * 68}")
    print("CONNECTIVITY ANALYSIS SUMMARY")
    print('=' * 68)
    print(header)
    print(sep)
    for m in methods:
        mx = metrics[m]
        label = METHOD_LABELS[m]
        rows = [
            ('W_in',  mx['norm_dW_in'],  mx['frac_in'],  mx['sparsity_in'],  None),
            ('W_rec', mx['norm_dW_rec'], mx['frac_rec'], mx['sparsity_rec'], mx['eff_rank_rec']),
            ('W_out', mx['norm_dW_out'], mx['frac_out'], mx['sparsity_out'], None),
        ]
        for i, (layer, norm, frac, sp, er) in enumerate(rows):
            meth_col = label if i == 0 else ''
            er_str = f'{er:>7d}' if er is not None else f"{'—':>7}"
            print(f"{meth_col:>6} | {layer:>5} | {norm:>9.4f} | "
                  f"{frac:>5.1%} | {sp:>8.1%} | {er_str}")
        print(sep)


# ── Figure helpers ────────────────────────────────────────────────────────────

def _subsample(arr: np.ndarray, seed: int = 0) -> np.ndarray:
    """Flatten and subsample large weight matrices for violin plots."""
    flat = arr.ravel()
    if len(flat) > MAX_VIOLIN_PTS:
        rng = np.random.default_rng(seed)
        flat = rng.choice(flat, MAX_VIOLIN_PTS, replace=False)
    return flat


def _violin_panel(ax, data_dict: dict, palette: dict, title: str,
                  ylabel: str = ''):
    """
    Draw a single violin panel.

    data_dict: {label: 1D array}  ordered dict
    palette:   {label: color}
    """
    labels = list(data_dict.keys())
    arrays = list(data_dict.values())

    if _SEABORN:
        sns.violinplot(
            data=data_dict,
            order=labels,
            palette=palette,
            inner='box',
            bw_adjust=0.8,
            cut=2,
            linewidth=0.8,
            ax=ax,
        )
    else:
        parts = ax.violinplot(arrays, positions=range(len(labels)),
                               showmedians=True, showextrema=False)
        for pc, lbl in zip(parts['bodies'], labels):
            pc.set_facecolor(palette[lbl])
            pc.set_alpha(0.75)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

    ax.axhline(0, color='k', linewidth=0.7, linestyle='--', alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel('')
    if ylabel:
        ax.set_ylabel(ylabel)


# ── Figure 1: Weight-change fractions (stacked bar) ──────────────────────────

def fig1_weight_change_fractions(metrics: dict, methods: list,
                                  fig_dir: str):
    fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(methods)), 4))
    x = np.arange(len(methods))
    labels = [METHOD_LABELS[m] for m in methods]
    bottom = np.zeros(len(methods))

    for key, color, name in zip(
        ['frac_in', 'frac_rec', 'frac_out'], LAYER_COLORS, LAYER_NAMES
    ):
        vals = np.array([metrics[m][key] for m in methods])
        ax.bar(x, vals, bottom=bottom, color=color, label=name,
               width=0.55, edgecolor='white', linewidth=0.5)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.06:
                ax.text(xi, b + v / 2, f'{v:.0%}',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Fraction of total weight change')
    ax.set_title('Per-layer weight change fractions')
    ax.set_ylim(0, 1.10)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9,
              bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    path = os.path.join(fig_dir, 'fig1_weight_change_fractions.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: ΔW distribution violins ────────────────────────────────────────

def fig2_delta_violins(metrics: dict, methods: list, fig_dir: str):
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in methods}
    layers = [
        ('dW_in',  'ΔW_in'),
        ('dW_rec', 'ΔW_rec'),
        ('dW_out', 'ΔW_out'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, layers):
        data_dict = {
            METHOD_LABELS[m]: _subsample(metrics[m][key])
            for m in methods
        }
        _violin_panel(ax, data_dict, palette, title,
                      ylabel='Weight change' if ax is axes[0] else '')

    fig.suptitle('ΔW distributions by layer and method', y=1.02, fontsize=12)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'fig2_delta_distributions.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: ΔW_rec heatmaps ─────────────────────────────────────────────────

def fig3_delta_rec_heatmaps(results: dict, metrics: dict, methods: list,
                              ei_ratio: float, fig_dir: str):
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2),
                              squeeze=False)
    axes = axes[0]

    # Symmetric colorscale: use 99th percentile of |ΔW_rec| across all methods
    all_vals = np.concatenate([metrics[m]['dW_rec'].ravel() for m in methods])
    vmax = float(np.percentile(np.abs(all_vals), 99)) or 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, m in zip(axes, methods):
        dW = metrics[m]['dW_rec']
        N  = dW.shape[0]
        ei_boundary = int(ei_ratio * N)

        im = ax.imshow(dW, cmap='RdBu_r', norm=norm, aspect='equal',
                       origin='upper')

        # E/I boundary lines
        for pos in [ei_boundary - 0.5]:
            ax.axvline(pos, color='k', linewidth=1.5, linestyle='--', alpha=0.7)
            ax.axhline(pos, color='k', linewidth=1.5, linestyle='--', alpha=0.7)

        # E/I region labels (top of axes, in axis coords)
        if ei_boundary > 0:
            ax.text(ei_boundary / (2 * N), 1.03, 'Exc',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8, color='#444')
        if ei_boundary < N:
            ax.text((ei_boundary + N) / (2 * N), 1.03, 'Inh',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8, color='#444')

        ax.set_title(f'{METHOD_LABELS[m]}  ΔW_rec')
        ax.set_xlabel('From neuron  j')
        ax.set_ylabel('To neuron  i')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ΔW')

    fig.suptitle('Recurrent weight changes  (dashed = E/I boundary)', y=1.06)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'fig3_delta_rec_heatmaps.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Final weight distribution violins ───────────────────────────────

def fig4_final_weight_violins(results: dict, methods: list, fig_dir: str):
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in methods}
    layers = [
        ('W_in_final',  'W_in  (final)'),
        ('W_rec_final', 'W_rec  (final)'),
        ('W_out_final', 'W_out  (final)'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, layers):
        data_dict = {
            METHOD_LABELS[m]: _subsample(results[m][key])
            for m in methods
        }
        _violin_panel(ax, data_dict, palette, title,
                      ylabel='Weight value' if ax is axes[0] else '')

    fig.suptitle('Final weight distributions by layer and method', y=1.02)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'fig4_final_weights.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: Sparsity comparison (grouped bar) ───────────────────────────────

def fig5_sparsity(metrics: dict, methods: list, fig_dir: str):
    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(methods)), 4))
    x   = np.arange(3)
    n   = len(methods)
    w   = 0.72 / n

    for i, m in enumerate(methods):
        mx   = metrics[m]
        vals = [mx['sparsity_in'], mx['sparsity_rec'], mx['sparsity_out']]
        offset = (i - n / 2 + 0.5) * w
        ax.bar(x + offset, vals, width=w * 0.9,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0%}')
    )
    ax.set_ylabel(f'Sparsity  (|w| < {SPARSITY_THR})')
    ax.set_title('Per-layer weight sparsity')
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'fig5_sparsity.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze(results: dict, output_dir: str, ei_ratio: float = 0.8):
    """
    Run connectivity analysis on training results and save figures.

    Args:
        results:    dict mapping method name → result dict (from trainers).
                    Each result dict must contain W_{rec,in,out}_{init,final}.
        output_dir: root output directory; figures go into output_dir/figures/.
        ei_ratio:   excitatory fraction (for E/I boundary in heatmaps).
    """
    # Only process methods that have complete weight data
    methods = [
        m for m in METHOD_ORDER
        if m in results
        and results[m] is not None
        and all(k in results[m] for k in
                ('W_rec_init', 'W_in_init', 'W_out_init',
                 'W_rec_final', 'W_in_final', 'W_out_final'))
    ]
    if not methods:
        print("  [analyze] No results with weight data — skipping figures.")
        return

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n[analyze] methods: {[METHOD_LABELS[m] for m in methods]}")
    print(f"[analyze] figures → {fig_dir}/")

    metrics = {m: compute_metrics(results[m]) for m in methods}

    print_summary(metrics, methods)

    fig1_weight_change_fractions(metrics, methods, fig_dir)
    fig2_delta_violins(metrics, methods, fig_dir)
    fig3_delta_rec_heatmaps(results, metrics, methods, ei_ratio, fig_dir)
    fig4_final_weight_violins(results, methods, fig_dir)
    fig5_sparsity(metrics, methods, fig_dir)

    print(f"[analyze] done.")


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_weights(results: dict, output_dir: str):
    """
    Save weight matrices as .npz files for offline analysis.

    Creates {output_dir}/{method}_weights.npz for each available method.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    weight_keys = ('W_rec_init', 'W_in_init', 'W_out_init',
                   'W_rec_final', 'W_in_final', 'W_out_final')
    for method, r in results.items():
        if r is None:
            continue
        arrays = {k: r[k] for k in weight_keys if k in r}
        if len(arrays) == 6:
            path = os.path.join(output_dir, f'{method}_weights.npz')
            np.savez(path, **arrays)
            saved.append(method)
    if saved:
        print(f"  [weights] Saved: {saved}")


def load_results(results_dir: str) -> tuple:
    """
    Load weight matrices and config from a saved experiment directory.

    Supports the current per-method directory layout:
        {results_dir}/{method}/weights_init.npz
        {results_dir}/{method}/weights_final.npz
        {results_dir}/config.json

    Returns:
        results:   dict {method: {weight_key: array}}  (keys: W_rec_init, etc.)
        conf_dict: config dict from config.json (or {})
    """
    from scripts.load_results import load_experiment, to_analyze_format
    exp = load_experiment(results_dir)
    return to_analyze_format(exp), exp.get('config') or {}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Connectivity analysis for trained RNN models'
    )
    p.add_argument(
        '--from-results', type=str, required=True,
        metavar='DIR',
        help='Results directory containing *_weights.npz and summary.json',
    )
    args = p.parse_args()

    results, conf_dict = load_results(args.from_results)
    if not results:
        print(f"No *_weights.npz files found in {args.from_results}")
        sys.exit(1)

    ei_ratio = conf_dict.get('ei_ratio', 0.8)
    analyze(results, args.from_results, ei_ratio=ei_ratio)
