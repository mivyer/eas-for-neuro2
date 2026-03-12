#!/usr/bin/env python3
# scripts/analyze_large_n.py
"""
Focused analysis for 128- and 256-neuron experiments at n-back 4, 5, 6.

Generates (per experiment directory):
  fig1_weight_change_fractions.png   — per-layer ΔW fractions (stacked bar)
  fig2_delta_distributions.png       — ΔW violin plots
  fig3_delta_rec_heatmaps.png        — ΔW_rec heatmaps
  fig4_final_weights.png             — final weight violins
  fig5_sparsity.png                  — sparsity comparison

Plus cross-condition summary figures saved to results/large_n_analysis/:
  figA_layer_norms.png               — ||ΔW|| per layer × condition × method
  figB_sv_spectra.png                — singular value spectra (all layers, init vs final)
  figC_eff_rank.png                  — effective rank (90% variance) for all layers
  figD_complexity_heatmap.png        — matrix complexity heatmap across conditions

Usage:
    python scripts/analyze_large_n.py
    python scripts/analyze_large_n.py --results-root results/ --out results/large_n_analysis
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

# ── Style ─────────────────────────────────────────────────────────────────────

METHOD_ORDER  = ['bptt', 'es', 'ga', 'ga_oja']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA-Oja'}
METHOD_COLORS = {'bptt': '#2166ac', 'es': '#d6604d', 'ga': '#4dac26', 'ga_oja': '#9970ab'}
LAYER_NAMES   = ['W_in', 'W_rec', 'W_out']
LAYER_COLORS  = ['#4393c3', '#f4a582', '#74c476']
SPARSITY_THR  = 0.01
MAX_VIOLIN_PTS = 30_000

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       100,
    'savefig.dpi':      300,
})

# Target experiment directories (relative to results root)
TARGET_DIRS = [
    'nback4_neurons128_seed42',
    'nback4_neurons256_seed42',
    'nback5_neurons128_seed42',
    'nback5_neurons256_seed42',
    'nback6_neurons128_seed42',
    'nback6_neurons256_seed42',
]

CONDITION_LABELS = {
    'nback4_neurons128_seed42': '4-back / 128n',
    'nback4_neurons256_seed42': '4-back / 256n',
    'nback5_neurons128_seed42': '5-back / 128n',
    'nback5_neurons256_seed42': '5-back / 256n',
    'nback6_neurons128_seed42': '6-back / 128n',
    'nback6_neurons256_seed42': '6-back / 256n',
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_exp(exp_dir: str) -> dict:
    """Load weights + history for all available methods in an experiment dir."""
    results = {}
    for method in METHOD_ORDER:
        mdir = os.path.join(exp_dir, method)
        if not os.path.isdir(mdir):
            continue
        m = {}
        for label in ('init', 'final'):
            npz = os.path.join(mdir, f'weights_{label}.npz')
            if os.path.exists(npz):
                d = np.load(npz)
                for k in d.files:
                    m[f'{k}_{label}'] = d[k]
        hist = os.path.join(mdir, 'history.json')
        if os.path.exists(hist):
            with open(hist) as f:
                m['history'] = json.load(f)
        required = ('W_rec_init','W_in_init','W_out_init',
                    'W_rec_final','W_in_final','W_out_final')
        if all(k in m for k in required):
            results[method] = m
    return results


def get_methods(results: dict) -> list:
    return [m for m in METHOD_ORDER if m in results]


# ── Metrics ───────────────────────────────────────────────────────────────────

def eff_rank(W: np.ndarray, var_threshold: float = 0.90) -> int:
    sv = np.linalg.svd(W, compute_uv=False)
    cumvar = np.cumsum(sv**2) / (sv**2).sum()
    return int(np.searchsorted(cumvar, var_threshold)) + 1


def condition_number(W: np.ndarray) -> float:
    sv = np.linalg.svd(W, compute_uv=False)
    return float(sv[0] / (sv[-1] + 1e-12))


def nuclear_norm(W: np.ndarray) -> float:
    sv = np.linalg.svd(W, compute_uv=False)
    return float(sv.sum())


def compute_metrics(r: dict) -> dict:
    dW_in  = r['W_in_final']  - r['W_in_init']
    dW_rec = r['W_rec_final'] - r['W_rec_init']
    dW_out = r['W_out_final'] - r['W_out_init']

    norm_in  = float(np.linalg.norm(dW_in))
    norm_rec = float(np.linalg.norm(dW_rec))
    norm_out = float(np.linalg.norm(dW_out))
    total    = norm_in + norm_rec + norm_out + 1e-12

    sp = lambda W: float((np.abs(W) < SPARSITY_THR).mean())

    return {
        'norm_dW_in':  norm_in,
        'norm_dW_rec': norm_rec,
        'norm_dW_out': norm_out,
        'frac_in':  norm_in  / total,
        'frac_rec': norm_rec / total,
        'frac_out': norm_out / total,
        'sparsity_in':  sp(r['W_in_final']),
        'sparsity_rec': sp(r['W_rec_final']),
        'sparsity_out': sp(r['W_out_final']),
        'eff_rank_in_init':   eff_rank(r['W_in_init']),
        'eff_rank_rec_init':  eff_rank(r['W_rec_init']),
        'eff_rank_out_init':  eff_rank(r['W_out_init']),
        'eff_rank_in_final':  eff_rank(r['W_in_final']),
        'eff_rank_rec_final': eff_rank(r['W_rec_final']),
        'eff_rank_out_final': eff_rank(r['W_out_final']),
        'cond_in_init':   condition_number(r['W_in_init']),
        'cond_rec_init':  condition_number(r['W_rec_init']),
        'cond_out_init':  condition_number(r['W_out_init']),
        'cond_in_final':  condition_number(r['W_in_final']),
        'cond_rec_final': condition_number(r['W_rec_final']),
        'cond_out_final': condition_number(r['W_out_final']),
        'nuc_dW_in':  nuclear_norm(dW_in),
        'nuc_dW_rec': nuclear_norm(dW_rec),
        'nuc_dW_out': nuclear_norm(dW_out),
        'dW_in':  dW_in,
        'dW_rec': dW_rec,
        'dW_out': dW_out,
        'sv_in_init':   np.linalg.svd(r['W_in_init'],  compute_uv=False),
        'sv_in_final':  np.linalg.svd(r['W_in_final'], compute_uv=False),
        'sv_rec_init':  np.linalg.svd(r['W_rec_init'], compute_uv=False),
        'sv_rec_final': np.linalg.svd(r['W_rec_final'],compute_uv=False),
        'sv_out_init':  np.linalg.svd(r['W_out_init'], compute_uv=False),
        'sv_out_final': np.linalg.svd(r['W_out_final'],compute_uv=False),
        'acc_final': r.get('history', {}).get('accuracy', [None])[-1],
    }


# ── Violin helper ─────────────────────────────────────────────────────────────

def _sub(arr, seed=0):
    flat = arr.ravel()
    if len(flat) > MAX_VIOLIN_PTS:
        rng = np.random.default_rng(seed)
        flat = rng.choice(flat, MAX_VIOLIN_PTS, replace=False)
    return flat


def _violin(ax, data_dict, palette, title, ylabel=''):
    labels = list(data_dict.keys())
    arrays = list(data_dict.values())
    if _SEABORN:
        sns.violinplot(data=data_dict, order=labels, palette=palette,
                       inner='box', bw_adjust=0.8, cut=2,
                       linewidth=0.8, ax=ax)
    else:
        parts = ax.violinplot(arrays, positions=range(len(labels)),
                               showmedians=True, showextrema=False)
        for pc, lbl in zip(parts['bodies'], labels):
            pc.set_facecolor(palette[lbl])
            pc.set_alpha(0.75)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


# ── Per-experiment figures (same as analyze_connectivity) ─────────────────────

def fig1_fractions(metrics, methods, fig_dir, title_suffix=''):
    fig, ax = plt.subplots(figsize=(max(4, 1.5*len(methods)), 4))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    for key, color, name in zip(['frac_in','frac_rec','frac_out'],
                                  LAYER_COLORS, LAYER_NAMES):
        vals = np.array([metrics[m][key] for m in methods])
        ax.bar(x, vals, bottom=bottom, color=color, label=name,
               width=0.55, edgecolor='white', lw=0.5)
        for xi,(v,b) in enumerate(zip(vals, bottom)):
            if v > 0.06:
                ax.text(xi, b+v/2, f'{v:.0%}', ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel('Fraction of total weight change')
    ax.set_title(f'Per-layer weight change fractions{title_suffix}')
    ax.set_ylim(0, 1.10)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig1_weight_change_fractions.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig2_delta_violins(metrics, methods, fig_dir, title_suffix=''):
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in methods}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, [('dW_in','ΔW_in'),
                                         ('dW_rec','ΔW_rec'),
                                         ('dW_out','ΔW_out')]):
        _violin(ax, {METHOD_LABELS[m]: _sub(metrics[m][key]) for m in methods},
                palette, title, ylabel='Weight change' if ax is axes[0] else '')
    fig.suptitle(f'ΔW distributions{title_suffix}', y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig2_delta_distributions.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig3_rec_heatmaps(results, metrics, methods, fig_dir, title_suffix=''):
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4.2*n, 4.2), squeeze=False)
    axes = axes[0]
    all_vals = np.concatenate([metrics[m]['dW_rec'].ravel() for m in methods])
    vmax = float(np.percentile(np.abs(all_vals), 99)) or 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for ax, m in zip(axes, methods):
        im = ax.imshow(metrics[m]['dW_rec'], cmap='RdBu_r', norm=norm,
                       aspect='equal', origin='upper')
        ax.set_title(f'{METHOD_LABELS[m]}  ΔW_rec')
        ax.set_xlabel('From  j'); ax.set_ylabel('To  i')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ΔW')
    fig.suptitle(f'Recurrent weight changes{title_suffix}', y=1.06)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig3_delta_rec_heatmaps.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig4_final_violins(results, methods, fig_dir, title_suffix=''):
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in methods}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, [('W_in_final','W_in (final)'),
                                         ('W_rec_final','W_rec (final)'),
                                         ('W_out_final','W_out (final)')]):
        _violin(ax, {METHOD_LABELS[m]: _sub(results[m][key]) for m in methods},
                palette, title, ylabel='Weight' if ax is axes[0] else '')
    fig.suptitle(f'Final weight distributions{title_suffix}', y=1.02)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig4_final_weights.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig5_sparsity(metrics, methods, fig_dir, title_suffix=''):
    fig, ax = plt.subplots(figsize=(max(4, 1.8*len(methods)), 4))
    x = np.arange(3); n = len(methods); w = 0.72/n
    for i, m in enumerate(methods):
        mx = metrics[m]
        vals = [mx['sparsity_in'], mx['sparsity_rec'], mx['sparsity_out']]
        ax.bar(x + (i - n/2 + 0.5)*w, vals, width=w*0.9,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               edgecolor='white', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_NAMES)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax.set_ylabel(f'Sparsity  (|w| < {SPARSITY_THR})')
    ax.set_title(f'Per-layer sparsity{title_suffix}')
    ax.set_ylim(0, 1.05); ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig5_sparsity.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig6_sv_spectra(results, metrics, methods, fig_dir, title_suffix=''):
    """Singular value spectra for W_in, W_rec, W_out — init vs final."""
    layers = [
        ('sv_in_init',  'sv_in_final',  'W_in'),
        ('sv_rec_init', 'sv_rec_final', 'W_rec'),
        ('sv_out_init', 'sv_out_final', 'W_out'),
    ]
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 3,
                              figsize=(13, 3.0*n_methods),
                              squeeze=False)
    for row, m in enumerate(methods):
        mx = metrics[m]
        for col, (ki, kf, lname) in enumerate(layers):
            ax = axes[row, col]
            sv_i = mx[ki]; sv_f = mx[kf]
            # Normalize by first singular value for shape comparison
            ax.plot(sv_i / sv_i[0], color='#999999', lw=1.2,
                    label='init', linestyle='--')
            ax.plot(sv_f / sv_f[0], color=METHOD_COLORS[m], lw=1.5,
                    label='final')
            ax.set_yscale('log')
            ax.set_xlabel('Singular value index')
            if col == 0:
                ax.set_ylabel(f'{METHOD_LABELS[m]}\nNorm. σ (log)')
            if row == 0:
                ax.set_title(lname)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f'Singular value spectra (normalized){title_suffix}',
                 y=1.01, fontsize=12)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig6_sv_spectra.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig7_effective_rank(metrics, methods, fig_dir, title_suffix=''):
    """Effective rank (90% variance) for all three layers, init vs final."""
    layers = [('in', 'W_in'), ('rec', 'W_rec'), ('out', 'W_out')]
    x = np.arange(len(methods))
    n_layers = len(layers)
    w = 0.7 / n_layers

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for col, (lkey, lname) in enumerate(layers):
        ax = axes[col]
        for i, stage in enumerate(['init', 'final']):
            vals = [metrics[m][f'eff_rank_{lkey}_{stage}'] for m in methods]
            offset = (i - 0.5) * w
            color = '#aaaaaa' if stage == 'init' else [METHOD_COLORS[m] for m in methods]
            if isinstance(color, list):
                for xi, (v, c) in enumerate(zip(vals, color)):
                    ax.bar(xi + offset, v, width=w*0.9, color=c,
                           edgecolor='white', lw=0.5,
                           label=METHOD_LABELS[methods[xi]] if col == 0 and stage == 'final' else '')
            else:
                ax.bar(x + offset, vals, width=w*0.9, color=color,
                       edgecolor='white', lw=0.5,
                       label='init' if col == 0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_title(f'Eff. rank — {lname}')
        ax.set_ylabel('Singular values for 90% variance')
    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color='#aaaaaa', label='init')] + \
              [Patch(color=METHOD_COLORS[m], label=f'{METHOD_LABELS[m]} final') for m in methods]
    axes[1].legend(handles=handles, fontsize=8, framealpha=0.8,
                   bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=len(methods)+1)
    fig.suptitle(f'Effective rank (90% variance){title_suffix}', fontsize=12)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig7_effective_rank.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def fig8_condition_numbers(metrics, methods, fig_dir, title_suffix=''):
    """Log-scale condition number for all layers, init vs final."""
    layers = [('in', 'W_in'), ('rec', 'W_rec'), ('out', 'W_out')]
    x = np.arange(len(methods))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for col, (lkey, lname) in enumerate(layers):
        ax = axes[col]
        for i, (stage, alpha) in enumerate([('init', 0.45), ('final', 1.0)]):
            vals = [metrics[m][f'cond_{lkey}_{stage}'] for m in methods]
            offset = (i - 0.5) * w
            ax.bar(x + offset, vals, width=w*0.9,
                   color=[METHOD_COLORS[m] for m in methods],
                   edgecolor='white', lw=0.5, alpha=alpha,
                   label=stage if col == 0 else '')
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_title(f'Condition number — {lname}')
        ax.set_ylabel('κ(W)  [log scale]')
    handles = [plt.Rectangle((0,0),1,1, color='#888888', alpha=0.45, label='init'),
               plt.Rectangle((0,0),1,1, color='#888888', alpha=1.0, label='final')]
    axes[1].legend(handles=handles, fontsize=9)
    fig.suptitle(f'Condition numbers{title_suffix}', fontsize=12)
    fig.tight_layout()
    p = os.path.join(fig_dir, 'fig8_condition_numbers.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


# ── Cross-condition summary figures ──────────────────────────────────────────

def figA_layer_norms(all_data: dict, out_dir: str):
    """
    ||ΔW||_F per layer across all 6 conditions × methods.
    3 subplots (one per layer), x=condition, bars=method.
    """
    conditions = list(all_data.keys())
    cond_labels = [CONDITION_LABELS[c] for c in conditions]
    layers = [('norm_dW_in','W_in'), ('norm_dW_rec','W_rec'), ('norm_dW_out','W_out')]

    all_methods = []
    for c in conditions:
        for m in get_methods(all_data[c]['results']):
            if m not in all_methods:
                all_methods.append(m)

    x = np.arange(len(conditions)); n = len(all_methods); w = 0.72/n
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for col, (mkey, lname) in enumerate(layers):
        ax = axes[col]
        for i, m in enumerate(all_methods):
            vals = []
            for c in conditions:
                mx = all_data[c]['metrics'].get(m)
                vals.append(mx[mkey] if mx else np.nan)
            ax.bar(x + (i - n/2 + 0.5)*w, vals, width=w*0.9,
                   color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                   edgecolor='white', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=9)
        ax.set_title(f'‖ΔW‖_F — {lname}')
        ax.set_ylabel('Frobenius norm of ΔW')
        if col == 1:
            ax.legend(fontsize=9, framealpha=0.9)
    fig.suptitle('Per-layer weight change norms across conditions', fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, 'figA_layer_norms.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def figB_eff_rank_summary(all_data: dict, out_dir: str):
    """
    Effective rank for W_in, W_rec, W_out (final) across all conditions × methods.
    Shows delta from init (Δeff_rank = final - init).
    """
    conditions = list(all_data.keys())
    cond_labels = [CONDITION_LABELS[c] for c in conditions]
    layers = [('in','W_in'), ('rec','W_rec'), ('out','W_out')]

    all_methods = []
    for c in conditions:
        for m in get_methods(all_data[c]['results']):
            if m not in all_methods:
                all_methods.append(m)

    x = np.arange(len(conditions)); n = len(all_methods); w = 0.72/n
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, (lkey, lname) in enumerate(layers):
        # Row 0: absolute final eff_rank
        ax0 = axes[0, col]
        # Row 1: delta (final - init)
        ax1 = axes[1, col]
        for i, m in enumerate(all_methods):
            finals = []; deltas = []
            for c in conditions:
                mx = all_data[c]['metrics'].get(m)
                if mx:
                    f = mx[f'eff_rank_{lkey}_final']
                    ini = mx[f'eff_rank_{lkey}_init']
                    finals.append(f); deltas.append(f - ini)
                else:
                    finals.append(np.nan); deltas.append(np.nan)
            offset = (i - n/2 + 0.5)*w
            ax0.bar(x + offset, finals, width=w*0.9, color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m], edgecolor='white', lw=0.5)
            ax1.bar(x + offset, deltas, width=w*0.9, color=METHOD_COLORS[m],
                    edgecolor='white', lw=0.5)
        for ax, ylabel, title in [
            (ax0, 'Eff. rank (final)',      f'Effective rank — {lname}  [final]'),
            (ax1, 'Δ Eff. rank (final−init)', f'Rank change — {lname}'),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, rotation=30, ha='right', fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            if col == 1 and ax is ax1:
                ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        if col == 0:
            ax0.legend(fontsize=9, framealpha=0.9)
    fig.suptitle('Matrix effective rank across conditions (90% variance threshold)',
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(out_dir, 'figB_eff_rank_summary.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def figC_sv_spectra_grid(all_data: dict, out_dir: str):
    """
    Singular value spectra for W_in and W_out (init vs final),
    faceted by condition (rows) × method (columns).
    """
    conditions = list(all_data.keys())
    all_methods = []
    for c in conditions:
        for m in get_methods(all_data[c]['results']):
            if m not in all_methods:
                all_methods.append(m)

    for layer_key, lname in [('in', 'W_in'), ('out', 'W_out')]:
        n_rows = len(conditions); n_cols = len(all_methods)
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.5*n_cols, 2.8*n_rows),
                                  squeeze=False)
        for row, c in enumerate(conditions):
            mx_all = all_data[c]['metrics']
            for col, m in enumerate(all_methods):
                ax = axes[row, col]
                mx = mx_all.get(m)
                if mx is None:
                    ax.set_visible(False); continue
                sv_i = mx[f'sv_{layer_key}_init']
                sv_f = mx[f'sv_{layer_key}_final']
                ax.plot(sv_i, color='#aaaaaa', lw=1.2, ls='--', label='init')
                ax.plot(sv_f, color=METHOD_COLORS[m], lw=1.5, label='final')
                ax.set_yscale('log')
                if row == 0:
                    ax.set_title(METHOD_LABELS[m], fontsize=11)
                if col == 0:
                    ax.set_ylabel(f'{CONDITION_LABELS[c]}\nσ  [log]', fontsize=9)
                ax.set_xlabel('Index', fontsize=8)
                if row == 0 and col == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f'Singular value spectra — {lname}', fontsize=13, y=1.01)
        fig.tight_layout()
        p = os.path.join(out_dir, f'figC_sv_spectra_{lname}.png')
        fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f'  Saved {p}')


def figD_complexity_heatmap(all_data: dict, out_dir: str):
    """
    Heatmap grid: rows = condition, cols = method × metric.
    Metrics: eff_rank_in, eff_rank_rec, eff_rank_out (final),
             cond_in, cond_rec, cond_out (final, log).
    """
    conditions = list(all_data.keys())
    all_methods = []
    for c in conditions:
        for m in get_methods(all_data[c]['results']):
            if m not in all_methods:
                all_methods.append(m)

    # Columns: method × metric
    metric_defs = [
        ('eff_rank_in_final',  'eff_rank\nW_in'),
        ('eff_rank_rec_final', 'eff_rank\nW_rec'),
        ('eff_rank_out_final', 'eff_rank\nW_out'),
        ('cond_in_final',      'κ(W_in)\n[log]'),
        ('cond_rec_final',     'κ(W_rec)\n[log]'),
        ('cond_out_final',     'κ(W_out)\n[log]'),
    ]
    log_metrics = {'cond_in_final','cond_rec_final','cond_out_final'}

    n_rows = len(conditions)
    n_cols = len(all_methods) * len(metric_defs)
    mat = np.full((n_rows, n_cols), np.nan)

    col_labels = []
    for m in all_methods:
        for (mkey, mlabel) in metric_defs:
            col_labels.append(f'{METHOD_LABELS[m]}\n{mlabel}')

    for row, c in enumerate(conditions):
        col = 0
        for m in all_methods:
            mx = all_data[c]['metrics'].get(m)
            for (mkey, _) in metric_defs:
                if mx:
                    v = mx[mkey]
                    if mkey in log_metrics:
                        v = np.log10(v + 1e-12)
                    mat[row, col] = v
                col += 1

    # Normalize each column to [0,1]
    col_min = np.nanmin(mat, axis=0, keepdims=True)
    col_max = np.nanmax(mat, axis=0, keepdims=True)
    mat_norm = (mat - col_min) / (col_max - col_min + 1e-12)

    fig, ax = plt.subplots(figsize=(max(10, 2.0*n_cols), max(4, 1.2*n_rows)))
    im = ax.imshow(mat_norm, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(n_rows)); ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=10)

    # Overlay raw values
    for r in range(n_rows):
        col = 0
        for m in all_methods:
            mx = all_data[c]['metrics'].get(m)  # use last c just for structure
            mx = all_data[conditions[r]]['metrics'].get(m)
            for j, (mkey, _) in enumerate(metric_defs):
                v = mat[r, col]
                if not np.isnan(v):
                    txt = f'{int(v)}' if 'eff_rank' in mkey else f'{10**v:.1f}' if mkey in log_metrics else f'{v:.2f}'
                    # recompute for raw display
                    raw = all_data[conditions[r]]['metrics'].get(m, {}).get(mkey, np.nan) if mx else np.nan
                    if not np.isnan(raw):
                        txt = f'{int(raw)}' if 'eff_rank' in mkey else f'{raw:.1f}'
                    brightness = mat_norm[r, col]
                    fc = 'black' if brightness > 0.4 else 'white'
                    ax.text(col, r, txt, ha='center', va='center', fontsize=7, color=fc)
                col += 1

    # Draw method separators
    for i in range(1, len(all_methods)):
        ax.axvline(i * len(metric_defs) - 0.5, color='k', lw=1.5)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='Normalized value')
    ax.set_title('Matrix complexity metrics across conditions\n(column-normalized; κ in log scale)',
                 fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, 'figD_complexity_heatmap.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


def figE_accuracy_bars(all_data: dict, out_dir: str):
    """Final accuracy bars across all conditions × methods."""
    conditions = list(all_data.keys())
    cond_labels = [CONDITION_LABELS[c] for c in conditions]
    all_methods = []
    for c in conditions:
        for m in get_methods(all_data[c]['results']):
            if m not in all_methods:
                all_methods.append(m)

    x = np.arange(len(conditions)); n = len(all_methods); w = 0.72/n
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, m in enumerate(all_methods):
        vals = []
        for c in conditions:
            acc = all_data[c]['metrics'].get(m, {}).get('acc_final')
            vals.append(acc if acc is not None else np.nan)
        ax.bar(x + (i - n/2 + 0.5)*w, vals, width=w*0.9,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               edgecolor='white', lw=0.5)
    ax.axhline(0.2, color='k', lw=0.8, ls=':', alpha=0.5, label='Chance (20%)')
    ax.set_xticks(x); ax.set_xticklabels(cond_labels, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
    ax.set_ylabel('Final accuracy')
    ax.set_title('Final accuracy across conditions (128n & 256n, n-back 4–6)')
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9, framealpha=0.9, ncol=len(all_methods))
    fig.tight_layout()
    p = os.path.join(out_dir, 'figE_accuracy.png')
    fig.savefig(p, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f'  Saved {p}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', default='results',
                        help='Root directory containing experiment subdirectories')
    parser.add_argument('--out', default='results/large_n_analysis',
                        help='Output directory for cross-condition summary figures')
    parser.add_argument('--skip-per-exp', action='store_true',
                        help='Skip regenerating per-experiment figures')
    args = parser.parse_args()

    root = args.results_root
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    all_data = {}  # exp_name → {results, metrics}

    for exp_name in TARGET_DIRS:
        exp_dir = os.path.join(root, exp_name)
        if not os.path.isdir(exp_dir):
            print(f'[skip] {exp_dir} not found')
            continue

        print(f'\n{"="*60}')
        print(f'Processing: {exp_name}')
        print('='*60)

        results = load_exp(exp_dir)
        methods = get_methods(results)
        if not methods:
            print('  No complete weight data found — skipping.')
            continue

        metrics = {m: compute_metrics(results[m]) for m in methods}
        cond_label = CONDITION_LABELS.get(exp_name, exp_name)
        suffix = f'\n{cond_label}'

        all_data[exp_name] = {'results': results, 'metrics': metrics}

        if not args.skip_per_exp:
            fig_dir = os.path.join(exp_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            fig1_fractions(metrics, methods, fig_dir, suffix)
            fig2_delta_violins(metrics, methods, fig_dir, suffix)
            fig3_rec_heatmaps(results, metrics, methods, fig_dir, suffix)
            fig4_final_violins(results, methods, fig_dir, suffix)
            fig5_sparsity(metrics, methods, fig_dir, suffix)
            fig6_sv_spectra(results, metrics, methods, fig_dir, suffix)
            fig7_effective_rank(metrics, methods, fig_dir, suffix)
            fig8_condition_numbers(metrics, methods, fig_dir, suffix)

    if len(all_data) < 2:
        print('\nNot enough experiments loaded for cross-condition figures.')
        return

    print(f'\n{"="*60}')
    print(f'Cross-condition summary → {out_dir}/')
    print('='*60)
    figA_layer_norms(all_data, out_dir)
    figB_eff_rank_summary(all_data, out_dir)
    figC_sv_spectra_grid(all_data, out_dir)
    figD_complexity_heatmap(all_data, out_dir)
    figE_accuracy_bars(all_data, out_dir)

    print(f'\n[done]')


if __name__ == '__main__':
    main()
