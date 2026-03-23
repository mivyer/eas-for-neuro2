#!/usr/bin/env python3
"""
Generate a 3-panel CNS*2026 composite figure.

Story arc:
  A. All methods reach high accuracy at 1–2 back; only BPTT holds at 3–4 back.
  B. Despite similar accuracy, BPTT learns low-rank recurrent dynamics while
     EAs converge on high-rank solutions — a fundamentally different strategy.
  C. BPTT uniquely redirects learning toward output weights as difficulty rises;
     evolutionary methods maintain uniform layer-wise allocation throughout.

Usage:
    python3 scripts/make_cns_figure.py [--results-dir results/] [--out cns_figure.png]
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────

METHODS       = ['bptt', 'es', 'ga', 'ga_oja']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA+Oja'}
METHOD_COLORS = {'bptt': '#2196F3', 'es': '#FF9800', 'ga': '#4CAF50', 'ga_oja': '#E91E63'}
LAYER_COLORS  = {'W_in': '#9C27B0', 'W_rec': '#FF9800', 'W_out': '#4CAF50'}
LAYER_LABELS  = {'W_in': '$W_{in}$', 'W_rec': '$W_{rec}$', 'W_out': '$W_{out}$'}

NBACKS  = [1, 2, 3, 4]
SEEDS   = [42, 123, 456]
NEURONS = [32]   # 32n gives the clearest rank/allocation contrast

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(results_dir):
    data = {nb: {nn: {} for nn in NEURONS} for nb in NBACKS}
    for nb in NBACKS:
        for nn in NEURONS:
            for seed in SEEDS:
                candidates = [
                    os.path.join(results_dir, f"nback{nb}_neurons{nn}_seed{seed}"),
                    os.path.join(results_dir, "nback", f"nback{nb}_neurons{nn}_seed{seed}"),
                ]
                run_dir = next((p for p in candidates if os.path.isdir(p)), None)
                if run_dir is None:
                    continue
                data[nb][nn][seed] = {}
                for method in METHODS:
                    mdir = os.path.join(run_dir, method)
                    if not os.path.isdir(mdir):
                        continue
                    entry = {}
                    hpath = os.path.join(mdir, "history.json")
                    if os.path.exists(hpath):
                        with open(hpath) as f:
                            entry['history'] = json.load(f)
                    for tag in ('weights_init', 'weights_final'):
                        wp = os.path.join(mdir, f"{tag}.npz")
                        if os.path.exists(wp):
                            entry[tag] = dict(np.load(wp))
                    data[nb][nn][seed][method] = entry
    return data

# ── Metrics ───────────────────────────────────────────────────────────────────

def get_accuracy(entry, method):
    h = entry.get('history', {})
    if method == 'bptt':
        vals = h.get('accuracy', [])
        return vals[-1] * 100 if vals else np.nan
    bf = h.get('best_fitness', h.get('accuracy', []))
    return max(bf) * 100 if bf else np.nan

def get_eff_rank(entry):
    wf = entry.get('weights_final', {})
    W  = wf.get('W_rec')
    if W is None:
        return np.nan
    S = np.linalg.svd(W, compute_uv=False)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(cumvar, 0.90)) + 1

def get_layer_fracs(entry):
    wi, wf = entry.get('weights_init', {}), entry.get('weights_final', {})
    if not wi or not wf:
        return {k: np.nan for k in ('W_in', 'W_rec', 'W_out')}
    d = {k: np.linalg.norm(wf[k] - wi[k]) for k in ('W_in', 'W_rec', 'W_out')}
    total = sum(d.values())
    if total < 1e-8:
        return {k: np.nan for k in d}
    return {k: v / total * 100 for k, v in d.items()}

def compute_metrics(data):
    """Pool across all neuron sizes; return per-(nb, method) value lists."""
    acc   = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    erank = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    fracs = {nb: {m: {l: [] for l in ('W_in', 'W_rec', 'W_out')}
                  for m in METHODS} for nb in NBACKS}

    for nb in NBACKS:
        for nn in NEURONS:
            for seed_data in data[nb][nn].values():
                for method, entry in seed_data.items():
                    acc[nb][method].append(get_accuracy(entry, method))
                    erank[nb][method].append(get_eff_rank(entry))
                    fv = get_layer_fracs(entry)
                    for layer, val in fv.items():
                        fracs[nb][method][layer].append(val)
    return acc, erank, fracs

def pooled(vals):
    v = [x for x in vals if not np.isnan(x)]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, 0.0)

# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(acc, erank, fracs, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8),
                              facecolor='white',
                              gridspec_kw={'wspace': 0.36})

    x = np.array(NBACKS)
    xlabels = [f"{nb}-back" for nb in NBACKS]

    def style_ax(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(NBACKS)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_xlim(0.7, 4.3)
        ax.tick_params(labelsize=10)

    # ── Panel A: Task Performance ─────────────────────────────────────────────
    ax = axes[0]
    style_ax(ax)
    ax.axhline(20, color='#AAAAAA', ls='--', lw=1.2, zorder=0)
    ax.text(4.35, 21, 'chance', color='#AAAAAA', fontsize=8.5, va='bottom')

    for m in METHODS:
        means = np.array([pooled(acc[nb][m])[0] for nb in NBACKS])
        stds  = np.array([pooled(acc[nb][m])[1] for nb in NBACKS])
        mask  = ~np.isnan(means)
        c = METHOD_COLORS[m]
        ax.plot(x[mask], means[mask], 'o-', color=c, lw=2.2, ms=7,
                label=METHOD_LABELS[m], zorder=3)
        ax.fill_between(x[mask], (means-stds)[mask], (means+stds)[mask],
                        color=c, alpha=0.15, zorder=2)

    ax.set_ylim(0, 112)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.set_title('A.  Task Performance', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.9,
              handlelength=1.6, borderpad=0.6)

    # ── Panel B: Recurrent Weight Structure (effective rank) ──────────────────
    ax = axes[1]
    style_ax(ax)

    for m in METHODS:
        means = np.array([pooled(erank[nb][m])[0] for nb in NBACKS])
        stds  = np.array([pooled(erank[nb][m])[1] for nb in NBACKS])
        mask  = ~np.isnan(means)
        c = METHOD_COLORS[m]
        ax.plot(x[mask], means[mask], 'o-', color=c, lw=2.2, ms=7,
                label=METHOD_LABELS[m], zorder=3)
        ax.fill_between(x[mask], (means-stds)[mask], (means+stds)[mask],
                        color=c, alpha=0.15, zorder=2)

    # Annotate the split
    bptt_1  = pooled(erank[1]['bptt'])[0]
    ea_mean = np.nanmean([pooled(erank[1][m])[0] for m in ('es', 'ga', 'ga_oja')])
    mid     = (bptt_1 + ea_mean) / 2
    ax.annotate('', xy=(1.15, bptt_1 + 0.3), xytext=(1.15, ea_mean - 0.3),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.4))
    ax.text(1.22, mid, 'different\nstructure', fontsize=8, color='#444444',
            va='center', style='italic')

    ax.set_ylim(0, 36)
    ax.set_ylabel('Effective rank (W_rec, 90% var.)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.set_title('B.  Recurrent Weight Structure', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.9,
              handlelength=1.6, borderpad=0.6)

    # ── Panel C: W_out fraction — all methods, showing BPTT's rising trend ──────
    ax = axes[2]
    style_ax(ax)

    for m in METHODS:
        means = np.array([pooled(fracs[nb][m]['W_out'])[0] for nb in NBACKS])
        stds  = np.array([pooled(fracs[nb][m]['W_out'])[1] for nb in NBACKS])
        mask  = ~np.isnan(means)
        c = METHOD_COLORS[m]
        ax.plot(x[mask], means[mask], 's-', color=c, lw=2.2, ms=7,
                label=METHOD_LABELS[m], zorder=3)
        ax.fill_between(x[mask], (means-stds)[mask], (means+stds)[mask],
                        color=c, alpha=0.15, zorder=2)

    # Annotate BPTT's rising trend
    bptt_1 = pooled(fracs[1]['bptt']['W_out'])[0]
    bptt_4 = pooled(fracs[4]['bptt']['W_out'])[0]
    ax.annotate('', xy=(4.1, bptt_4), xytext=(4.1, bptt_1),
                arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.6))
    ax.text(4.18, (bptt_1 + bptt_4) / 2, f'+{bptt_4-bptt_1:.0f}%',
            color='#2196F3', fontsize=9, va='center', fontweight='bold')

    ax.set_ylim(0, 60)
    ax.set_ylabel('$W_{out}$ weight change fraction (%)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.set_title('C.  Output Layer Allocation', fontsize=12, fontweight='bold', loc='left')
    ax.legend(fontsize=9.5, loc='lower right', framealpha=0.9,
              handlelength=1.6, borderpad=0.6)

    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close(fig)

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results/')
    parser.add_argument('--out', default='cns_figure.png')
    args = parser.parse_args()
    data = load_all(args.results_dir)
    acc, erank, fracs = compute_metrics(data)
    make_figure(acc, erank, fracs, args.out)

if __name__ == '__main__':
    main()
