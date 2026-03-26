#!/usr/bin/env python3
"""
Publication-quality multipanel thesis figures.

Generates four figures to results/thesis_figures/:
  figure1_core_finding.png     — 2×2 grid: performance + representation divergence
  figure2_cross_task.png       — 1×3: robot arm generalisation
  figure3_learning_dynamics.png — 1×4: learning curves 1-4 back
  figure4_scaling_challenge.png — heatmaps + ES collapse curve

Usage:
    python3 scripts/make_thesis_figures.py [--results-dir results/] [--out-dir results/thesis_figures/]
"""

import argparse, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

METHODS       = ['bptt', 'es', 'ga', 'ga_oja']
METHOD_LABELS = {'bptt': 'BPTT', 'es': 'ES', 'ga': 'GA', 'ga_oja': 'GA+Oja'}
METHOD_COLORS = {'bptt': '#2196F3', 'es': '#FF9800', 'ga': '#4CAF50', 'ga_oja': '#E91E63'}
LAYER_COLORS  = {'W_in': '#9C27B0', 'W_rec': '#FF9800', 'W_out': '#4CAF50'}
LAYER_LABELS  = {'W_in': '$W_{in}$', 'W_rec': '$W_{rec}$', 'W_out': '$W_{out}$'}
LAYER_KEYS    = ('W_in', 'W_rec', 'W_out')

NBACKS        = [1, 2, 3, 4]
ALL_NBACKS    = [0, 1, 2, 3, 4, 5, 6]
SEEDS         = [42, 123, 456]
NEURON_SIZES  = [32, 64, 128, 256]


def _find_dir(results_dir, task, nb, nn, seed):
    if task == 'nback':
        cands = [
            os.path.join(results_dir, f"nback{nb}_neurons{nn}_seed{seed}"),
            os.path.join(results_dir, 'nback', f"nback{nb}_neurons{nn}_seed{seed}"),
        ]
    else:
        cands = [
            os.path.join(results_dir, f"robot_T20_neurons{nn}_seed{seed}"),
            os.path.join(results_dir, 'robot', f"robot_T20_neurons{nn}_seed{seed}"),
        ]
    return next((p for p in cands if os.path.isdir(p)), None)


def _load_entry(run_dir, method):
    mdir = os.path.join(run_dir, method)
    if not os.path.isdir(mdir):
        return None
    entry = {}
    hpath = os.path.join(mdir, 'history.json')
    if os.path.exists(hpath):
        with open(hpath) as f:
            entry['history'] = json.load(f)
    for tag in ('weights_init', 'weights_final'):
        wp = os.path.join(mdir, f"{tag}.npz")
        if os.path.exists(wp):
            entry[tag] = dict(np.load(wp))
    return entry or None


def load_nback(results_dir, nbacks=NBACKS, neurons=None, seeds=SEEDS):
    if neurons is None:
        neurons = [32]
    data = {nb: {nn: {} for nn in neurons} for nb in nbacks}
    missing = []
    for nb in nbacks:
        for nn in neurons:
            for seed in seeds:
                rd = _find_dir(results_dir, 'nback', nb, nn, seed)
                if rd is None:
                    missing.append(f"nback{nb}_neurons{nn}_seed{seed}")
                    continue
                data[nb][nn][seed] = {}
                for method in METHODS:
                    e = _load_entry(rd, method)
                    if e is not None:
                        data[nb][nn][seed][method] = e
    if missing:
        print(f"  [warn] {len(missing)} nback run dir(s) not found: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))
    return data


def load_robot(results_dir, neurons=None, seeds=SEEDS):
    if neurons is None:
        neurons = NEURON_SIZES
    data = {nn: {} for nn in neurons}
    missing = []
    for nn in neurons:
        for seed in seeds:
            rd = _find_dir(results_dir, 'robot', None, nn, seed)
            if rd is None:
                missing.append(f"robot_T20_neurons{nn}_seed{seed}")
                continue
            data[nn][seed] = {}
            for method in METHODS:
                e = _load_entry(rd, method)
                if e is not None:
                    data[nn][seed][method] = e
    if missing:
        print(f"  [warn] {len(missing)} robot run dir(s) not found: {missing[:5]}"
              + (" ..." if len(missing) > 5 else ""))
    return data


def get_accuracy(entry, method):
    """Best accuracy in [0, 100]; returns NaN if unavailable."""
    h = entry.get('history', {})
    if method == 'bptt':
        for key in ('accuracy', 'reward', 'fitness'):
            vals = h.get(key, [])
            if vals:
                return float(vals[-1]) * 100
        return np.nan
    for key in ('best_fitness', 'accuracy', 'reward', 'fitness'):
        vals = h.get(key, [])
        if vals:
            return float(max(vals)) * 100
    return np.nan


def eff_rank(W):
    """Participation-ratio effective rank: exp(H(p)), p_i = sigma_i / sum(sigma)."""
    S = np.linalg.svd(W, compute_uv=False)
    S = S[S > 1e-12]
    if len(S) == 0:
        return np.nan
    p = S / S.sum()
    H = -np.sum(p * np.log(p + 1e-300))
    return float(np.exp(H))


def layer_fracs(entry):
    """||ΔW||_F per layer as fraction of total; returns dict with NaN on failure."""
    wi = entry.get('weights_init', {})
    wf = entry.get('weights_final', {})
    nan_dict = {k: np.nan for k in LAYER_KEYS}
    if not wi or not wf:
        return nan_dict
    deltas = {}
    for k in LAYER_KEYS:
        if k in wi and k in wf:
            deltas[k] = float(np.linalg.norm(wf[k] - wi[k]))
        else:
            return nan_dict
    total = sum(deltas.values())
    if total < 1e-8:
        return nan_dict
    return {k: v / total * 100 for k, v in deltas.items()}


def pooled(vals):
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return np.nan, 0.0
    return float(np.mean(v)), float(np.std(v))


def nback_param_count(nn):
    """EA gene parameter count for n-back (obs_dim=5, action_dim=5)."""
    return nn * nn + nn * 5 + 5 * nn


def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=10)


def panel_label(ax, label, x=-0.13, y=1.07):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')


def stacked_bar_ax(ax, methods, fracs_by_method, bar_w=0.6, annotate=True):
    for i, m in enumerate(methods):
        fv = fracs_by_method.get(m, {})
        fin  = fv.get('W_in',  np.nan)
        frec = fv.get('W_rec', np.nan)
        fout = fv.get('W_out', np.nan)
        if np.isnan(fin):
            continue
        ax.bar(i, fin,  bar_w, color=LAYER_COLORS['W_in'],  alpha=0.85)
        ax.bar(i, frec, bar_w, bottom=fin,
               color=LAYER_COLORS['W_rec'], alpha=0.85)
        ax.bar(i, fout, bar_w, bottom=fin + frec,
               color=LAYER_COLORS['W_out'], alpha=0.85)
        if annotate:
            for val, base, label in [(fin, 0, 'W_in'),
                                     (frec, fin, 'W_rec'),
                                     (fout, fin + frec, 'W_out')]:
                if val >= 5:
                    ax.text(i, base + val / 2, f'{val:.0f}%',
                            ha='center', va='center',
                            fontsize=8.5, fontweight='bold', color='white')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], fontsize=10)
    ax.set_ylim(0, 105)



def make_figure1(results_dir, out_dir):
    print("Figure 1: The core finding (2×2)...")
    data = load_nback(results_dir, nbacks=NBACKS, neurons=[32])

    acc_vals   = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    erank_vals = {nb: {m: [] for m in METHODS} for nb in NBACKS}
    frac_vals  = {nb: {m: {k: [] for k in LAYER_KEYS} for m in METHODS} for nb in NBACKS}

    for nb in NBACKS:
        for seed_data in data[nb][32].values():
            for method, entry in seed_data.items():
                acc_vals[nb][method].append(get_accuracy(entry, method))
                wf = entry.get('weights_final', {})
                if 'W_rec' in wf:
                    erank_vals[nb][method].append(eff_rank(wf['W_rec']))
                fv = layer_fracs(entry)
                for k in LAYER_KEYS:
                    frac_vals[nb][method][k].append(fv[k])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor='white',
                              gridspec_kw={'hspace': 0.44, 'wspace': 0.34})
    x      = np.array(NBACKS)
    xlbls  = [f"{nb}-back" for nb in NBACKS]

    ax = axes[0, 0]
    clean_ax(ax)
    ax.axhline(20, color='#AAAAAA', ls='--', lw=1.2, zorder=0)
    ax.text(4.38, 21.5, 'chance', color='#999999', fontsize=8.5, va='bottom')
    for m in METHODS:
        mn = np.array([pooled(acc_vals[nb][m])[0] for nb in NBACKS])
        sd = np.array([pooled(acc_vals[nb][m])[1] for nb in NBACKS])
        mask = ~np.isnan(mn)
        c = METHOD_COLORS[m]
        ax.plot(x[mask], mn[mask], 'o-', color=c, lw=2.2, ms=7,
                label=METHOD_LABELS[m], zorder=3)
        ax.fill_between(x[mask], (mn - sd)[mask], (mn + sd)[mask],
                        color=c, alpha=0.15, zorder=2)
    ax.set_ylim(0, 115)
    ax.set_xlim(0.7, 4.5)
    ax.set_xticks(NBACKS); ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.legend(fontsize=9.5, loc='lower left', framealpha=0.9,
              handlelength=1.6, borderpad=0.6)
    panel_label(ax, 'A')

    ax = axes[0, 1]
    clean_ax(ax)
    bar_w = 0.17
    offs  = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w
    for i, m in enumerate(METHODS):
        mn = [pooled(erank_vals[nb][m])[0] for nb in NBACKS]
        sd = [pooled(erank_vals[nb][m])[1] for nb in NBACKS]
        xs = np.arange(len(NBACKS)) + offs[i]
        ax.bar(xs, mn, bar_w, yerr=sd, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], alpha=0.85, capsize=3, ecolor='#444444',
               error_kw={'elinewidth': 1.2})
    ax.set_xticks(np.arange(len(NBACKS)))
    ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel('Effective rank of $W_{rec}$\n(participation ratio)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9, handlelength=1.4)
    panel_label(ax, 'B')

    ax = axes[1, 0]
    clean_ax(ax)
    for k in LAYER_KEYS:
        mn = np.array([pooled(frac_vals[nb]['bptt'][k])[0] for nb in NBACKS])
        sd = np.array([pooled(frac_vals[nb]['bptt'][k])[1] for nb in NBACKS])
        mask = ~np.isnan(mn)
        ax.plot(x[mask], mn[mask], 's-', color=LAYER_COLORS[k], lw=2.2, ms=7,
                label=LAYER_LABELS[k], zorder=3)
        ax.fill_between(x[mask], (mn - sd)[mask], (mn + sd)[mask],
                        color=LAYER_COLORS[k], alpha=0.15, zorder=2)
    ax.set_ylim(0, 62)
    ax.set_xlim(0.7, 4.5)
    ax.set_xticks(NBACKS); ax.set_xticklabels(xlbls, fontsize=10)
    ax.set_ylabel('Fraction of $\\|\\Delta W\\|_F$ (%)', fontsize=11)
    ax.set_xlabel('N-back level', fontsize=11)
    ax.set_title('BPTT only', fontsize=10, color='#666666', style='italic', pad=3)
    ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left')
    panel_label(ax, 'C')

    ax = axes[1, 1]
    clean_ax(ax)
    fracs_4 = {}
    for m in METHODS:
        fracs_4[m] = {k: pooled(frac_vals[4][m][k])[0] for k in LAYER_KEYS}
    stacked_bar_ax(ax, METHODS, fracs_4)
    ax.set_ylabel('Fraction of $\\|\\Delta W\\|_F$ (%)', fontsize=11)
    ax.set_title('4-back: layer allocation per method', fontsize=10,
                 color='#666666', style='italic', pad=3)
    legend_hdl = [Patch(fc=LAYER_COLORS[k], alpha=0.85, label=LAYER_LABELS[k])
                  for k in LAYER_KEYS]
    ax.legend(handles=legend_hdl, fontsize=9.5, loc='upper right', framealpha=0.9)
    panel_label(ax, 'D')

    fig.suptitle(
        'Figure 1 — The Core Finding\n'
        'BPTT adapts its learning strategy with task difficulty; evolutionary methods do not.\n'
        '(32 neurons, mean ± std across seeds 42 / 123 / 456)',
        fontsize=12, fontweight='bold', y=1.02)

    _save(fig, out_dir, 'figure1_core_finding')
    plt.close(fig)



def make_figure2(results_dir, out_dir):
    print("Figure 2: Cross-task generalisation (1×3)...")
    robot = load_robot(results_dir, neurons=NEURON_SIZES)

    acc_r   = {nn: {m: [] for m in METHODS} for nn in NEURON_SIZES}
    erank_r = {nn: {m: [] for m in METHODS} for nn in NEURON_SIZES}
    frac128 = {m: {k: [] for k in LAYER_KEYS} for m in METHODS}

    for nn in NEURON_SIZES:
        for seed_data in robot[nn].values():
            for method, entry in seed_data.items():
                acc_r[nn][method].append(get_accuracy(entry, method))
                wf = entry.get('weights_final', {})
                if 'W_rec' in wf:
                    erank_r[nn][method].append(eff_rank(wf['W_rec']))
                if nn == 128:
                    fv = layer_fracs(entry)
                    for k in LAYER_KEYS:
                        frac128[method][k].append(fv[k])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white',
                              gridspec_kw={'wspace': 0.36})
    nn_labels = [f"{nn}n" for nn in NEURON_SIZES]
    bar_w     = 0.17
    offs      = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_w

    ax = axes[0]
    clean_ax(ax)
    for i, m in enumerate(METHODS):
        mn = [pooled(acc_r[nn][m])[0] for nn in NEURON_SIZES]
        sd = [pooled(acc_r[nn][m])[1] for nn in NEURON_SIZES]
        xs = np.arange(len(NEURON_SIZES)) + offs[i]
        ax.bar(xs, mn, bar_w, yerr=sd, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], alpha=0.85, capsize=3, ecolor='#444444',
               error_kw={'elinewidth': 1.2})
    ax.set_xticks(np.arange(len(NEURON_SIZES)))
    ax.set_xticklabels(nn_labels, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlabel('Network size', fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9, handlelength=1.4)
    panel_label(ax, 'A')

    ax = axes[1]
    clean_ax(ax)
    for i, m in enumerate(METHODS):
        mn = [pooled(erank_r[nn][m])[0] for nn in NEURON_SIZES]
        sd = [pooled(erank_r[nn][m])[1] for nn in NEURON_SIZES]
        xs = np.arange(len(NEURON_SIZES)) + offs[i]
        ax.bar(xs, mn, bar_w, yerr=sd, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], alpha=0.85, capsize=3, ecolor='#444444',
               error_kw={'elinewidth': 1.2})
    ax.set_xticks(np.arange(len(NEURON_SIZES)))
    ax.set_xticklabels(nn_labels, fontsize=10)
    ax.set_ylabel('Effective rank of $W_{rec}$', fontsize=11)
    ax.set_xlabel('Network size', fontsize=11)
    ax.legend(fontsize=9.5, framealpha=0.9, handlelength=1.4)
    panel_label(ax, 'B')

    ax = axes[2]
    clean_ax(ax)
    fracs_128 = {m: {k: pooled(frac128[m][k])[0] for k in LAYER_KEYS} for m in METHODS}
    stacked_bar_ax(ax, METHODS, fracs_128)
    ax.set_ylabel('Fraction of $\\|\\Delta W\\|_F$ (%)', fontsize=11)
    ax.set_title('128 neurons', fontsize=10, color='#666666', style='italic', pad=3)
    legend_hdl = [Patch(fc=LAYER_COLORS[k], alpha=0.85, label=LAYER_LABELS[k])
                  for k in LAYER_KEYS]
    ax.legend(handles=legend_hdl, fontsize=9.5, loc='upper right', framealpha=0.9)
    panel_label(ax, 'C')

    fig.suptitle(
        'Figure 2 — Cross-Task Generalisation: Robot Arm\n'
        'BPTT concentrates learning in $W_{rec}$ on robot; EAs maintain uniform allocation.\n'
        '(mean ± std across seeds 42 / 123 / 456)',
        fontsize=12, fontweight='bold', y=1.03)

    _save(fig, out_dir, 'figure2_cross_task')
    plt.close(fig)



def make_figure3(results_dir, out_dir):
    print("Figure 3: Learning dynamics (1×4)...")
    data = load_nback(results_dir, nbacks=NBACKS, neurons=[32])

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor='white',
                              gridspec_kw={'wspace': 0.08}, sharey=True)

    for idx, nb in enumerate(NBACKS):
        ax = axes[idx]
        clean_ax(ax)
        ax.axhline(20, color='#AAAAAA', ls='--', lw=1.0, zorder=0)

        for method in METHODS:
            curves = []
            for seed_data in data[nb][32].values():
                if method not in seed_data:
                    continue
                h = seed_data[method].get('history', {})
                if method == 'bptt':
                    c = np.array(h.get('accuracy', [])) * 100
                else:
                    raw = np.array(h.get('best_fitness', h.get('accuracy', []))) * 100
                    c = np.maximum.accumulate(raw) if len(raw) > 0 else np.array([])
                if len(c) > 0:
                    curves.append(c)

            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            aligned = np.array([c[:min_len] for c in curves])
            mean_c  = aligned.mean(axis=0)
            std_c   = aligned.std(axis=0)
            xvals   = np.arange(min_len)
            c_col   = METHOD_COLORS[method]
            ax.plot(xvals, mean_c, lw=2.0, color=c_col,
                    label=METHOD_LABELS[method], zorder=3)
            ax.fill_between(xvals,
                            np.maximum(mean_c - std_c, 0),
                            np.minimum(mean_c + std_c, 105),
                            alpha=0.15, color=c_col, zorder=2)

        ax.set_ylim(0, 108)
        ax.set_xlabel('Iteration / generation', fontsize=10)
        ax.set_title(f'{nb}-back', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
        ax.tick_params(labelsize=9)
        panel_label(ax, chr(ord('A') + idx))

    fig.suptitle(
        'Figure 3 — Learning Dynamics\n'
        'BPTT converges in ~50 iterations; EAs require 200–500 generations.\n'
        '(32 neurons, mean ± std across seeds 42 / 123 / 456)',
        fontsize=12, fontweight='bold', y=1.04)

    _save(fig, out_dir, 'figure3_learning_dynamics')
    plt.close(fig)



def make_figure4(results_dir, out_dir):
    print("Figure 4: The scaling challenge (4 heatmaps + ES collapse)...")
    data = load_nback(results_dir, nbacks=ALL_NBACKS, neurons=NEURON_SIZES)

    nb_list = ALL_NBACKS
    nn_list = NEURON_SIZES
    acc_grid = {}
    for m in METHODS:
        grid = np.full((len(nb_list), len(nn_list)), np.nan)
        for ni, nb in enumerate(nb_list):
            for nni, nn in enumerate(nn_list):
                vals = []
                for seed_data in data[nb][nn].values():
                    if m in seed_data:
                        vals.append(get_accuracy(seed_data[m], m))
                mn, _ = pooled(vals)
                grid[ni, nni] = mn
        acc_grid[m] = grid

    fig = plt.figure(figsize=(22, 5.5), facecolor='white')
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.22,
                            width_ratios=[1, 1, 1, 1, 1.3])

    hm_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_line = fig.add_subplot(gs[0, 4])

    vmin, vmax = 0, 100
    cmap = 'YlOrRd'

    for hi, m in enumerate(METHODS):
        ax = hm_axes[hi]
        im = ax.imshow(acc_grid[m], aspect='auto', vmin=vmin, vmax=vmax,
                       cmap=cmap, interpolation='nearest')
        ax.set_xticks(range(len(nn_list)))
        ax.set_xticklabels([f"{nn}n" for nn in nn_list], fontsize=9)
        ax.set_yticks(range(len(nb_list)))
        ax.set_yticklabels([f"{nb}-back" for nb in nb_list], fontsize=9)
        ax.set_title(METHOD_LABELS[m], fontsize=11, fontweight='bold',
                     color=METHOD_COLORS[m])
        ax.set_xlabel('Network size', fontsize=9)
        if hi == 0:
            ax.set_ylabel('N-back level', fontsize=10)
            panel_label(ax, 'A', x=-0.18, y=1.08)
        # Annotate cells with accuracy value
        for ni in range(len(nb_list)):
            for nni in range(len(nn_list)):
                v = acc_grid[m][ni, nni]
                if not np.isnan(v):
                    txt_col = 'white' if v < 50 else 'black'
                    ax.text(nni, ni, f'{v:.0f}', ha='center', va='center',
                            fontsize=7, color=txt_col, fontweight='bold')
        # Colorbar on last heatmap only
        if hi == 3:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label('Accuracy (%)', fontsize=9)
            cb.ax.tick_params(labelsize=8)

    ax = ax_line
    clean_ax(ax)
    nb_target = 3
    nb_idx = nb_list.index(nb_target)
    for m in METHODS:
        mn_list, sd_list, nn_valid = [], [], []
        for nni, nn in enumerate(nn_list):
            v = acc_grid[m][nb_idx, nni]
            if not np.isnan(v):
                # Collect raw vals for std
                vals = []
                for seed_data in data[nb_target][nn].values():
                    if m in seed_data:
                        vals.append(get_accuracy(seed_data[m], m))
                mn_, sd_ = pooled(vals)
                mn_list.append(mn_)
                sd_list.append(sd_)
                nn_valid.append(nn)
        if mn_list:
            mn_arr = np.array(mn_list)
            sd_arr = np.array(sd_list)
            ax.plot(nn_valid, mn_arr, 'o-', lw=2.0, ms=6,
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m], zorder=3)
            ax.fill_between(nn_valid, mn_arr - sd_arr, mn_arr + sd_arr,
                            color=METHOD_COLORS[m], alpha=0.15, zorder=2)

    # Annotate x-axis with param counts
    param_labels = [f"{nn}n\n({nback_param_count(nn):,}p)" for nn in nn_list]
    ax.set_xticks(nn_list)
    ax.set_xticklabels(param_labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_xlabel('Network size (parameter count)', fontsize=10)
    ax.set_title(f'{nb_target}-back: accuracy vs. scale', fontsize=11, fontweight='bold')
    ax.axhline(20, color='#AAAAAA', ls='--', lw=1.0)
    ax.text(nn_list[-1] * 1.01, 21, 'chance', color='#999999', fontsize=8, va='bottom')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9.5, framealpha=0.9, loc='upper right')
    panel_label(ax, 'B', x=-0.14, y=1.08)

    fig.suptitle(
        'Figure 4 — The Scaling Challenge\n'
        'ES collapses at larger network sizes; GA is more robust. '
        'Panel B: 3-back accuracy vs. network size with parameter counts.',
        fontsize=11, fontweight='bold', y=1.04)

    _save(fig, out_dir, 'figure4_scaling_challenge')
    plt.close(fig)


def _save(fig, out_dir, stem):
    for ext in ('png', 'pdf'):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='results/')
    parser.add_argument('--out-dir', default='results/thesis_figures/')
    parser.add_argument('--figures', default='1,2,3,4')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    to_run = {int(x.strip()) for x in args.figures.split(',')}

    generated = []
    failed    = []

    for fig_num, func in [(1, make_figure1), (2, make_figure2),
                          (3, make_figure3), (4, make_figure4)]:
        if fig_num not in to_run:
            continue
        try:
            func(args.results_dir, args.out_dir)
            generated.append(fig_num)
        except Exception as exc:
            print(f"  [ERROR] Figure {fig_num} failed: {exc}")
            import traceback; traceback.print_exc()
            failed.append(fig_num)

    print("\n" + "=" * 60)
    print(f"Generated figures: {generated}")
    if failed:
        print(f"Failed figures:    {failed}")
    print(f"Output directory:  {args.out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
