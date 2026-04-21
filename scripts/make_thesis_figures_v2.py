#!/usr/bin/env python3
"""
Publication-quality thesis figures from 10-seed sweep.

Outputs to results/thesis_figures_v2/ at 300 DPI, tight layout, no titles.

Figures:
  fig1_accuracy.png          — 3-panel accuracy vs n-back (N=32/64/128)
  fig2_effective_rank.png    — grouped bar: W_rec eff. rank at N=32
  fig3_wout_fraction.png     — 2-panel W_out fraction trend
  fig4_cross_task.png        — n-back vs robot arm frac_rec allocation
  fig5_heatmap.png           — accuracy heatmap (3× N sizes)
  fig6_learning_curves.png   — learning curves at N=32

Tables:
  table1_accuracy.csv
  table2_connectivity.csv

Usage:
  python3 scripts/make_thesis_figures_v2.py
  python3 scripts/make_thesis_figures_v2.py --results-dir results/pub --out results/thesis_figures_v2
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# Make repo root importable for task env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent

# ── style ──────────────────────────────────────────────────────────────────────

METHODS      = ["bptt", "es", "ga", "ga_oja"]
METHOD_LABEL = {"bptt": "BPTT", "es": "ES", "ga": "GA", "ga_oja": "GA+Oja"}
METHOD_COLOR = {
    "bptt":   "#1565C0",   # deep blue      (cool end)
    "es":     "#00838F",   # dark cyan/teal
    "ga":     "#EF6C00",   # amber orange
    "ga_oja": "#B71C1C",   # deep red        (warm end)
}
LAYER_COLOR  = {"W_in": "#78909C", "W_rec": "#546E7A", "W_out": "#263238"}

NBACKS  = [1, 2, 3, 4]
NEURONS = [32, 64, 128]
SEEDS   = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Spearman ρ for BPTT frac_out trend — from stats_10seed_report.txt
BPTT_FRAC_RHO = {32: 0.969, 64: 0.969, 128: 0.941}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":          10,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          False,
})


# ── metrics ────────────────────────────────────────────────────────────────────

def effective_rank(W, threshold=0.90):
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        return 1
    cumvar = np.cumsum(s2) / total
    return int(np.searchsorted(cumvar, threshold)) + 1


def weight_fractions(init_w, final_w):
    deltas = {k: np.linalg.norm(final_w[k] - init_w[k])
              for k in ("W_in", "W_rec", "W_out")}
    total = sum(deltas.values())
    if total < 1e-12:
        return {"W_in": 0.0, "W_rec": 0.0, "W_out": 0.0}
    return {k: v / total for k, v in deltas.items()}


# ── data loading ────────────────────────────────────────────────────────────────

def load_run(pub_dir, task, nb, nn, seed, method):
    """Returns dict of metrics for one (task, nb, nn, seed, method) combo, or None."""
    if task == "nback":
        run_dir = pub_dir / f"nback{nb}_neurons{nn}_seed{seed}"
    else:
        run_dir = pub_dir / f"robot_T20_neurons{nn}_seed{seed}"

    mdir = run_dir / method
    if not mdir.is_dir():
        return None

    hist_path = mdir / "history.json"
    if not hist_path.exists():
        return None
    with open(hist_path) as f:
        hist = json.load(f)
    acc_list = hist.get("accuracy", [])
    if not acc_list:
        return None

    init_path  = mdir / "weights_init.npz"
    final_path = mdir / "weights_final.npz"
    if not init_path.exists() or not final_path.exists():
        return None

    init_w  = {k: v.astype(np.float64) for k, v in np.load(init_path).items()}
    final_w = {k: v.astype(np.float64) for k, v in np.load(final_path).items()}
    fracs   = weight_fractions(init_w, final_w)

    total_norm = sum(
        np.linalg.norm(final_w[k] - init_w[k])
        for k in ("W_in", "W_rec", "W_out")
    )

    return {
        "accuracy":      float(acc_list[-1]),
        "history":       acc_list,
        "eff_rank":      effective_rank(final_w["W_rec"]),
        "eff_rank_wout": effective_rank(final_w["W_out"]),
        "frac_in":       fracs["W_in"],
        "frac_rec":      fracs["W_rec"],
        "frac_out":      fracs["W_out"],
        "total_norm":    total_norm,
    }


def collect(pub_dir, task="nback", seeds=None):
    """
    Returns data[nb][nn][method] = list of per-seed dicts.
    nb is None for robot task.  Pass seeds= to restrict which seeds are loaded.
    """
    seed_list = seeds if seeds is not None else SEEDS
    data = {}
    nbacks_iter = NBACKS if task == "nback" else [None]

    for nb in nbacks_iter:
        for nn in NEURONS:
            for method in METHODS:
                vals = []
                for seed in seed_list:
                    r = load_run(pub_dir, task, nb, nn, seed, method)
                    if r is None:
                        warnings.warn(
                            f"  WARNING: missing {task} nb={nb} N={nn} seed={seed} {method}",
                            stacklevel=2)
                    else:
                        vals.append(r)
                if vals:
                    data.setdefault(nb, {}).setdefault(nn, {})[method] = vals

    return data


def vals(data, nb, nn, method, metric):
    """Extract list of scalar values for one metric across seeds."""
    return [r[metric] for r in data.get(nb, {}).get(nn, {}).get(method, [])
            if metric in r]


def curves(data, nb, nn, method):
    """Extract list of accuracy history lists across seeds."""
    return [r["history"] for r in data.get(nb, {}).get(nn, {}).get(method, [])
            if "history" in r]


def mean_std(v):
    a = np.array(v)
    if len(a) == 0:
        return np.nan, 0.0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


# ── shared helpers ─────────────────────────────────────────────────────────────

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def method_legend(ax, methods=None, **kw):
    if methods is None:
        methods = METHODS
    handles = [plt.Line2D([0], [0], color=METHOD_COLOR[m], linewidth=2.2,
                          label=METHOD_LABEL[m]) for m in methods]
    ax.legend(handles=handles, **kw)


# ── Figure 1: Accuracy vs N-back ──────────────────────────────────────────────

def fig1_accuracy(data, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for col, nn in enumerate(NEURONS):
        ax = axes[col]

        for method in METHODS:
            ms, ss = [], []
            for nb in NBACKS:
                v = vals(data, nb, nn, method, "accuracy")
                m, s = mean_std(v)
                ms.append(m * 100)
                ss.append(s * 100)

            ax.errorbar(NBACKS, ms, yerr=ss,
                        color=METHOD_COLOR[method],
                        marker="o", markersize=5, linewidth=1.8,
                        capsize=3, label=METHOD_LABEL[method])

        # Chance level
        ax.axhline(20, color="0.55", linestyle="--", linewidth=1.0,
                   label="Chance (20%)")

        # # Annotate ES collapse at N=64 n-back 4
        # if nn == 64:
        #     v = vals(data, 4, 64, "es", "accuracy")
        #     if v:
        #         y = np.mean(v) * 100
        #         ax.annotate(f"ES collapse\n({y:.1f}%)",
        #                     xy=(4, y), xytext=(3.15, y + 18),
        #                     fontsize=8, color=METHOD_COLOR["es"],
        #                     arrowprops=dict(arrowstyle="->",
        #                                     color=METHOD_COLOR["es"], lw=0.9))

        ax.set_xlabel("n-back level")
        ax.set_xlim(0.7, 4.3)
        ax.set_xticks(NBACKS)
        ax.set_ylim(-2, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.text(0.5, 1.02, f"N = {nn}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        despine(ax)

        if col == 0:
            ax.set_ylabel("Accuracy (%)")
            method_legend(ax, loc="lower left", fontsize=9, frameon=True,
                          framealpha=0.9, edgecolor="0.8")
            # Add chance label separately
            ax.legend(
                handles=[plt.Line2D([0],[0], color=METHOD_COLOR[m], lw=2.2,
                                    label=METHOD_LABEL[m]) for m in METHODS]
                      + [plt.Line2D([0],[0], color="0.55", lw=1, ls="--",
                                    label="Chance (20%)")],
                loc="lower left", fontsize=9, frameon=True,
                framealpha=0.9, edgecolor="0.8")

    plt.tight_layout(w_pad=1.5)
    fpath = out_dir / "fig1_accuracy.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 2: Effective Rank ───────────────────────────────────────────────────

def fig2_effective_rank(data, out_dir):
    nn = 32
    x      = np.arange(len(NBACKS))
    width  = 0.18
    shifts = np.linspace(-(len(METHODS)-1)/2, (len(METHODS)-1)/2, len(METHODS)) * width

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(METHODS):
        ms, ss = [], []
        for nb in NBACKS:
            v = vals(data, nb, nn, method, "eff_rank")
            m, s = mean_std(v)
            ms.append(m)
            ss.append(s)

        bars = ax.bar(x + shifts[i], ms, width,
                      color=METHOD_COLOR[method], label=METHOD_LABEL[method],
                      alpha=0.92, zorder=3)
        ax.errorbar(x + shifts[i], ms, yerr=ss,
                    fmt="none", color="black",
                    capsize=2.5, linewidth=0.9, zorder=4)

   
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nb}-back" for nb in NBACKS])
    ax.set_xlabel("n-back level")
    ax.set_ylabel("Effective rank of $W_{rec}$  (90% variance)")
    ax.set_ylim(0, 22)
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.25, zorder=0, linewidth=0.7)
    despine(ax)

    plt.tight_layout()
    fpath = out_dir / "fig2_effective_rank.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 3: Per-layer Weight Change Fractions ────────────────────────────────

# Distinct layer colours used only in fig3
_LAYER_LINE = {"W_in": "#78909C", "W_rec": "#455A64", "W_out": "#B71C1C"}
_LAYER_LABEL = {"W_in": "$\\Delta W_{in}$", "W_rec": "$\\Delta W_{rec}$",
                "W_out": "$\\Delta W_{out}$"}
_LAYERS = ["W_in", "W_rec", "W_out"]
_FRAC_KEY = {"W_in": "frac_in", "W_rec": "frac_rec", "W_out": "frac_out"}

SEEDS_3 = [42, 123, 456]   # 3-seed subset used for this figure


def fig3_per_layer_fractions(out_dir, pub_dir):
    """1×4 line plots — per-layer Δweight fractions, 3 seeds, N=32.

    BPTT: W_out rises with difficulty; EA: all layers flat.
    """
    nn   = 32
    data = collect(pub_dir, task="nback", seeds=SEEDS_3)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=True)

    for col, method in enumerate(METHODS):
        ax = axes[col]

        for layer in _LAYERS:
            key = _FRAC_KEY[layer]
            ms, ss = [], []
            for nb in NBACKS:
                v = vals(data, nb, nn, method, key)
                m, s = mean_std(v)
                ms.append(m * 100)
                ss.append(s * 100)

            ms = np.array(ms)
            ss = np.array(ss)
            color = _LAYER_LINE[layer]

            ax.plot(NBACKS, ms,
                    color=color, marker="o", markersize=5,
                    linewidth=1.8, label=_LAYER_LABEL[layer])
            ax.fill_between(NBACKS, ms - ss, ms + ss,
                            color=color, alpha=0.15)

        ax.set_xticks(NBACKS)
        ax.set_xlim(0.7, 4.3)
        ax.set_xlabel("n-back level")
        ax.text(0.5, 1.03, METHOD_LABEL[method], transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold",
                color=METHOD_COLOR[method])
        despine(ax)

        if col == 0:
            ax.set_ylabel("Weight change fraction (%)")
            handles = [plt.Line2D([0], [0], color=_LAYER_LINE[l], linewidth=2,
                                  label=_LAYER_LABEL[l]) for l in _LAYERS]
            ax.legend(handles=handles, fontsize=9, frameon=True,
                      framealpha=0.9, edgecolor="0.8", loc="upper right")

    # unified y range with a little padding
    ax0 = axes[0]
    ylo, yhi = ax0.get_ylim()
    for ax in axes:
        lo, hi = ax.get_ylim()
        ylo = min(ylo, lo)
        yhi = max(yhi, hi)
    for ax in axes:
        ax.set_ylim(max(0, ylo - 1), yhi + 1)

    plt.tight_layout(w_pad=0.8)
    fpath = out_dir / "fig3_per_layer_fractions.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 4: Cross-Task Layer Allocation ─────────────────────────────────────

def fig4_cross_task(data_nb, data_rb, out_dir):
    nn    = 32
    x     = np.arange(len(METHODS))
    width = 0.30

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, method in enumerate(METHODS):
        # n-back: pool across all 4 levels
        nback_v = []
        for nb in NBACKS:
            nback_v.extend(vals(data_nb, nb, nn, method, "frac_rec"))
        robot_v = vals(data_rb, None, nn, method, "frac_rec")

        nb_m, nb_s = mean_std(nback_v)
        rb_m, rb_s = mean_std(robot_v)
        nb_m *= 100; nb_s *= 100
        rb_m *= 100; rb_s *= 100

        # n-back bar (solid)
        ax.bar(x[i] - width / 2, nb_m, width,
               color=METHOD_COLOR[method], alpha=0.95, zorder=3)
        ax.errorbar(x[i] - width / 2, nb_m, yerr=nb_s,
                    fmt="none", color="black", capsize=3, lw=1.1, zorder=5)

        # robot bar (hatched, lighter)
        ax.bar(x[i] + width / 2, rb_m, width,
               color=METHOD_COLOR[method], alpha=0.45,
               hatch="///", edgecolor=METHOD_COLOR[method], zorder=3)
        ax.errorbar(x[i] + width / 2, rb_m, yerr=rb_s,
                    fmt="none", color="black", capsize=3, lw=1.1, zorder=5)

        # Shift arrow drawn above both bars
        if not (np.isnan(nb_m) or np.isnan(rb_m)):
            y_arrow = max(nb_m + nb_s, rb_m + rb_s) + 3.5
            ax.annotate(
                "", xy=(x[i] + width / 2, y_arrow),
                xytext=(x[i] - width / 2, y_arrow),
                arrowprops=dict(arrowstyle="-|>",
                                color=METHOD_COLOR[method],
                                lw=1.4, mutation_scale=11))

        # p-value (all p_adj < 0.001)
        y_star = max(nb_m + nb_s, rb_m + rb_s) + 7.5
        ax.text(x[i], y_star, "***", ha="center", fontsize=11, color="black")

    # Legend: task type
    leg_els = [
        Patch(facecolor="0.55", alpha=0.95, label="n-back (pooled 1–4)"),
        Patch(facecolor="0.55", alpha=0.45, hatch="///",
              edgecolor="0.55", label="Robot arm"),
    ]
    leg1 = ax.legend(handles=leg_els, loc="upper left", fontsize=9,
                     frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.add_artist(leg1)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS])
    for tick, method in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(METHOD_COLOR[method])
        tick.set_fontweight("semibold")

    ax.set_ylabel("$\\Delta W_{rec}$ fraction (%)")
    ax.set_ylim(40, 85)
    ax.set_yticks([40, 50, 60, 70, 80])
    ax.grid(axis="y", alpha=0.25, linewidth=0.7, zorder=0)
    despine(ax)

    plt.tight_layout()
    fpath = out_dir / "fig4_cross_task.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 5: Accuracy Heatmap ────────────────────────────────────────────────

def fig5_heatmap(data, out_dir):
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.20, vmax=1.0)
    cmap = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    im_ref = None
    for col, nn in enumerate(NEURONS):
        ax = axes[col]
        grid = np.full((len(METHODS), len(NBACKS)), np.nan)
        cell_text = {}

        for ri, method in enumerate(METHODS):
            for ci, nb in enumerate(NBACKS):
                v = vals(data, nb, nn, method, "accuracy")
                if v:
                    m, s = mean_std(v)
                    grid[ri, ci] = m
                    cell_text[(ri, ci)] = f"{m*100:.1f}\n±{s*100:.1f}"

        im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm)
        im_ref = im

        for ri in range(len(METHODS)):
            for ci in range(len(NBACKS)):
                txt = cell_text.get((ri, ci), "—")
                fg  = "white" if (np.isnan(grid[ri, ci]) or grid[ri, ci] < 0.40) else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=7.5, color=fg)

        # Box around ES N=64 n-back-4 (38.9% collapse)
        if nn == 64:
            es_ri  = METHODS.index("es")
            nb4_ci = NBACKS.index(4)
            rect = Rectangle((nb4_ci - 0.5, es_ri - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="#1565C0",
                              facecolor="none", zorder=5)
            ax.add_patch(rect)

        ax.set_xticks(range(len(NBACKS)))
        ax.set_xticklabels([f"{nb}-back" for nb in NBACKS])
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABEL[m] for m in METHODS])
        ax.text(0.5, 1.03, f"N = {nn}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")

        for tick, method in zip(ax.get_yticklabels(), METHODS):
            tick.set_color(METHOD_COLOR[method])
            tick.set_fontweight("semibold")

    # Shared colorbar
    cbar = plt.colorbar(im_ref, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Accuracy", fontsize=10)
    cbar.set_ticks([0.0, 0.20, 0.40, 0.60, 0.80, 1.0])
    cbar.set_ticklabels(["0%", "20%\n(chance)", "40%", "60%", "80%", "100%"])

    fig.subplots_adjust(right=0.88, wspace=0.08)
    fpath = out_dir / "fig5_heatmap.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 6: Learning Curves ─────────────────────────────────────────────────

def fig6_learning_curves(data, out_dir, smooth=20):
    nn  = 32
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)

    for col, nb in enumerate(NBACKS):
        ax = axes[col]

        for method in METHODS:
            c_list = curves(data, nb, nn, method)
            if not c_list:
                continue

            min_len = min(len(c) for c in c_list)
            arr = np.array([c[:min_len] for c in c_list], dtype=float) * 100

            # Rolling average for readability
            if smooth > 1:
                kernel = np.ones(smooth) / smooth
                arr = np.array([np.convolve(row, kernel, mode="valid")
                                for row in arr])

            steps = np.arange(1, arr.shape[1] + 1)
            m     = arr.mean(axis=0)
            s     = arr.std(axis=0, ddof=1)

            ax.plot(steps, m, color=METHOD_COLOR[method],
                    linewidth=1.6, label=METHOD_LABEL[method])
            ax.fill_between(steps, m - s, m + s,
                            color=METHOD_COLOR[method], alpha=0.14)

        ax.axhline(20, color="0.55", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Training step")
        ax.set_xlim(1, None)
        ax.set_ylim(-2, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.text(0.5, 1.02, f"{nb}-back", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        despine(ax)

        if col == 0:
            ax.set_ylabel("Accuracy (%)")
            method_legend(ax, loc="lower right", fontsize=9,
                          frameon=True, framealpha=0.9, edgecolor="0.8")

    plt.tight_layout(w_pad=1.0)
    fpath = out_dir / "fig6_learning_curves.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 7: W_out Effective Rank ────────────────────────────────────────────

def fig7_effective_rank_wout(data, out_dir):
    """Grouped bar chart of W_out effective rank at N=32, by n-back level × method.

    W_out shape is (5, N) so max rank = 5 (bounded by action_dim).
    BPTT shows a progressive decrease (4→2) while EA methods remain near-maximal.
    """
    nn    = 32
    x     = np.arange(len(NBACKS))
    width = 0.18
    shifts = np.linspace(-(len(METHODS)-1)/2, (len(METHODS)-1)/2,
                         len(METHODS)) * width

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(METHODS):
        ms, ss = [], []
        for nb in NBACKS:
            v = vals(data, nb, nn, method, "eff_rank_wout")
            m, s = mean_std(v)
            ms.append(m)
            ss.append(s)

        bars = ax.bar(x + shifts[i], ms, width,
                      color=METHOD_COLOR[method], label=METHOD_LABEL[method],
                      alpha=0.92, zorder=3)
        ax.errorbar(x + shifts[i], ms, yerr=ss,
                    fmt="none", color="black",
                    capsize=2.5, linewidth=0.9, zorder=4)

        # Annotate BPTT bar values to highlight the decreasing trend
        if method == "bptt":
            for bar, m_val in zip(bars, ms):
                if not np.isnan(m_val):
                    ax.text(bar.get_x() + bar.get_width() / 2, m_val + 0.07,
                            f"{m_val:.1f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold",
                            color=METHOD_COLOR["bptt"])

    # Max possible rank line
    ax.axhline(5, color="0.45", linestyle="--", linewidth=1.1,
               label="Max rank (= 5)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{nb}-back" for nb in NBACKS])
    ax.set_xlabel("n-back level")
    ax.set_ylabel("Effective rank of $W_{out}$  (90% variance)")
    ax.set_ylim(0, 6.5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.25, zorder=0, linewidth=0.7)
    despine(ax)

    plt.tight_layout()
    fpath = out_dir / "fig7_effective_rank_wout.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 8: PCA Dimensionality of Recurrent Dynamics ───────────────────────

def _pca_n_dims(W_rec, W_in, n_back, n_trials=100, threshold=0.90):
    """Run RNN on n_trials letter sequences; return # PCs for `threshold` variance."""
    from envs.letter_nback import LetterNBackTask
    task  = LetterNBackTask(n_back=n_back)
    N     = W_rec.shape[0]
    all_h = []
    for _ in range(n_trials):
        obs, _, _ = task.get_trial()   # (T, 5)
        h = np.zeros(N)
        for t in range(len(obs)):
            h = np.tanh(W_rec @ h + W_in @ obs[t])
            all_h.append(h.copy())
    X  = np.array(all_h, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    s2  = s ** 2
    cum = np.cumsum(s2) / s2.sum()
    return int(np.searchsorted(cum, threshold)) + 1


def fig8_pca_dims(pub_dir, out_dir, n_trials=100):
    """3-panel line plot: PCs needed for 90% variance of h_t vs n-back level.

    Directly shows that BPTT produces low-dimensional recurrent dynamics
    while EA methods occupy higher-dimensional state spaces.
    """
    print("  Computing PCA dimensionality (this may take ~30s) ...")

    # Cache dims: dims_cache[nn][nb][method] = list of ints across seeds
    dims_cache = {}
    for nn in NEURONS:
        dims_cache[nn] = {}
        for nb in NBACKS:
            dims_cache[nn][nb] = {m: [] for m in METHODS}
            for method in METHODS:
                for seed in SEEDS:
                    run_dir = pub_dir / f"nback{nb}_neurons{nn}_seed{seed}" / method
                    wp = run_dir / "weights_final.npz"
                    if not wp.exists():
                        continue
                    fw = {k: v.astype(np.float64)
                          for k, v in np.load(wp).items()}
                    if "W_rec" not in fw or "W_in" not in fw:
                        continue
                    d = _pca_n_dims(fw["W_rec"], fw["W_in"], nb,
                                    n_trials=n_trials)
                    dims_cache[nn][nb][method].append(d)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for col, nn in enumerate(NEURONS):
        ax = axes[col]

        for method in METHODS:
            ms, ss = [], []
            for nb in NBACKS:
                v = dims_cache[nn][nb][method]
                m, s = mean_std(v)
                ms.append(m)
                ss.append(s)

            ms = np.array(ms, dtype=float)
            ss = np.array(ss, dtype=float)

            ax.errorbar(NBACKS, ms, yerr=ss,
                        color=METHOD_COLOR[method],
                        marker="o", markersize=5, linewidth=1.8,
                        capsize=3, label=METHOD_LABEL[method])

        ax.set_xlabel("n-back level")
        ax.set_xticks(NBACKS)
        ax.set_xlim(0.7, 4.3)
        ax.set_ylim(0, None)
        ax.text(0.5, 1.02, f"N = {nn}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6, zorder=0)
        despine(ax)

        if col == 0:
            ax.set_ylabel("PCs for 90% variance of $h_t$")
            method_legend(ax, loc="upper left", fontsize=9,
                          frameon=True, framealpha=0.9, edgecolor="0.8")

    plt.tight_layout(w_pad=1.5)
    fpath = out_dir / "fig8_pca_dims.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 8b: PCA Summary — both tasks 

def fig_pca_all_tasks(pub_dir, out_dir, n_trials=50):
    """Polished PCA summary across n-back + robot arm tasks.

    3 panels (N=32, 64, 128). Each panel shows grouped bars for 5 conditions
    (1-back, 2-back, 3-back, 4-back, Robot) with 4 method bars per group.
    A dashed vertical divider separates n-back from robot conditions.

    Key message: BPTT consistently requires fewer PCs than EA methods,
    and this generalises across tasks and network sizes.
    """
    from models.rsnn_policy import RSNNPolicy
    from envs.letter_nback import LetterNBackTask
    from envs.robot_arm import RobotArmTask

    print("  Computing PCA dims across all tasks (may take ~60s) ...")

    THRESHOLD = 0.90

    CONDITIONS = [
        ("1-back", "nback",  1,    "nback1"),
        ("2-back", "nback",  2,    "nback2"),
        ("3-back", "nback",  3,    "nback3"),
        ("4-back", "nback",  4,    "nback4"),
        ("Robot",  "robot",  None, "robot_T20"),
    ]

    def _pca_dims_policy(wp, task, seed):
        d   = np.load(wp)
        pol = RSNNPolicy(d["W_rec"], d["W_in"], d["W_out"])
        rng = np.random.default_rng(seed)
        states = []
        for _ in range(n_trials):
            inputs, *_ = task.get_trial(rng=rng)
            pol.reset()
            for t in range(len(inputs)):
                pol.act(inputs[t])
                states.append(pol.h.copy())
        X  = np.array(states, dtype=np.float64)
        Xc = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        s2 = s ** 2
        cum = np.cumsum(s2) / s2.sum()
        return int(np.searchsorted(cum, THRESHOLD)) + 1

    # dims_data[nn][label][method] = list of ints (one per seed)
    dims_data: dict = {}
    for nn in NEURONS:
        dims_data[nn] = {c[0]: {m: [] for m in METHODS} for c in CONDITIONS}
        for label, task_type, nb, prefix in CONDITIONS:
            task = (LetterNBackTask(n_back=nb, seq_length=20)
                    if task_type == "nback"
                    else RobotArmTask(seq_length=20))
            for method in METHODS:
                for seed in SEEDS:
                    wp = pub_dir / f"{prefix}_neurons{nn}_seed{seed}" / method / "weights_final.npz"
                    if not wp.exists():
                        continue
                    try:
                        dims_data[nn][label][method].append(
                            _pca_dims_policy(wp, task, seed)
                        )
                    except Exception:
                        pass

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_conds   = len(CONDITIONS)
    bar_w     = 0.17
    offsets   = np.arange(len(METHODS)) * bar_w - (len(METHODS) - 1) * bar_w / 2
    x         = np.arange(n_conds)
    xlabels   = [c[0] for c in CONDITIONS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for col, nn in enumerate(NEURONS):
        ax = axes[col]

        for mi, method in enumerate(METHODS):
            means, stds = [], []
            for label, *_ in CONDITIONS:
                v = dims_data[nn][label][method]
                means.append(np.mean(v) if v else np.nan)
                stds.append(np.std(v)  if v else 0.0)

            bars = ax.bar(x + offsets[mi], means, width=bar_w,
                          yerr=stds, color=METHOD_COLOR[method],
                          label=METHOD_LABEL[method], alpha=0.88,
                          error_kw={"elinewidth": 1.0, "capsize": 2.5,
                                    "ecolor": "0.35"})

        # Vertical divider between n-back and robot
        ax.axvline(3.5, color="0.70", linewidth=0.9, linestyle="--", zorder=1)

        # N reference line (maximum possible = N dims)
        ax.axhline(nn, color="0.80", linewidth=0.7, linestyle=":", zorder=0,
                   label=f"Max (N={nn})" if col == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.text(0.5, 1.02, f"N = {nn}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        ax.set_ylim(0, max(nn * 0.65, 12))
        ax.grid(axis="y", alpha=0.18, linewidth=0.6, zorder=0)
        despine(ax)

        if col == 0:
            ax.set_ylabel("PCs for 90% variance of $h_t$")
            method_legend(ax, loc="upper left", fontsize=9,
                          frameon=True, framealpha=0.92, edgecolor="0.8")

    # Shared x annotation
    fig.text(0.52, 0.01, "Task condition", ha="center", fontsize=11)

    plt.tight_layout(w_pad=1.5)
    plt.subplots_adjust(bottom=0.14)
    fpath = out_dir / "fig_pca_all_tasks.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Figure 9: Total Frobenius Norm ─────────────────────────────────────────────

def fig9_total_frobenius_norm(data, out_dir):
    """3-panel line plot of total ||ΔW||_F (sum over W_in, W_rec, W_out) vs
    n-back difficulty, one panel per network size, all 10 seeds.

    Key patterns:
      BPTT:    norm DECREASES with difficulty (more efficient / targeted updates)
      GA/GA+Oja: norm INCREASES substantially with difficulty
      ES:      relatively flat
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for col, nn in enumerate(NEURONS):
        ax = axes[col]

        for method in METHODS:
            ms, ss = [], []
            for nb in NBACKS:
                v = vals(data, nb, nn, method, "total_norm")
                m, s = mean_std(v)
                ms.append(m)
                ss.append(s)

            ms = np.array(ms, dtype=float)
            ss = np.array(ss, dtype=float)

            ax.errorbar(NBACKS, ms, yerr=ss,
                        color=METHOD_COLOR[method],
                        marker="o", markersize=5, linewidth=1.8,
                        capsize=3, label=METHOD_LABEL[method])

        ax.set_xlabel("N-back level")
        ax.set_xlim(0.7, 4.3)
        ax.set_xticks(NBACKS)
        ax.text(0.5, 1.02, f"N = {nn}", transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        ax.grid(axis="y", alpha=0.20, linewidth=0.6, zorder=0)
        despine(ax)

        if col == 0:
            ax.set_ylabel("Total $\\|\\Delta W\\|_F$")
            method_legend(ax, loc="upper left", fontsize=9,
                          frameon=True, framealpha=0.9, edgecolor="0.8")

    plt.tight_layout(w_pad=1.5)
    fpath = out_dir / "fig9_total_frobenius_norm.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Table 1: Accuracy ─────────────────────────────────────────────────────────

def table1_accuracy(data, out_dir):
    header = ["n-back", "N"] + [METHOD_LABEL[m] for m in METHODS]
    rows   = []

    for nb in NBACKS:
        for nn in NEURONS:
            row = [nb, nn]
            for method in METHODS:
                v = vals(data, nb, nn, method, "accuracy")
                if v:
                    m, s = mean_std(v)
                    row.append(f"{m*100:.1f}±{s*100:.1f}")
                else:
                    row.append("—")
            rows.append(row)

    fpath = out_dir / "table1_accuracy.csv"
    with open(fpath, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)

    # Print to stdout
    widths = [7, 5] + [18] * len(METHODS)
    print("\n" + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    print(f"\nSaved: {fpath}")


# ── Table 2: Connectivity ──────────────────────────────────────────────────────

def table2_connectivity(data, out_dir):
    header = ["Method", "n-back",
              "frac_W_in (%)", "frac_W_rec (%)", "frac_W_out (%)",
              "eff_rank_W_rec"]
    rows = []
    nn = 32

    def fmt(v, scale=100):
        if not v:
            return "—"
        m, s = mean_std(v)
        return f"{m*scale:.1f}±{s*scale:.1f}"

    def fmt_rank(v):
        if not v:
            return "—"
        m, s = mean_std(v)
        return f"{m:.1f}±{s:.1f}"

    for method in METHODS:
        for nb in NBACKS:
            rows.append([
                METHOD_LABEL[method], nb,
                fmt(vals(data, nb, nn, method, "frac_in")),
                fmt(vals(data, nb, nn, method, "frac_rec")),
                fmt(vals(data, nb, nn, method, "frac_out")),
                fmt_rank(vals(data, nb, nn, method, "eff_rank")),
            ])

    fpath = out_dir / "table2_connectivity.csv"
    with open(fpath, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"Saved: {fpath}")


# ── Figure 9: Scaling Summary (accuracy vs N, both tasks) ────────────────────

def fig10_scaling_summary(data_nback, data_robot, out_dir):
    """1×2 panels: accuracy vs N size for all methods, n-back (pooled) and robot arm.

    Puts ES N=64 anomaly on both tasks in direct visual context.
    n-back accuracy is pooled (mean across n-back levels 1–4) per seed then averaged.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    x     = np.array([0, 1, 2])   # positions for N=32, 64, 128
    xlabs = ["32", "64", "128"]

    for method in METHODS:
        # ── left: n-back pooled ──────────────────────────────────────────────
        ms, ss = [], []
        for nn in NEURONS:
            # pool all seeds × all n-back levels into one list
            v = []
            for nb in NBACKS:
                v.extend(vals(data_nback, nb, nn, method, "accuracy"))
            m, s = mean_std([a * 100 for a in v])
            ms.append(m); ss.append(s)

        ms = np.array(ms); ss = np.array(ss)
        axL.errorbar(x, ms, yerr=ss,
                     color=METHOD_COLOR[method], marker="o",
                     markersize=5, linewidth=1.8, capsize=3,
                     label=METHOD_LABEL[method])

        # ── right: robot arm ─────────────────────────────────────────────────
        ms_r, ss_r = [], []
        for nn in NEURONS:
            v = [a * 100 for a in vals(data_robot, None, nn, method, "accuracy")]
            m, s = mean_std(v)
            ms_r.append(m); ss_r.append(s)

        ms_r = np.array(ms_r); ss_r = np.array(ss_r)
        axR.errorbar(x, ms_r, yerr=ss_r,
                     color=METHOD_COLOR[method], marker="o",
                     markersize=5, linewidth=1.8, capsize=3,
                     label=METHOD_LABEL[method])

    for ax, title in [(axL, "N-back (levels 1–4 pooled)"),
                      (axR, "Robot arm")]:
        ax.axhline(20, color="0.55", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabs)
        ax.set_xlabel("Network size  N")
        ax.set_xlim(-0.4, 2.4)
        ax.set_ylim(-2, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.text(0.5, 1.03, title, transform=ax.transAxes,
                ha="center", fontsize=11, fontweight="semibold")
        ax.grid(axis="y", alpha=0.18, linewidth=0.6, zorder=0)
        despine(ax)

    axL.set_ylabel("Accuracy (%)")
    method_legend(axL, loc="lower right", fontsize=9,
                  frameon=True, framealpha=0.9, edgecolor="0.8")

    # Annotate ES collapse on robot arm N=64
    v_es64 = [a * 100 for a in vals(data_robot, None, 64, "es", "accuracy")]
    if v_es64:
        y_es = np.mean(v_es64)
        axR.annotate(f"ES  ({y_es:.1f}%)",
                     xy=(1, y_es), xytext=(1.25, y_es + 22),
                     fontsize=8, color=METHOD_COLOR["es"],
                     arrowprops=dict(arrowstyle="->",
                                     color=METHOD_COLOR["es"], lw=0.9))

    plt.tight_layout(w_pad=2.5)
    fpath = out_dir / "fig10_scaling_summary.png"
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ── Table 3: Robot Arm Accuracy ────────────────────────────────────────────────

def table3_robot_accuracy(data_robot, out_dir):
    """Accuracy by method × N size for the robot arm task."""
    header = ["N"] + [METHOD_LABEL[m] for m in METHODS]
    rows   = []

    for nn in NEURONS:
        row = [nn]
        for method in METHODS:
            v = [a * 100 for a in vals(data_robot, None, nn, method, "accuracy")]
            if v:
                m, s = mean_std(v)
                row.append(f"{m:.1f}±{s:.1f}")
            else:
                row.append("—")
        rows.append(row)

    fpath = out_dir / "table3_robot_accuracy.csv"
    with open(fpath, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [5] + [14] * len(METHODS)
    print("\n" + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    print(f"\nSaved: {fpath}")


# ── Table 4: Effective Rank Cross-Task ────────────────────────────────────────

def table4_eff_rank_cross_task(data_nback, data_robot, out_dir):
    """W_rec effective rank side-by-side: n-back (pooled) vs robot arm."""
    header = ["N", "Task"] + [METHOD_LABEL[m] for m in METHODS]
    rows   = []

    def fmt_rank(v):
        if not v:
            return "—"
        m, s = mean_std(v)
        return f"{m:.1f}±{s:.1f}"

    for nn in NEURONS:
        # n-back: pool across all n-back levels
        nb_row = [nn, "n-back"]
        for method in METHODS:
            v = []
            for nb in NBACKS:
                v.extend(vals(data_nback, nb, nn, method, "eff_rank"))
            nb_row.append(fmt_rank(v))
        rows.append(nb_row)

        # robot arm
        rb_row = ["", "robot"]
        for method in METHODS:
            v = vals(data_robot, None, nn, method, "eff_rank")
            rb_row.append(fmt_rank(v))
        rows.append(rb_row)

    fpath = out_dir / "table4_eff_rank_cross_task.csv"
    with open(fpath, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [5, 7] + [14] * len(METHODS)
    print("\n" + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    print(f"\nSaved: {fpath}")


# ── Table 5: GA vs GA+Oja ─────────────────────────────────────────────────────

# p_adj values for GA vs GA+Oja tests, read from stats_10seed.json
_GA_OJA_PADJ = {
    "acc_32":   8.687e-05,
    "acc_64":   1.102e-06,
    "acc_128":  1.074e-12,
    "rank_32":  3.805e-01,
    "rank_64":  1.000e+00,
    "rank_128": 1.000e+00,
}


def _fmt_padj(p):
    if p < 0.001:
        return "<0.001"
    if p < 0.05:
        return f"{p:.3f}"
    return "ns"


def table5_ga_vs_gaoja(data_nback, out_dir):
    """GA vs GA+Oja: pooled accuracy and W_rec eff_rank by N with p_adj and r_rb.

    p_adj values sourced from stats_10seed.json (Holm-Šidák correction over 78 tests).
    """
    # r_rb from ga_vs_oja section (pooled = same as nback-pooled)
    R_RB_ACC  = {32: -0.631, 64: -0.736, 128: -1.000}
    R_RB_RANK = {32:  0.235, 64:  0.063, 128: -0.150}

    header = ["N", "Metric", "GA", "GA+Oja", "p_adj", "r_rb"]
    rows   = []

    for nn in NEURONS:
        # accuracy (pooled across n-back levels)
        acc_ga, acc_oja = [], []
        rank_ga, rank_oja = [], []
        for nb in NBACKS:
            acc_ga.extend( [a * 100 for a in vals(data_nback, nb, nn, "ga",     "accuracy")])
            acc_oja.extend([a * 100 for a in vals(data_nback, nb, nn, "ga_oja", "accuracy")])
            rank_ga.extend( vals(data_nback, nb, nn, "ga",     "eff_rank"))
            rank_oja.extend(vals(data_nback, nb, nn, "ga_oja", "eff_rank"))

        m_ga,  s_ga  = mean_std(acc_ga)
        m_oja, s_oja = mean_std(acc_oja)
        rows.append([
            nn, "Accuracy (%)",
            f"{m_ga:.1f}±{s_ga:.1f}",
            f"{m_oja:.1f}±{s_oja:.1f}",
            _fmt_padj(_GA_OJA_PADJ[f"acc_{nn}"]),
            f"{R_RB_ACC[nn]:+.3f}",
        ])

        m_rga, s_rga  = mean_std(rank_ga)
        m_roja, s_roja = mean_std(rank_oja)
        rows.append([
            "", "W_rec eff. rank",
            f"{m_rga:.1f}±{s_rga:.1f}",
            f"{m_roja:.1f}±{s_roja:.1f}",
            _fmt_padj(_GA_OJA_PADJ[f"rank_{nn}"]),
            f"{R_RB_RANK[nn]:+.3f}",
        ])

    fpath = out_dir / "table5_ga_vs_gaoja.csv"
    with open(fpath, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)

    widths = [5, 18, 14, 14, 8, 7]
    print("\n" + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "-" * (sum(widths) + 2 * len(widths)))
    for row in rows:
        print("  " + "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))
    print(f"\nSaved: {fpath}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir",
                        default=str(ROOT / "results" / "pub"))
    parser.add_argument("--out",
                        default=str(ROOT / "results" / "thesis_figures_v2"))
    args = parser.parse_args()

    pub_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pub_dir.is_dir():
        sys.exit(f"ERROR: {pub_dir} not found")

    print("Loading n-back data ...")
    data_nback = collect(pub_dir, task="nback")
    print("Loading robot arm data ...")
    data_robot = collect(pub_dir, task="robot")

    print("\n── Figures ──────────────────────────────────────────")
    fig1_accuracy(data_nback, out_dir)
    fig2_effective_rank(data_nback, out_dir)
    fig3_per_layer_fractions(out_dir, pub_dir)
    fig4_cross_task(data_nback, data_robot, out_dir)
    fig5_heatmap(data_nback, out_dir)
    fig6_learning_curves(data_nback, out_dir)
    fig7_effective_rank_wout(data_nback, out_dir)
    fig8_pca_dims(pub_dir, out_dir)
    fig_pca_all_tasks(pub_dir, out_dir)
    fig9_total_frobenius_norm(data_nback, out_dir)
    fig10_scaling_summary(data_nback, data_robot, out_dir)

    print("\n── Tables ───────────────────────────────────────────")
    table1_accuracy(data_nback, out_dir)
    table2_connectivity(data_nback, out_dir)
    table3_robot_accuracy(data_robot, out_dir)
    table4_eff_rank_cross_task(data_nback, data_robot, out_dir)
    table5_ga_vs_gaoja(data_nback, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
