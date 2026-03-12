"""
Comprehensive overview figure:
  - Performance vs n-back / task across network sizes and methods
  - Effective rank of W_in, W_rec, W_out  (init vs final, Δ)
  - ||ΔW||_F per layer
  - Singular value spectra of W_rec at key configs

Layout (4 rows × 4 cols + extras):
  Row 0: Accuracy heatmaps (nback × neurons) per method  [4 panels]
  Row 1: Accuracy curves vs n-back, one line per neuron size [4 panels: ES, GA, GA-Oja, BPTT]
  Row 2: Effective rank W_rec (final) & ΔRank W_rec, W_in, W_out [4 panels]
  Row 3: ||ΔW||_F per layer [3 panels] + SV spectra [1 panel]
"""

import os, json, re, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

warnings.filterwarnings("ignore")

# ── helpers ───────────────────────────────────────────────────────────────────

def effective_rank(W):
    """Participation-ratio effective rank: exp(H(p)) where p = σ/Σσ."""
    sv = np.linalg.svd(W, compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def load_run(path, method):
    """Return dict with weights_init, weights_final, history, config."""
    mdir = Path(path) / method
    if not mdir.exists():
        return None
    try:
        wi  = np.load(mdir / "weights_init.npz")
        wf  = np.load(mdir / "weights_final.npz")
        his = json.load(open(mdir / "history.json"))
        cfg = json.load(open(Path(path) / "config.json"))
        return dict(W_in_i=wi["W_in"], W_rec_i=wi["W_rec"], W_out_i=wi["W_out"],
                    W_in_f=wf["W_in"], W_rec_f=wf["W_rec"], W_out_f=wf["W_out"],
                    history=his, config=cfg)
    except Exception:
        return None

def best_acc(run):
    h = run["history"]
    # prefer explicit accuracy key (EA methods track this)
    if "accuracy" in h and h["accuracy"]:
        return float(max(h["accuracy"]))
    # BPTT tracks accuracy in 'val_accuracy' or 'train_accuracy'
    for key in ("val_accuracy", "train_accuracy", "best_accuracy"):
        if key in h and h[key]:
            return float(max(h[key]))
    return np.nan

def delta_rank(run, layer):
    ki, kf = f"W_{layer}_i", f"W_{layer}_f"
    return effective_rank(run[kf]) - effective_rank(run[ki])

def delta_frob(run, layer):
    ki, kf = f"W_{layer}_i", f"W_{layer}_f"
    return float(np.linalg.norm(run[kf] - run[ki], "fro"))

# ── data collection ───────────────────────────────────────────────────────────

RESULTS = Path("results")
METHODS = ["es", "ga", "ga_oja", "bptt"]
METHOD_LABELS = {"es": "ES", "ga": "GA", "ga_oja": "GA-Oja", "bptt": "BPTT"}
METHOD_COLORS = {"es": "#4C72B0", "ga": "#DD8452", "ga_oja": "#55A868", "bptt": "#C44E52"}

NBACK_NEURONS = [32, 64, 128, 256]
NBACK_LEVELS  = list(range(7))   # 0–6
SEEDS = [42, 123, 456]

# nback data: {n_back: {n_neurons: {method: [acc, ...]}}}
nback_data = {}
for nb in NBACK_LEVELS:
    nback_data[nb] = {}
    for nn in NBACK_NEURONS:
        nback_data[nb][nn] = {m: [] for m in METHODS}
        for seed in SEEDS:
            d = RESULTS / "nback" / f"nback{nb}_neurons{nn}_seed{seed}"
            if not d.exists():
                continue
            for m in METHODS:
                run = load_run(d, m)
                if run:
                    nback_data[nb][nn][m].append(run)

# robot data: {n_neurons: {method: [run, ...]}}
ROBOT_NEURONS = [32, 64, 128, 256]
robot_data = {}
for nn in ROBOT_NEURONS:
    robot_data[nn] = {m: [] for m in METHODS}
    for seed in SEEDS:
        d = RESULTS / "robot" / f"robot_T20_neurons{nn}_seed{seed}"
        if not d.exists():
            continue
        for m in METHODS:
            run = load_run(d, m)
            if run:
                robot_data[nn][m].append(run)

print("Data loaded.")
print(f"  nback runs: { {nb: {nn: {m: len(v) for m,v in mdict.items()} for nn,mdict in ndict.items()} for nb,ndict in nback_data.items()} }")

# ── figure setup ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 26))
fig.patch.set_facecolor("#F7F7F7")

gs_top  = gridspec.GridSpec(1, 4, figure=fig, left=0.05, right=0.97,
                             top=0.97, bottom=0.80, wspace=0.35)
gs_mid  = gridspec.GridSpec(1, 4, figure=fig, left=0.05, right=0.97,
                             top=0.77, bottom=0.60, wspace=0.35)
gs_rank = gridspec.GridSpec(1, 4, figure=fig, left=0.05, right=0.97,
                             top=0.57, bottom=0.39, wspace=0.35)
gs_bot  = gridspec.GridSpec(1, 4, figure=fig, left=0.05, right=0.97,
                             top=0.36, bottom=0.05, wspace=0.35)

axes_heatmap  = [fig.add_subplot(gs_top[i])  for i in range(4)]
axes_curves   = [fig.add_subplot(gs_mid[i])  for i in range(4)]
axes_rank     = [fig.add_subplot(gs_rank[i]) for i in range(4)]
axes_bot      = [fig.add_subplot(gs_bot[i])  for i in range(4)]

PANEL_LABELS = list("ABCDEFGHIJKLMNOP")
panel_idx = [0]

def label_panel(ax, txt=None):
    t = txt or PANEL_LABELS[panel_idx[0]]
    ax.text(-0.12, 1.05, t, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")
    panel_idx[0] += 1

def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.5, color="#CCCCCC")

# ── ROW 0: accuracy heatmaps (nback × neurons), one per method ───────────────

for mi, m in enumerate(METHODS):
    ax = axes_heatmap[mi]
    # rows = n_back 0..6, cols = neurons [32,64,128,256]
    grid = np.full((len(NBACK_LEVELS), len(NBACK_NEURONS)), np.nan)
    for ni, nb in enumerate(NBACK_LEVELS):
        for nni, nn in enumerate(NBACK_NEURONS):
            runs = nback_data[nb][nn][m]
            if runs:
                grid[ni, nni] = np.nanmean([best_acc(r) for r in runs])

    im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                   origin="upper", interpolation="nearest")

    # annotate cells
    for ni in range(len(NBACK_LEVELS)):
        for nni in range(len(NBACK_NEURONS)):
            v = grid[ni, nni]
            if not np.isnan(v):
                ax.text(nni, ni, f"{v:.0%}", ha="center", va="center",
                        fontsize=7.5, color="black" if 0.25 < v < 0.85 else "white",
                        fontweight="bold")
            else:
                ax.text(nni, ni, "—", ha="center", va="center",
                        fontsize=8, color="#AAAAAA")

    ax.set_xticks(range(len(NBACK_NEURONS)))
    ax.set_xticklabels([str(n) for n in NBACK_NEURONS], fontsize=8)
    ax.set_yticks(range(len(NBACK_LEVELS)))
    ax.set_yticklabels([f"{nb}-back" for nb in NBACK_LEVELS], fontsize=8)
    ax.set_xlabel("Neurons", fontsize=9)
    ax.set_title(f"{METHOD_LABELS[m]}", fontsize=11, fontweight="bold",
                 color=METHOD_COLORS[m])
    if mi == 0:
        ax.set_ylabel("N-back level", fontsize=9)

    label_panel(ax)

# shared colorbar for heatmaps
cax = fig.add_axes([0.98, 0.80, 0.008, 0.17])
fig.colorbar(ScalarMappable(Normalize(0, 1), "RdYlGn"), cax=cax, label="Accuracy")

# ── ROW 1: accuracy vs n-back level, per method, lines = neuron sizes ─────────

NN_COLORS = {32: "#2C7BB6", 64: "#ABD9E9", 128: "#FDAE61", 256: "#D7191C"}

for mi, m in enumerate(METHODS):
    ax = axes_curves[mi]
    style_ax(ax)
    for nn in NBACK_NEURONS:
        means, stds = [], []
        for nb in NBACK_LEVELS:
            runs = nback_data[nb][nn][m]
            accs = [best_acc(r) for r in runs if not np.isnan(best_acc(r))]
            means.append(np.mean(accs) if accs else np.nan)
            stds.append(np.std(accs)  if len(accs) > 1 else 0)
        xs = NBACK_LEVELS
        ys = np.array(means)
        se = np.array(stds)
        mask = ~np.isnan(ys)
        ax.plot(np.array(xs)[mask], ys[mask], "o-", color=NN_COLORS[nn],
                lw=1.8, ms=5, label=f"{nn}n")
        ax.fill_between(np.array(xs)[mask],
                        (ys - se)[mask], (ys + se)[mask],
                        color=NN_COLORS[nn], alpha=0.15)

    # robot point (T20) as star at x=7, one per neuron size
    for rnn in ROBOT_NEURONS:
        r_accs = [best_acc(r) for r in robot_data[rnn][m]
                  if not np.isnan(best_acc(r))]
        if r_accs:
            ax.plot(7, np.mean(r_accs), "*", color=NN_COLORS[rnn],
                    ms=10, zorder=5)

    ax.axvline(6.5, color="#888888", lw=0.8, ls="--", alpha=0.5)
    ax.text(6.6, 0.05, "robot\nT20", fontsize=6.5, color="#555555", va="bottom")
    ax.set_xticks(list(range(7)) + [7])
    ax.set_xticklabels([str(x) for x in range(7)] + ["R"], fontsize=8)
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("N-back level  (R=robot)", fontsize=9)
    ax.set_title(METHOD_LABELS[m], fontsize=11, fontweight="bold",
                 color=METHOD_COLORS[m])
    if mi == 0:
        ax.set_ylabel("Best accuracy", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    label_panel(ax)

# ── ROW 2: effective rank panels ──────────────────────────────────────────────
# [0] W_rec final rank vs n-back (lines=neurons, one method=all)
# [1] ΔRank W_rec  (final - init)
# [2] ΔRank W_in   (final - init)
# [3] ΔRank W_out  (final - init)

RANK_LAYERS = ["rec", "in", "out"]
RANK_TITLES = ["ΔRank  W_rec", "ΔRank  W_in", "ΔRank  W_out"]

# panel [0]: W_rec effective rank (final) by n-back, all methods averaged per neuron size
ax = axes_rank[0]
style_ax(ax)
for nn in NBACK_NEURONS:
    means = []
    for nb in NBACK_LEVELS:
        vals = []
        for m in METHODS:
            for r in nback_data[nb][nn][m]:
                vals.append(effective_rank(r["W_rec_f"]))
        means.append(np.nanmean(vals) if vals else np.nan)
    mask = ~np.isnan(np.array(means))
    ax.plot(np.array(NBACK_LEVELS)[mask], np.array(means)[mask],
            "o-", color=NN_COLORS[nn], lw=1.8, ms=5, label=f"{nn}n")

ax.set_xlabel("N-back level", fontsize=9)
ax.set_ylabel("Effective rank (W_rec final)", fontsize=9)
ax.set_title("W_rec Effective Rank", fontsize=10, fontweight="bold")
ax.legend(fontsize=7.5, framealpha=0.7)
label_panel(ax)

# panels [1-3]: ΔRank per layer, per method, grouped by neuron size
for li, layer in enumerate(RANK_LAYERS):
    ax = axes_rank[li + 1]
    style_ax(ax)
    x = np.arange(len(NBACK_NEURONS))
    width = 0.18
    for mi, m in enumerate(METHODS):
        vals = []
        for nn in NBACK_NEURONS:
            drs = []
            for nb in NBACK_LEVELS:
                for r in nback_data[nb][nn][m]:
                    drs.append(delta_rank(r, layer))
            vals.append(np.nanmean(drs) if drs else np.nan)
        offset = (mi - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[m],
                      color=METHOD_COLORS[m], alpha=0.85, zorder=3)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NBACK_NEURONS], fontsize=8)
    ax.set_xlabel("Neurons", fontsize=9)
    ax.set_ylabel("Δ Effective rank", fontsize=9)
    ax.set_title(RANK_TITLES[li], fontsize=10, fontweight="bold")
    if li == 0:
        ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    label_panel(ax)

# ── ROW 3: ||ΔW||_F per layer + SV spectra ────────────────────────────────────

# panels [0-2]: ||ΔW||_F for W_in, W_rec, W_out
BOT_LAYERS = ["in", "rec", "out"]
BOT_TITLES = ["‖ΔW_in‖_F", "‖ΔW_rec‖_F", "‖ΔW_out‖_F"]

for li, layer in enumerate(BOT_LAYERS):
    ax = axes_bot[li]
    style_ax(ax)
    x = np.arange(len(NBACK_NEURONS))
    width = 0.18
    for mi, m in enumerate(METHODS):
        vals = []
        for nn in NBACK_NEURONS:
            dfs = []
            for nb in NBACK_LEVELS:
                for r in nback_data[nb][nn][m]:
                    dfs.append(delta_frob(r, layer))
            vals.append(np.nanmean(dfs) if dfs else np.nan)
        offset = (mi - 1.5) * width
        ax.bar(x + offset, vals, width, label=METHOD_LABELS[m],
               color=METHOD_COLORS[m], alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NBACK_NEURONS], fontsize=8)
    ax.set_xlabel("Neurons", fontsize=9)
    ax.set_ylabel("Frobenius norm of ΔW", fontsize=9)
    ax.set_title(BOT_TITLES[li], fontsize=10, fontweight="bold")
    if li == 0:
        ax.legend(fontsize=7, framealpha=0.7, ncol=2)
    label_panel(ax)

# panel [3]: SV spectra of W_rec at select configs
ax = axes_bot[3]
style_ax(ax)
ax.set_title("W_rec Singular Value Spectra", fontsize=10, fontweight="bold")
ax.set_xlabel("Singular value index", fontsize=9)
ax.set_ylabel("Singular value", fontsize=9)

spec_configs = [
    ("nback", 1, 32,  "es",    "--"),
    ("nback", 1, 128, "es",    "-"),
    ("nback", 3, 32,  "bptt",  "--"),
    ("nback", 3, 128, "bptt",  "-"),
]
for task, nb, nn, m, ls in spec_configs:
    runs = nback_data[nb][nn][m]
    if not runs:
        continue
    r = runs[0]
    sv = np.linalg.svd(r["W_rec_f"], compute_uv=False)
    label = f"{nb}-back, {nn}n, {METHOD_LABELS[m]}"
    ax.plot(sv, ls=ls, lw=1.4,
            color="#2C7BB6" if m == "es" else "#C44E52",
            alpha=0.9 if "128" in str(nn) else 0.55,
            label=label)

ax.legend(fontsize=6.5, framealpha=0.7)
ax.set_yscale("log")
label_panel(ax)

# ── title & save ─────────────────────────────────────────────────────────────

fig.suptitle(
    "Network performance, weight change, and effective rank\n"
    "across n-back levels, network sizes, and training methods",
    fontsize=14, fontweight="bold", y=0.995
)

out = Path("results/fig_overview.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved {out}")
plt.close(fig)
