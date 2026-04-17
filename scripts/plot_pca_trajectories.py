#!/usr/bin/env python3
"""
PCA state-cloud visualization: representational dimensionality across methods.

Loads final trained weights for each method, runs many trials, collects all
hidden states, fits PCA, and plots the cloud of states in the top-3-PC space.

The key visual question: do the hidden states occupy a tight low-D manifold
(BPTT) or spread diffusely through high-D space (EA)?  The subtitle reports
what fraction of total activity variance is captured by the top 3 PCs — the
direct complement to the W_rec effective-rank finding.

Two figures are produced:
  1. pca_cloud_* : 2×2 grid of 3-D state clouds, one per method
  2. pca_cumvar_*: cumulative-variance-explained curve (all methods overlaid)

Usage
-----
  python scripts/plot_pca_trajectories.py --n-back 2 --neurons 64 --seed 42
  python scripts/plot_pca_trajectories.py --n-back 4 --neurons 32 --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3-D projection)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from envs.letter_nback import LetterNBackTask  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────

METHODS      = ["bptt", "es", "ga", "ga_oja"]
METHOD_LABEL = {"bptt": "BPTT", "es": "ES", "ga": "GA", "ga_oja": "GA+Oja"}
METHOD_COLOR = {"bptt": "#e41a1c", "es": "#377eb8", "ga": "#4daf4a", "ga_oja": "#984ea3"}


# ── PCA (numpy only) ──────────────────────────────────────────────────────────

def pca_fit(X: np.ndarray, n_components: int = 3):
    """
    Fit PCA on X (n_samples × n_features).

    Returns
    -------
    mean       : (n_features,)
    components : (n_components, n_features)
    evr        : (n_components,)   explained-variance ratios for top PCs
    all_evr    : (min(n,p),)       full spectrum, for the cumvar plot
    """
    mean = X.mean(axis=0)
    Xc   = X - mean
    n, p = Xc.shape

    if n >= p:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        all_var  = (S ** 2) / max(n - 1, 1)
        components = Vt
    else:
        C = Xc @ Xc.T / max(n - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(C)
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        vecs = Xc.T @ eigvecs
        vecs /= np.maximum(np.linalg.norm(vecs, axis=0, keepdims=True), 1e-10)
        components = vecs.T
        all_var = eigvals

    total_var = all_var.sum()
    all_evr   = all_var / max(total_var, 1e-10)
    return mean, components[:n_components], all_evr[:n_components], all_evr


def pca_transform(X, mean, components):
    return (X - mean) @ components.T


# ── model helpers ─────────────────────────────────────────────────────────────

def load_weights(npz_path: Path) -> dict:
    d = np.load(npz_path)
    return {k: d[k].astype(np.float64) for k in d.files}


def rollout(W_rec, W_in, h0, inputs: np.ndarray) -> np.ndarray:
    """inputs: (T, obs_dim) → returns (T, N) hidden states."""
    h = h0.copy()
    states = []
    for t in range(len(inputs)):
        h = np.tanh(W_rec @ h + W_in @ inputs[t])
        states.append(h.copy())
    return np.array(states, dtype=np.float64)


def load_weights_for_method(method_dir: Path, method: str):
    candidates = []
    if method == "ga_oja":
        candidates.append(method_dir / "weights_post_oja.npz")
    candidates.append(method_dir / "weights_final.npz")
    npz = next((p for p in candidates if p.exists()), None)
    if npz is None:
        return None
    w  = load_weights(npz)
    N  = w["W_rec"].shape[0]
    h0 = w.get("h0", np.zeros(N, dtype=np.float64))
    return w, h0


# ── 3-D state-cloud plot ──────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: float = 0.80):
    """Mix hex_color toward white by `amount` (0 = original, 1 = white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1-r)*amount, g + (1-g)*amount, b + (1-b)*amount)


def plot_state_cloud(ax, proj_all: np.ndarray, seq_len: int,
                     var_exp: np.ndarray, title: str, color: str):
    """
    Scatter all hidden states in PC1-PC2-PC3 space.

    proj_all : (n_trials * seq_len, 3)   — all states projected to top 3 PCs
    seq_len  : timesteps per trial        — used to color by time-within-trial
    var_exp  : (3,)                       — explained-variance ratios
    """
    T        = seq_len
    n_total  = len(proj_all)
    n_trials = n_total // T

    # Color points by time within trial: early = light, late = full method color
    t_norm = np.tile(np.linspace(0, 1, T), n_trials)
    cmap   = mcolors.LinearSegmentedColormap.from_list(
                 "", [_lighten(color, 0.80), color])

    ax.scatter(proj_all[:, 0], proj_all[:, 1], proj_all[:, 2],
               c=t_norm, cmap=cmap,
               s=4, alpha=0.35, edgecolors="none", depthshade=False)

    total_var = var_exp.sum()
    ax.set_title(f"{title}\n{total_var:.0%} of variance in top 3 PCs",
                 fontsize=10, pad=4)
    ax.set_xlabel("PC 1", fontsize=8, labelpad=1)
    ax.set_ylabel("PC 2", fontsize=8, labelpad=1)
    ax.set_zlabel("PC 3", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=6, pad=0)


# ── cumulative-variance figure ────────────────────────────────────────────────

def plot_cumvar(all_evr_dict: dict, n_back: int, neurons: int,
                out_dir: Path, seed: int):
    fig, ax = plt.subplots(figsize=(6, 4))

    for method, evr in all_evr_dict.items():
        cumvar = np.cumsum(evr)
        k = np.arange(1, len(cumvar) + 1)
        ax.plot(k, cumvar, color=METHOD_COLOR[method],
                label=METHOD_LABEL[method], linewidth=2)
        cross = np.searchsorted(cumvar, 0.90)
        if cross < len(cumvar):
            ax.scatter(cross + 1, cumvar[cross],
                       color=METHOD_COLOR[method], s=60, zorder=5)

    ax.axhline(0.90, color="0.4", linestyle="--", linewidth=0.9,
               label="90% threshold")
    ax.set_xlabel("Number of PCs", fontsize=11)
    ax.set_ylabel("Cumulative explained variance", fontsize=11)
    ax.set_title(
        f"Representational dimensionality  |  {n_back}-back  N={neurons} neurons",
        fontsize=11)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, None)
    ax.legend(fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    stem = f"pca_cumvar_nback{n_back}_N{neurons}_seed{seed}"
    for ext in ("pdf", "png"):
        fpath = out_dir / f"{stem}.{ext}"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-back",   type=int, default=2)
    parser.add_argument("--neurons",  type=int, default=64)
    parser.add_argument("--seed",     type=int, default=42,
                        help="Model seed (selects weights directory)")
    parser.add_argument("--seq-len",  type=int, default=25)
    parser.add_argument("--n-trials", type=int, default=200,
                        help="Trials to collect states from (default 200)")
    parser.add_argument("--elev",     type=float, default=25)
    parser.add_argument("--azim",     type=float, default=-60)
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "pub"))
    parser.add_argument("--out",      default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()

    pub_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"nback{args.n_back}_neurons{args.neurons}_seed{args.seed}"
    exp_dir  = pub_dir / exp_name
    if not exp_dir.is_dir():
        sys.exit(f"ERROR: {exp_dir} not found.\n"
                 f"Available: {[d.name for d in pub_dir.iterdir() if d.is_dir()]}")

    task = LetterNBackTask(n_back=args.n_back, seq_length=args.seq_len)

    # ── per-method: collect states, fit PCA, project ──────────────────────────
    results   = {}
    all_evr_d = {}

    # Same trial pool for every method so comparison is fair
    rng = np.random.default_rng(99999)
    trial_inputs = []
    for _ in range(args.n_trials):
        inp, _, _ = task.get_trial(rng=rng)
        trial_inputs.append(inp)

    for method in METHODS:
        method_dir = exp_dir / method
        loaded = load_weights_for_method(method_dir, method)
        if loaded is None:
            print(f"  skipping {method}: weights not found")
            continue
        w, h0 = loaded

        # Collect hidden states across all trials → (n_trials * T, N)
        all_states = np.vstack([
            rollout(w["W_rec"], w["W_in"], h0, inp.astype(np.float64))
            for inp in trial_inputs
        ])

        mean, components, var_exp, all_evr = pca_fit(all_states, n_components=3)
        proj_all = pca_transform(all_states, mean, components)

        results[method]   = {"proj_all": proj_all, "var_exp": var_exp}
        all_evr_d[method] = all_evr

        eff_rank = int(np.searchsorted(np.cumsum(all_evr), 0.90)) + 1
        print(f"  {METHOD_LABEL[method]:8s}  "
              f"top-3 PCs: {var_exp[0]:.1%} / {var_exp[1]:.1%} / {var_exp[2]:.1%}  "
              f"(cumulative {var_exp.sum():.1%})   eff-rank@90%: {eff_rank}")

    if not results:
        sys.exit("No methods found — check --results-dir and --seed.")

    # ── Figure 1: state-cloud grid ────────────────────────────────────────────
    n_methods = len(results)
    ncols = 2
    nrows = (n_methods + 1) // 2

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows + 0.8))
    fig.suptitle(
        f"{args.n_back}-back  ·  N={args.neurons} neurons  ·  model seed {args.seed}  "
        f"·  {args.n_trials} trials × T={args.seq_len}\n"
        f"Each point = one hidden state  ·  color = time within trial "
        f"(light → dark = early → late)",
        fontsize=10
    )

    for idx, (method, res) in enumerate(results.items()):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.view_init(elev=args.elev, azim=args.azim)
        plot_state_cloud(
            ax,
            proj_all = res["proj_all"],
            seq_len  = args.seq_len,
            var_exp  = res["var_exp"],
            title    = METHOD_LABEL[method],
            color    = METHOD_COLOR[method],
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    stem = f"pca_cloud_nback{args.n_back}_N{args.neurons}_seed{args.seed}"
    for ext in ("pdf", "png"):
        fpath = out_dir / f"{stem}.{ext}"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")
    plt.close()

    # ── Figure 2: cumulative variance explained ───────────────────────────────
    plot_cumvar(all_evr_d, args.n_back, args.neurons, out_dir, args.seed)


if __name__ == "__main__":
    main()
