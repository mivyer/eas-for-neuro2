"""PCA dimensionality analysis of recurrent hidden-state trajectories.

Loads trained weights, runs a few trials through RSNNPolicy, and performs PCA
on the concatenated activity matrix to quantify effective dimensionality per method.
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import LetterNBackTask
from envs.robot_arm import RobotArmTask

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METHODS = ["bptt", "es", "ga", "ga_oja"]
METHOD_LABELS = {"bptt": "BPTT", "es": "ES", "ga": "GA", "ga_oja": "GA+Oja"}
COLORS = {"bptt": "#2196F3", "es": "#FF9800", "ga": "#4CAF50", "ga_oja": "#E91E63"}

NBACK_LEVELS = [1, 2, 4]
NEURON_SIZES = [32, 64, 128, 256]
N_TRIALS = 8
SEED = 42

# Search paths for weights (checked in order)
def _weight_path(task_tag: str, nn: int, seed: int, method: str) -> Path | None:
    candidates = [
        ROOT / "results" / "nback" / f"{task_tag}_neurons{nn}_seed{seed}" / method / "weights_final.npz",
        ROOT / "results" / "robot" / f"{task_tag}_neurons{nn}_seed{seed}" / method / "weights_final.npz",
        ROOT / "results" / "pub" / f"{task_tag}_neurons{nn}_seed{seed}" / method / "weights_final.npz",
        ROOT / "results" / f"{task_tag}_neurons{nn}_seed{seed}" / method / "weights_final.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_policy(path: Path) -> RSNNPolicy:
    d = np.load(path)
    return RSNNPolicy(d["W_rec"], d["W_in"], d["W_out"])


def collect_activity(policy: RSNNPolicy, task, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Return (total_timesteps, N) hidden-state matrix across n_trials."""
    states = []
    for _ in range(n_trials):
        inputs, *_ = task.get_trial(rng=rng)
        policy.reset()
        for t in range(len(inputs)):
            policy.act(inputs[t])
            states.append(policy.h.copy())
    return np.array(states, dtype=np.float32)


def pca_thresholds(activity: np.ndarray) -> dict:
    """Run PCA; return explained variance and PC counts for 80/90/95%."""
    X = activity - activity.mean(axis=0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    ev = s ** 2
    ev_ratio = ev / ev.sum()
    cum = np.cumsum(ev_ratio)
    def n_for(thr):
        idx = np.searchsorted(cum, thr)
        return int(min(idx + 1, len(cum)))
    return {
        "cum_var": cum,
        "ev_ratio": ev_ratio,
        "pc80": n_for(0.80),
        "pc90": n_for(0.90),
        "pc95": n_for(0.95),
        "n_dims": activity.shape[1],
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(save: bool = False):
    rng = np.random.default_rng(SEED)
    results = []  # list of dicts

    # --- N-back conditions ---
    for nb in NBACK_LEVELS:
        task = LetterNBackTask(n_back=nb, seq_length=20)
        tag = f"nback{nb}"
        for nn in NEURON_SIZES:
            row = {"task": "nback", "n_back": nb, "neurons": nn}
            found_any = False
            for method in METHODS:
                path = _weight_path(tag, nn, SEED, method)
                if path is None:
                    continue
                found_any = True
                policy = load_policy(path)
                activity = collect_activity(policy, task, N_TRIALS, rng)
                stats = pca_thresholds(activity)
                results.append({**row, "method": method, **stats})
            if not found_any:
                print(f"  [skip] {tag} neurons={nn} — no weights found")

    # --- Robot arm conditions ---
    task_robot = RobotArmTask(seq_length=20)
    tag_robot = "robot_T20"
    for nn in NEURON_SIZES:
        row = {"task": "robot", "n_back": None, "neurons": nn}
        found_any = False
        for method in METHODS:
            path = _weight_path(tag_robot, nn, SEED, method)
            if path is None:
                continue
            found_any = True
            policy = load_policy(path)
            activity = collect_activity(policy, task_robot, N_TRIALS, rng)
            stats = pca_thresholds(activity)
            results.append({**row, "method": method, **stats})
        if not found_any:
            print(f"  [skip] robot neurons={nn} — no weights found")

    return results


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(results):
    header = f"{'Method':<10} {'Task':<8} {'N-back':>6} {'Neurons':>7} {'PC80':>5} {'PC90':>5} {'PC95':>5}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        nb = str(r["n_back"]) if r["n_back"] is not None else "—"
        print(f"{METHOD_LABELS[r['method']]:<10} {r['task']:<8} {nb:>6} {r['neurons']:>7} "
              f"{r['pc80']:>5} {r['pc90']:>5} {r['pc95']:>5}")
    print()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(results, save: bool = False):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Figure 1: cumulative variance curves ---
    # rows = neuron sizes, cols = (nback1, nback2, nback4, robot)
    conditions = [("nback", 1), ("nback", 2), ("nback", 4), ("robot", None)]
    col_labels = ["N-back 1", "N-back 2", "N-back 4", "Robot arm"]
    nrows = len(NEURON_SIZES)
    ncols = len(conditions)

    fig1, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows), sharey=True)
    fig1.suptitle("Cumulative explained variance of hidden-state PCA", fontsize=14, y=1.01)

    for i, nn in enumerate(NEURON_SIZES):
        for j, (task_name, nb) in enumerate(conditions):
            ax = axes[i, j]
            ax.set_xlim(0, nn)
            ax.set_ylim(0, 1.05)
            ax.axhline(0.9, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.set_xlabel("PCs" if i == nrows - 1 else "")
            ax.set_ylabel("Cum. var." if j == 0 else "")
            if i == 0:
                ax.set_title(col_labels[j], fontsize=11)
            ax.text(0.98, 0.04, f"N={nn}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="gray")

            for r in results:
                if r["task"] == task_name and r["n_back"] == nb and r["neurons"] == nn:
                    m = r["method"]
                    cum = r["cum_var"]
                    xs = np.arange(1, len(cum) + 1)
                    ax.plot(xs, cum, color=COLORS[m], lw=1.8, label=METHOD_LABELS[m])
                    ax.axvline(r["pc90"], color=COLORS[m], lw=0.8, ls=":", alpha=0.7)
                    ax.text(r["pc90"] + 0.3, 0.91, str(r["pc90"]),
                            color=COLORS[m], fontsize=7, va="bottom")

            # legend only on first row, last column
            if i == 0 and j == ncols - 1:
                ax.legend(fontsize=8, loc="lower right")

    fig1.tight_layout()

    # --- Figure 2: bar chart summary (PCs for 90%) ---
    # Build x-axis: one group per (task, neurons)
    group_keys = []
    for nn in NEURON_SIZES:
        for task_name, nb in conditions:
            group_keys.append((task_name, nb, nn))

    group_labels = []
    for task_name, nb, nn in group_keys:
        if task_name == "nback":
            group_labels.append(f"nback{nb}\nN={nn}")
        else:
            group_labels.append(f"robot\nN={nn}")

    n_groups = len(group_keys)
    n_methods = len(METHODS)
    bar_w = 0.18
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_w
    xs = np.arange(n_groups)

    fig2, ax2 = plt.subplots(figsize=(max(14, n_groups * 0.8), 5))
    ax2.set_title("PCs required for 90% explained variance", fontsize=13)

    for mi, method in enumerate(METHODS):
        vals = []
        for task_name, nb, nn in group_keys:
            match = [r for r in results
                     if r["method"] == method and r["task"] == task_name
                     and r["n_back"] == nb and r["neurons"] == nn]
            vals.append(match[0]["pc90"] if match else np.nan)
        ax2.bar(xs + offsets[mi], vals, width=bar_w,
                color=COLORS[method], label=METHOD_LABELS[method], alpha=0.85)

    ax2.set_xticks(xs)
    ax2.set_xticklabels(group_labels, fontsize=8)
    ax2.set_ylabel("PCs for 90% variance")
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()

    if save:
        out = ROOT / "results"
        out.mkdir(exist_ok=True)
        fig1.savefig(out / "fig_pca_activity.png", dpi=300, bbox_inches="tight")
        fig2.savefig(out / "fig_pca_summary.png", dpi=300, bbox_inches="tight")
        print(f"Saved: results/fig_pca_activity.png")
        print(f"Saved: results/fig_pca_summary.png")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PCA dimensionality analysis of RNN activity")
    p.add_argument("--save", action="store_true", help="Save figures to results/")
    p.add_argument("--no-fig", action="store_true", help="Skip figure generation")
    p.add_argument("--nback-levels", nargs="+", type=int, default=NBACK_LEVELS,
                   metavar="N", help="N-back levels to include")
    p.add_argument("--neurons", nargs="+", type=int, default=NEURON_SIZES,
                   metavar="N", help="Network sizes to include")
    p.add_argument("--trials", type=int, default=N_TRIALS,
                   help="Trials per condition")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    NBACK_LEVELS[:] = args.nback_levels
    NEURON_SIZES[:] = args.neurons
    N_TRIALS = args.trials
    SEED = args.seed

    print("Running PCA analysis...")
    results = run_analysis(save=args.save)

    if not results:
        print("No results found — check that weight files exist in results/")
        sys.exit(1)

    print_table(results)

    if not args.no_fig:
        make_figures(results, save=args.save)
