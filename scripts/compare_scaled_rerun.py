#!/usr/bin/env python3
"""
Before/after comparison of EA accuracy and W_rec effective rank
after applying dimension-aware scaling (--scale-pop, --scale-sigma).

Compares the stored stats_10seed.json (old, unscaled) against
fresh metrics loaded from results/pub/ (new, post-rerun).

Usage:
    python3 scripts/compare_scaled_rerun.py
    python3 scripts/compare_scaled_rerun.py --results-dir results/pub \
        --stats results/stats_10seed.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

METHODS_EA  = ["es", "ga", "ga_oja"]
METHOD_LABEL = {"es": "ES", "ga": "GA", "ga_oja": "GA+Oja"}
NBACKS  = [1, 2, 3, 4]
NEURONS = [32, 64]   # 128 was already correct — not re-examined
SEEDS   = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

THRESHOLD_PP = 5.0   # flag accuracy change ≥ 5pp


# ── helpers ───────────────────────────────────────────────────────────────────

def eff_rank(W, thr=0.90):
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s ** 2
    t = s2.sum()
    if t < 1e-12:
        return 1
    return int(np.searchsorted(np.cumsum(s2) / t, thr)) + 1


def load_new_metrics(pub_dir, task, nb, nn, method):
    """Load accuracy and eff_rank from fresh results for one condition."""
    accs, ranks = [], []
    for seed in SEEDS:
        if task == "nback":
            run_dir = pub_dir / f"nback{nb}_neurons{nn}_seed{seed}"
        else:
            run_dir = pub_dir / f"robot_T20_neurons{nn}_seed{seed}"
        mdir = run_dir / method
        hist_path  = mdir / "history.json"
        final_path = mdir / "weights_final.npz"
        if not hist_path.exists() or not final_path.exists():
            warnings.warn(f"  missing: {mdir}")
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        acc_list = hist.get("accuracy", [])
        if acc_list:
            accs.append(float(acc_list[-1]) * 100)
        fw = {k: v.astype(np.float64) for k, v in np.load(final_path).items()}
        if "W_rec" in fw:
            ranks.append(eff_rank(fw["W_rec"]))
    return accs, ranks


def mean_std(v):
    a = np.array(v, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def fmt(m, s):
    if np.isnan(m):
        return "       —"
    return f"{m:6.1f}±{s:.1f}"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "pub"))
    parser.add_argument("--stats",       default=str(ROOT / "results" / "stats_10seed.json"))
    args = parser.parse_args()

    pub_dir    = Path(args.results_dir)
    stats_path = Path(args.stats)

    if not pub_dir.is_dir():
        raise SystemExit(f"ERROR: {pub_dir} not found")

    # ── load old stats ────────────────────────────────────────────────────────
    old_stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            raw = json.load(f)
        # expected key pattern: "nback_{nb}_neurons_{nn}_{method}"
        for key, val in raw.items():
            old_stats[key] = val
    else:
        print(f"WARNING: {stats_path} not found — old values will show as NaN\n")

    def get_old(nb, nn, method, metric):
        key = f"nback_{nb}_neurons_{nn}_{method}"
        if key not in old_stats:
            return float("nan"), float("nan")
        entry = old_stats[key]
        vals = entry.get(metric, [])
        return mean_std([v * 100 if metric == "accuracy" else v for v in vals])

    # ── comparison table ──────────────────────────────────────────────────────
    header = (f"{'Condition':<32s}  {'Method':<8s}  "
              f"{'Old acc':>12s}  {'New acc':>12s}  {'Δacc':>7s}  "
              f"{'Old rank':>9s}  {'New rank':>9s}  {'Δrank':>6s}  Notes")
    sep = "─" * len(header)
    print(f"\n{header}")
    print(sep)

    significant_rows = []

    for nn in NEURONS:
        for nb in NBACKS:
            for method in METHODS_EA:
                label  = f"nback{nb}  N={nn}"
                mlabel = METHOD_LABEL[method]

                old_acc_m,  old_acc_s  = get_old(nb, nn, method, "accuracy")
                old_rank_m, old_rank_s = get_old(nb, nn, method, "eff_rank")

                new_accs, new_ranks = load_new_metrics(pub_dir, "nback", nb, nn, method)
                new_acc_m,  new_acc_s  = mean_std(new_accs)
                new_rank_m, new_rank_s = mean_std(new_ranks)

                d_acc  = new_acc_m  - old_acc_m
                d_rank = new_rank_m - old_rank_m

                notes = []
                if abs(d_acc) >= THRESHOLD_PP:
                    notes.append(f"SIGNIFICANT (acc {'+' if d_acc>0 else ''}{d_acc:.1f}pp)")
                if (old_acc_m < 20 and new_acc_m >= 20) or (old_acc_m >= 20 and new_acc_m < 20):
                    notes.append("DIRECTION CHANGE (above/below chance)")

                note_str = "  ← " + "; ".join(notes) if notes else ""

                d_acc_str  = f"{d_acc:+.1f}" if not np.isnan(d_acc)  else "   ?"
                d_rank_str = f"{d_rank:+.1f}" if not np.isnan(d_rank) else "  ?"

                print(f"{label:<32s}  {mlabel:<8s}  "
                      f"{fmt(old_acc_m, old_acc_s):>12s}  "
                      f"{fmt(new_acc_m, new_acc_s):>12s}  "
                      f"{d_acc_str:>7s}  "
                      f"{fmt(old_rank_m, old_rank_s):>9s}  "
                      f"{fmt(new_rank_m, new_rank_s):>9s}  "
                      f"{d_rank_str:>6s}"
                      f"{note_str}")

                if notes:
                    significant_rows.append((label, mlabel, d_acc, notes))

        print()  # blank line between N sizes

    # ── robot arm ─────────────────────────────────────────────────────────────
    print("── Robot arm ────────────────────────────────────────────────────────")
    for nn in NEURONS:
        for method in METHODS_EA:
            label  = f"robot  N={nn}"
            mlabel = METHOD_LABEL[method]

            # old robot stats key pattern may differ — try both
            key1 = f"robot_neurons_{nn}_{method}"
            key2 = f"robot_T20_neurons_{nn}_{method}"
            entry = old_stats.get(key1, old_stats.get(key2, {}))
            old_acc_vals = entry.get("accuracy", [])
            old_acc_m, old_acc_s = mean_std([v * 100 for v in old_acc_vals])

            new_accs, _ = load_new_metrics(pub_dir, "robot", None, nn, method)
            new_acc_m, new_acc_s = mean_std(new_accs)

            d_acc     = new_acc_m - old_acc_m
            d_acc_str = f"{d_acc:+.1f}" if not np.isnan(d_acc) else "   ?"
            notes = []
            if abs(d_acc) >= THRESHOLD_PP:
                notes.append(f"SIGNIFICANT ({'+' if d_acc>0 else ''}{d_acc:.1f}pp)")
            note_str = "  ← " + "; ".join(notes) if notes else ""

            print(f"{label:<32s}  {mlabel:<8s}  "
                  f"{fmt(old_acc_m, old_acc_s):>12s}  "
                  f"{fmt(new_acc_m, new_acc_s):>12s}  "
                  f"{d_acc_str:>7s}"
                  f"{note_str}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    if significant_rows:
        print(f"\n  {len(significant_rows)} condition(s) with |Δacc| ≥ {THRESHOLD_PP:.0f}pp:")
        for label, method, d_acc, notes in significant_rows:
            print(f"    {label}  {method}  {'+' if d_acc > 0 else ''}{d_acc:.1f}pp")
    else:
        print(f"\n  No conditions with |Δacc| ≥ {THRESHOLD_PP:.0f}pp.")
    print()


if __name__ == "__main__":
    main()
