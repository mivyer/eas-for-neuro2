"""
make_summary.py — generates a compact results_summary.json (and .txt) suitable
for pasting into Claude web for thesis analysis.

Usage:
    python3 scripts/make_summary.py
    python3 scripts/make_summary.py --results-dir results/nback --out results/results_summary
"""
import json
import os
import argparse
import numpy as np
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def effective_rank(W):
    """Effective rank via normalised entropy of squared singular values."""
    sv = np.linalg.svd(W, compute_uv=False)
    p = sv ** 2
    p = p / (p.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def downsample(arr, n=40):
    """Keep at most n evenly-spaced points from a list."""
    if len(arr) <= n:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return [arr[i] for i in idx]


def round_list(lst, decimals=4):
    return [round(float(x), decimals) for x in lst]


def weight_stats(npz_path):
    """Extract layer norms and effective ranks from a weights .npz file."""
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path)
    stats = {}
    for key in d.files:
        W = d[key]
        if W.ndim == 2:
            stats[key] = {
                "shape": list(W.shape),
                "frobenius_norm": round(float(np.linalg.norm(W, "fro")), 4),
                "effective_rank": round(effective_rank(W), 3),
                "sparsity": round(float(np.mean(np.abs(W) < 0.01)), 4),
            }
        elif W.ndim == 1:
            stats[key] = {
                "shape": list(W.shape),
                "mean": round(float(W.mean()), 4),
                "std": round(float(W.std()), 4),
            }
    return stats


def parse_history(method_dir, method):
    """Return a compact dict of key metrics from history.json."""
    h_path = method_dir / "history.json"
    if not h_path.exists():
        return None
    with open(h_path) as f:
        h = json.load(f)

    acc = h.get("accuracy", [])
    result = {}

    if method == "bptt":
        result["final_accuracy"] = round(float(acc[-1]), 4) if acc else None
        result["best_accuracy"] = round(float(max(acc)), 4) if acc else None
        result["accuracy_curve"] = round_list(downsample(acc))
        loss = h.get("loss", [])
        result["final_loss"] = round(float(loss[-1]), 4) if loss else None
        result["n_iters"] = len(acc)
    else:  # es / ga / ga_oja
        best = h.get("best_fitness", [])
        mean = h.get("fitness", acc)  # fitness == accuracy for EAs
        result["final_best_accuracy"] = round(float(best[-1]), 4) if best else None
        result["final_mean_accuracy"] = round(float(mean[-1]), 4) if mean else None
        result["best_accuracy_curve"] = round_list(downsample(best))
        result["mean_accuracy_curve"] = round_list(downsample(mean))
        result["n_gens"] = len(best) if best else len(mean)

    return result


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/nback")
    p.add_argument("--out", default="results/results_summary")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    experiments = {}
    methods = ["bptt", "es", "ga", "ga_oja"]

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        cfg_path = exp_dir / "config.json"
        if not cfg_path.exists():
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        task = cfg.get("task", "nback")
        # Skip robot arm entries — they share n_back values with nback runs
        # and would corrupt the aggregated accuracy table
        if task != "nback":
            continue

        key = exp_dir.name  # e.g. nback2_neurons32_seed42
        entry = {
            "n_back": cfg.get("n_back"),
            "n_neurons": cfg.get("n_neurons"),
            "seed": cfg.get("seed"),
            "ea_generations": cfg.get("ea_generations"),
            "bptt_iterations": cfg.get("bptt_iterations"),
            "methods": {},
        }

        for m in methods:
            m_dir = exp_dir / m
            if not m_dir.exists():
                continue

            hist = parse_history(m_dir, m)
            if hist is None:
                continue

            # Weight stats: prefer final weights; fall back to init
            wf = m_dir / "weights_final.npz"
            wi = m_dir / "weights_init.npz"
            wp = m_dir / "weights_post_oja.npz"  # ga_oja only

            w_entry = {"history": hist}
            ws = weight_stats(str(wf))
            if ws:
                w_entry["weights_final"] = ws
            ws_init = weight_stats(str(wi))
            if ws_init:
                w_entry["weights_init"] = ws_init
            if m == "ga_oja":
                ws_p = weight_stats(str(wp))
                if ws_p:
                    w_entry["weights_post_oja"] = ws_p

            entry["methods"][m] = w_entry

        if entry["methods"]:
            experiments[key] = entry

    # ── aggregate summary table ───────────────────────────────────────────────
    rows = []
    for exp_key, exp in experiments.items():
        row = {
            "exp": exp_key,
            "n_back": exp["n_back"],
            "n_neurons": exp["n_neurons"],
            "seed": exp["seed"],
        }
        for m in methods:
            if m not in exp["methods"]:
                row[f"{m}_acc"] = None
                continue
            h = exp["methods"][m]["history"]
            if m == "bptt":
                row[f"{m}_acc"] = h.get("final_accuracy")
            else:
                row[f"{m}_acc"] = h.get("final_best_accuracy")
        rows.append(row)

    # Sort by n_back, n_neurons, seed for readability
    rows.sort(key=lambda r: (r["n_back"] or 0, r["n_neurons"] or 0, r["seed"] or 0))

    # ── pivot: per (n_back, n_neurons) → mean ± std across seeds ─────────────
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        gkey = (r["n_back"], r["n_neurons"])
        for m in methods:
            v = r.get(f"{m}_acc")
            if v is not None:
                grouped[gkey][m].append(v)

    aggregated = {}
    for (nb, nn), mdata in sorted(grouped.items()):
        agg = {}
        for m, vals in mdata.items():
            agg[m] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "n_seeds": len(vals),
                "values": [round(v, 4) for v in vals],
            }
        aggregated[f"nback{nb}_neurons{nn}"] = agg

    output = {
        "meta": {
            "description": "Compact results summary for N-back task across methods/conditions",
            "methods": methods,
            "n_experiments": len(experiments),
        },
        "aggregated_accuracy": aggregated,
        "per_experiment_table": rows,
        "full_experiments": experiments,
    }

    # ── write JSON ────────────────────────────────────────────────────────────
    out_json = args.out + ".json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    size_kb = os.path.getsize(out_json) / 1024
    print(f"Wrote {out_json}  ({size_kb:.1f} KB)")

    # ── write human-readable .txt for Claude web ──────────────────────────────
    out_txt = args.out + ".txt"
    with open(out_txt, "w") as f:
        f.write("# N-BACK EXPERIMENT RESULTS SUMMARY\n\n")
        f.write("## AGGREGATED ACCURACY (mean ± std across seeds)\n")
        f.write(f"{'Condition':<30} {'BPTT':>10} {'ES':>10} {'GA':>10} {'GA+Oja':>10}\n")
        f.write("-" * 70 + "\n")
        for cond, mdata in sorted(aggregated.items()):
            def fmt(m):
                if m not in mdata:
                    return "  n/a   "
                v = mdata[m]
                return f"{v['mean']:.3f}±{v['std']:.3f}"
            f.write(f"{cond:<30} {fmt('bptt'):>10} {fmt('es'):>10} {fmt('ga'):>10} {fmt('ga_oja'):>10}\n")

        f.write("\n## PER-EXPERIMENT TABLE\n")
        header = f"{'experiment':<40} {'bptt':>7} {'es':>7} {'ga':>7} {'ga_oja':>8}\n"
        f.write(header)
        f.write("-" * 72 + "\n")
        for r in rows:
            def fv(m):
                v = r.get(f"{m}_acc")
                return f"{v:.3f}" if v is not None else "  n/a "
            f.write(f"{r['exp']:<40} {fv('bptt'):>7} {fv('es'):>7} {fv('ga'):>7} {fv('ga_oja'):>8}\n")

        f.write("\n## WEIGHT STRUCTURE (final weights, first seed available per condition)\n")
        seen = set()
        for exp_key, exp in experiments.items():
            cond = (exp["n_back"], exp["n_neurons"])
            if cond in seen:
                continue
            seen.add(cond)
            f.write(f"\n### {exp_key}\n")
            for m in methods:
                if m not in exp["methods"]:
                    continue
                wf = exp["methods"][m].get("weights_final", {})
                if not wf:
                    continue
                f.write(f"  [{m}]\n")
                for layer, stats in wf.items():
                    if "frobenius_norm" in stats:
                        f.write(f"    {layer}: shape={stats['shape']}  "
                                f"||W||_F={stats['frobenius_norm']}  "
                                f"eff_rank={stats['effective_rank']}  "
                                f"sparsity={stats['sparsity']}\n")

        f.write("\n## LEARNING CURVES (downsampled to ≤40 points)\n")
        f.write("(bptt: accuracy per iteration; ea: best_accuracy per generation)\n\n")
        for exp_key, exp in sorted(experiments.items())[:20]:  # limit for readability
            f.write(f"### {exp_key}\n")
            for m in methods:
                if m not in exp["methods"]:
                    continue
                h = exp["methods"][m]["history"]
                if m == "bptt":
                    curve = h.get("accuracy_curve", [])
                else:
                    curve = h.get("best_accuracy_curve", [])
                f.write(f"  {m}: {[round(v,3) for v in curve]}\n")
            f.write("\n")

    size_kb_txt = os.path.getsize(out_txt) / 1024
    print(f"Wrote {out_txt}  ({size_kb_txt:.1f} KB)")
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Conditions (n_back x n_neurons): {len(aggregated)}")
    print(f"\nAggregated accuracy table preview:")
    for cond, mdata in sorted(aggregated.items()):
        parts = [f"{m}={mdata[m]['mean']:.3f}" for m in methods if m in mdata]
        print(f"  {cond:<30} {', '.join(parts)}")


if __name__ == "__main__":
    main()
