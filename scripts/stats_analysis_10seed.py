"""
10-seed statistical analysis for thesis: BPTT vs ES vs GA vs GA+Oja on n-back / robot arm.

Data source: results/pub/ (consistent hyperparameters across all seeds).
"""

import argparse
import json
import os
import re
import sys
import io
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["bptt", "es", "ga", "ga_oja"]
METHOD_LABELS = {"bptt": "BPTT", "es": "ES", "ga": "GA", "ga_oja": "GA+Oja"}

# ── Metrics ──────────────────────────────────────────────────────────────────

def effective_rank(W, threshold=0.9):
    s = np.linalg.svd(W, compute_uv=False)
    s2 = s ** 2
    cumvar = np.cumsum(s2) / np.sum(s2)
    return int(np.searchsorted(cumvar, threshold)) + 1


def weight_change_fractions(init_d, final_d):
    deltas = {layer: np.linalg.norm(final_d[layer] - init_d[layer])
              for layer in ("W_in", "W_rec", "W_out")}
    total = sum(deltas.values())
    if total == 0:
        return {"W_in": 0.0, "W_rec": 0.0, "W_out": 0.0}
    return {k: v / total for k, v in deltas.items()}


def bootstrap_ci(values, stat=np.mean, n_boot=2000, ci=0.95, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    vals = np.array(values)
    boots = [stat(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo = (1 - ci) / 2
    return float(np.percentile(boots, lo * 100)), float(np.percentile(boots, (1 - lo) * 100))


def rank_biserial(U, n1, n2):
    return 1 - 2 * U / (n1 * n2)


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) * (len(a) - 1) + np.var(b, ddof=1) * (len(b) - 1))
                         / (len(a) + len(b) - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def holm_correction(p_values):
    """Holm-Šidák step-down correction. Returns adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = min(1.0, p_values[idx] * (n - rank))
        running_max = max(running_max, adj_p)
        adj[idx] = running_max
    return adj


# ── Data Loading ─────────────────────────────────────────────────────────────

NBACK_PAT = re.compile(r"nback(\d+)_neurons(\d+)_seed(\d+)$")
ROBOT_PAT = re.compile(r"robot_T20_neurons(\d+)_seed(\d+)$")


def _load_weights(path):
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k].astype(np.float64) for k in d.files}


def _load_method(method_dir: Path, method: str):
    if not method_dir.is_dir():
        return None
    hist_path = method_dir / "history.json"
    if not hist_path.exists():
        return None
    with open(hist_path) as f:
        history = json.load(f)
    acc = history.get("accuracy", [None])
    if not acc:
        return None
    accuracy = acc[-1]

    init_w = _load_weights(method_dir / "weights_init.npz")
    final_w = _load_weights(method_dir / "weights_final.npz")
    if init_w is None or final_w is None:
        return None

    fracs = weight_change_fractions(init_w, final_w)
    total_delta = sum(np.linalg.norm(final_w[k] - init_w[k]) for k in ("W_in", "W_rec", "W_out"))

    result = {
        "accuracy": float(accuracy),
        "eff_rank_rec": effective_rank(final_w["W_rec"]),
        "frac_in": fracs["W_in"],
        "frac_rec": fracs["W_rec"],
        "frac_out": fracs["W_out"],
        "total_delta_norm": float(total_delta),
    }

    if method == "ga_oja":
        post = _load_weights(method_dir / "weights_post_oja.npz")
        if post is not None:
            result["eff_rank_rec_post_oja"] = effective_rank(post["W_rec"])

    return result


def load_pub_data(pub_dir: Path):
    """
    Returns dict: data[(task, n_back, n_neurons)][method] = list of per-seed metric dicts.
    task is 'nback' or 'robot'. n_back is None for robot.
    """
    data = {}

    for entry in sorted(pub_dir.iterdir()):
        if not entry.is_dir() or entry.name == "logs":
            continue
        m = NBACK_PAT.match(entry.name)
        if m:
            key = ("nback", int(m.group(1)), int(m.group(2)))
        else:
            m = ROBOT_PAT.match(entry.name)
            if m:
                key = ("robot", None, int(m.group(1)))
            else:
                continue

        for method in METHODS:
            result = _load_method(entry / method, method)
            if result is None:
                continue
            if key not in data:
                data[key] = {m: [] for m in METHODS}
            data[key][method].append(result)

    return data


def get_metric(data, key, method, metric):
    return [r[metric] for r in data.get(key, {}).get(method, []) if metric in r]


# ── Report Helpers ────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self._buf = io.StringIO()

    def p(self, s=""):
        print(s)
        self._buf.write(s + "\n")

    def section(self, title):
        self.p()
        self.p("═" * 70)
        self.p(title)
        self.p("═" * 70)

    def sub(self, title):
        self.p()
        self.p(f"── {title}")

    def text(self):
        return self._buf.getvalue()


def fmt_p(p):
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def fmt_stat(name, val, p, extra=""):
    sig = "*" if p < 0.05 else " "
    return f"  {name}: {val:.3f}  {fmt_p(p)} {sig}  {extra}"


# ── Statistical Tests ─────────────────────────────────────────────────────────

def mwu(a, b, alternative="two-sided"):
    """Mann-Whitney U, returns (U, p, rank_biserial_r)."""
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    res = stats.mannwhitneyu(a, b, alternative=alternative)
    r = rank_biserial(res.statistic, len(a), len(b))
    return float(res.statistic), float(res.pvalue), float(r)


def kruskal(*groups):
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None, None
    res = stats.kruskal(*groups)
    return float(res.statistic), float(res.pvalue)


def spearman(x, y):
    if len(x) < 3:
        return None, None
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run(pub_dir: Path, out_dir: Path):
    rpt = Report()
    all_results = {}  # for JSON export

    rpt.p("═" * 70)
    rpt.p("THESIS STATISTICAL ANALYSIS — 10 Seeds")
    rpt.p(f"Data: {pub_dir}")
    rpt.p("═" * 70)

    data = load_pub_data(pub_dir)

    # ── SECTION 1: Sample sizes ───────────────────────────────────────────────
    rpt.section("SECTION 1: SAMPLE SIZES")
    rpt.p(f"  {'Condition':<30} {'BPTT':>5} {'ES':>5} {'GA':>5} {'GA+Oja':>7}")
    rpt.p("  " + "-" * 52)
    for key in sorted(data.keys()):
        task, nb, nn = key
        label = f"nback{nb}_N{nn}" if task == "nback" else f"robot_N{nn}"
        ns = [len(data[key].get(m, [])) for m in METHODS]
        rpt.p(f"  {label:<30} {ns[0]:>5} {ns[1]:>5} {ns[2]:>5} {ns[3]:>7}")
        for mi, m in enumerate(METHODS):
            if 0 < ns[mi] < 5:
                rpt.p(f"    WARNING: {label}/{m} has only {ns[mi]} seeds")

    # ── SECTION 2: Descriptive statistics ────────────────────────────────────
    rpt.section("SECTION 2: DESCRIPTIVE STATISTICS")
    desc_out = {}

    for metric, label in [("accuracy", "Accuracy"), ("eff_rank_rec", "Eff. Rank W_rec"),
                           ("frac_out", "Frac ΔW_out"), ("frac_rec", "Frac ΔW_rec")]:
        rpt.sub(label)
        rpt.p(f"  {'Condition':<28} {'Method':<10} {'Mean':>7} {'Std':>7} {'Median':>7} {'IQR':>9}")
        rpt.p("  " + "-" * 68)
        for key in sorted(data.keys()):
            task, nb, nn = key
            label_k = f"nback{nb}_N{nn}" if task == "nback" else f"robot_N{nn}"
            for m in METHODS:
                vals = get_metric(data, key, m, metric)
                if not vals:
                    continue
                a = np.array(vals)
                q25, q75 = np.percentile(a, [25, 75])
                rpt.p(f"  {label_k:<28} {METHOD_LABELS[m]:<10} "
                      f"{a.mean():>7.3f} {a.std(ddof=1):>7.3f} "
                      f"{np.median(a):>7.3f} {q75-q25:>9.3f}")
                desc_out.setdefault(str(key), {}).setdefault(m, {})[metric] = {
                    "mean": float(a.mean()), "std": float(a.std(ddof=1)),
                    "median": float(np.median(a)), "iqr": float(q75 - q25),
                    "n": len(vals),
                }
    all_results["descriptive"] = desc_out

    # ── SECTION 3: Effective rank — BPTT vs EA ───────────────────────────────
    rpt.section("SECTION 3: EFFECTIVE RANK — BPTT vs EVOLUTIONARY METHODS")
    rpt.p("  (all neuron sizes, n-back 1–4; Mann-Whitney U, two-sided)")

    eff_rank_tests = {}
    pvals_for_correction = []
    pval_labels = []

    for nn in [32, 64, 128]:
        rpt.sub(f"N={nn} neurons")
        eff_rank_tests[nn] = {}
        for nb in [1, 2, 3, 4]:
            key = ("nback", nb, nn)
            bptt_r = get_metric(data, key, "bptt", "eff_rank_rec")
            if not bptt_r:
                continue

            all_groups = [get_metric(data, key, m, "eff_rank_rec") for m in METHODS]
            H, p_kw = kruskal(*all_groups)
            kw_str = f"KW H={H:.2f} {fmt_p(p_kw)}" + (" *" if p_kw < 0.05 else "") if H is not None else ""

            ea_pool = []
            pairwise_strs = []
            nb_tests = {}
            for m in ["es", "ga", "ga_oja"]:
                ea_r = get_metric(data, key, m, "eff_rank_rec")
                ea_pool.extend(ea_r)
                U, p, r_rb = mwu(bptt_r, ea_r)
                if U is None:
                    continue
                sig = "*" if p < 0.05 else " "
                pairwise_strs.append(f"vs {METHOD_LABELS[m]} U={U:.0f} {fmt_p(p)}{sig} r={r_rb:.2f}")
                nb_tests[m] = {"U": U, "p": p, "r_rb": r_rb}
                pvals_for_correction.append(p)
                pval_labels.append(f"eff_rank N{nn}_nback{nb}_vs_{m}")

            U_pool, p_pool, r_pool = mwu(bptt_r, ea_pool)
            pool_str = ""
            if U_pool is not None:
                pool_str = f"  pooled U={U_pool:.0f} {fmt_p(p_pool)}" + (" *" if p_pool < 0.05 else "") + f" r={r_pool:.2f}"
                nb_tests["ea_pooled"] = {"U": U_pool, "p": p_pool, "r_rb": r_pool}

            bptt_mean = np.mean(bptt_r) if bptt_r else float("nan")
            ea_mean = np.mean(ea_pool) if ea_pool else float("nan")
            rpt.p(f"  nback{nb}  BPTT={bptt_mean:.1f} EA={ea_mean:.1f}  {kw_str}{pool_str}")
            for s in pairwise_strs:
                rpt.p(f"         {s}")

            eff_rank_tests[nn][nb] = nb_tests

    all_results["eff_rank_tests"] = eff_rank_tests

    # ── SECTION 4: W_out fraction trend with n-back difficulty ───────────────
    rpt.section("SECTION 4: W_out FRACTION TREND WITH N-BACK DIFFICULTY")
    rpt.p("  (all neuron sizes; Spearman ρ using all individual seed values)")

    frac_trend = {}
    for nn in [32, 64, 128]:
        rpt.sub(f"frac_out trend — N={nn} neurons")
        frac_trend[nn] = {}
        for m in METHODS:
            xs, ys = [], []
            for nb in [1, 2, 3, 4]:
                vals = get_metric(data, ("nback", nb, nn), m, "frac_out")
                xs.extend([nb] * len(vals))
                ys.extend(vals)
            rho, p_sp = spearman(xs, ys)
            if rho is None:
                continue
            sig = "*" if p_sp < 0.05 else " "
            rpt.p(f"  {METHOD_LABELS[m]:<10}  ρ={rho:+.3f}  {fmt_p(p_sp)} {sig}  (n={len(xs)})")
            frac_trend[nn][m] = {"rho": rho, "p": p_sp, "n": len(xs)}
            pvals_for_correction.append(p_sp)
            pval_labels.append(f"frac_out_trend_N{nn}_{m}")

    rpt.sub("W_rec fraction trend (all neuron sizes)")
    for nn in [32, 64, 128]:
        rpt.p(f"  N={nn}:")
        for m in METHODS:
            xs, ys = [], []
            for nb in [1, 2, 3, 4]:
                vals = get_metric(data, ("nback", nb, nn), m, "frac_rec")
                xs.extend([nb] * len(vals))
                ys.extend(vals)
            rho, p_sp = spearman(xs, ys)
            if rho is not None:
                sig = "*" if p_sp < 0.05 else " "
                rpt.p(f"    {METHOD_LABELS[m]:<10}  ρ={rho:+.3f}  {fmt_p(p_sp)} {sig}")

    all_results["frac_out_trend"] = frac_trend

    # ── SECTION 5: Cross-task comparison (n-back vs robot arm) ───────────────
    rpt.section("SECTION 5: CROSS-TASK COMPARISON — N-BACK vs ROBOT ARM")
    rpt.p("  (all neuron sizes; frac_rec nback pooled 1-4 vs robot)")

    cross_task = {}
    for nn in [32, 64, 128]:
        rpt.sub(f"N={nn} neurons")
        rpt.p(f"  {'Method':<10} {'Task':<8} {'frac_in':>8} {'frac_rec':>9} {'frac_out':>9}")
        rpt.p("  " + "-" * 46)
        cross_task[nn] = {}

        for m in METHODS:
            nback_fin, nback_frec, nback_fout = [], [], []
            for nb in [1, 2, 3, 4]:
                nback_fin.extend(get_metric(data, ("nback", nb, nn), m, "frac_in"))
                nback_frec.extend(get_metric(data, ("nback", nb, nn), m, "frac_rec"))
                nback_fout.extend(get_metric(data, ("nback", nb, nn), m, "frac_out"))
            robot_fin  = get_metric(data, ("robot", None, nn), m, "frac_in")
            robot_frec = get_metric(data, ("robot", None, nn), m, "frac_rec")
            robot_fout = get_metric(data, ("robot", None, nn), m, "frac_out")

            for task_label, fin, frec, fout in [
                ("nback", nback_fin, nback_frec, nback_fout),
                ("robot", robot_fin, robot_frec, robot_fout),
            ]:
                if frec:
                    rpt.p(f"  {METHOD_LABELS[m]:<10} {task_label:<8} "
                          f"{np.mean(fin):>8.3f} {np.mean(frec):>9.3f} {np.mean(fout):>9.3f}")

            U, p, r = mwu(nback_frec, robot_frec)
            sig = "*" if (p is not None and p < 0.05) else " "
            if U is not None:
                rpt.p(f"    → {METHOD_LABELS[m]} frac_rec nback vs robot: U={U:.0f} {fmt_p(p)} {sig} r_rb={r:.3f}")
                cross_task[nn][m] = {"U_frec": U, "p_frec": p, "r_rb_frec": r}
                pvals_for_correction.append(p)
                pval_labels.append(f"cross_task_N{nn}_frec_{m}")

    all_results["cross_task"] = cross_task

    # ── SECTION 5b: Magnitude of shift — does BPTT shift more than EA? ───────
    rpt.section("SECTION 5b: MAGNITUDE OF frac_rec SHIFT (robot − n-back)")
    rpt.p("  Per-seed delta = frac_rec_robot − mean(frac_rec_nback 1–4)")
    rpt.p("  Seeds are paired by index (same 10 seeds across all conditions)")
    rpt.p("  Mann-Whitney: BPTT delta vs each EA method's delta")

    shift_results = {}
    for nn in [32, 64, 128]:
        rpt.sub(f"N={nn} neurons")
        shift_results[nn] = {}
        method_deltas = {}

        for m in METHODS:
            # Per-seed mean frac_rec across n-back levels
            nback_per_seed = []
            for nb in [1, 2, 3, 4]:
                vals = get_metric(data, ("nback", nb, nn), m, "frac_rec")
                if not nback_per_seed:
                    nback_per_seed = [[v] for v in vals]
                else:
                    for i, v in enumerate(vals):
                        if i < len(nback_per_seed):
                            nback_per_seed[i].append(v)
            nback_means = [np.mean(s) for s in nback_per_seed]

            robot_vals = get_metric(data, ("robot", None, nn), m, "frac_rec")
            n = min(len(nback_means), len(robot_vals))
            if n < 2:
                continue
            deltas = [robot_vals[i] - nback_means[i] for i in range(n)]
            method_deltas[m] = deltas
            rpt.p(f"  {METHOD_LABELS[m]:<10}  delta={np.mean(deltas):+.3f} ± {np.std(deltas, ddof=1):.3f}"
                  f"  (robot {np.mean(robot_vals[:n]):.3f} − nback {np.mean(nback_means[:n]):.3f})")

        # BPTT delta vs each EA method's delta
        bptt_d = method_deltas.get("bptt", [])
        rpt.p("")
        for m in ["es", "ga", "ga_oja"]:
            ea_d = method_deltas.get(m, [])
            U, p, r = mwu(bptt_d, ea_d)
            if U is not None:
                sig = "*" if p < 0.05 else " "
                rpt.p(f"  BPTT vs {METHOD_LABELS[m]:<8} shift: U={U:.0f} {fmt_p(p)} {sig} r_rb={r:.3f}")
                shift_results[nn][m] = {"U": U, "p": p, "r_rb": r,
                                        "bptt_delta": float(np.mean(bptt_d)),
                                        "ea_delta": float(np.mean(ea_d))}
                pvals_for_correction.append(p)
                pval_labels.append(f"shift_magnitude_N{nn}_bptt_vs_{m}")

    all_results["shift_magnitude"] = shift_results

    # ── SECTION 6: Scaling analysis ───────────────────────────────────────────
    rpt.section("SECTION 6: SCALING ANALYSIS (N-back 4, neurons 32→128)")

    scaling = {}
    rpt.p(f"  {'Neurons':>8}  {'BPTT':>10} {'ES':>10} {'GA':>10} {'GA+Oja':>10}")
    rpt.p("  " + "-" * 52)
    for nn in [32, 64, 128]:
        key = ("nback", 4, nn)
        row = []
        for m in METHODS:
            vals = get_metric(data, key, m, "accuracy")
            row.append(f"{np.mean(vals):.3f}±{np.std(vals, ddof=1):.3f}" if vals else "  —  ")
        rpt.p(f"  {nn:>8}  {'  '.join(row)}")

    rpt.sub("Kruskal-Wallis on accuracy (all methods, each network size)")
    for nn in [32, 64, 128]:
        key = ("nback", 4, nn)
        groups = [get_metric(data, key, m, "accuracy") for m in METHODS]
        H, p_kw = kruskal(*groups)
        if H is not None:
            rpt.p(f"  N={nn}: H={H:.2f}  {fmt_p(p_kw)}" + (" *" if p_kw < 0.05 else ""))

    rpt.sub("Spearman: accuracy vs network size per method (n-back 4)")
    for m in METHODS:
        xs, ys = [], []
        for nn in [32, 64, 128]:
            vals = get_metric(data, ("nback", 4, nn), m, "accuracy")
            xs.extend([nn] * len(vals))
            ys.extend(vals)
        rho, p_sp = spearman(xs, ys)
        if rho is not None:
            sig = "*" if p_sp < 0.05 else " "
            rpt.p(f"  {METHOD_LABELS[m]:<10}  ρ={rho:+.3f}  {fmt_p(p_sp)} {sig}")
            scaling[m] = {"rho": rho, "p": p_sp}

    all_results["scaling"] = scaling

    # ── SECTION 7: GA vs GA+Oja ──────────────────────────────────────────────
    rpt.section("SECTION 7: GA vs GA+OJA")
    rpt.p("  (n-back 1–4 pooled; Mann-Whitney U; all neuron sizes)")

    oja_results = {}
    for nn in [32, 64, 128]:
        rpt.sub(f"N={nn} neurons (pooled n-back 1–4)")
        oja_results[nn] = {}
        for metric, label in [("accuracy", "Accuracy"), ("eff_rank_rec", "Eff. Rank"), ("frac_out", "Frac W_out")]:
            ga_vals, oja_vals = [], []
            for nb in [1, 2, 3, 4]:
                ga_vals.extend(get_metric(data, ("nback", nb, nn), "ga", metric))
                oja_vals.extend(get_metric(data, ("nback", nb, nn), "ga_oja", metric))
            U, p, r = mwu(ga_vals, oja_vals)
            sig = "*" if (p is not None and p < 0.05) else " "
            if U is not None:
                rpt.p(f"  {label:<14}  GA={np.mean(ga_vals):.3f}  GA+Oja={np.mean(oja_vals):.3f}"
                      f"  U={U:.0f}  {fmt_p(p)} {sig}  r_rb={r:.3f}")
                oja_results[nn][metric] = {"U": U, "p": p, "r_rb": r,
                                           "ga_mean": float(np.mean(ga_vals)),
                                           "oja_mean": float(np.mean(oja_vals))}
                pvals_for_correction.append(p)
                pval_labels.append(f"ga_vs_oja_N{nn}_{metric}")

    all_results["ga_vs_oja"] = oja_results

    # ── SECTION 8: Bootstrap CIs ─────────────────────────────────────────────
    rpt.section("SECTION 8: BOOTSTRAP 95% CONFIDENCE INTERVALS")
    rpt.p("  (Key metrics, 32 neurons, n-back 2; 2000 bootstrap resamples)")

    boot_results = {}
    for metric, label in [("accuracy", "Accuracy"), ("eff_rank_rec", "Eff. Rank"), ("frac_out", "Frac W_out")]:
        rpt.sub(label)
        for m in METHODS:
            for nb in [1, 2, 4]:
                vals = get_metric(data, ("nback", nb, 32), m, metric)
                if len(vals) < 3:
                    continue
                lo, hi = bootstrap_ci(vals)
                mean = np.mean(vals)
                rpt.p(f"  nback{nb} {METHOD_LABELS[m]:<10}  {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
                boot_results.setdefault(f"nback{nb}", {}).setdefault(m, {})[metric] = {
                    "mean": float(mean), "ci95_lo": lo, "ci95_hi": hi
                }

    all_results["bootstrap_ci"] = boot_results

    # ── SECTION 9: Multiple comparison correction ────────────────────────────
    rpt.section("SECTION 9: MULTIPLE COMPARISON CORRECTION (Holm-Šidák)")
    rpt.p(f"  {len(pvals_for_correction)} tests corrected")
    rpt.p()
    rpt.p(f"  {'Test':<42} {'p_raw':>8} {'p_adj':>8} {'sig':>4}")
    rpt.p("  " + "-" * 64)

    if pvals_for_correction:
        adj = holm_correction(pvals_for_correction)
        correction_out = []
        for label_c, p_raw, p_adj in sorted(
                zip(pval_labels, pvals_for_correction, adj), key=lambda x: x[1]):
            sig = "*" if p_adj < 0.05 else " "
            rpt.p(f"  {label_c:<42} {p_raw:>8.4f} {p_adj:>8.4f} {sig:>4}")
            correction_out.append({"test": label_c, "p_raw": p_raw, "p_adj": float(p_adj)})
        all_results["multiple_correction"] = correction_out

    # ── SECTION 10: Thesis claim summary ────────────────────────────────────
    rpt.section("SECTION 10: THESIS CLAIM SUMMARY")

    rpt.sub("Claim 1: BPTT produces lower effective rank than EA methods")
    rpt.p("  (r_rb > 0 means BPTT ranks lower = smaller eff. rank)")
    rpt.p(f"  {'':6} {'N=32':>16} {'N=64':>16} {'N=128':>16}")
    for nb in [1, 2, 3, 4]:
        row_parts = []
        for nn in [32, 64, 128]:
            pool = eff_rank_tests.get(nn, {}).get(nb, {}).get("ea_pooled", {})
            p = pool.get("p")
            r = pool.get("r_rb")
            if p is not None:
                sup = "SUP" if p < 0.05 and r > 0 else "NO"
                row_parts.append(f"{sup} p={p:.4f} r={r:.2f}")
            else:
                row_parts.append("—")
        rpt.p(f"  nback{nb}  {row_parts[0]:>16} {row_parts[1]:>16} {row_parts[2]:>16}")

    rpt.sub("Claim 2: BPTT W_out fraction increases with n-back difficulty")
    for nn in [32, 64, 128]:
        bt = frac_trend.get(nn, {}).get("bptt", {})
        if bt:
            supported = "SUPPORTED" if bt["p"] < 0.01 and bt["rho"] > 0.5 else "PARTIAL/NOT SUPPORTED"
            rpt.p(f"  N={nn}: BPTT ρ={bt['rho']:.3f}  p={bt['p']:.4f}  → {supported}")

    rpt.sub("Claim 3: EA methods do NOT show W_out trend")
    for nn in [32, 64, 128]:
        rpt.p(f"  N={nn}:")
        for m in ["es", "ga", "ga_oja"]:
            t = frac_trend.get(nn, {}).get(m, {})
            if t:
                supported = "no trend" if t["p"] > 0.05 else "TREND PRESENT"
                rpt.p(f"    {METHOD_LABELS[m]:<10} ρ={t['rho']:.3f}  p={t['p']:.4f}  → {supported}")

    rpt.sub("Claim 4: BPTT layer allocation reverses N-back→Robot arm")
    for nn in [32, 64, 128]:
        bptt_ct = cross_task.get(nn, {}).get("bptt", {})
        if bptt_ct:
            supported = "SUPPORTED" if bptt_ct["p_frec"] < 0.05 else "NOT SUPPORTED"
            rpt.p(f"  N={nn}: frac_rec diff p={bptt_ct['p_frec']:.4f}  r_rb={bptt_ct['r_rb_frec']:.3f}  → {supported}")

    rpt.sub("Claim 5: EA accuracy vs network size (n-back 4)")
    for m in ["es", "ga", "ga_oja"]:
        row = []
        for nn in [32, 64, 128]:
            vals = get_metric(data, ("nback", 4, nn), m, "accuracy")
            row.append(f"N{nn}={np.mean(vals):.3f}" if vals else f"N{nn}=—")
        rpt.p(f"  {METHOD_LABELS[m]}: {', '.join(row)}")

    return rpt, all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(ROOT / "results" / "pub"),
                   help="Directory containing experiment subdirectories")
    p.add_argument("--out", default=str(ROOT / "results"),
                   help="Output directory for report and JSON")
    args = p.parse_args()

    pub_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pub_dir.is_dir():
        print(f"ERROR: {pub_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    rpt, all_results = run(pub_dir, out_dir)

    report_path = out_dir / "stats_10seed_report.txt"
    with open(report_path, "w") as f:
        f.write(rpt.text())
    print(f"\nReport saved: {report_path}")

    json_path = out_dir / "stats_10seed.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved:   {json_path}")
