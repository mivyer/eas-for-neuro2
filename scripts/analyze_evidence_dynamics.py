#!/usr/bin/env python3
# scripts/analyze_evidence_dynamics.py
"""
Analyze and compare network dynamics for evidence accumulation task.

Loads saved best_gene (EA) or model.pt (BPTT) from a results directory,
then runs diagnostic trials to expose:

  1. Hidden-state trajectories — does the network actually integrate?
  2. W_rec eigenvalue spectrum — eigenvalues near 1 → slow integration
  3. Channel selectivity of W_in — does the network learn to amplify signal?
  4. Within-trial accumulation curves — does evidence build monotonically?
  5. Accuracy vs integration time — ablate early timesteps to test reliance on integration

Usage:
    python3 scripts/analyze_evidence_dynamics.py --result-dir results/evidence_s0.1_n0.5_neurons32_seed42
    python3 scripts/analyze_evidence_dynamics.py --result-dir results/evidence_s0.1_n0.5_neurons32_seed42 --save-figs
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.evidence_accumulation import EvidenceAccumulationTask
from models.rsnn_policy import RSNNPolicy


# ── helpers ──────────────────────────────────────────────────────────────────

def load_ea_weights(method_dir):
    """Load W_rec, W_in, W_out from weights_final.npz."""
    path = os.path.join(method_dir, 'weights_final.npz')
    if not os.path.exists(path):
        return None
    w = np.load(path)
    return w['W_rec'], w['W_in'], w['W_out']


def load_bptt_weights(method_dir):
    """Load weights from model.pt via torch (or fall back to weights_final.npz)."""
    pt_path = os.path.join(method_dir, 'model.pt')
    npz_path = os.path.join(method_dir, 'weights_final.npz')
    if os.path.exists(pt_path):
        import torch
        state = torch.load(pt_path, map_location='cpu', weights_only=True)
        W_rec = state.get('W_rec', state.get('rnn.weight_hh_l0', None))
        W_in  = state.get('W_in',  state.get('rnn.weight_ih_l0', None))
        W_out = state.get('W_out', state.get('fc.weight', None))
        if W_rec is not None:
            return (W_rec.numpy().astype(np.float32),
                    W_in.numpy().astype(np.float32),
                    W_out.numpy().astype(np.float32))
    if os.path.exists(npz_path):
        w = np.load(npz_path)
        return w['W_rec'], w['W_in'], w['W_out']
    return None


def run_recorded_trial(W_rec, W_in, W_out, task, rng, correct=None):
    """
    Run one trial, recording hidden states at every step.

    Returns:
        h_traj: (T, N) hidden states
        outputs: (T, K) raw logits before softmax
        inputs: (T, K) noisy inputs
        targets: (T,) int
        correct: int
    """
    if correct is None:
        inputs, targets, cats = task.get_trial(rng=rng)
        correct = int(cats[0])
    else:
        T, K = task.trial_length, task.n_categories
        inputs = (task.noise_std * rng.standard_normal((T, K))).astype(np.float32)
        inputs[:, correct] += task.evidence_strength
        targets = np.full(T, -1, dtype=np.int32)
        targets[task._accum_steps:] = correct

    T, N = task.trial_length, W_rec.shape[0]
    h = np.zeros(N, dtype=np.float32)
    h_traj   = np.zeros((T, N), dtype=np.float32)
    outputs  = np.zeros((T, task.n_categories), dtype=np.float32)

    for t in range(T):
        h = np.tanh(W_rec @ h + W_in @ inputs[t])
        h_traj[t]  = h
        outputs[t] = np.tanh(W_out @ h)

    return h_traj, outputs, inputs, targets, correct


def compute_accuracy_by_response_time(W_rec, W_in, W_out, task, rng,
                                       n_trials=200, response_onset_steps=None):
    """
    Accuracy if the network is forced to respond at different points in the trial.
    Tests whether early timesteps actually contribute to performance.

    response_onset_steps: list of ints — how many accumulation steps to use.
    Returns dict {onset: accuracy}.
    """
    if response_onset_steps is None:
        response_onset_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, task._accum_steps]

    results = {s: [] for s in response_onset_steps}

    for _ in range(n_trials):
        correct = int(rng.integers(0, task.n_categories))
        T = task.trial_length
        inputs = (task.noise_std * rng.standard_normal((T, task.n_categories))).astype(np.float32)
        inputs[:, correct] += task.evidence_strength

        h = np.zeros(W_rec.shape[0], dtype=np.float32)
        h_at = {}

        for t in range(task._accum_steps):
            h = np.tanh(W_rec @ h + W_in @ inputs[t])
            if (t + 1) in response_onset_steps:
                h_at[t + 1] = h.copy()

        for onset, h_snap in h_at.items():
            out = np.tanh(W_out @ h_snap)
            results[onset].append(int(np.argmax(out) == correct))

    return {k: float(np.mean(v)) for k, v in results.items() if v}


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze(result_dir, save_figs=False, n_trials=500):
    import json

    cfg_path = os.path.join(result_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        task = EvidenceAccumulationTask(
            evidence_strength=cfg.get('evidence_strength', 0.1),
            noise_std=cfg.get('noise_std', 0.5),
            trial_length=cfg.get('trial_length', 50),
            response_length=cfg.get('response_length', 5),
        )
    else:
        task = EvidenceAccumulationTask()

    rng = np.random.default_rng(0)

    methods = {}
    for m in ['bptt', 'es', 'ga', 'ga_oja']:
        mdir = os.path.join(result_dir, m)
        if not os.path.isdir(mdir):
            continue
        loader = load_bptt_weights if m == 'bptt' else load_ea_weights
        weights = loader(mdir)
        if weights is None:
            continue
        methods[m] = weights
        print(f"Loaded {m}: W_rec {weights[0].shape}")

    if not methods:
        print(f"No saved weights found in {result_dir}")
        print("Re-run with --save to persist weights.")
        return

    print(f"\nTask: T={task.trial_length}, resp={task.response_length}, "
          f"strength={task.evidence_strength}, noise={task.noise_std}")
    print(f"Chance = {1/task.n_categories:.0%}  |  Perfect integrator ≈ 80-90%\n")

    # ── 1. W_rec eigenvalue spectra ───────────────────────────────────────────
    print("=" * 60)
    print("W_rec eigenvalue spectra")
    print("  Near-1 eigenvalues → slow integration (good for evidence)")
    print("  All small eigenvalues → fast forgetting (bad for evidence)")
    print(f"{'Method':<10} {'max|λ|':>8} {'mean|λ|':>9} {'n_eigs>0.9':>12} {'n_eigs>0.5':>12}")
    print("-" * 55)
    eig_data = {}
    for m, (Wr, Wi, Wo) in methods.items():
        eigs = np.abs(np.linalg.eigvals(Wr))
        eig_data[m] = eigs
        print(f"{m:<10} {eigs.max():>8.4f} {eigs.mean():>9.4f} "
              f"{(eigs > 0.9).sum():>12d} {(eigs > 0.5).sum():>12d}")

    # ── 2. W_in channel selectivity ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("W_in channel selectivity")
    print("  Higher variance across input channels → more selective encoding")
    print(f"{'Method':<10} {'col_std_mean':>14} {'max_col_std':>12} {'SNR amplification':>20}")
    print("-" * 60)
    for m, (Wr, Wi, Wo) in methods.items():
        col_norms = np.linalg.norm(Wi, axis=0)          # (K,) — input weight norm per channel
        col_std   = Wi.std(axis=0)                       # (K,) — variation per input channel
        # SNR amplification: do high-noise channels get suppressed?
        snr_amp   = col_norms.max() / (col_norms.mean() + 1e-8)
        print(f"{m:<10} {col_std.mean():>14.4f} {col_std.max():>12.4f} {snr_amp:>20.3f}x")

    # ── 3. Accumulation curves: mean hidden-state magnitude vs time ───────────
    print("\n" + "=" * 60)
    print("Hidden-state drift: mean |h| over time (20 fixed-correct trials)")
    print("  Monotonic increase → network is integrating evidence")
    print(f"{'Method':<10}", end="")
    checkpoints = [1, 5, 10, 20, 30, 40, 45, 49]
    for t in checkpoints:
        print(f"  t={t:2d}", end="")
    print()
    print("-" * (10 + 7 * len(checkpoints)))
    for m, (Wr, Wi, Wo) in methods.items():
        all_h = []
        for _ in range(20):
            h_traj, outputs, inputs, targets, correct = run_recorded_trial(
                Wr, Wi, Wo, task, rng)
            all_h.append(h_traj)
        mean_h_mag = np.mean([np.abs(h).mean(axis=1) for h in all_h], axis=0)  # (T,)
        print(f"{m:<10}", end="")
        for t in checkpoints:
            print(f"  {mean_h_mag[t]:.3f}", end="")
        print()

    # ── 4. Accuracy vs evidence integration time ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"Accuracy vs integration steps (n={n_trials} trials per method/onset)")
    print("  If acc rises with more steps → genuine integration")
    print("  Flat curve → not using temporal accumulation")
    onset_steps = [1, 5, 10, 20, 30, 40, task._accum_steps]
    print(f"{'Method':<10}", end="")
    for s in onset_steps:
        print(f"  {s:4d}t", end="")
    print()
    print("-" * (10 + 7 * len(onset_steps)))
    for m, (Wr, Wi, Wo) in methods.items():
        acc_by_onset = compute_accuracy_by_response_time(
            Wr, Wi, Wo, task, rng, n_trials=n_trials, response_onset_steps=onset_steps)
        print(f"{m:<10}", end="")
        for s in onset_steps:
            v = acc_by_onset.get(s, float('nan'))
            print(f"  {v:.2f}", end="")
        print()
    print(f"  chance    ", end="")
    for s in onset_steps:
        print(f"  0.20", end="")
    print()

    # ── 5. Final accuracy (re-evaluated with many trials) ─────────────────────
    print("\n" + "=" * 60)
    print(f"Re-evaluated accuracy ({n_trials} trials, full integration):")
    for m, (Wr, Wi, Wo) in methods.items():
        policy = RSNNPolicy(Wr, Wi, Wo)
        r = task.evaluate_policy(policy, n_trials=n_trials, rng=rng)
        print(f"  {m:<10} acc={r['accuracy']:.1%} ± {r['fitness_std']:.3f}")

    # ── 6. Figures ────────────────────────────────────────────────────────────
    if save_figs:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig_dir = os.path.join(result_dir, 'figures_evidence')
            os.makedirs(fig_dir, exist_ok=True)

            colors = {'bptt': '#2196F3', 'es': '#FF9800', 'ga': '#4CAF50', 'ga_oja': '#9C27B0'}

            # Fig 1: Eigenvalue spectra
            fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5))
            if len(methods) == 1:
                axes = [axes]
            for ax, (m, eigs) in zip(axes, eig_data.items()):
                ax.hist(eigs, bins=20, color=colors.get(m, 'gray'), alpha=0.8)
                ax.axvline(1.0, color='red', ls='--', lw=1, label='λ=1')
                ax.axvline(0.9, color='orange', ls=':', lw=1, label='λ=0.9')
                ax.set_title(m.upper())
                ax.set_xlabel('|eigenvalue|')
                ax.set_ylabel('count' if ax == axes[0] else '')
                ax.legend(fontsize=8)
            fig.suptitle('W_rec Eigenvalue Spectra', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'fig_eigenspectra.png'), dpi=120)
            plt.close()

            # Fig 2: Accuracy vs integration time
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for m, (Wr, Wi, Wo) in methods.items():
                acc_curve = compute_accuracy_by_response_time(
                    Wr, Wi, Wo, task, rng, n_trials=200, response_onset_steps=list(range(1, task._accum_steps + 1)))
                xs = sorted(acc_curve.keys())
                ys = [acc_curve[x] for x in xs]
                ax.plot(xs, ys, label=m.upper(), color=colors.get(m, 'gray'), lw=2)
            ax.axhline(1 / task.n_categories, color='gray', ls='--', lw=1, label='chance')
            ax.set_xlabel('Integration steps before response')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy vs Evidence Integration Time')
            ax.legend()
            ax.set_ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'fig_integration_curve.png'), dpi=120)
            plt.close()

            # Fig 3: Sample hidden-state trajectories (best method only)
            best_m = max(methods.keys(), key=lambda m: sum(
                1 for eig in eig_data[m] if eig > 0.9))  # most integrator-like
            Wr, Wi, Wo = methods[best_m]
            fig, axes = plt.subplots(2, 3, figsize=(12, 6))
            for ax, cat in zip(axes.flat, range(task.n_categories)):
                h_traj, outputs, inputs, targets, correct = run_recorded_trial(
                    Wr, Wi, Wo, task, rng, correct=cat)
                # Plot first 5 PCs of h_traj
                from numpy.linalg import svd
                U, S, Vt = svd(h_traj - h_traj.mean(0), full_matrices=False)
                ax.plot(U[:, 0] * S[0], label='PC1', color='blue', lw=1.5)
                ax.plot(U[:, 1] * S[1], label='PC2', color='orange', lw=1.5)
                ax.axvline(task._accum_steps, color='red', ls='--', lw=1, label='response')
                ax.set_title(f'Correct={chr(65+cat)}  |{best_m.upper()}|')
                ax.set_xlabel('time')
                ax.legend(fontsize=7)
            axes.flat[-1].set_visible(False)
            fig.suptitle(f'Hidden-State Trajectories ({best_m.upper()})', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'fig_hidden_trajectories.png'), dpi=120)
            plt.close()

            print(f"\nFigures saved → {fig_dir}/")
        except ImportError:
            print("(matplotlib not available — skipping figures)")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Analyze evidence accumulation network dynamics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--result-dir', required=True,
                   help='Experiment directory (e.g. results/evidence_s0.1_n0.5_neurons32_seed42)')
    p.add_argument('--save-figs', action='store_true',
                   help='Save diagnostic figures to result_dir/figures_evidence/')
    p.add_argument('--n-trials', type=int, default=500)
    args = p.parse_args()

    analyze(args.result_dir, save_figs=args.save_figs, n_trials=args.n_trials)
