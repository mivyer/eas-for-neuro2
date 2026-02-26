# sweep_nback.py
"""
Progressive N-Back Sweep

Train EA and BPTT separately at each n-back level (1, 2, 3, 4).
Produces:
  1. Accuracy vs n-back level (the key figure)
  2. Learning curves per level
  3. Sample trial outputs per level
"""

import numpy as np
import time
import os
import json
from dataclasses import asdict

from run_full_analysis import Config, train_ea, train_bptt
from envs.letter_nback import (
    LetterNBackTask, SYMBOL_VALUES, SYMBOL_LABELS,
    decode_output, N_SYMBOLS
)
from models.rsnn_policy import RSNNPolicy

try:
    import torch
    from models.bptt_rnn import RNNPolicy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ============================================================================
# Helper
# ============================================================================

def _make_bptt_wrapper(model):
    class W:
        def __init__(s, m):
            s.m = m; s.m.eval(); s.h = None
        def reset(s):
            with torch.no_grad(): s.h = s.m.h0.detach().clone()
        def act(s, obs):
            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32)
                s.h = torch.tanh(s.h @ s.m.W_rec.T + o @ s.m.W_in.T)
                return torch.tanh(s.h @ s.m.W_out.T).numpy()
    return W(model)


def _run_trial(policy, task, rng):
    inputs, targets, letters = task.get_trial(rng=rng)
    policy.reset()
    outputs = []
    for t in range(task.total_steps):
        obs = np.array([inputs[t]], dtype=np.float32)
        a = policy.act(obs)
        outputs.append(float(a[0]) if hasattr(a, '__len__') else float(a))
    return np.array(outputs), inputs, targets, letters


# ============================================================================
# Sweep
# ============================================================================

def run_sweep(n_values=(1, 2, 3, 4), n_neurons=32, seed=42, output_dir="results/sweep"):
    """Train EA and BPTT at each n-back level. Returns all results."""

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for n in n_values:
        print("\n" + "=" * 60)
        print(f"  N-BACK LEVEL: {n}")
        print("=" * 60)

        conf = Config(
            task="nback",
            n_neurons=n_neurons,
            n_back=n,
            seq_length=20,
            seed=seed,
            output_dir=os.path.join(output_dir, f"nback_{n}"),
        )
        os.makedirs(conf.output_dir, exist_ok=True)

        # EA
        print(f"\n--- EA ({n}-back) ---")
        t0 = time.time()
        ea = train_ea(conf)
        ea_time = time.time() - t0
        print(f"EA time: {ea_time:.1f}s")

        # BPTT
        bptt = None
        bptt_time = 0
        if TORCH_AVAILABLE:
            print(f"\n--- BPTT ({n}-back) ---")
            t0 = time.time()
            bptt = train_bptt(conf)
            bptt_time = time.time() - t0
            print(f"BPTT time: {bptt_time:.1f}s")

        # Test both on 100 trials at this level
        task = LetterNBackTask(n_back=n, seq_length=conf.seq_length)
        rng = np.random.default_rng(seed + 999)

        ea_policy = RSNNPolicy(ea['W_rec_final'], ea['W_in_final'], ea['W_out_final'])
        ea_test = task.evaluate_policy(ea_policy, n_trials=100, rng=rng)

        bptt_test = None
        if bptt and TORCH_AVAILABLE:
            bptt_policy = _make_bptt_wrapper(bptt['model'])
            rng = np.random.default_rng(seed + 999)  # same trials
            bptt_test = task.evaluate_policy(bptt_policy, n_trials=100, rng=rng)

        all_results[n] = {
            'ea': ea, 'bptt': bptt, 'conf': conf,
            'ea_test': ea_test, 'bptt_test': bptt_test,
            'ea_time': ea_time, 'bptt_time': bptt_time,
        }

        print(f"\n  EA:   acc={ea_test['accuracy']:.1%}  fit={ea_test['fitness']:+.4f}")
        if bptt_test:
            print(f"  BPTT: acc={bptt_test['accuracy']:.1%}  fit={bptt_test['fitness']:+.4f}")

    # Save summary
    summary = {}
    for n, r in all_results.items():
        summary[n] = {
            'ea_accuracy': r['ea_test']['accuracy'],
            'ea_fitness': r['ea_test']['fitness'],
            'ea_time': r['ea_time'],
        }
        if r['bptt_test']:
            summary[n]['bptt_accuracy'] = r['bptt_test']['accuracy']
            summary[n]['bptt_fitness'] = r['bptt_test']['fitness']
            summary[n]['bptt_time'] = r['bptt_time']

    with open(os.path.join(output_dir, 'sweep_summary.json'), 'w') as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    return all_results, summary


# ============================================================================
# Plots
# ============================================================================

def plot_sweep(all_results, summary, output_dir="results/sweep"):
    if not PLOT_AVAILABLE:
        return

    n_values = sorted(all_results.keys())

    # ------------------------------------------------------------------
    # Figure 1: Accuracy vs n-back level (THE figure for the thesis)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ea_accs = [summary[n]['ea_accuracy'] for n in n_values]
    axes[0].plot(n_values, ea_accs, 'b-o', lw=2.5, ms=10, label='EA', zorder=3)

    if 'bptt_accuracy' in summary[n_values[0]]:
        bptt_accs = [summary[n]['bptt_accuracy'] for n in n_values]
        axes[0].plot(n_values, bptt_accs, 'r-s', lw=2.5, ms=10, label='BPTT', zorder=3)

    axes[0].axhline(1 / N_SYMBOLS, color='gray', ls='--', lw=1.5,
                     alpha=0.6, label=f'Chance ({1/N_SYMBOLS:.0%})')
    axes[0].set_xlabel('N-back Level', fontsize=12)
    axes[0].set_ylabel('Recall Accuracy', fontsize=12)
    axes[0].set_title('Performance vs Working Memory Demand', fontweight='bold', fontsize=13)
    axes[0].set_xticks(n_values)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Fitness
    ea_fits = [summary[n]['ea_fitness'] for n in n_values]
    axes[1].plot(n_values, ea_fits, 'b-o', lw=2.5, ms=10, label='EA')
    if 'bptt_fitness' in summary[n_values[0]]:
        bptt_fits = [summary[n]['bptt_fitness'] for n in n_values]
        axes[1].plot(n_values, bptt_fits, 'r-s', lw=2.5, ms=10, label='BPTT')
    axes[1].set_xlabel('N-back Level', fontsize=12)
    axes[1].set_ylabel('Fitness (neg MSE)', fontsize=12)
    axes[1].set_title('Fitness vs Working Memory Demand', fontweight='bold', fontsize=13)
    axes[1].set_xticks(n_values)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_accuracy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Figure 2: Learning curves at each level (2x4 grid)
    # ------------------------------------------------------------------
    n_levels = len(n_values)
    fig, axes = plt.subplots(2, n_levels, figsize=(4.5 * n_levels, 8), squeeze=False)

    for col, n in enumerate(n_values):
        r = all_results[n]

        # Fitness curves
        ax = axes[0, col]
        ea_hist = r['ea']['history']
        ax.plot(ea_hist['fitness'], 'b-', alpha=0.3, lw=1)
        ax.plot(ea_hist['best_fitness'], 'b-', lw=2, label='EA best')
        if r['bptt']:
            bptt_hist = r['bptt']['history']
            bx = np.linspace(0, len(ea_hist['fitness']),
                             len(bptt_hist['fitness']))
            ax.plot(bx, bptt_hist['fitness'], 'r-', lw=2, label='BPTT')
        ax.set_title(f'{n}-back', fontweight='bold', fontsize=12)
        ax.set_ylabel('Fitness')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

        # Accuracy curves
        ax = axes[1, col]
        ax.plot(ea_hist['accuracy'], 'b-', lw=2, alpha=0.7, label='EA')
        if r['bptt']:
            ax.plot(bx, bptt_hist['accuracy'], 'r-', lw=2, label='BPTT')
        ax.axhline(1 / N_SYMBOLS, color='gray', ls='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)

    plt.suptitle('Learning Curves by N-back Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Figure 3: Sample trial outputs at each level
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(n_levels, 2, figsize=(14, 3.5 * n_levels), squeeze=False)

    for row, n in enumerate(n_values):
        r = all_results[n]
        task = LetterNBackTask(n_back=n, seq_length=20)
        trial_rng = np.random.default_rng(42)

        # EA
        ea_policy = RSNNPolicy(
            r['ea']['W_rec_final'], r['ea']['W_in_final'], r['ea']['W_out_final'])
        outputs_ea, inputs, targets, letters = _run_trial(ea_policy, task, trial_rng)

        ax = axes[row, 0]
        t = np.arange(task.total_steps)
        ax.step(t, targets, 'g--', lw=1.5, alpha=0.6, where='mid', label='Target')
        ax.step(t, outputs_ea, 'b-', lw=2, where='mid', label='Output')
        ax.axvspan(0, n - 0.5, alpha=0.1, color='gray')
        for sv in SYMBOL_VALUES:
            ax.axhline(sv, color='gray', ls=':', alpha=0.2, lw=0.5)
        for ti in range(n, task.total_steps):
            ok = decode_output(outputs_ea[ti]) == decode_output(targets[ti])
            ax.plot(ti, outputs_ea[ti], 'o',
                    color='#2ecc71' if ok else '#e74c3c', ms=4, zorder=3)
        acc = task.compute_accuracy(outputs_ea, targets)
        ax.set_title(f'EA | {n}-back | acc={acc:.0%}', fontweight='bold')
        ax.set_ylim(-0.1, 1.2); ax.grid(True, alpha=0.2)
        ax.set_yticks(list(SYMBOL_VALUES))
        ax.set_yticklabels(SYMBOL_LABELS)
        if row == 0:
            ax.legend(fontsize=8)
        if row == n_levels - 1:
            ax.set_xlabel('Step')

        # BPTT
        ax = axes[row, 1]
        if r['bptt'] and TORCH_AVAILABLE:
            bptt_policy = _make_bptt_wrapper(r['bptt']['model'])
            trial_rng = np.random.default_rng(42)  # same trial
            outputs_bptt, _, _, _ = _run_trial(bptt_policy, task, trial_rng)
            ax.step(t, targets, 'g--', lw=1.5, alpha=0.6, where='mid', label='Target')
            ax.step(t, outputs_bptt, 'r-', lw=2, where='mid', label='Output')
            ax.axvspan(0, n - 0.5, alpha=0.1, color='gray')
            for sv in SYMBOL_VALUES:
                ax.axhline(sv, color='gray', ls=':', alpha=0.2, lw=0.5)
            for ti in range(n, task.total_steps):
                ok = decode_output(outputs_bptt[ti]) == decode_output(targets[ti])
                ax.plot(ti, outputs_bptt[ti], 'o',
                        color='#2ecc71' if ok else '#e74c3c', ms=4, zorder=3)
            acc = task.compute_accuracy(outputs_bptt, targets)
            ax.set_title(f'BPTT | {n}-back | acc={acc:.0%}', fontweight='bold')
        else:
            ax.set_title(f'BPTT | {n}-back | N/A')
        ax.set_ylim(-0.1, 1.2); ax.grid(True, alpha=0.2)
        ax.set_yticks(list(SYMBOL_VALUES))
        ax.set_yticklabels(SYMBOL_LABELS)
        if row == 0:
            ax.legend(fontsize=8)
        if row == n_levels - 1:
            ax.set_xlabel('Step')

    plt.suptitle('Sample Trials by N-back Level', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_sample_trials.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Figure 4: Training time comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ea_times = [all_results[n]['ea_time'] for n in n_values]
    ax.bar([n - 0.15 for n in n_values], ea_times, 0.3, color='steelblue', label='EA')
    if all_results[n_values[0]]['bptt_time']:
        bptt_times = [all_results[n]['bptt_time'] for n in n_values]
        ax.bar([n + 0.15 for n in n_values], bptt_times, 0.3, color='indianred', label='BPTT')
    ax.set_xlabel('N-back Level'); ax.set_ylabel('Time (s)')
    ax.set_title('Training Time', fontweight='bold')
    ax.set_xticks(n_values); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sweep_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# Print summary table
# ============================================================================

def print_summary_table(summary):
    n_values = sorted(int(k) for k in summary.keys())
    print("\n" + "=" * 65)
    print(f"{'N-back':>7} | {'EA acc':>8} {'EA fit':>9} {'EA time':>8} | "
          f"{'BPTT acc':>9} {'BPTT fit':>9} {'BPTT time':>9}")
    print("-" * 65)
    for n in n_values:
        s = summary[str(n)] if str(n) in summary else summary[n]
        ea_str = f"{s['ea_accuracy']:.1%}  {s['ea_fitness']:+.4f}  {s['ea_time']:7.1f}s"
        if 'bptt_accuracy' in s:
            bptt_str = f"{s['bptt_accuracy']:.1%}   {s['bptt_fitness']:+.4f}  {s['bptt_time']:8.1f}s"
        else:
            bptt_str = "   —         —          —"
        print(f"{n:>7} | {ea_str} | {bptt_str}")
    print("=" * 65)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Progressive n-back sweep")
    p.add_argument('--n-values', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--neurons', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default='results/sweep')
    args = p.parse_args()

    all_results, summary = run_sweep(
        n_values=tuple(args.n_values),
        n_neurons=args.neurons,
        seed=args.seed,
        output_dir=args.output,
    )

    print_summary_table(summary)

    if PLOT_AVAILABLE:
        plot_sweep(all_results, summary, output_dir=args.output)

    print(f"\nAll results saved to {args.output}/")