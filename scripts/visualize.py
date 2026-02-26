# visualize_outputs.py
"""
Visualization for EA vs BPTT on the n-back recall task.

Key figures:
  1. Output evolution: how network outputs change over training (snapshots)
  2. Learning dynamics: fitness/accuracy/loss curves
  3. Sample trial comparison: EA vs BPTT on same trial
  4. Progressive difficulty: accuracy vs n-back level
  5. Weight structure: how W_rec changes
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available")

from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import (
    LetterNBackTask, SYMBOL_VALUES, SYMBOL_LABELS,
    decode_output, N_SYMBOLS
)

try:
    import torch
    from models.bptt_rnn import RNNPolicy
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Helpers
# ============================================================================

def _run_trial(policy, task, rng):
    """Run one trial, return (outputs, inputs, targets, letters)."""
    inputs, targets, letters = task.get_trial(rng=rng)
    policy.reset()
    outputs = []
    for t in range(task.total_steps):
        obs = np.array([inputs[t]], dtype=np.float32)
        a = policy.act(obs)
        outputs.append(float(a[0]) if hasattr(a, '__len__') else float(a))
    return np.array(outputs), inputs, targets, letters


def _make_bptt_wrapper(model):
    """Wrap a torch RNNPolicy for numpy-style act/reset."""
    class W:
        def __init__(s, m):
            s.m = m; s.m.eval(); s.h = None
        def reset(s):
            with torch.no_grad():
                s.h = s.m.h0.detach().clone()
        def act(s, obs):
            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32)
                s.h = torch.tanh(s.h @ s.m.W_rec.T + o @ s.m.W_in.T)
                return torch.tanh(s.h @ s.m.W_out.T).numpy()
    return W(model)


# ============================================================================
# 1. OUTPUT EVOLUTION — the money plot
#    Shows actual network output on a FIXED trial at different points
#    during training. You can literally see the network learning to recall.
# ============================================================================

def plot_output_evolution_from_training(conf, train_fn, method_name="EA",
                                         snapshot_points=None, seed=42):
    """
    Re-train and capture full output traces at snapshot points.
    This is the most informative plot: you see the raw outputs change.

    For use in a notebook or standalone. For post-hoc plotting from
    saved weights, use plot_output_evolution_from_snapshots() instead.
    """
    if not PLOT_AVAILABLE:
        return

    # This function is a template — the actual snapshots come from
    # train_ea / train_bptt which already capture them.
    # See plot_output_evolution_from_results() below.
    print("Use plot_output_evolution_from_results() with saved training results.")


def plot_output_evolution_from_results(ea_results, bptt_results, conf,
                                        save_dir=None):
    """
    THE KEY FIGURE: For EA and BPTT, show network output on a fixed trial
    at early/mid/late training, compared to the target.

    Uses the best weights at snapshot generations/iterations.
    Since we only save the final best weights (not at every snapshot),
    this shows initial vs final. For full evolution, see the dedicated
    training-with-snapshots functions.
    """
    if not PLOT_AVAILABLE:
        return
    if save_dir is None:
        save_dir = conf.output_dir

    task = LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)
    rng = np.random.default_rng(42)

    # Build policies: initial and final for each method
    policies = {}
    if ea_results:
        policies['EA init'] = RSNNPolicy(
            ea_results['W_rec_init'], ea_results['W_in_init'], ea_results['W_out_init'])
        policies['EA final'] = RSNNPolicy(
            ea_results['W_rec_final'], ea_results['W_in_final'], ea_results['W_out_final'])
    if bptt_results and TORCH_AVAILABLE:
        policies['BPTT init'] = RSNNPolicy(
            bptt_results['W_rec_init'], bptt_results['W_in_init'], bptt_results['W_out_init'])
        policies['BPTT final'] = _make_bptt_wrapper(bptt_results['model'])

    n_policies = len(policies)
    if n_policies == 0:
        return

    # Fixed trial
    trial_rng = np.random.default_rng(42)
    inputs, targets, letters = task.get_trial(rng=trial_rng)

    fig, axes = plt.subplots(n_policies + 1, 1,
                              figsize=(12, 2.5 * (n_policies + 1)),
                              sharex=True)

    t = np.arange(task.total_steps)
    colors_target = plt.cm.Set2(np.linspace(0, 1, N_SYMBOLS))

    # Row 0: Input sequence + target
    ax = axes[0]
    ax.step(t, inputs, 'k-', lw=2, where='mid', label='Input (current letter)')
    ax.step(t, targets, 'g--', lw=2, where='mid', label='Target (recall)')
    ax.axvspan(0, conf.n_back - 0.5, alpha=0.15, color='gray')
    # Mark symbol levels
    for i, sv in enumerate(SYMBOL_VALUES):
        ax.axhline(sv, color='gray', ls=':', alpha=0.3, lw=0.5)
    ax.set_ylabel('Letter')
    ax.set_yticks(list(SYMBOL_VALUES))
    ax.set_yticklabels(SYMBOL_LABELS)
    ax.set_title(f'{conf.n_back}-back Recall | Input & Target', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-0.1, 1.2)
    ax.grid(True, alpha=0.2)

    # Subsequent rows: each policy's output
    for row, (name, policy) in enumerate(policies.items(), start=1):
        trial_rng = np.random.default_rng(42)  # same trial every time
        outputs, _, _, _ = _run_trial(policy, task, trial_rng)

        ax = axes[row]
        ax.step(t, targets, 'g--', lw=1.5, alpha=0.5, where='mid', label='Target')
        ax.step(t, outputs, 'b-', lw=2, where='mid', label='Output')
        ax.axvspan(0, conf.n_back - 0.5, alpha=0.15, color='gray')

        # Color-code correctness
        for ti in range(conf.n_back, task.total_steps):
            correct = decode_output(outputs[ti]) == decode_output(targets[ti])
            color = '#2ecc71' if correct else '#e74c3c'
            ax.plot(ti, outputs[ti], 'o', color=color, markersize=5, zorder=3)

        for sv in SYMBOL_VALUES:
            ax.axhline(sv, color='gray', ls=':', alpha=0.3, lw=0.5)

        acc = task.compute_accuracy(outputs, targets)
        fit = task.evaluate_outputs(outputs, targets)
        ax.set_ylabel('Letter')
        ax.set_yticks(list(SYMBOL_VALUES))
        ax.set_yticklabels(SYMBOL_LABELS)
        ax.set_title(f'{name}  |  acc={acc:.0%}  fit={fit:+.4f}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-0.1, 1.2)
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    path = os.path.join(save_dir, 'output_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# 2. LEARNING DYNAMICS — fitness, accuracy, loss over training
# ============================================================================

def plot_learning_dynamics(ea_results, bptt_results, conf, save_dir=None):
    """
    3-panel figure: fitness, accuracy, loss/sparsity over training.
    EA and BPTT on same axes for direct comparison.
    """
    if not PLOT_AVAILABLE:
        return
    if save_dir is None:
        save_dir = conf.output_dir

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Fitness ---
    ax = axes[0]
    if ea_results:
        gens = np.arange(len(ea_results['history']['fitness']))
        ax.plot(gens, ea_results['history']['fitness'],
                'b-', alpha=0.3, lw=1, label='EA pop mean')
        ax.plot(gens, ea_results['history']['best_fitness'],
                'b-', lw=2, label='EA best-so-far')
    if bptt_results:
        iters = np.arange(len(bptt_results['history']['fitness']))
        # Scale BPTT x-axis to match EA generations
        if ea_results:
            scale = len(ea_results['history']['fitness']) / len(iters)
            bx = iters * scale
        else:
            bx = iters
        ax.plot(bx, bptt_results['history']['fitness'],
                'r-', lw=2, label='BPTT')
    ax.set_xlabel('Generation (EA) / Scaled Iter (BPTT)')
    ax.set_ylabel('Fitness (neg MSE)')
    ax.set_title('Fitness', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax = axes[1]
    if ea_results:
        ax.plot(gens, ea_results['history']['accuracy'],
                'b-', lw=2, alpha=0.7, label='EA')
    if bptt_results:
        ax.plot(bx, bptt_results['history']['accuracy'],
                'r-', lw=2, label='BPTT')
    # Chance line (1/5 = 20%)
    ax.axhline(1.0 / N_SYMBOLS, color='gray', ls='--', alpha=0.5, label=f'Chance ({1/N_SYMBOLS:.0%})')
    ax.set_xlabel('Generation / Scaled Iter')
    ax.set_ylabel('Symbol Recall Accuracy')
    ax.set_title('Accuracy', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Loss (BPTT) / Weight norm (EA) ---
    ax = axes[2]
    if bptt_results:
        ax.plot(np.arange(len(bptt_results['history']['loss'])),
                bptt_results['history']['loss'],
                'r-', lw=1.5, alpha=0.7, label='BPTT loss')
        ax.set_ylabel('MSE Loss', color='r')
        ax.tick_params(axis='y', labelcolor='r')
    if ea_results:
        ax2 = ax.twinx() if bptt_results else ax
        # Compute weight norm over training from best_fitness as proxy
        # (we don't have per-gen weight norms, so show fitness envelope)
        best = ea_results['history']['best_fitness']
        ax2.plot(gens, best, 'b-', lw=2, label='EA best fitness')
        ax2.set_ylabel('EA Best Fitness', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
    ax.set_xlabel('Iteration / Generation')
    ax.set_title('Loss / Fitness Progression', fontweight='bold')
    ax.grid(True, alpha=0.3)

    task_label = f"{conf.n_back}-back" if conf.task == "nback" else "WM"
    plt.suptitle(f'{task_label} | {conf.n_neurons}N | Learning Dynamics',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'learning_dynamics.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# 3. MULTI-TRIAL COMPARISON — multiple trials side by side
# ============================================================================

def plot_multi_trial(ea_results, bptt_results, conf, n_trials=4, save_dir=None):
    """
    Show EA and BPTT outputs on multiple different random trials.
    Reveals whether the model generalizes or just memorizes one pattern.
    """
    if not PLOT_AVAILABLE:
        return
    if save_dir is None:
        save_dir = conf.output_dir

    task = LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)

    methods = {}
    if ea_results:
        methods['EA'] = RSNNPolicy(
            ea_results['W_rec_final'], ea_results['W_in_final'], ea_results['W_out_final'])
    if bptt_results and TORCH_AVAILABLE:
        methods['BPTT'] = _make_bptt_wrapper(bptt_results['model'])

    n_methods = len(methods)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(n_trials, n_methods, figsize=(7 * n_methods, 3 * n_trials),
                              squeeze=False)

    for trial_idx in range(n_trials):
        rng = np.random.default_rng(100 + trial_idx)

        for col, (name, policy) in enumerate(methods.items()):
            trial_rng = np.random.default_rng(100 + trial_idx)  # same trial both methods
            outputs, inputs, targets, letters = _run_trial(policy, task, trial_rng)

            ax = axes[trial_idx, col]
            t = np.arange(task.total_steps)

            ax.step(t, targets, 'g--', lw=1.5, alpha=0.6, where='mid', label='Target')
            ax.step(t, outputs, 'b-', lw=2, where='mid', label='Output')
            ax.axvspan(0, conf.n_back - 0.5, alpha=0.1, color='gray')

            # Correctness dots
            for ti in range(conf.n_back, task.total_steps):
                correct = decode_output(outputs[ti]) == decode_output(targets[ti])
                ax.plot(ti, outputs[ti], 'o',
                        color='#2ecc71' if correct else '#e74c3c',
                        markersize=4, zorder=3)

            for sv in SYMBOL_VALUES:
                ax.axhline(sv, color='gray', ls=':', alpha=0.2, lw=0.5)

            acc = task.compute_accuracy(outputs, targets)
            ax.set_ylim(-0.1, 1.2)
            ax.set_yticks(list(SYMBOL_VALUES))
            ax.set_yticklabels(SYMBOL_LABELS)
            ax.grid(True, alpha=0.2)

            if trial_idx == 0:
                ax.set_title(f'{name}', fontweight='bold', fontsize=12)
            # Letter sequence annotation
            letter_str = ''.join(SYMBOL_LABELS[l] for l in letters)
            ax.text(0.02, 0.95, f'Trial {trial_idx+1}: {letter_str}  acc={acc:.0%}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    fontfamily='monospace', alpha=0.7)

            if trial_idx == n_trials - 1:
                ax.set_xlabel('Step')
            if col == 0:
                ax.set_ylabel('Value')

    plt.suptitle(f'{conf.n_back}-back Recall | {conf.n_neurons}N | Multiple Trials',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'multi_trial.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# 4. PROGRESSIVE DIFFICULTY — accuracy vs n-back level
# ============================================================================

def plot_difficulty_sweep(ea_results, bptt_results, conf,
                          n_values=(1, 2, 3, 4), n_trials=50, save_dir=None):
    """
    Train on one n-back level, test on multiple.
    Shows where each method breaks down.
    """
    if not PLOT_AVAILABLE:
        return
    if save_dir is None:
        save_dir = conf.output_dir

    methods = {}
    if ea_results:
        methods['EA'] = RSNNPolicy(
            ea_results['W_rec_final'], ea_results['W_in_final'], ea_results['W_out_final'])
    if bptt_results and TORCH_AVAILABLE:
        methods['BPTT'] = _make_bptt_wrapper(bptt_results['model'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for name, policy in methods.items():
        accs, fits = [], []
        for n in n_values:
            task = LetterNBackTask(n_back=n, seq_length=conf.seq_length)
            rng = np.random.default_rng(42)
            r = task.evaluate_policy(policy, n_trials=n_trials, rng=rng)
            accs.append(r['accuracy'])
            fits.append(r['fitness'])

        color = 'b' if name == 'EA' else 'r'
        axes[0].plot(n_values, accs, f'{color}-o', lw=2, ms=8, label=name)
        axes[1].plot(n_values, fits, f'{color}-o', lw=2, ms=8, label=name)

    axes[0].axhline(1/N_SYMBOLS, color='gray', ls='--', alpha=0.5, label='Chance')
    axes[0].set_xlabel('N-back level'); axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Recall Accuracy vs Difficulty', fontweight='bold')
    axes[0].set_ylim(-0.05, 1.05); axes[0].set_xticks(list(n_values))
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('N-back level'); axes[1].set_ylabel('Fitness (neg MSE)')
    axes[1].set_title('Fitness vs Difficulty', fontweight='bold')
    axes[1].set_xticks(list(n_values))
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    trained_n = conf.n_back
    for ax in axes:
        ax.axvline(trained_n, color='green', ls='--', alpha=0.3, lw=2)
        ax.text(trained_n + 0.1, ax.get_ylim()[1] * 0.9, f'trained on {trained_n}-back',
                fontsize=8, color='green')

    plt.suptitle(f'{conf.n_neurons}N | Trained on {trained_n}-back | Tested across levels',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'difficulty_sweep.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# 5. WEIGHT STRUCTURE — how recurrent weights change
# ============================================================================

def plot_weight_analysis(ea_results, bptt_results, conf, save_dir=None):
    """
    Weight matrix heatmaps + distribution histograms.
    Shows what kind of structure each method finds.
    """
    if not PLOT_AVAILABLE:
        return
    if save_dir is None:
        save_dir = conf.output_dir

    panels = []
    if ea_results:
        panels.append(('EA init', ea_results['W_rec_init']))
        panels.append(('EA final', ea_results['W_rec_final']))
    if bptt_results:
        panels.append(('BPTT init', bptt_results['W_rec_init']))
        panels.append(('BPTT final', bptt_results['W_rec_final']))

    n = len(panels)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    if n == 1:
        axes = axes.reshape(-1, 1)

    vmax = max(np.abs(w).max() for _, w in panels)

    for col, (name, W) in enumerate(panels):
        # Heatmap
        ax = axes[0, col]
        im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('From'); ax.set_ylabel('To')

        # Histogram
        ax = axes[1, col]
        w_flat = W.flatten()
        ax.hist(w_flat, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
        sparsity = (np.abs(w_flat) < conf.sparsity_threshold).mean()
        ax.axvline(0, color='red', ls='--', lw=1)
        ax.set_title(f'sparsity={sparsity:.1%}  std={w_flat.std():.3f}', fontsize=9)
        ax.set_xlabel('Weight value')
        ax.set_ylabel('Count')

    fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.8, label='Weight')
    plt.suptitle(f'{conf.n_back}-back | {conf.n_neurons}N | Weight Structure',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'weight_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================================
# MASTER: generate all figures
# ============================================================================

def generate_all_figures(ea_results, bptt_results, conf):
    """Generate the full figure set from training results."""
    os.makedirs(conf.output_dir, exist_ok=True)

    print(f"\nGenerating figures in {conf.output_dir}/")
    print("-" * 40)

    plot_output_evolution_from_results(ea_results, bptt_results, conf)
    plot_learning_dynamics(ea_results, bptt_results, conf)
    plot_multi_trial(ea_results, bptt_results, conf, n_trials=4)
    plot_difficulty_sweep(ea_results, bptt_results, conf, n_values=(1, 2, 3, 4))
    plot_weight_analysis(ea_results, bptt_results, conf)

    print("-" * 40)
    print("All figures generated.")


# ============================================================================
# Standalone usage: load saved results and regenerate plots
# ============================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default='results/nback_32n')
    args = p.parse_args()

    d = args.results_dir

    # Load config
    with open(os.path.join(d, 'config.json')) as f:
        conf_dict = json.load(f)

    from run_full_analysis import Config
    conf = Config(**conf_dict)

    # Load EA
    ea = dict(np.load(os.path.join(d, 'ea_weights.npz')))
    with open(os.path.join(d, 'ea_history.json')) as f:
        ea_hist = json.load(f)
    ea['history'] = ea_hist['history']
    ea['snapshots'] = ea_hist.get('snapshots', {})
    ea['best_fitness'] = max(ea['history']['best_fitness'])

    # Load BPTT
    bptt = None
    bptt_path = os.path.join(d, 'bptt_weights.npz')
    if os.path.exists(bptt_path):
        bptt = dict(np.load(bptt_path))
        with open(os.path.join(d, 'bptt_history.json')) as f:
            bptt_hist = json.load(f)
        bptt['history'] = bptt_hist['history']
        bptt['snapshots'] = bptt_hist.get('snapshots', {})
        # Rebuild model from saved weights
        if TORCH_AVAILABLE:
            model = RNNPolicy(conf.n_neurons, conf.obs_dim, conf.action_dim)
            with torch.no_grad():
                model.W_rec.copy_(torch.tensor(bptt['W_rec_final']))
                model.W_in.copy_(torch.tensor(bptt['W_in_final']))
                model.W_out.copy_(torch.tensor(bptt['W_out_final']))
            bptt['model'] = model

    generate_all_figures(ea, bptt, conf)
