#!/usr/bin/env python3
# scripts/sanity_check_ga_stdp.py
"""
Sanity check: What is GA+STDP actually doing?

Produces diagnostic figures:
  1. Spike raster: which LIF neurons fire during a trial
  2. STDP weight changes: how W_rec changes within a single trial
  3. Membrane potential traces: are neurons actually spiking or silent?
  4. Output comparison: GA alone vs GA+STDP on same trial
  5. Evolved STDP parameters: what did the GA discover?

Run:
    python scripts/sanity_check_ga_stdp.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT = True
except ImportError:
    PLOT = False
    print("matplotlib not available")

from config import Config
from models.lif_rsnn import LIF_RSNN_NP, make_dale_mask, enforce_dale_weights
from models.stdp import STDP_Rule
from envs.letter_nback import LetterNBackTask, SYMBOL_LABELS, SYMBOL_VALUES, N_SYMBOLS


def run_trial_with_diagnostics(W_rec, W_in, W_out, stdp, task, rng,
                                beta=0.85, threshold=1.0):
    """
    Run one trial with full diagnostic recording.
    Returns dict with outputs, spikes, voltages, weight changes.
    """
    dale_mask = make_dale_mask(W_rec.shape[0])
    W_rec_start = W_rec.copy()
    W_rec_trial = W_rec.copy()

    net = LIF_RSNN_NP(W_rec_trial, W_in.copy(), W_out.copy(),
                       beta=beta, threshold=threshold, dale_mask=dale_mask)
    stdp.reset()

    inputs, targets, letters = task.get_trial(rng=rng)

    outputs = []
    voltages = []    # (T, N)
    all_spikes = []  # (T, N)
    w_rec_snapshots = [W_rec_start.copy()]

    for t in range(task.total_steps):
        obs = np.array([inputs[t]], dtype=np.float32)
        y = net.step(obs)
        outputs.append(float(y[0]) if hasattr(y, '__len__') else float(y))
        voltages.append(net.v.copy())
        all_spikes.append(net.s.copy())

        # STDP update
        net.W_rec = stdp.update(net.W_rec, net.s, net.s)

        if t % 5 == 0:
            w_rec_snapshots.append(net.W_rec.copy())

    W_rec_end = net.W_rec.copy()

    return {
        'outputs': np.array(outputs),
        'inputs': inputs,
        'targets': targets,
        'letters': letters,
        'spikes': np.stack(all_spikes),       # (T, N)
        'voltages': np.stack(voltages),       # (T, N)
        'W_rec_start': W_rec_start,
        'W_rec_end': W_rec_end,
        'W_rec_delta': W_rec_end - W_rec_start,
        'w_rec_snapshots': w_rec_snapshots,
    }


def plot_diagnostics(diag, title_prefix="", save_dir="results/sanity_check"):
    """Generate diagnostic figures."""
    if not PLOT:
        return

    os.makedirs(save_dir, exist_ok=True)
    T, N = diag['spikes'].shape

    fig = plt.figure(figsize=(16, 20))
    gs = gridspec.GridSpec(6, 2, hspace=0.4, wspace=0.3)

    # 1. Spike raster
    ax1 = fig.add_subplot(gs[0, :])
    spike_times, spike_neurons = np.where(diag['spikes'] > 0)
    if len(spike_times) > 0:
        ax1.scatter(spike_times, spike_neurons, s=1, c='black', marker='|')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Neuron')
    ax1.set_title(f'{title_prefix}Spike Raster')
    total_spikes = diag['spikes'].sum()
    ax1.text(0.02, 0.95, f'Total spikes: {int(total_spikes)}',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Membrane potentials (first 8 neurons)
    ax2 = fig.add_subplot(gs[1, :])
    n_show = min(8, N)
    for i in range(n_show):
        ax2.plot(diag['voltages'][:, i], alpha=0.7, linewidth=0.8, label=f'N{i}')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='threshold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Membrane potential')
    ax2.set_title(f'{title_prefix}Membrane Potentials (first {n_show} neurons)')
    ax2.legend(fontsize=7, ncol=5, loc='upper right')

    # 3. Input / Target / Output
    ax3 = fig.add_subplot(gs[2, :])
    ax3.plot(diag['inputs'], 'b-', alpha=0.5, label='input', linewidth=1)
    ax3.plot(diag['targets'], 'g--', alpha=0.8, label='target', linewidth=2)
    ax3.plot(diag['outputs'], 'r-', alpha=0.8, label='output', linewidth=1.5)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Value')
    ax3.set_title(f'{title_prefix}Input → Target → Output')
    ax3.legend(fontsize=9)
    # Add letter labels
    ax3.set_yticks(SYMBOL_VALUES)
    ax3.set_yticklabels(SYMBOL_LABELS)

    # 4. W_rec before/after STDP
    ax4a = fig.add_subplot(gs[3, 0])
    vmax = max(np.abs(diag['W_rec_start']).max(), 0.01)
    ax4a.imshow(diag['W_rec_start'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax4a.set_title('W_rec BEFORE trial')
    ax4a.set_xlabel('Pre')
    ax4a.set_ylabel('Post')

    ax4b = fig.add_subplot(gs[3, 1])
    ax4b.imshow(diag['W_rec_end'], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax4b.set_title('W_rec AFTER trial (STDP applied)')
    ax4b.set_xlabel('Pre')

    # 5. Weight change (delta)
    ax5 = fig.add_subplot(gs[4, 0])
    delta = diag['W_rec_delta']
    dvmax = max(np.abs(delta).max(), 1e-6)
    im = ax5.imshow(delta, cmap='RdBu_r', vmin=-dvmax, vmax=dvmax)
    ax5.set_title(f'ΔW_rec from STDP (max={dvmax:.4f})')
    plt.colorbar(im, ax=ax5, shrink=0.8)

    # 6. Weight change histogram
    ax6 = fig.add_subplot(gs[4, 1])
    flat_delta = delta.ravel()
    nonzero = flat_delta[np.abs(flat_delta) > 1e-8]
    if len(nonzero) > 0:
        ax6.hist(nonzero, bins=50, color='steelblue', edgecolor='none')
        ax6.set_title(f'ΔW distribution ({len(nonzero)} nonzero of {len(flat_delta)})')
    else:
        ax6.text(0.5, 0.5, 'No weight changes\n(no spikes?)',
                 ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('ΔW distribution')
    ax6.set_xlabel('ΔW')

    # 7. Firing rate per neuron
    ax7 = fig.add_subplot(gs[5, 0])
    firing_rates = diag['spikes'].mean(axis=0)  # per neuron
    colors = ['tab:blue' if i < int(0.8 * N) else 'tab:red' for i in range(N)]
    ax7.bar(range(N), firing_rates, color=colors, width=1.0)
    ax7.set_xlabel('Neuron')
    ax7.set_ylabel('Firing rate')
    ax7.set_title('Firing rate (blue=E, red=I)')

    # 8. Firing rate over time
    ax8 = fig.add_subplot(gs[5, 1])
    window = 5
    if T > window:
        pop_rate = np.convolve(diag['spikes'].sum(axis=1),
                               np.ones(window)/window, mode='valid')
        ax8.plot(pop_rate, 'k-', linewidth=1)
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Pop. spike count')
    ax8.set_title('Population firing rate')

    plt.suptitle(f'{title_prefix}GA+STDP Diagnostic', fontsize=14, fontweight='bold')
    path = os.path.join(save_dir, 'ga_stdp_diagnostic.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    rng = np.random.default_rng(42)
    N = 32
    task = LetterNBackTask(n_back=1, seq_length=20)
    dale_mask = make_dale_mask(N)

    # Random initial weights — scaled up so LIF neurons actually fire
    # With threshold=1.0 and beta=0.9, we need input current > 0.1 per step
    # to accumulate past threshold. Xavier scale (~0.25) is too small.
    scale = np.sqrt(2.0 / N)
    W_rec = enforce_dale_weights(
        scale * rng.standard_normal((N, N)).astype(np.float32), dale_mask)
    W_in = 2.0 * scale * rng.standard_normal((N, 1)).astype(np.float32)  # 2x input drive
    W_out = scale * rng.standard_normal((1, N)).astype(np.float32)

    # LIF params tuned for spiking activity
    beta = 0.9         # slower leak → voltage accumulates more
    threshold = 0.5    # lower threshold → easier to spike

    # Default STDP
    stdp = STDP_Rule(N, dale_mask=dale_mask)

    print("=" * 60)
    print("SANITY CHECK: GA+STDP Diagnostics")
    print("=" * 60)
    print(f"Network: {N} neurons, LIF, 80/20 E/I")
    print(f"LIF: beta={beta}, threshold={threshold}")
    print(f"STDP: A+={stdp.A_plus} A-={stdp.A_minus} "
          f"τ+={stdp.tau_plus} τ-={stdp.tau_minus}")
    print()

    # --- Run with random weights (no training) ---
    print("Running trial with random weights + STDP...")
    diag = run_trial_with_diagnostics(W_rec, W_in, W_out, stdp, task, rng,
                                       beta=beta, threshold=threshold)

    total_spikes = diag['spikes'].sum()
    print(f"  Total spikes: {int(total_spikes)}")
    print(f"  Mean firing rate: {diag['spikes'].mean():.4f}")
    print(f"  Max |ΔW|: {np.abs(diag['W_rec_delta']).max():.6f}")
    print(f"  Nonzero ΔW: {(np.abs(diag['W_rec_delta']) > 1e-8).sum()} "
          f"of {N*N}")

    if total_spikes == 0:
        print("\n⚠ NO SPIKES — LIF neurons not firing.")
        print("  This means STDP has nothing to work with.")
        print("  Likely cause: initial weights too small for threshold=1.0")
        print("  Fix: scale up W_in, lower threshold, or higher beta")

    plot_diagnostics(diag, title_prefix="Random weights: ")

    # --- Also run without STDP for comparison ---
    print("\nRunning same trial WITHOUT STDP for comparison...")
    no_stdp = STDP_Rule(N, A_plus=0, A_minus=0, dale_mask=dale_mask)
    diag_no_stdp = run_trial_with_diagnostics(W_rec, W_in, W_out, no_stdp, task, rng,
                                               beta=beta, threshold=threshold)
    print(f"  Output range: [{diag_no_stdp['outputs'].min():.3f}, "
          f"{diag_no_stdp['outputs'].max():.3f}]")

    print("\nDone. Check results/sanity_check/")


if __name__ == "__main__":
    main()