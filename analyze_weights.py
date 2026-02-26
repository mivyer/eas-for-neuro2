# analyze_weights.py
"""
Analyze where weight changes occur in BPTT vs EA trained networks.

Questions:
- Do BPTT and EA modify recurrent weights differently than input/output weights?
- Which method relies more on recurrent dynamics vs input/output transformations?
- Are there structural differences in what each method learns?
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from train_ea import train_ea, EAConfig, params_to_weights, init_params
from train_bptt import train_bptt, BPTTConfig


def analyze_ea_weights(conf: EAConfig, results: dict) -> dict:
    """Extract weight statistics from EA training."""
    
    # Get initial weights
    rng = np.random.default_rng(conf.seed)
    init_p = init_params(conf, rng)
    W_rec_init, W_in_init, W_out_init = params_to_weights(init_p, conf)
    
    # Get final weights
    W_rec_final, W_in_final, W_out_final = params_to_weights(results['best_params'], conf)
    
    # Compute changes
    delta_rec = W_rec_final - W_rec_init
    delta_in = W_in_final - W_in_init
    delta_out = W_out_final - W_out_init
    
    return {
        'init': {'W_rec': W_rec_init, 'W_in': W_in_init, 'W_out': W_out_init},
        'final': {'W_rec': W_rec_final, 'W_in': W_in_final, 'W_out': W_out_final},
        'delta': {'W_rec': delta_rec, 'W_in': delta_in, 'W_out': delta_out},
        'stats': {
            'rec_change_norm': np.linalg.norm(delta_rec),
            'in_change_norm': np.linalg.norm(delta_in),
            'out_change_norm': np.linalg.norm(delta_out),
            'rec_change_mean': np.abs(delta_rec).mean(),
            'in_change_mean': np.abs(delta_in).mean(),
            'out_change_mean': np.abs(delta_out).mean(),
            'rec_final_norm': np.linalg.norm(W_rec_final),
            'in_final_norm': np.linalg.norm(W_in_final),
            'out_final_norm': np.linalg.norm(W_out_final),
        }
    }


def analyze_bptt_weights(conf: BPTTConfig, results: dict) -> dict:
    """Extract weight statistics from BPTT training."""
    import torch
    
    model = results['model']
    
    with torch.no_grad():
        W_rec_final = model.W_rec.cpu().numpy()
        W_in_final = model.W_in.cpu().numpy()
        W_out_final = model.W_out.cpu().numpy()
    
    # Recreate initial weights (same seed)
    torch.manual_seed(conf.seed)
    from models.bptt_rnn import RNNPolicy
    init_model = RNNPolicy(conf.n_neurons, conf.obs_dim, conf.action_dim)
    
    with torch.no_grad():
        W_rec_init = init_model.W_rec.cpu().numpy()
        W_in_init = init_model.W_in.cpu().numpy()
        W_out_init = init_model.W_out.cpu().numpy()
    
    # Compute changes
    delta_rec = W_rec_final - W_rec_init
    delta_in = W_in_final - W_in_init
    delta_out = W_out_final - W_out_init
    
    return {
        'init': {'W_rec': W_rec_init, 'W_in': W_in_init, 'W_out': W_out_init},
        'final': {'W_rec': W_rec_final, 'W_in': W_in_final, 'W_out': W_out_final},
        'delta': {'W_rec': delta_rec, 'W_in': delta_in, 'W_out': delta_out},
        'stats': {
            'rec_change_norm': np.linalg.norm(delta_rec),
            'in_change_norm': np.linalg.norm(delta_in),
            'out_change_norm': np.linalg.norm(delta_out),
            'rec_change_mean': np.abs(delta_rec).mean(),
            'in_change_mean': np.abs(delta_in).mean(),
            'out_change_mean': np.abs(delta_out).mean(),
            'rec_final_norm': np.linalg.norm(W_rec_final),
            'in_final_norm': np.linalg.norm(W_in_final),
            'out_final_norm': np.linalg.norm(W_out_final),
        }
    }


def plot_weight_comparison(ea_analysis: dict, bptt_analysis: dict, save_path: str = 'weight_analysis.png'):
    """Create visualization comparing weight changes."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # =========================================================================
    # Row 1: Weight change magnitude comparison (bar chart)
    # =========================================================================
    ax1 = fig.add_subplot(3, 3, 1)
    
    labels = ['W_rec\n(recurrent)', 'W_in\n(input)', 'W_out\n(output)']
    x = np.arange(len(labels))
    width = 0.35
    
    # Normalize by number of parameters for fair comparison
    n_rec = ea_analysis['delta']['W_rec'].size
    n_in = ea_analysis['delta']['W_in'].size
    n_out = ea_analysis['delta']['W_out'].size
    
    ea_changes = [
        ea_analysis['stats']['rec_change_norm'] / np.sqrt(n_rec),
        ea_analysis['stats']['in_change_norm'] / np.sqrt(n_in),
        ea_analysis['stats']['out_change_norm'] / np.sqrt(n_out),
    ]
    bptt_changes = [
        bptt_analysis['stats']['rec_change_norm'] / np.sqrt(n_rec),
        bptt_analysis['stats']['in_change_norm'] / np.sqrt(n_in),
        bptt_analysis['stats']['out_change_norm'] / np.sqrt(n_out),
    ]
    
    bars1 = ax1.bar(x - width/2, ea_changes, width, label='EA', color='#3498db')
    bars2 = ax1.bar(x + width/2, bptt_changes, width, label='BPTT', color='#e74c3c')
    
    ax1.set_ylabel('RMS Weight Change')
    ax1.set_title('Weight Change Magnitude\n(normalized by √n_params)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Row 1: Relative contribution (pie charts)
    # =========================================================================
    ax2 = fig.add_subplot(3, 3, 2)
    ea_total = sum([ea_analysis['stats']['rec_change_norm'],
                    ea_analysis['stats']['in_change_norm'],
                    ea_analysis['stats']['out_change_norm']])
    ea_fracs = [ea_analysis['stats']['rec_change_norm'] / ea_total,
                ea_analysis['stats']['in_change_norm'] / ea_total,
                ea_analysis['stats']['out_change_norm'] / ea_total]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax2.pie(ea_fracs, labels=['Recurrent', 'Input', 'Output'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('EA: Where Changes Occur', fontweight='bold')
    
    ax3 = fig.add_subplot(3, 3, 3)
    bptt_total = sum([bptt_analysis['stats']['rec_change_norm'],
                      bptt_analysis['stats']['in_change_norm'],
                      bptt_analysis['stats']['out_change_norm']])
    bptt_fracs = [bptt_analysis['stats']['rec_change_norm'] / bptt_total,
                  bptt_analysis['stats']['in_change_norm'] / bptt_total,
                  bptt_analysis['stats']['out_change_norm'] / bptt_total]
    ax3.pie(bptt_fracs, labels=['Recurrent', 'Input', 'Output'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax3.set_title('BPTT: Where Changes Occur', fontweight='bold')
    
    # =========================================================================
    # Row 2: Weight matrices heatmaps (EA)
    # =========================================================================
    ax4 = fig.add_subplot(3, 3, 4)
    im = ax4.imshow(ea_analysis['delta']['W_rec'], cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(ea_analysis['delta']['W_rec']).max(),
                    vmax=np.abs(ea_analysis['delta']['W_rec']).max())
    ax4.set_title('EA: ΔW_rec (recurrent)', fontweight='bold')
    ax4.set_xlabel('From neuron')
    ax4.set_ylabel('To neuron')
    plt.colorbar(im, ax=ax4, label='Δweight')
    
    ax5 = fig.add_subplot(3, 3, 5)
    # W_in is (N, 1), reshape for visualization
    delta_in_ea = ea_analysis['delta']['W_in'].flatten()
    ax5.bar(range(len(delta_in_ea)), delta_in_ea, color='#2ecc71', alpha=0.7)
    ax5.axhline(0, color='black', linewidth=0.5)
    ax5.set_title('EA: ΔW_in (input)', fontweight='bold')
    ax5.set_xlabel('Neuron')
    ax5.set_ylabel('Δweight')
    ax5.grid(True, alpha=0.3, axis='y')
    
    ax6 = fig.add_subplot(3, 3, 6)
    # W_out is (1, N), reshape for visualization
    delta_out_ea = ea_analysis['delta']['W_out'].flatten()
    ax6.bar(range(len(delta_out_ea)), delta_out_ea, color='#e74c3c', alpha=0.7)
    ax6.axhline(0, color='black', linewidth=0.5)
    ax6.set_title('EA: ΔW_out (output)', fontweight='bold')
    ax6.set_xlabel('Neuron')
    ax6.set_ylabel('Δweight')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Row 3: Weight matrices heatmaps (BPTT)
    # =========================================================================
    ax7 = fig.add_subplot(3, 3, 7)
    im = ax7.imshow(bptt_analysis['delta']['W_rec'], cmap='RdBu_r', aspect='auto',
                    vmin=-np.abs(bptt_analysis['delta']['W_rec']).max(),
                    vmax=np.abs(bptt_analysis['delta']['W_rec']).max())
    ax7.set_title('BPTT: ΔW_rec (recurrent)', fontweight='bold')
    ax7.set_xlabel('From neuron')
    ax7.set_ylabel('To neuron')
    plt.colorbar(im, ax=ax7, label='Δweight')
    
    ax8 = fig.add_subplot(3, 3, 8)
    delta_in_bptt = bptt_analysis['delta']['W_in'].flatten()
    ax8.bar(range(len(delta_in_bptt)), delta_in_bptt, color='#2ecc71', alpha=0.7)
    ax8.axhline(0, color='black', linewidth=0.5)
    ax8.set_title('BPTT: ΔW_in (input)', fontweight='bold')
    ax8.set_xlabel('Neuron')
    ax8.set_ylabel('Δweight')
    ax8.grid(True, alpha=0.3, axis='y')
    
    ax9 = fig.add_subplot(3, 3, 9)
    delta_out_bptt = bptt_analysis['delta']['W_out'].flatten()
    ax9.bar(range(len(delta_out_bptt)), delta_out_bptt, color='#e74c3c', alpha=0.7)
    ax9.axhline(0, color='black', linewidth=0.5)
    ax9.set_title('BPTT: ΔW_out (output)', fontweight='bold')
    ax9.set_xlabel('Neuron')
    ax9.set_ylabel('Δweight')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_weight_distributions(ea_analysis: dict, bptt_analysis: dict, save_path: str = 'weight_distributions.png'):
    """Plot histograms of weight changes."""
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    weight_types = ['W_rec', 'W_in', 'W_out']
    titles = ['Recurrent (W_rec)', 'Input (W_in)', 'Output (W_out)']
    
    for col, (wtype, title) in enumerate(zip(weight_types, titles)):
        # EA
        ax = axes[0, col]
        delta = ea_analysis['delta'][wtype].flatten()
        ax.hist(delta, bins=50, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(delta.mean(), color='green', linestyle='-', linewidth=2, label=f'mean={delta.mean():.3f}')
        ax.set_title(f'EA: Δ{title}', fontweight='bold')
        ax.set_xlabel('Weight change')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # BPTT
        ax = axes[1, col]
        delta = bptt_analysis['delta'][wtype].flatten()
        ax.hist(delta, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1)
        ax.axvline(delta.mean(), color='green', linestyle='-', linewidth=2, label=f'mean={delta.mean():.3f}')
        ax.set_title(f'BPTT: Δ{title}', fontweight='bold')
        ax.set_xlabel('Weight change')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_analysis_summary(ea_analysis: dict, bptt_analysis: dict):
    """Print a text summary of the weight analysis."""
    
    print("\n" + "=" * 70)
    print("WEIGHT CHANGE ANALYSIS: EA vs BPTT")
    print("=" * 70)
    
    # Compute percentages
    ea_total = (ea_analysis['stats']['rec_change_norm'] + 
                ea_analysis['stats']['in_change_norm'] + 
                ea_analysis['stats']['out_change_norm'])
    bptt_total = (bptt_analysis['stats']['rec_change_norm'] + 
                  bptt_analysis['stats']['in_change_norm'] + 
                  bptt_analysis['stats']['out_change_norm'])
    
    print("\n### Frobenius Norm of Weight Changes ###")
    print(f"{'':20} {'EA':>12} {'BPTT':>12}")
    print("-" * 46)
    print(f"{'W_rec (recurrent)':20} {ea_analysis['stats']['rec_change_norm']:>12.4f} {bptt_analysis['stats']['rec_change_norm']:>12.4f}")
    print(f"{'W_in (input)':20} {ea_analysis['stats']['in_change_norm']:>12.4f} {bptt_analysis['stats']['in_change_norm']:>12.4f}")
    print(f"{'W_out (output)':20} {ea_analysis['stats']['out_change_norm']:>12.4f} {bptt_analysis['stats']['out_change_norm']:>12.4f}")
    print("-" * 46)
    print(f"{'Total':20} {ea_total:>12.4f} {bptt_total:>12.4f}")
    
    print("\n### Percentage of Total Change ###")
    print(f"{'':20} {'EA':>12} {'BPTT':>12}")
    print("-" * 46)
    print(f"{'W_rec (recurrent)':20} {100*ea_analysis['stats']['rec_change_norm']/ea_total:>11.1f}% {100*bptt_analysis['stats']['rec_change_norm']/bptt_total:>11.1f}%")
    print(f"{'W_in (input)':20} {100*ea_analysis['stats']['in_change_norm']/ea_total:>11.1f}% {100*bptt_analysis['stats']['in_change_norm']/bptt_total:>11.1f}%")
    print(f"{'W_out (output)':20} {100*ea_analysis['stats']['out_change_norm']/ea_total:>11.1f}% {100*bptt_analysis['stats']['out_change_norm']/bptt_total:>11.1f}%")
    
    print("\n### Mean Absolute Weight Change ###")
    print(f"{'':20} {'EA':>12} {'BPTT':>12}")
    print("-" * 46)
    print(f"{'W_rec (recurrent)':20} {ea_analysis['stats']['rec_change_mean']:>12.4f} {bptt_analysis['stats']['rec_change_mean']:>12.4f}")
    print(f"{'W_in (input)':20} {ea_analysis['stats']['in_change_mean']:>12.4f} {bptt_analysis['stats']['in_change_mean']:>12.4f}")
    print(f"{'W_out (output)':20} {ea_analysis['stats']['out_change_mean']:>12.4f} {bptt_analysis['stats']['out_change_mean']:>12.4f}")
    
    # Key insights
    print("\n### Key Observations ###")
    
    ea_rec_pct = 100 * ea_analysis['stats']['rec_change_norm'] / ea_total
    bptt_rec_pct = 100 * bptt_analysis['stats']['rec_change_norm'] / bptt_total
    
    if ea_rec_pct > bptt_rec_pct + 5:
        print(f"• EA modifies recurrent weights MORE than BPTT ({ea_rec_pct:.1f}% vs {bptt_rec_pct:.1f}%)")
    elif bptt_rec_pct > ea_rec_pct + 5:
        print(f"• BPTT modifies recurrent weights MORE than EA ({bptt_rec_pct:.1f}% vs {ea_rec_pct:.1f}%)")
    else:
        print(f"• Both methods modify recurrent weights similarly ({ea_rec_pct:.1f}% vs {bptt_rec_pct:.1f}%)")
    
    ea_io_pct = 100 * (ea_analysis['stats']['in_change_norm'] + ea_analysis['stats']['out_change_norm']) / ea_total
    bptt_io_pct = 100 * (bptt_analysis['stats']['in_change_norm'] + bptt_analysis['stats']['out_change_norm']) / bptt_total
    
    if ea_io_pct > bptt_io_pct + 5:
        print(f"• EA relies more on input/output weights ({ea_io_pct:.1f}% vs {bptt_io_pct:.1f}%)")
    elif bptt_io_pct > ea_io_pct + 5:
        print(f"• BPTT relies more on input/output weights ({bptt_io_pct:.1f}% vs {ea_io_pct:.1f}%)")
    
    print("=" * 70)


def run_analysis(train_new: bool = True, ea_generations: int = 200, bptt_iterations: int = 500):
    """Run full weight analysis."""
    
    print("=" * 70)
    print("RUNNING WEIGHT ANALYSIS: BPTT vs EA")
    print("=" * 70)
    
    # Common settings
    n_neurons = 32
    delay_duration = 15
    seed = 42
    
    if train_new:
        # Train EA
        print("\n[1/2] Training EA...")
        ea_conf = EAConfig(
            n_neurons=n_neurons,
            pop_size=64,
            n_generations=ea_generations,
            lr=0.03,
            sigma=0.1,
            delay_duration=delay_duration,
            n_eval_trials=20,
            seed=seed,
            print_every=50,
        )
        ea_results = train_ea(ea_conf)
        
        # Train BPTT
        print("\n[2/2] Training BPTT...")
        bptt_conf = BPTTConfig(
            n_neurons=n_neurons,
            n_iterations=bptt_iterations,
            batch_size=64,
            lr=1e-3,
            delay_duration=delay_duration,
            use_lif=False,
            seed=seed,
            print_every=100,
        )
        bptt_results = train_bptt(bptt_conf)
    else:
        raise ValueError("Must train new models for analysis")
    
    # Analyze weights
    print("\nAnalyzing weight changes...")
    ea_analysis = analyze_ea_weights(ea_conf, ea_results)
    bptt_analysis = analyze_bptt_weights(bptt_conf, bptt_results)
    
    # Print summary
    print_analysis_summary(ea_analysis, bptt_analysis)
    
    # Generate figures
    print("\nGenerating figures...")
    plot_weight_comparison(ea_analysis, bptt_analysis, 'weight_analysis.png')
    plot_weight_distributions(ea_analysis, bptt_analysis, 'weight_distributions.png')
    
    return {
        'ea_analysis': ea_analysis,
        'bptt_analysis': bptt_analysis,
        'ea_results': ea_results,
        'bptt_results': bptt_results,
    }


if __name__ == "__main__":
    results = run_analysis(train_new=True, ea_generations=200, bptt_iterations=500)
