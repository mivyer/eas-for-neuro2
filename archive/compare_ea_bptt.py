# compare_ea_bptt.py
"""
Run both EA and BPTT on the working memory task and compare results.
"""

import numpy as np
import time
from dataclasses import dataclass

from train_ea import train_ea, test_policy_from_params, EAConfig
from train_bptt import train_bptt, compute_ea_fitness, BPTTConfig

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available, skipping plots")


@dataclass  
class ComparisonConfig:
    """Shared settings for fair comparison."""
    n_neurons: int = 32
    cue_duration: int = 5
    delay_duration: int = 15
    response_duration: int = 5
    seed: int = 42
    
    # EA-specific
    ea_pop_size: int = 64
    ea_generations: int = 200
    ea_lr: float = 0.03
    ea_sigma: float = 0.1
    ea_n_eval_trials: int = 20
    
    # BPTT-specific  
    bptt_iterations: int = 500
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3


def run_comparison(conf: ComparisonConfig, save_plot: bool = True):
    """Run EA and BPTT, compare learning curves and final performance."""
    
    print("=" * 60)
    print("EA vs BPTT Comparison on Working Memory Task")
    print("=" * 60)
    print(f"Network: {conf.n_neurons} neurons")
    print(f"Task: cue={conf.cue_duration}, delay={conf.delay_duration}, response={conf.response_duration}")
    print()
    
    # =========================================================================
    # Train EA
    # =========================================================================
    print("-" * 60)
    print("Training EA...")
    print("-" * 60)
    
    ea_conf = EAConfig(
        n_neurons=conf.n_neurons,
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        pop_size=conf.ea_pop_size,
        n_generations=conf.ea_generations,
        lr=conf.ea_lr,
        sigma=conf.ea_sigma,
        n_eval_trials=conf.ea_n_eval_trials,
        seed=conf.seed,
        print_every=20,
    )
    
    t0 = time.time()
    ea_results = train_ea(ea_conf)
    ea_time = time.time() - t0
    
    # Test EA
    ea_test = test_policy_from_params(ea_results['best_params'], ea_conf, n_trials=100)
    print(f"\nEA training time: {ea_time:.1f}s")
    print(f"EA test - fitness: {ea_test['fitness']:.3f}, accuracy: {ea_test['accuracy']:.1%}")
    
    # =========================================================================
    # Train BPTT
    # =========================================================================
    print("\n" + "-" * 60)
    print("Training BPTT...")
    print("-" * 60)
    
    bptt_conf = BPTTConfig(
        n_neurons=conf.n_neurons,
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        n_iterations=conf.bptt_iterations,
        batch_size=conf.bptt_batch_size,
        lr=conf.bptt_lr,
        use_lif=False,
        seed=conf.seed,
        print_every=100,
    )
    
    t0 = time.time()
    bptt_results = train_bptt(bptt_conf)
    bptt_time = time.time() - t0
    
    # Test BPTT with EA metric
    bptt_test = compute_ea_fitness(bptt_results['model'], bptt_conf, n_trials=100)
    print(f"\nBPTT training time: {bptt_time:.1f}s")
    print(f"BPTT test - fitness: {bptt_test['fitness']:.3f}, accuracy: {bptt_test['accuracy']:.1%}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Final Comparison")
    print("=" * 60)
    print(f"{'Method':<10} {'Fitness':>10} {'Accuracy':>10} {'Time':>10}")
    print("-" * 42)
    print(f"{'EA':<10} {ea_test['fitness']:>+10.3f} {ea_test['accuracy']:>9.1%} {ea_time:>9.1f}s")
    print(f"{'BPTT':<10} {bptt_test['fitness']:>+10.3f} {bptt_test['accuracy']:>9.1%} {bptt_time:>9.1f}s")
    
    # =========================================================================
    # Plot
    # =========================================================================
    if save_plot and PLOT_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # EA learning curve
        ax = axes[0]
        gens = np.arange(len(ea_results['history']['mean_fitness']))
        ax.fill_between(
            gens,
            ea_results['history']['min_fitness'],
            ea_results['history']['max_fitness'],
            alpha=0.3, label='min-max'
        )
        ax.plot(gens, ea_results['history']['mean_fitness'], 'b-', label='mean')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('EA Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # BPTT loss
        ax = axes[1]
        iters = np.arange(len(bptt_results['history']['loss']))
        ax.plot(iters, bptt_results['history']['loss'], 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('BPTT Loss')
        ax.grid(True, alpha=0.3)
        
        # BPTT accuracy
        ax = axes[2]
        ax.plot(iters, bptt_results['history']['accuracy'], 'g-')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_title('BPTT Accuracy')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150)
        print(f"\nSaved plot to comparison_results.png")
        plt.close()
    
    return {
        'ea': ea_results,
        'bptt': bptt_results,
        'ea_test': ea_test,
        'bptt_test': bptt_test,
        'ea_time': ea_time,
        'bptt_time': bptt_time,
    }


def main():
    """Run comparison with default settings."""
    conf = ComparisonConfig(
        n_neurons=32,
        delay_duration=15,
        ea_generations=200,
        bptt_iterations=500,
        seed=42,
    )
    
    results = run_comparison(conf, save_plot=True)
    return results


if __name__ == "__main__":
    main()
