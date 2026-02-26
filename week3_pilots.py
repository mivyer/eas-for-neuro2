# week3_pilots.py
"""
Week 3 Pilot Experiments

Three conditions:
1. EA-connectivity (sample): Evolve P, sample binary connectivity each generation
2. EA-connectivity (P as W): Evolve P, use P directly as weights
3. BPTT baseline: Standard gradient-based training

Goals:
- Verify all three methods learn
- Identify good hyperparameter ranges
- Track sparsity behavior
- Scale model size
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
import json

from models.rsnn_policy import RSNNPolicy
from envs.working_memory import WorkingMemoryTask

# Check for torch
try:
    import torch
    from models.bptt_rnn import RNNPolicy, count_parameters
    from envs.working_memory import WorkingMemoryTaskTorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - BPTT experiments will be skipped")

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PilotConfig:
    """Configuration for pilot experiments."""
    # Network (scalable)
    n_neurons: int = 64  # Scale up from 32
    obs_dim: int = 1
    action_dim: int = 1
    
    # Task
    cue_duration: int = 5
    delay_duration: int = 15
    response_duration: int = 5
    response_weight: float = 0.75  # Weighted loss
    
    # EA settings
    ea_pop_size: int = 64
    ea_generations: int = 200
    ea_lr: float = 0.03
    ea_sigma: float = 0.1
    ea_n_eval_trials: int = 20
    
    # BPTT settings
    bptt_iterations: int = 500
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3
    
    # Sparsity
    sparsity_threshold: float = 0.01  # Weights below this are "zero"
    
    # Misc
    seed: int = 42
    print_every: int = 20


# ============================================================================
# EA with Sampled Connectivity
# ============================================================================

def train_ea_sample(conf: PilotConfig) -> dict:
    """
    EA-connectivity (sample): Evolve probability matrix P, sample binary 
    connectivity each generation.
    """
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # Initialize P (connection probabilities)
    P = np.full((N, N), 0.5, dtype=np.float32)
    P += 0.1 * rng.standard_normal(P.shape).astype(np.float32)
    P = np.clip(P, 0.05, 0.95)
    
    # Fixed input/output weights (only evolve recurrent structure)
    W_in = 0.3 * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out = 0.3 * rng.standard_normal((conf.action_dim, N)).astype(np.float32)
    
    history = {'fitness': [], 'accuracy': [], 'sparsity': [], 'connectivity': []}
    
    print("EA-Sample: Evolving connectivity probabilities")
    print(f"  Network: {N} neurons, P matrix: {N*N} parameters")
    
    best_fitness = -np.inf
    best_P = P.copy()
    
    for gen in range(conf.ea_generations):
        # Sample population of binary connectivity matrices
        pop_fitness = []
        pop_connectivity = []
        
        for i in range(conf.ea_pop_size):
            # Sample connectivity from P
            connectivity = (rng.random(P.shape) < P).astype(np.float32)
            W_rec = connectivity * 0.2 * rng.standard_normal(P.shape).astype(np.float32)
            
            policy = RSNNPolicy(W_rec, W_in.copy(), W_out.copy())
            result = task.evaluate_policy(policy, n_trials=conf.ea_n_eval_trials, rng=rng)
            
            pop_fitness.append(result['fitness'])
            pop_connectivity.append(connectivity)
        
        pop_fitness = np.array(pop_fitness)
        
        # Track best
        best_idx = np.argmax(pop_fitness)
        if pop_fitness[best_idx] > best_fitness:
            best_fitness = pop_fitness[best_idx]
            best_P = P.copy()
        
        # NES update for P
        ranks = np.argsort(np.argsort(pop_fitness)).astype(np.float32)
        ranks = ranks / (len(ranks) - 1) - 0.5
        
        # Gradient estimation
        grad = np.zeros_like(P)
        for i, conn in enumerate(pop_connectivity):
            grad += ranks[i] * (conn - P)
        grad /= conf.ea_pop_size
        
        # Update with variance normalization
        var = P * (1 - P) + 1e-8
        P = P + conf.ea_lr * grad / var
        P = np.clip(P, 0.05, 0.95)
        
        # Compute sparsity (how many connections have low probability)
        sparsity = (P < 0.3).mean()
        mean_connectivity = P.mean()
        
        # Log
        history['fitness'].append(float(pop_fitness.mean()))
        history['accuracy'].append(float(np.mean([1 if f > 0 else 0 for f in pop_fitness])))
        history['sparsity'].append(float(sparsity))
        history['connectivity'].append(float(mean_connectivity))
        
        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            print(f"Gen {gen:4d} | fitness: {pop_fitness.mean():+.3f} | "
                  f"conn: {mean_connectivity:.2f} | sparsity: {sparsity:.2f}")
    
    return {
        'P': P,
        'best_P': best_P,
        'W_in': W_in,
        'W_out': W_out,
        'history': history,
        'best_fitness': best_fitness,
    }


# ============================================================================
# EA with P as Weights
# ============================================================================

def train_ea_p_as_w(conf: PilotConfig) -> dict:
    """
    EA-connectivity (P as W): Evolve all weights directly.
    Standard OpenAI-ES on the full parameter vector.
    """
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # Initialize all weights
    scale = np.sqrt(2.0 / N)
    W_rec = scale * rng.standard_normal((N, N)).astype(np.float32)
    W_in = scale * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out = scale * rng.standard_normal((conf.action_dim, N)).astype(np.float32)
    
    # Flatten to parameter vector
    def flatten(W_rec, W_in, W_out):
        return np.concatenate([W_rec.flatten(), W_in.flatten(), W_out.flatten()])
    
    def unflatten(params):
        n_rec = N * N
        n_in = N * conf.obs_dim
        W_rec = params[:n_rec].reshape(N, N)
        W_in = params[n_rec:n_rec+n_in].reshape(N, conf.obs_dim)
        W_out = params[n_rec+n_in:].reshape(conf.action_dim, N)
        return W_rec, W_in, W_out
    
    params = flatten(W_rec, W_in, W_out)
    n_params = len(params)
    
    history = {'fitness': [], 'accuracy': [], 'sparsity': [], 'weight_norm': []}
    
    print("EA-P-as-W: Evolving all weights directly")
    print(f"  Network: {N} neurons, {n_params} parameters")
    
    best_fitness = -np.inf
    best_params = params.copy()
    
    half_pop = conf.ea_pop_size // 2
    
    for gen in range(conf.ea_generations):
        # Mirrored sampling
        noise = rng.standard_normal((half_pop, n_params)).astype(np.float32)
        
        fitness_pos = []
        fitness_neg = []
        
        for i in range(half_pop):
            # Positive direction
            params_pos = params + conf.ea_sigma * noise[i]
            W_rec, W_in, W_out = unflatten(params_pos)
            policy = RSNNPolicy(W_rec, W_in, W_out)
            result = task.evaluate_policy(policy, n_trials=conf.ea_n_eval_trials, rng=rng)
            fitness_pos.append(result['fitness'])
            
            # Negative direction
            params_neg = params - conf.ea_sigma * noise[i]
            W_rec, W_in, W_out = unflatten(params_neg)
            policy = RSNNPolicy(W_rec, W_in, W_out)
            result = task.evaluate_policy(policy, n_trials=conf.ea_n_eval_trials, rng=rng)
            fitness_neg.append(result['fitness'])
        
        fitness_pos = np.array(fitness_pos)
        fitness_neg = np.array(fitness_neg)
        all_fitness = np.concatenate([fitness_pos, fitness_neg])
        
        # Track best
        max_idx = np.argmax(all_fitness)
        max_fit = all_fitness[max_idx]
        if max_fit > best_fitness:
            best_fitness = max_fit
            if max_idx < half_pop:
                best_params = params + conf.ea_sigma * noise[max_idx]
            else:
                best_params = params - conf.ea_sigma * noise[max_idx - half_pop]
        
        # Gradient update
        fitness_diff = fitness_pos - fitness_neg
        grad = np.mean(fitness_diff[:, None] * noise, axis=0) / conf.ea_sigma
        params = params + conf.ea_lr * grad
        
        # Compute sparsity
        W_rec, W_in, W_out = unflatten(params)
        sparsity = (np.abs(W_rec) < conf.sparsity_threshold).mean()
        weight_norm = np.linalg.norm(W_rec)
        
        # Log
        history['fitness'].append(float(all_fitness.mean()))
        history['accuracy'].append(float(np.mean([1 if f > 0 else 0 for f in all_fitness])))
        history['sparsity'].append(float(sparsity))
        history['weight_norm'].append(float(weight_norm))
        
        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            print(f"Gen {gen:4d} | fitness: {all_fitness.mean():+.3f} | "
                  f"best: {best_fitness:+.3f} | sparsity: {sparsity:.2f}")
    
    W_rec, W_in, W_out = unflatten(best_params)
    return {
        'W_rec': W_rec,
        'W_in': W_in,
        'W_out': W_out,
        'params': best_params,
        'history': history,
        'best_fitness': best_fitness,
    }


# ============================================================================
# BPTT Baseline
# ============================================================================

def train_bptt_baseline(conf: PilotConfig) -> dict:
    """BPTT baseline with same task and metrics."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping BPTT")
        return None
    
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    
    N = conf.n_neurons
    device = "cpu"
    
    model = RNNPolicy(N, conf.obs_dim, conf.action_dim).to(device)
    
    task = WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
        device=device,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.bptt_lr)
    
    history = {'loss': [], 'fitness': [], 'accuracy': [], 'sparsity': []}
    
    print("BPTT Baseline: Gradient-based training")
    print(f"  Network: {N} neurons, {count_parameters(model)} parameters")
    
    for iteration in range(conf.bptt_iterations):
        model.train()
        
        inputs, targets = task.get_batch(conf.bptt_batch_size)
        outputs = model(inputs)
        loss = task.compute_loss(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            accuracy = task.compute_accuracy(outputs, targets)
            fitness = task.compute_fitness(outputs, targets)
            
            # Compute sparsity
            W_rec = model.W_rec.cpu().numpy()
            sparsity = (np.abs(W_rec) < conf.sparsity_threshold).mean()
        
        history['loss'].append(float(loss.item()))
        history['fitness'].append(float(fitness))
        history['accuracy'].append(float(accuracy))
        history['sparsity'].append(float(sparsity))
        
        if iteration % (conf.print_every * 5) == 0 or iteration == conf.bptt_iterations - 1:
            print(f"Iter {iteration:4d} | loss: {loss.item():.4f} | "
                  f"fitness: {fitness:+.3f} | acc: {accuracy:.1%}")
    
    return {
        'model': model,
        'history': history,
    }


# ============================================================================
# Run All Pilots
# ============================================================================

def run_all_pilots(conf: PilotConfig, save_results: bool = True) -> dict:
    """Run all three pilot conditions and compare."""
    
    print("=" * 70)
    print("WEEK 3 PILOT EXPERIMENTS")
    print("=" * 70)
    print(f"Network size: {conf.n_neurons} neurons")
    print(f"Task: cue={conf.cue_duration}, delay={conf.delay_duration}, response={conf.response_duration}")
    print(f"Loss weighting: {conf.response_weight:.0%} response, {1-conf.response_weight:.0%} other")
    print()
    
    results = {}
    
    # Condition 1: EA-sample
    print("\n" + "-" * 70)
    print("CONDITION 1: EA-connectivity (sample)")
    print("-" * 70)
    t0 = time.time()
    results['ea_sample'] = train_ea_sample(conf)
    results['ea_sample']['time'] = time.time() - t0
    print(f"Time: {results['ea_sample']['time']:.1f}s")
    
    # Condition 2: EA-P-as-W
    print("\n" + "-" * 70)
    print("CONDITION 2: EA-connectivity (P as W)")
    print("-" * 70)
    t0 = time.time()
    results['ea_p_as_w'] = train_ea_p_as_w(conf)
    results['ea_p_as_w']['time'] = time.time() - t0
    print(f"Time: {results['ea_p_as_w']['time']:.1f}s")
    
    # Condition 3: BPTT
    print("\n" + "-" * 70)
    print("CONDITION 3: BPTT Baseline")
    print("-" * 70)
    t0 = time.time()
    results['bptt'] = train_bptt_baseline(conf)
    if results['bptt']:
        results['bptt']['time'] = time.time() - t0
        print(f"Time: {results['bptt']['time']:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<20} {'Final Fitness':>15} {'Final Sparsity':>15} {'Time':>10}")
    print("-" * 62)
    
    ea_sample_fit = results['ea_sample']['history']['fitness'][-1]
    ea_sample_spar = results['ea_sample']['history']['sparsity'][-1]
    print(f"{'EA-sample':<20} {ea_sample_fit:>+15.3f} {ea_sample_spar:>15.2f} {results['ea_sample']['time']:>9.1f}s")
    
    ea_paw_fit = results['ea_p_as_w']['history']['fitness'][-1]
    ea_paw_spar = results['ea_p_as_w']['history']['sparsity'][-1]
    print(f"{'EA-P-as-W':<20} {ea_paw_fit:>+15.3f} {ea_paw_spar:>15.2f} {results['ea_p_as_w']['time']:>9.1f}s")
    
    if results['bptt']:
        bptt_fit = results['bptt']['history']['fitness'][-1]
        bptt_spar = results['bptt']['history']['sparsity'][-1]
        print(f"{'BPTT':<20} {bptt_fit:>+15.3f} {bptt_spar:>15.2f} {results['bptt']['time']:>9.1f}s")
    
    # Plot
    if PLOT_AVAILABLE:
        plot_pilot_results(results, conf)
    
    return results


def plot_pilot_results(results: dict, conf: PilotConfig):
    """Generate comparison plots for pilot experiments."""
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Row 1: Learning curves (fitness)
    ax = axes[0, 0]
    gens = np.arange(len(results['ea_sample']['history']['fitness']))
    ax.plot(gens, results['ea_sample']['history']['fitness'], 'b-', label='EA-sample', linewidth=2)
    ax.plot(gens, results['ea_p_as_w']['history']['fitness'], 'g-', label='EA-P-as-W', linewidth=2)
    if results['bptt']:
        # Subsample BPTT to match EA generations
        bptt_fit = results['bptt']['history']['fitness']
        bptt_x = np.linspace(0, len(gens)-1, len(bptt_fit))
        ax.plot(bptt_x, bptt_fit, 'r-', label='BPTT', linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation / Iteration (scaled)')
    ax.set_ylabel('Fitness')
    ax.set_title('Learning Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 1: Sparsity over training
    ax = axes[0, 1]
    ax.plot(gens, results['ea_sample']['history']['sparsity'], 'b-', label='EA-sample', linewidth=2)
    ax.plot(gens, results['ea_p_as_w']['history']['sparsity'], 'g-', label='EA-P-as-W', linewidth=2)
    if results['bptt']:
        bptt_spar = results['bptt']['history']['sparsity']
        bptt_x = np.linspace(0, len(gens)-1, len(bptt_spar))
        ax.plot(bptt_x, bptt_spar, 'r-', label='BPTT', linewidth=2, alpha=0.7)
    ax.set_xlabel('Generation / Iteration (scaled)')
    ax.set_ylabel('Sparsity (fraction < threshold)')
    ax.set_title('Sparsity Evolution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 1: Final comparison bars
    ax = axes[0, 2]
    conditions = ['EA-sample', 'EA-P-as-W']
    fitness_vals = [results['ea_sample']['history']['fitness'][-1],
                    results['ea_p_as_w']['history']['fitness'][-1]]
    colors = ['#3498db', '#2ecc71']
    if results['bptt']:
        conditions.append('BPTT')
        fitness_vals.append(results['bptt']['history']['fitness'][-1])
        colors.append('#e74c3c')
    
    bars = ax.bar(conditions, fitness_vals, color=colors, edgecolor='black')
    ax.set_ylabel('Final Fitness')
    ax.set_title('Final Performance', fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, fitness_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Row 2: EA-sample P matrix
    ax = axes[1, 0]
    im = ax.imshow(results['ea_sample']['P'], cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title('EA-sample: Final P (connectivity prob)', fontweight='bold')
    ax.set_xlabel('From neuron')
    ax.set_ylabel('To neuron')
    plt.colorbar(im, ax=ax)
    
    # Row 2: EA-P-as-W weight matrix
    ax = axes[1, 1]
    W_rec = results['ea_p_as_w']['W_rec']
    vmax = np.abs(W_rec).max()
    im = ax.imshow(W_rec, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_title('EA-P-as-W: Final W_rec', fontweight='bold')
    ax.set_xlabel('From neuron')
    ax.set_ylabel('To neuron')
    plt.colorbar(im, ax=ax)
    
    # Row 2: BPTT weight matrix
    ax = axes[1, 2]
    if results['bptt']:
        W_rec = results['bptt']['model'].W_rec.detach().cpu().numpy()
        vmax = np.abs(W_rec).max()
        im = ax.imshow(W_rec, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_title('BPTT: Final W_rec', fontweight='bold')
        ax.set_xlabel('From neuron')
        ax.set_ylabel('To neuron')
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, 'BPTT not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('BPTT: Final W_rec', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('week3_pilot_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: week3_pilot_results.png")


# ============================================================================
# Hyperparameter Sweep
# ============================================================================

def sweep_hyperparameters(base_conf: PilotConfig, param_name: str, values: list) -> dict:
    """
    Sweep over one hyperparameter to find good ranges.
    """
    print(f"\nSweeping {param_name} over {values}")
    
    results = []
    for val in values:
        conf = PilotConfig(**{**base_conf.__dict__, param_name: val})
        print(f"\n{param_name} = {val}")
        
        # Quick EA-P-as-W run
        conf.ea_generations = 100  # Shorter for sweep
        result = train_ea_p_as_w(conf)
        
        results.append({
            param_name: val,
            'fitness': result['best_fitness'],
            'final_sparsity': result['history']['sparsity'][-1],
        })
    
    return results


def main():
    """Run Week 3 pilot experiments."""
    conf = PilotConfig(
        n_neurons=64,  # Scaled up
        delay_duration=15,
        response_weight=0.75,  # Weighted loss
        ea_generations=200,
        bptt_iterations=500,
        seed=42,
    )
    
    results = run_all_pilots(conf)
    
    return results


if __name__ == "__main__":
    main()
