# run_full_analysis.py
"""
Full Analysis Pipeline for Week 3

1. Runs EA and BPTT conditions
2. Scales to 256 neurons
3. Generates summary plots including OUTPUT EVOLUTION over generations
4. Analyzes weight changes (rec vs in/out)
5. Saves all data for later analysis
"""

import numpy as np
import time
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from models.rsnn_policy import RSNNPolicy
from envs.working_memory import WorkingMemoryTask

try:
    import torch
    from models.bptt_rnn import RNNPolicy, count_parameters
    from envs.working_memory import WorkingMemoryTaskTorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - BPTT will be skipped")

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available - plots will be skipped")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnalysisConfig:
    """Full analysis configuration."""
    # Network - SCALED UP
    n_neurons: int = 32
    obs_dim: int = 1
    action_dim: int = 1
    
    # Task
    cue_duration: int = 5
    delay_duration: int = 10      # Shorter delay
    response_duration: int = 10   # Longer response
    response_weight: float =.75  # 75% on response (no silence penalty)
    
    # EA settings
    ea_pop_size: int = 128
    ea_generations: int = 300   
    ea_lr: float = 0.03
    ea_sigma: float = 0.1
    ea_n_eval_trials: int = 20
    
    # BPTT settings
    bptt_iterations: int = 1000
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3
    
    # Analysis
    sparsity_threshold: float = 0.01
    
    # Misc
    seed: int = 42
    print_every: int = 25
    
    # Output
    output_dir: str = "analysis_results"


# ============================================================================
# Helper Functions
# ============================================================================

def run_single_trial(policy, task, cue, rng):
    """Run a single trial and return outputs at each timestep."""
    inputs, target = task.get_trial(cue=cue, rng=rng)
    
    policy.reset()
    outputs = []
    for t in range(task.total_steps):
        obs = np.array([inputs[t]], dtype=np.float32)
        action = policy.act(obs)
        if hasattr(action, '__len__'):
            outputs.append(float(action[0]))
        else:
            outputs.append(float(action))
    
    return np.array(outputs), inputs


# ============================================================================
# Training Functions
# ============================================================================

def train_ea_p_as_w(conf: AnalysisConfig) -> dict:
    """
    Hybrid EA: Evolve connection probabilities P, sample binary connectivity.
    Now captures output snapshots at specific generations for visualization.
    """
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    
    # E/I SPLIT (80% excitatory, 20% inhibitory)
    n_exc = int(0.8 * N)
    n_inh = N - n_exc
    
    print(f"E/I split: {n_exc} excitatory, {n_inh} inhibitory")
    
    # TASK SETUP
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # BIOLOGICALLY-INFORMED CONNECTION PROBABILITIES
    P_init = np.zeros((N, N), dtype=np.float32)
    P_init[:n_exc, :n_exc] = 0.16   # E -> E: sparse
    P_init[n_exc:, :n_exc] = 0.30   # E -> I: medium
    P_init[:n_exc, n_exc:] = 0.40   # I -> E: dense
    P_init[n_exc:, n_exc:] = 0.20   # I -> I: moderate
    
    P_init = P_init + 0.05 * rng.standard_normal(P_init.shape).astype(np.float32)
    P_init = np.clip(P_init, 0.05, 0.95)
    np.fill_diagonal(P_init, 0.0)
    
    P = P_init.copy()
    
    # FIXED WEIGHT MAGNITUDES
    mu, sigma = -0.64, 0.51
    weight_magnitudes = np.random.lognormal(mu, sigma, (N, N)).astype(np.float32)
    weight_magnitudes = weight_magnitudes / np.sqrt(N)
    fixed_weights = weight_magnitudes.copy()
    np.fill_diagonal(fixed_weights, 0.0)
    
    # FIXED INPUT/OUTPUT WEIGHTS
    W_in_fixed = 0.5 * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out_fixed = 0.5 * rng.standard_normal((conf.action_dim, N)).astype(np.float32)
    
    # ADAM OPTIMIZER STATE
    adam_m = np.zeros_like(P)
    adam_v = np.zeros_like(P)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_eps = 1e-8
    
    # TRACKING
    n_params = N * N
    print(f"Hybrid EA: {N} neurons")
    print(f"  Evolving: P (connection probabilities), {n_params} params")
    print(f"  Fixed: weight magnitudes, W_in, W_out")
    print(f"  Task: {task.total_steps} steps (cue={conf.cue_duration}, delay={conf.delay_duration}, response={conf.response_duration})")
    print(f"  Fitness: {conf.response_weight:.0%} response, {1-conf.response_weight:.0%} silence")
    
    history = {
        'fitness_mean': [], 'fitness_max': [], 'fitness_min': [],
        'accuracy': [],
        'mean_P': [],
        'sparsity': [],
        'P_ee': [], 'P_ei': [], 'P_ie': [], 'P_ii': [],
    }
    
    # SNAPSHOTS: capture outputs at specific generations
    snapshot_gens = [0, 25, 50, 75, 100, 150, 200, 250, 299]
    # Make sure last generation is included
    if conf.ea_generations - 1 not in snapshot_gens:
        snapshot_gens.append(conf.ea_generations - 1)
    snapshot_gens = sorted(list(set(snapshot_gens)))
    snapshots = {}
    
    best_fitness = -np.inf
    best_P = P.copy()
    
    eps = 0.02
    
    # EVOLUTION LOOP
    for gen in range(conf.ea_generations):
        
        # SAMPLE BINARY CONNECTIVITY FROM P
        fitness_all = []
        acc_all = []
        connectivity_all = []
        
        for i in range(conf.ea_pop_size):
            connectivity = (rng.random(P.shape) < P).astype(np.float32)
            W_rec = connectivity * fixed_weights
            np.fill_diagonal(W_rec, 0.0)
            
            policy = RSNNPolicy(W_rec, W_in_fixed, W_out_fixed)
            result = task.evaluate_policy(policy, n_trials=conf.ea_n_eval_trials, rng=rng)
            
            fitness_all.append(result['fitness'])
            acc_all.append(result['accuracy'])
            connectivity_all.append(connectivity)
        
        fitness_all = np.array(fitness_all)
        acc_all = np.array(acc_all)
        connectivity_all = np.array(connectivity_all)
        
        # Track best
        max_idx = np.argmax(fitness_all)
        if fitness_all[max_idx] > best_fitness:
            best_fitness = fitness_all[max_idx]
            best_P = P.copy()
        
        # CAPTURE SNAPSHOT
        if gen in snapshot_gens:
            snapshot_conn = (P > 0.5).astype(np.float32)
            W_rec_snap = snapshot_conn * fixed_weights
            np.fill_diagonal(W_rec_snap, 0.0)
            snap_policy = RSNNPolicy(W_rec_snap, W_in_fixed, W_out_fixed)
            
            # Use fixed seed for consistent comparison
            snap_rng = np.random.default_rng(999)
            out_pos, inp_pos = run_single_trial(snap_policy, task, cue=+1, rng=snap_rng)
            snap_rng = np.random.default_rng(999)
            out_neg, inp_neg = run_single_trial(snap_policy, task, cue=-1, rng=snap_rng)
            
            # Response phase statistics
            resp_start = conf.cue_duration + conf.delay_duration
            resp_pos_mean = out_pos[resp_start:].mean()
            resp_neg_mean = out_neg[resp_start:].mean()
            
            snapshots[gen] = {
                'out_pos': out_pos.tolist(),
                'out_neg': out_neg.tolist(),
                'inp_pos': inp_pos.tolist(),
                'inp_neg': inp_neg.tolist(),
                'fitness_mean': float(fitness_all.mean()),
                'fitness_max': float(fitness_all.max()),
                'resp_pos_mean': float(resp_pos_mean),
                'resp_neg_mean': float(resp_neg_mean),
            }
        
        # CENTERED RANK TRANSFORM
        ranks = np.argsort(np.argsort(fitness_all)).astype(np.float32)
        ranks = ranks / (len(ranks) - 1) - 0.5
        
        # GRADIENT ESTIMATION
        grad = np.zeros_like(P)
        for i in range(conf.ea_pop_size):
            grad += ranks[i] * (connectivity_all[i] - P)
        grad = -grad / conf.ea_pop_size
        
        # ADAM UPDATE
        t = gen + 1
        adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * grad
        adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * (grad ** 2)
        m_hat = adam_m / (1 - adam_beta1 ** t)
        v_hat = adam_v / (1 - adam_beta2 ** t)
        P = P - conf.ea_lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        
        # CLIP P
        P = np.clip(P, eps, 1 - eps)
        np.fill_diagonal(P, 0.0)
        
        # LOGGING
        sparsity = (P < 0.1).mean()
        
        history['fitness_mean'].append(float(fitness_all.mean()))
        history['fitness_max'].append(float(fitness_all.max()))
        history['fitness_min'].append(float(fitness_all.min()))
        history['accuracy'].append(float(acc_all.mean()))
        history['mean_P'].append(float(P.mean()))
        history['sparsity'].append(float(sparsity))
        history['P_ee'].append(float(P[:n_exc, :n_exc].mean()))
        history['P_ei'].append(float(P[n_exc:, :n_exc].mean()))
        history['P_ie'].append(float(P[:n_exc, n_exc:].mean()))
        history['P_ii'].append(float(P[n_exc:, n_exc:].mean()))
        
        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            # Get response phase means for logging
            if gen in snapshots:
                resp_pos = snapshots[gen]['resp_pos_mean']
                resp_neg = snapshots[gen]['resp_neg_mean']
            else:
                resp_pos = 0
                resp_neg = 0
            
            print(f"Gen {gen:4d} | fit: {fitness_all.mean():+.3f} | "
                  f"best: {best_fitness:+.3f} | "
                  f"resp(+1): {resp_pos:+.3f} | resp(-1): {resp_neg:+.3f}")
    
    # FINAL EVALUATION
    final_connectivity = (best_P > 0.5).astype(np.float32)
    W_rec_final = final_connectivity * fixed_weights
    np.fill_diagonal(W_rec_final, 0.0)
    
    init_connectivity = (P_init > 0.5).astype(np.float32)
    W_rec_init = init_connectivity * fixed_weights
    np.fill_diagonal(W_rec_init, 0.0)
    
    return {
        'W_rec_init': W_rec_init,
        'W_in_init': W_in_fixed,
        'W_out_init': W_out_fixed,
        'W_rec_final': W_rec_final,
        'W_in_final': W_in_fixed,
        'W_out_final': W_out_fixed,
        'P_init': P_init,
        'P_final': best_P,
        'fixed_weights': fixed_weights,
        'history': history,
        'best_fitness': best_fitness,
        'n_exc': n_exc,
        'n_inh': n_inh,
        'snapshots': snapshots,  # NEW: output snapshots
        'snapshot_gens': snapshot_gens,  # NEW: which generations
    }


def train_bptt(conf: AnalysisConfig) -> dict:
    """BPTT baseline with output snapshots."""
    if not TORCH_AVAILABLE:
        return None
    
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    
    N = conf.n_neurons
    device = "cpu"
    
    model = RNNPolicy(N, conf.obs_dim, conf.action_dim).to(device)
    
    with torch.no_grad():
        W_rec_init = model.W_rec.cpu().numpy().copy()
        W_in_init = model.W_in.cpu().numpy().copy()
        W_out_init = model.W_out.cpu().numpy().copy()
    
    task = WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
        device=device,
    )
    
    # NumPy task for snapshot evaluation
    task_np = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.bptt_lr)
    
    print(f"BPTT: {N} neurons, {count_parameters(model)} parameters")
    print(f"  Task: {task_np.total_steps} steps")
    
    history = {
        'loss': [], 'fitness': [], 'accuracy': [],
        'sparsity_rec': [], 'sparsity_in': [], 'sparsity_out': [],
    }
    
    # Snapshots for BPTT
    snapshot_iters = [0, 50, 100, 200, 300, 500, 750, 999]
    if conf.bptt_iterations - 1 not in snapshot_iters:
        snapshot_iters.append(conf.bptt_iterations - 1)
    snapshot_iters = sorted(list(set([i for i in snapshot_iters if i < conf.bptt_iterations])))
    snapshots = {}
    
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
            
            W_rec = model.W_rec.cpu().numpy()
            W_in = model.W_in.cpu().numpy()
            W_out = model.W_out.cpu().numpy()
        
        history['loss'].append(float(loss.item()))
        history['fitness'].append(float(fitness))
        history['accuracy'].append(float(accuracy))
        history['sparsity_rec'].append(float((np.abs(W_rec) < conf.sparsity_threshold).mean()))
        history['sparsity_in'].append(float((np.abs(W_in) < conf.sparsity_threshold).mean()))
        history['sparsity_out'].append(float((np.abs(W_out) < conf.sparsity_threshold).mean()))
        
        # Capture snapshot
        if iteration in snapshot_iters:
            model.eval()
            
            # Create wrapper for BPTT model
            class BPTTWrapper:
                def __init__(self, m):
                    self.model = m
                    self.h = None
                def reset(self):
                    with torch.no_grad():
                        self.h = self.model.h0.detach().clone()
                def act(self, obs):
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, dtype=torch.float32)
                        self.h = torch.tanh(self.h @ self.model.W_rec.T + obs_t @ self.model.W_in.T)
                        action = torch.tanh(self.h @ self.model.W_out.T)
                        return action.numpy()
            
            wrapper = BPTTWrapper(model)
            
            snap_rng = np.random.default_rng(999)
            out_pos, inp_pos = run_single_trial(wrapper, task_np, cue=+1, rng=snap_rng)
            snap_rng = np.random.default_rng(999)
            out_neg, inp_neg = run_single_trial(wrapper, task_np, cue=-1, rng=snap_rng)
            
            resp_start = conf.cue_duration + conf.delay_duration
            
            snapshots[iteration] = {
                'out_pos': out_pos.tolist(),
                'out_neg': out_neg.tolist(),
                'fitness': float(fitness),
                'resp_pos_mean': float(out_pos[resp_start:].mean()),
                'resp_neg_mean': float(out_neg[resp_start:].mean()),
            }
            
            model.train()
        
        if iteration % 100 == 0 or iteration == conf.bptt_iterations - 1:
            if iteration in snapshots:
                resp_pos = snapshots[iteration]['resp_pos_mean']
                resp_neg = snapshots[iteration]['resp_neg_mean']
            else:
                resp_pos = 0
                resp_neg = 0
            print(f"Iter {iteration:4d} | loss: {loss.item():.4f} | "
                  f"fit: {fitness:+.3f} | resp(+1): {resp_pos:+.3f} | resp(-1): {resp_neg:+.3f}")
    
    with torch.no_grad():
        W_rec_final = model.W_rec.cpu().numpy()
        W_in_final = model.W_in.cpu().numpy()
        W_out_final = model.W_out.cpu().numpy()
    
    return {
        'model': model,
        'W_rec_init': W_rec_init,
        'W_in_init': W_in_init,
        'W_out_init': W_out_init,
        'W_rec_final': W_rec_final,
        'W_in_final': W_in_final,
        'W_out_final': W_out_final,
        'history': history,
        'snapshots': snapshots,
        'snapshot_iters': snapshot_iters,
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_output_evolution(ea_results: dict, bptt_results: dict, conf: AnalysisConfig):
    """
    THE KEY PLOT: Shows how network outputs change over training.
    This tells you if the network is actually learning vs just being quiet.
    """
    if not PLOT_AVAILABLE:
        return
    
    ea_snapshots = ea_results.get('snapshots', {})
    ea_gens = sorted(ea_snapshots.keys())
    
    if not ea_gens:
        print("No EA snapshots to plot")
        return
    
    n_cols = len(ea_gens)
    fig, axes = plt.subplots(4, n_cols, figsize=(2.5 * n_cols, 12))
    
    cue_end = conf.cue_duration
    delay_end = conf.cue_duration + conf.delay_duration
    total = conf.cue_duration + conf.delay_duration + conf.response_duration
    t = np.arange(total)
    
    # Row labels
    row_labels = ['Cue=+1\n(target=+1)', 'Cue=-1\n(target=-1)', 'Both\noverlaid', 'Response\nphase only']
    
    for col, gen in enumerate(ea_gens):
        snap = ea_snapshots[gen]
        out_pos = np.array(snap['out_pos'])
        out_neg = np.array(snap['out_neg'])
        
        # Row 0: Cue = +1
        ax = axes[0, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_pos, 'b-', linewidth=2)
        ax.axhline(+1, color='darkgreen', linestyle='--', linewidth=1.5, label='target')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Gen {gen}\nfit={snap['fitness_mean']:+.2f}", fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel(row_labels[0], fontsize=10, fontweight='bold')
        ax.set_xticks([])
        
        # Row 1: Cue = -1
        ax = axes[1, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_neg, 'r-', linewidth=2)
        ax.axhline(-1, color='darkgreen', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        if col == 0:
            ax.set_ylabel(row_labels[1], fontsize=10, fontweight='bold')
        ax.set_xticks([])
        
        # Row 2: Both overlaid
        ax = axes[2, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_pos, 'b-', linewidth=2, label='+1')
        ax.plot(t, out_neg, 'r-', linewidth=2, label='-1')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        if col == 0:
            ax.set_ylabel(row_labels[2], fontsize=10, fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
        ax.set_xticks([])
        
        # Row 3: Response phase bar chart
        ax = axes[3, col]
        resp_pos = out_pos[delay_end:]
        resp_neg = out_neg[delay_end:]
        x = np.arange(len(resp_pos))
        width = 0.4
        ax.bar(x - width/2, resp_pos, width, color='blue', alpha=0.7, label='+1')
        ax.bar(x + width/2, resp_neg, width, color='red', alpha=0.7, label='-1')
        ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
        ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Response step')
        if col == 0:
            ax.set_ylabel(row_labels[3], fontsize=10, fontweight='bold')
    
    plt.suptitle('EA: Network Output Evolution Over Generations\n'
                 '(Blue region=cue, Gray=delay, Green=response | Dashed line=target)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'ea_output_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/ea_output_evolution.png")
    
    # Also plot BPTT if available
    if bptt_results and 'snapshots' in bptt_results:
        plot_bptt_output_evolution(bptt_results, conf)


def plot_bptt_output_evolution(bptt_results: dict, conf: AnalysisConfig):
    """Plot BPTT output evolution."""
    if not PLOT_AVAILABLE:
        return
    
    snapshots = bptt_results.get('snapshots', {})
    iters = sorted(snapshots.keys())
    
    if not iters:
        return
    
    n_cols = len(iters)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 9))
    
    cue_end = conf.cue_duration
    delay_end = conf.cue_duration + conf.delay_duration
    total = conf.cue_duration + conf.delay_duration + conf.response_duration
    t = np.arange(total)
    
    for col, iteration in enumerate(iters):
        snap = snapshots[iteration]
        out_pos = np.array(snap['out_pos'])
        out_neg = np.array(snap['out_neg'])
        
        # Row 0: Cue = +1
        ax = axes[0, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_pos, 'b-', linewidth=2)
        ax.axhline(+1, color='darkgreen', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Iter {iteration}\nfit={snap['fitness']:+.2f}", fontsize=10, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Cue=+1', fontweight='bold')
        ax.set_xticks([])
        
        # Row 1: Cue = -1
        ax = axes[1, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_neg, 'r-', linewidth=2)
        ax.axhline(-1, color='darkgreen', linestyle='--', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        if col == 0:
            ax.set_ylabel('Cue=-1', fontweight='bold')
        ax.set_xticks([])
        
        # Row 2: Both
        ax = axes[2, col]
        ax.axvspan(0, cue_end, alpha=0.3, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.3, color='gray')
        ax.axvspan(delay_end, total, alpha=0.3, color='green')
        ax.plot(t, out_pos, 'b-', linewidth=2, label='+1')
        ax.plot(t, out_neg, 'r-', linewidth=2, label='-1')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Time')
        if col == 0:
            ax.set_ylabel('Both', fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('BPTT: Network Output Evolution Over Training', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'bptt_output_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/bptt_output_evolution.png")


def plot_ea_vs_bptt_comparison(ea_results: dict, bptt_results: dict, conf: AnalysisConfig):
    """Side-by-side comparison of final EA and BPTT outputs."""
    if not PLOT_AVAILABLE or not bptt_results:
        return
    
    ea_snaps = ea_results.get('snapshots', {})
    bptt_snaps = bptt_results.get('snapshots', {})
    
    if not ea_snaps or not bptt_snaps:
        return
    
    # Get final snapshots
    ea_final_gen = max(ea_snaps.keys())
    bptt_final_iter = max(bptt_snaps.keys())
    
    ea_snap = ea_snaps[ea_final_gen]
    bptt_snap = bptt_snaps[bptt_final_iter]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cue_end = conf.cue_duration
    delay_end = conf.cue_duration + conf.delay_duration
    total = conf.cue_duration + conf.delay_duration + conf.response_duration
    t = np.arange(total)
    
    # EA output
    ax = axes[0, 0]
    ax.axvspan(0, cue_end, alpha=0.2, color='blue', label='Cue')
    ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray', label='Delay')
    ax.axvspan(delay_end, total, alpha=0.2, color='green', label='Response')
    ax.plot(t, ea_snap['out_pos'], 'b-', linewidth=2, label='cue=+1')
    ax.plot(t, ea_snap['out_neg'], 'r-', linewidth=2, label='cue=-1')
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Output')
    ax.set_title(f'EA (Gen {ea_final_gen})\nFitness: {ea_results["best_fitness"]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # BPTT output
    ax = axes[0, 1]
    ax.axvspan(0, cue_end, alpha=0.2, color='blue')
    ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray')
    ax.axvspan(delay_end, total, alpha=0.2, color='green')
    ax.plot(t, bptt_snap['out_pos'], 'b-', linewidth=2, label='cue=+1')
    ax.plot(t, bptt_snap['out_neg'], 'r-', linewidth=2, label='cue=-1')
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Output')
    ax.set_title(f'BPTT (Iter {bptt_final_iter})\nFitness: {bptt_results["history"]["fitness"][-1]:.3f}', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Response phase comparison - EA
    ax = axes[1, 0]
    resp_pos = np.array(ea_snap['out_pos'])[delay_end:]
    resp_neg = np.array(ea_snap['out_neg'])[delay_end:]
    x = np.arange(len(resp_pos))
    width = 0.35
    ax.bar(x - width/2, resp_pos, width, label='cue=+1', color='blue', alpha=0.7)
    ax.bar(x + width/2, resp_neg, width, label='cue=-1', color='red', alpha=0.7)
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Response Step')
    ax.set_ylabel('Output')
    ax.set_title(f'EA Response: mean(+1)={resp_pos.mean():+.2f}, mean(-1)={resp_neg.mean():+.2f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Response phase comparison - BPTT
    ax = axes[1, 1]
    resp_pos = np.array(bptt_snap['out_pos'])[delay_end:]
    resp_neg = np.array(bptt_snap['out_neg'])[delay_end:]
    ax.bar(x - width/2, resp_pos, width, label='cue=+1', color='blue', alpha=0.7)
    ax.bar(x + width/2, resp_neg, width, label='cue=-1', color='red', alpha=0.7)
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Response Step')
    ax.set_ylabel('Output')
    ax.set_title(f'BPTT Response: mean(+1)={resp_pos.mean():+.2f}, mean(-1)={resp_neg.mean():+.2f}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('EA vs BPTT: Final Network Outputs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'ea_vs_bptt_outputs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/ea_vs_bptt_outputs.png")


def plot_response_evolution(ea_results: dict, bptt_results: dict, conf: AnalysisConfig):
    """Plot how response strength evolves over training for both methods."""
    if not PLOT_AVAILABLE:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # EA response evolution
    ax = axes[0]
    ea_snaps = ea_results.get('snapshots', {})
    if ea_snaps:
        gens = sorted(ea_snaps.keys())
        resp_pos = [ea_snaps[g]['resp_pos_mean'] for g in gens]
        resp_neg = [ea_snaps[g]['resp_neg_mean'] for g in gens]
        fitness = [ea_snaps[g]['fitness_mean'] for g in gens]
        
        ax.plot(gens, resp_pos, 'b-o', linewidth=2, markersize=8, label='Response to +1 cue')
        ax.plot(gens, resp_neg, 'r-o', linewidth=2, markersize=8, label='Response to -1 cue')
        ax.axhline(+1, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(-1, color='red', linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add fitness on secondary axis
        ax2 = ax.twinx()
        ax2.plot(gens, fitness, 'g--', linewidth=1.5, alpha=0.7, label='Fitness')
        ax2.set_ylabel('Fitness', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Response Output')
    ax.set_title('EA: Response Strength Over Training', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    
    # BPTT response evolution
    ax = axes[1]
    if bptt_results and 'snapshots' in bptt_results:
        bptt_snaps = bptt_results['snapshots']
        iters = sorted(bptt_snaps.keys())
        resp_pos = [bptt_snaps[i]['resp_pos_mean'] for i in iters]
        resp_neg = [bptt_snaps[i]['resp_neg_mean'] for i in iters]
        fitness = [bptt_snaps[i]['fitness'] for i in iters]
        
        ax.plot(iters, resp_pos, 'b-o', linewidth=2, markersize=8, label='Response to +1 cue')
        ax.plot(iters, resp_neg, 'r-o', linewidth=2, markersize=8, label='Response to -1 cue')
        ax.axhline(+1, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(-1, color='red', linestyle='--', alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(iters, fitness, 'g--', linewidth=1.5, alpha=0.7, label='Fitness')
        ax2.set_ylabel('Fitness', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Response Output')
    ax.set_title('BPTT: Response Strength Over Training', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'response_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/response_evolution.png")


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_weight_change_stats(init_weights: dict, final_weights: dict, 
                                 threshold: float = 0.01) -> dict:
    """Compute statistics about weight changes."""
    stats = {}
    
    for name in ['W_rec', 'W_in', 'W_out']:
        init = init_weights[f'{name}_init']
        final = final_weights[f'{name}_final']
        delta = final - init
        
        stats[name] = {
            'init_norm': float(np.linalg.norm(init)),
            'final_norm': float(np.linalg.norm(final)),
            'delta_norm': float(np.linalg.norm(delta)),
            'delta_mean': float(np.mean(delta)),
            'delta_std': float(np.std(delta)),
            'delta_abs_mean': float(np.mean(np.abs(delta))),
            'init_sparsity': float((np.abs(init) < threshold).mean()),
            'final_sparsity': float((np.abs(final) < threshold).mean()),
            'n_params': int(init.size),
        }
    
    total_change = sum(stats[name]['delta_norm'] for name in ['W_rec', 'W_in', 'W_out'])
    for name in ['W_rec', 'W_in', 'W_out']:
        stats[name]['change_fraction'] = stats[name]['delta_norm'] / total_change if total_change > 0 else 0
    
    return stats


# ============================================================================
# Data Saving Functions
# ============================================================================

def save_results(ea_results: dict, bptt_results: dict, conf: AnalysisConfig):
    """Save all results for later analysis."""
    
    os.makedirs(conf.output_dir, exist_ok=True)
    
    # Save config
    config_dict = asdict(conf)
    with open(os.path.join(conf.output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved: {conf.output_dir}/config.json")
    
    # Save EA weights
    np.savez(os.path.join(conf.output_dir, 'ea_weights.npz'),
             W_rec_init=ea_results['W_rec_init'],
             W_in_init=ea_results['W_in_init'],
             W_out_init=ea_results['W_out_init'],
             W_rec_final=ea_results['W_rec_final'],
             W_in_final=ea_results['W_in_final'],
             W_out_final=ea_results['W_out_final'])
    print(f"Saved: {conf.output_dir}/ea_weights.npz")
    
    # Save EA history + snapshots
    ea_save = {
        'history': ea_results['history'],
        'snapshots': ea_results.get('snapshots', {}),
        'snapshot_gens': ea_results.get('snapshot_gens', []),
    }
    with open(os.path.join(conf.output_dir, 'ea_history.json'), 'w') as f:
        json.dump(ea_save, f)
    print(f"Saved: {conf.output_dir}/ea_history.json")
    
    # Save BPTT results
    if bptt_results:
        np.savez(os.path.join(conf.output_dir, 'bptt_weights.npz'),
                 W_rec_init=bptt_results['W_rec_init'],
                 W_in_init=bptt_results['W_in_init'],
                 W_out_init=bptt_results['W_out_init'],
                 W_rec_final=bptt_results['W_rec_final'],
                 W_in_final=bptt_results['W_in_final'],
                 W_out_final=bptt_results['W_out_final'])
        print(f"Saved: {conf.output_dir}/bptt_weights.npz")
        
        bptt_save = {
            'history': bptt_results['history'],
            'snapshots': bptt_results.get('snapshots', {}),
            'snapshot_iters': bptt_results.get('snapshot_iters', []),
        }
        with open(os.path.join(conf.output_dir, 'bptt_history.json'), 'w') as f:
            json.dump(bptt_save, f)
        print(f"Saved: {conf.output_dir}/bptt_history.json")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_neurons': conf.n_neurons,
        'ea_best_fitness': float(ea_results['best_fitness']),
        'ea_final_accuracy': float(ea_results['history']['accuracy'][-1]),
        'bptt_final_fitness': float(bptt_results['history']['fitness'][-1]) if bptt_results else None,
        'bptt_final_accuracy': float(bptt_results['history']['accuracy'][-1]) if bptt_results else None,
    }
    with open(os.path.join(conf.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {conf.output_dir}/summary.json")


# ============================================================================
# Main
# ============================================================================

def run_full_analysis(n_neurons: int = 256, output_dir: str = "analysis_results"):
    """Run the complete analysis pipeline."""
    
    conf = AnalysisConfig(
        n_neurons=n_neurons,
        output_dir=output_dir,
    )
    
    os.makedirs(conf.output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"FULL ANALYSIS: {conf.n_neurons} NEURONS")
    print("=" * 70)
    print(f"Output directory: {conf.output_dir}")
    print(f"Task: cue={conf.cue_duration}, delay={conf.delay_duration}, response={conf.response_duration}")
    print(f"Fitness weighting: {conf.response_weight:.0%} response, {1-conf.response_weight:.0%} silence")
    print()
    
    # Train EA
    print("\n" + "-" * 70)
    print("Training EA...")
    print("-" * 70)
    t0 = time.time()
    ea_results = train_ea_p_as_w(conf)
    ea_time = time.time() - t0
    print(f"EA time: {ea_time:.1f}s")
    
    # Train BPTT
    print("\n" + "-" * 70)
    print("Training BPTT...")
    print("-" * 70)
    t0 = time.time()
    bptt_results = train_bptt(conf)
    bptt_time = time.time() - t0 if bptt_results else 0
    if bptt_results:
        print(f"BPTT time: {bptt_time:.1f}s")
    
    # Save data
    print("\n" + "-" * 70)
    print("Saving results...")
    print("-" * 70)
    save_results(ea_results, bptt_results, conf)
    
    # Generate plots
    print("\n" + "-" * 70)
    print("Generating plots...")
    print("-" * 70)
    
    # THE KEY PLOTS - output evolution
    plot_output_evolution(ea_results, bptt_results, conf)
    plot_ea_vs_bptt_comparison(ea_results, bptt_results, conf)
    plot_response_evolution(ea_results, bptt_results, conf)
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"EA:   fitness={ea_results['best_fitness']:.3f}")
    if bptt_results:
        print(f"BPTT: fitness={bptt_results['history']['fitness'][-1]:.3f}")
    print(f"\nKey plots:")
    print(f"  - {conf.output_dir}/ea_output_evolution.png  <- IS EA LEARNING?")
    print(f"  - {conf.output_dir}/ea_vs_bptt_outputs.png   <- SIDE BY SIDE COMPARISON")
    print(f"  - {conf.output_dir}/response_evolution.png   <- RESPONSE STRENGTH OVER TIME")
    
    return {
        'ea': ea_results,
        'bptt': bptt_results,
        'config': conf,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--neurons', type=int, default=256, help='Number of neurons')
    parser.add_argument('--output', type=str, default='analysis_results', help='Output directory')
    args = parser.parse_args()
    
    run_full_analysis(n_neurons=args.neurons, output_dir=args.output)
