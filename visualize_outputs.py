# At the end of your run_full_analysis.py, add:

from visualize_outputs import visualize_learning_over_generations, compare_ea_bptt_outputs

# See EA output over generations
snapshots = visualize_learning_over_generations(conf)

# Compare final EA vs BPTT
results = compare_ea_bptt_outputs(conf)

def visualize_learning_over_generations(conf: AnalysisConfig):
    """
    Visualize how network outputs change over training.
    Saves snapshots at different generations to see learning progression.
    """
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    
    # E/I split
    n_exc = int(0.8 * N)
    n_inh = N - n_exc
    
    # Task
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # Initialize P
    P_init = np.zeros((N, N), dtype=np.float32)
    P_init[:n_exc, :n_exc] = 0.16
    P_init[n_exc:, :n_exc] = 0.30
    P_init[:n_exc, n_exc:] = 0.40
    P_init[n_exc:, n_exc:] = 0.20
    P_init = P_init + 0.05 * rng.standard_normal(P_init.shape).astype(np.float32)
    P_init = np.clip(P_init, 0.05, 0.95)
    np.fill_diagonal(P_init, 0.0)
    P = P_init.copy()
    
    # Fixed weights
    mu, sigma = -0.64, 0.51
    weight_magnitudes = np.random.lognormal(mu, sigma, (N, N)).astype(np.float32)
    weight_magnitudes = weight_magnitudes / np.sqrt(N)
    fixed_weights = weight_magnitudes.copy()
    fixed_weights[:, n_exc:] = -np.abs(fixed_weights[:, n_exc:])
    fixed_weights[:, :n_exc] = np.abs(fixed_weights[:, :n_exc])
    np.fill_diagonal(fixed_weights, 0.0)
    
    W_in_fixed = 0.5 * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out_fixed = 0.5 * rng.standard_normal((conf.action_dim, N)).astype(np.float32)
    W_out_fixed[:, n_exc:] = -np.abs(W_out_fixed[:, n_exc:])
    W_out_fixed[:, :n_exc] = np.abs(W_out_fixed[:, :n_exc])
    
    # Adam state
    adam_m = np.zeros_like(P)
    adam_v = np.zeros_like(P)
    adam_beta1, adam_beta2, adam_eps = 0.9, 0.999, 1e-8
    
    # Track snapshots at specific generations
    snapshot_gens = [0, 25, 50, 100, 150, 200, 250, 299]
    snapshots = {}  # gen -> {'outputs_pos': [...], 'outputs_neg': [...], 'fitness': ...}
    
    half_pop = conf.ea_pop_size // 2
    eps = 0.02
    
    print("Training EA and capturing output snapshots...")
    
    for gen in range(conf.ea_generations):
        # Sample and evaluate population
        fitness_all = []
        connectivity_all = []
        
        for i in range(conf.ea_pop_size):
            connectivity = (rng.random(P.shape) < P).astype(np.float32)
            W_rec = connectivity * fixed_weights
            np.fill_diagonal(W_rec, 0.0)
            
            policy = RSNNPolicy(W_rec, W_in_fixed, W_out_fixed)
            result = task.evaluate_policy(policy, n_trials=conf.ea_n_eval_trials, rng=rng)
            
            fitness_all.append(result['fitness'])
            connectivity_all.append(connectivity)
        
        fitness_all = np.array(fitness_all)
        connectivity_all = np.array(connectivity_all)
        
        # Save snapshot at specific generations
        if gen in snapshot_gens:
            # Use deterministic P > 0.5 for snapshot
            snapshot_connectivity = (P > 0.5).astype(np.float32)
            W_rec_snapshot = snapshot_connectivity * fixed_weights
            np.fill_diagonal(W_rec_snapshot, 0.0)
            policy = RSNNPolicy(W_rec_snapshot, W_in_fixed, W_out_fixed)
            
            # Run trials for both cues
            outputs_pos = run_single_trial(policy, task, cue=+1, rng=rng)
            outputs_neg = run_single_trial(policy, task, cue=-1, rng=rng)
            
            snapshots[gen] = {
                'outputs_pos': outputs_pos,
                'outputs_neg': outputs_neg,
                'fitness': float(fitness_all.mean()),
                'best_fitness': float(fitness_all.max()),
                'P_mean': float(P.mean()),
            }
            print(f"  Gen {gen}: fitness={fitness_all.mean():+.3f}, captured snapshot")
        
        # Update P
        ranks = np.argsort(np.argsort(fitness_all)).astype(np.float32)
        ranks = ranks / (len(ranks) - 1) - 0.5
        
        grad = np.zeros_like(P)
        for i in range(conf.ea_pop_size):
            grad += ranks[i] * (connectivity_all[i] - P)
        grad = -grad / conf.ea_pop_size
        
        t = gen + 1
        adam_m = adam_beta1 * adam_m + (1 - adam_beta1) * grad
        adam_v = adam_beta2 * adam_v + (1 - adam_beta2) * (grad ** 2)
        m_hat = adam_m / (1 - adam_beta1 ** t)
        v_hat = adam_v / (1 - adam_beta2 ** t)
        P = P - conf.ea_lr * m_hat / (np.sqrt(v_hat) + adam_eps)
        P = np.clip(P, eps, 1 - eps)
        np.fill_diagonal(P, 0.0)
    
    # Plot snapshots
    plot_output_snapshots(snapshots, task, snapshot_gens)
    
    return snapshots


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
    
    return np.array(outputs)


def plot_output_snapshots(snapshots, task, snapshot_gens):
    """Plot network outputs across generations."""
    n_snapshots = len(snapshot_gens)
    
    fig, axes = plt.subplots(3, n_snapshots, figsize=(3 * n_snapshots, 10))
    
    cue_end = task.cue_duration
    delay_end = task.cue_duration + task.delay_duration
    total = task.total_steps
    
    for col, gen in enumerate(snapshot_gens):
        snap = snapshots[gen]
        t = np.arange(total)
        
        # Row 1: Cue = +1
        ax = axes[0, col]
        ax.axvspan(0, cue_end, alpha=0.2, color='blue', label='Cue')
        ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray', label='Delay')
        ax.axvspan(delay_end, total, alpha=0.2, color='green', label='Response')
        ax.plot(t, snap['outputs_pos'], 'b-', linewidth=2)
        ax.axhline(+1, color='green', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Gen {gen}\nfit={snap['fitness']:+.3f}", fontsize=10)
        if col == 0:
            ax.set_ylabel('Output\n(cue=+1)', fontweight='bold')
        ax.set_xticks([])
        
        # Row 2: Cue = -1
        ax = axes[1, col]
        ax.axvspan(0, cue_end, alpha=0.2, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray')
        ax.axvspan(delay_end, total, alpha=0.2, color='green')
        ax.plot(t, snap['outputs_neg'], 'r-', linewidth=2)
        ax.axhline(-1, color='green', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        if col == 0:
            ax.set_ylabel('Output\n(cue=-1)', fontweight='bold')
        ax.set_xticks([])
        
        # Row 3: Both overlaid
        ax = axes[2, col]
        ax.axvspan(0, cue_end, alpha=0.2, color='blue')
        ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray')
        ax.axvspan(delay_end, total, alpha=0.2, color='green')
        ax.plot(t, snap['outputs_pos'], 'b-', linewidth=2, label='cue=+1')
        ax.plot(t, snap['outputs_neg'], 'r-', linewidth=2, label='cue=-1')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('Time')
        if col == 0:
            ax.set_ylabel('Both', fontweight='bold')
            ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('EA Output Evolution Over Generations\n(Blue=cue phase, Gray=delay, Green=response)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ea_output_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ea_output_evolution.png")


def compare_ea_bptt_outputs(conf: AnalysisConfig):
    """
    Train both EA and BPTT, then compare their outputs side-by-side.
    """
    print("=" * 60)
    print("Comparing EA vs BPTT Outputs")
    print("=" * 60)
    
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # Train EA
    print("\nTraining EA...")
    ea_results = train_ea_hybrid(conf)
    
    # Build EA policy from best P
    N = conf.n_neurons
    n_exc = int(0.8 * N)
    final_connectivity = (ea_results['P_final'] > 0.5).astype(np.float32)
    W_rec_ea = final_connectivity * ea_results['fixed_weights']
    np.fill_diagonal(W_rec_ea, 0.0)
    ea_policy = RSNNPolicy(W_rec_ea, ea_results['W_in_final'], ea_results['W_out_final'])
    
    # Train BPTT
    print("\nTraining BPTT...")
    bptt_results = train_bptt(conf)
    
    # Build BPTT policy wrapper
    class BPTTWrapper:
        def __init__(self, model):
            self.model = model
            self.model.eval()
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
    
    bptt_policy = BPTTWrapper(bptt_results['model'])
    
    # Run trials
    rng = np.random.default_rng(conf.seed + 999)
    
    ea_out_pos = run_single_trial(ea_policy, task, cue=+1, rng=rng)
    ea_out_neg = run_single_trial(ea_policy, task, cue=-1, rng=rng)
    bptt_out_pos = run_single_trial(bptt_policy, task, cue=+1, rng=rng)
    bptt_out_neg = run_single_trial(bptt_policy, task, cue=-1, rng=rng)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cue_end = task.cue_duration
    delay_end = task.cue_duration + task.delay_duration
    total = task.total_steps
    t = np.arange(total)
    
    # EA outputs
    ax = axes[0, 0]
    ax.axvspan(0, cue_end, alpha=0.2, color='blue', label='Cue')
    ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray', label='Delay')
    ax.axvspan(delay_end, total, alpha=0.2, color='green', label='Response')
    ax.plot(t, ea_out_pos, 'b-', linewidth=2, label='cue=+1')
    ax.plot(t, ea_out_neg, 'r-', linewidth=2, label='cue=-1')
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Output')
    ax.set_title(f'EA (Hybrid)\nFitness: {ea_results["best_fitness"]:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # BPTT outputs
    ax = axes[0, 1]
    ax.axvspan(0, cue_end, alpha=0.2, color='blue')
    ax.axvspan(cue_end, delay_end, alpha=0.2, color='gray')
    ax.axvspan(delay_end, total, alpha=0.2, color='green')
    ax.plot(t, bptt_out_pos, 'b-', linewidth=2, label='cue=+1')
    ax.plot(t, bptt_out_neg, 'r-', linewidth=2, label='cue=-1')
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Output')
    ax.set_title(f'BPTT\nFitness: {bptt_results["history"]["fitness"][-1]:.3f}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Response phase zoom - EA
    ax = axes[1, 0]
    resp_t = np.arange(task.response_duration)
    width = 0.35
    ax.bar(resp_t - width/2, ea_out_pos[delay_end:], width, label='cue=+1', color='blue', alpha=0.7)
    ax.bar(resp_t + width/2, ea_out_neg[delay_end:], width, label='cue=-1', color='red', alpha=0.7)
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Response Step')
    ax.set_ylabel('Output')
    ax.set_title('EA: Response Phase', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Response phase zoom - BPTT
    ax = axes[1, 1]
    ax.bar(resp_t - width/2, bptt_out_pos[delay_end:], width, label='cue=+1', color='blue', alpha=0.7)
    ax.bar(resp_t + width/2, bptt_out_neg[delay_end:], width, label='cue=-1', color='red', alpha=0.7)
    ax.axhline(+1, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(-1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Response Step')
    ax.set_ylabel('Output')
    ax.set_title('BPTT: Response Phase', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('EA vs BPTT: Network Outputs on Working Memory Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ea_vs_bptt_outputs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: ea_vs_bptt_outputs.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    print(f"\nEA Response (cue=+1):  mean={ea_out_pos[delay_end:].mean():+.3f}")
    print(f"EA Response (cue=-1):  mean={ea_out_neg[delay_end:].mean():+.3f}")
    print(f"BPTT Response (cue=+1): mean={bptt_out_pos[delay_end:].mean():+.3f}")
    print(f"BPTT Response (cue=-1): mean={bptt_out_neg[delay_end:].mean():+.3f}")
    
    return {
        'ea': {'pos': ea_out_pos, 'neg': ea_out_neg, 'fitness': ea_results['best_fitness']},
        'bptt': {'pos': bptt_out_pos, 'neg': bptt_out_neg, 'fitness': bptt_results['history']['fitness'][-1]},
    }


# Quick runner
if __name__ == "__main__":
    from run_full_analysis import AnalysisConfig, train_ea_hybrid, train_bptt, RSNNPolicy
    import torch
    
    conf = AnalysisConfig(
        n_neurons=256,
        cue_duration=5,
        delay_duration=10,
        response_duration=10,
        response_weight=1.0,  # 100% on response
        ea_generations=300,
        ea_pop_size=128,
        seed=42,
    )
    
    # Option 1: See EA learning over time
    # snapshots = visualize_learning_over_generations(conf)
    
    # Option 2: Compare final EA vs BPTT
    # results = compare_ea_bptt_outputs(conf)