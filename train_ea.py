# train_ea.py
"""
Train RSNN on working memory task using evolutionary algorithm.
Uses OpenAI-ES style with mirrored sampling.
"""

import numpy as np
from dataclasses import dataclass

from models.rsnn_policy import RSNNPolicy
from envs.working_memory import WorkingMemoryTask


@dataclass
class EAConfig:
    n_neurons: int = 64
    obs_dim: int = 1
    action_dim: int = 1
    
    pop_size: int = 64
    n_generations: int = 200
    lr: float = 0.03
    sigma: float = 0.1
    
    cue_duration: int = 5
    delay_duration: int = 15
    response_duration: int = 5
    response_weight: float = 0.75  # Weighted loss
    n_eval_trials: int = 20
    
    seed: int = 42
    print_every: int = 10


def train_ea(conf: EAConfig) -> dict:
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    
    # Initialize weights
    scale = np.sqrt(2.0 / N)
    W_rec = scale * rng.standard_normal((N, N)).astype(np.float32)
    W_in = scale * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out = scale * rng.standard_normal((conf.action_dim, N)).astype(np.float32)
    
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
    
    print(f"EA Training: {N} neurons, {n_params} parameters")
    print(f"Loss weighting: {conf.response_weight:.0%} response")
    
    history = {'fitness': [], 'accuracy': [], 'best_fitness': []}
    best_fitness = -np.inf
    best_params = params.copy()
    
    half_pop = conf.pop_size // 2
    
    for gen in range(conf.n_generations):
        noise = rng.standard_normal((half_pop, n_params)).astype(np.float32)
        
        fitness_pos, fitness_neg = [], []
        acc_pos, acc_neg = [], []
        
        for i in range(half_pop):
            # Positive
            W_rec, W_in, W_out = unflatten(params + conf.sigma * noise[i])
            policy = RSNNPolicy(W_rec, W_in, W_out)
            result = task.evaluate_policy(policy, n_trials=conf.n_eval_trials, rng=rng)
            fitness_pos.append(result['fitness'])
            acc_pos.append(result['accuracy'])
            
            # Negative
            W_rec, W_in, W_out = unflatten(params - conf.sigma * noise[i])
            policy = RSNNPolicy(W_rec, W_in, W_out)
            result = task.evaluate_policy(policy, n_trials=conf.n_eval_trials, rng=rng)
            fitness_neg.append(result['fitness'])
            acc_neg.append(result['accuracy'])
        
        fitness_pos = np.array(fitness_pos)
        fitness_neg = np.array(fitness_neg)
        all_fitness = np.concatenate([fitness_pos, fitness_neg])
        all_acc = np.concatenate([acc_pos, acc_neg])
        
        # Track best
        max_idx = np.argmax(all_fitness)
        if all_fitness[max_idx] > best_fitness:
            best_fitness = all_fitness[max_idx]
            if max_idx < half_pop:
                best_params = params + conf.sigma * noise[max_idx]
            else:
                best_params = params - conf.sigma * noise[max_idx - half_pop]
        
        # Update
        grad = np.mean((fitness_pos - fitness_neg)[:, None] * noise, axis=0) / conf.sigma
        params = params + conf.lr * grad
        
        history['fitness'].append(float(all_fitness.mean()))
        history['accuracy'].append(float(all_acc.mean()))
        history['best_fitness'].append(float(best_fitness))
        
        if gen % conf.print_every == 0 or gen == conf.n_generations - 1:
            print(f"Gen {gen:4d} | fitness: {all_fitness.mean():+.3f} | "
                  f"acc: {all_acc.mean():.1%} | best: {best_fitness:+.3f}")
    
    W_rec, W_in, W_out = unflatten(best_params)
    return {
        'W_rec': W_rec, 'W_in': W_in, 'W_out': W_out,
        'params': best_params,
        'history': history,
        'best_fitness': best_fitness,
    }


def main():
    conf = EAConfig(n_neurons=64, n_generations=200, seed=42)
    results = train_ea(conf)
    
    # Final test
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    policy = RSNNPolicy(results['W_rec'], results['W_in'], results['W_out'])
    rng = np.random.default_rng(conf.seed + 1000)
    test = task.evaluate_policy(policy, n_trials=100, rng=rng)
    
    print(f"\nFinal test: fitness={test['fitness']:.3f}, accuracy={test['accuracy']:.1%}")
    return results


if __name__ == "__main__":
    main()
