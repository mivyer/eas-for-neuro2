# run_full_analysis.py
"""
Full Analysis Pipeline — v2

Runs EA and BPTT on two working memory tasks:
  1. Delayed match-to-sample (original, --task wm)
  2. Letter n-back recall (new, --task nback)

EA: evolves ALL weights directly (same param space as BPTT)
BPTT: standard gradient-based training

Default: 32 neurons, n-back recall
"""

import numpy as np
import time
import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime

from models.rsnn_policy import RSNNPolicy
from envs.working_memory import WorkingMemoryTask
from envs.letter_nback import LetterNBackTask, SYMBOL_VALUES, SYMBOL_LABELS

try:
    import torch
    from models.bptt_rnn import RNNPolicy, count_parameters
    from envs.working_memory import WorkingMemoryTaskTorch
    from envs.letter_nback import LetterNBackTaskTorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available — BPTT will be skipped")

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("matplotlib not available — plots will be skipped")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Network
    n_neurons: int = 32
    obs_dim: int = 1
    action_dim: int = 1

    # Task: "nback" or "wm"
    task: str = "nback"

    # N-back params
    n_back: int = 2
    seq_length: int = 20

    # Working memory params (used when task="wm")
    cue_duration: int = 5
    delay_duration: int = 10
    response_duration: int = 10

    # EA (Evolution Strategy — OpenAI-ES)
    ea_pop_size: int = 128
    ea_generations: int = 300
    ea_lr: float = 0.03
    ea_sigma: float = 0.02      # tuned: was 0.1, smaller = cleaner gradient estimate
    ea_n_eval_trials: int = 20

    # GA (Genetic Algorithm — tournament + crossover + mutation)
    ga_mutation_rate: float = 0.05
    ga_mutation_std: float = 0.3

    # BPTT
    bptt_iterations: int = 1000
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3

    # Analysis
    sparsity_threshold: float = 0.01

    # Misc
    seed: int = 42
    print_every: int = 25
    output_dir: str = "results"


# ============================================================================
# Task Factory
# ============================================================================

def make_task_np(conf: Config):
    if conf.task == "nback":
        return LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)
    return WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
    )


def make_task_torch(conf: Config):
    if conf.task == "nback":
        return LetterNBackTaskTorch(n_back=conf.n_back, seq_length=conf.seq_length)
    return WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
    )


# ============================================================================
# EA: Direct Weight Evolution (OpenAI-ES, mirrored sampling)
# ============================================================================

def train_ea(conf: Config) -> dict:
    """Evolve W_rec + W_in + W_out directly. Same param space as BPTT."""
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    task = make_task_np(conf)

    scale = np.sqrt(2.0 / N)
    W_rec = scale * rng.standard_normal((N, N)).astype(np.float32)
    W_in = scale * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
    W_out = scale * rng.standard_normal((conf.action_dim, N)).astype(np.float32)

    def flatten(Wr, Wi, Wo):
        return np.concatenate([Wr.ravel(), Wi.ravel(), Wo.ravel()])

    def unflatten(p):
        nr = N * N; ni = N * conf.obs_dim
        return (p[:nr].reshape(N, N),
                p[nr:nr + ni].reshape(N, conf.obs_dim),
                p[nr + ni:].reshape(conf.action_dim, N))

    params = flatten(W_rec, W_in, W_out)
    n_params = len(params)
    W_rec_init, W_in_init, W_out_init = unflatten(params.copy())

    history = {'fitness': [], 'accuracy': [], 'best_fitness': []}
    snapshot_gens = sorted(set(
        [0, 25, 50, 100, 150, 200, 250] + [conf.ea_generations - 1]
    ))
    snapshots = {}

    best_fitness = -np.inf
    best_params = params.copy()
    half_pop = conf.ea_pop_size // 2

    print(f"EA (direct weights): {N} neurons, {n_params} params")
    print(f"Task: {conf.task} | pop={conf.ea_pop_size}, gens={conf.ea_generations}")

    for gen in range(conf.ea_generations):
        noise = rng.standard_normal((half_pop, n_params)).astype(np.float32)

        fitness_pos = np.zeros(half_pop)
        fitness_neg = np.zeros(half_pop)
        acc_pos = np.zeros(half_pop)
        acc_neg = np.zeros(half_pop)

        for i in range(half_pop):
            for sign, fit_arr, acc_arr in [(+1, fitness_pos, acc_pos),
                                           (-1, fitness_neg, acc_neg)]:
                Wr, Wi, Wo = unflatten(params + sign * conf.ea_sigma * noise[i])
                r = task.evaluate_policy(RSNNPolicy(Wr, Wi, Wo),
                                         n_trials=conf.ea_n_eval_trials, rng=rng)
                fit_arr[i] = r['fitness']
                acc_arr[i] = r['accuracy']

        all_fitness = np.concatenate([fitness_pos, fitness_neg])
        all_acc = np.concatenate([acc_pos, acc_neg])

        idx = np.argmax(all_fitness)
        if all_fitness[idx] > best_fitness:
            best_fitness = all_fitness[idx]
            sign = +1 if idx < half_pop else -1
            noise_idx = idx if idx < half_pop else idx - half_pop
            best_params = params + sign * conf.ea_sigma * noise[noise_idx]

        grad = np.mean((fitness_pos - fitness_neg)[:, None] * noise, axis=0) / conf.ea_sigma
        params += conf.ea_lr * grad

        history['fitness'].append(float(all_fitness.mean()))
        history['accuracy'].append(float(all_acc.mean()))
        history['best_fitness'].append(float(best_fitness))

        if gen in snapshot_gens:
            Wr, Wi, Wo = unflatten(best_params)
            sr = task.evaluate_policy(RSNNPolicy(Wr, Wi, Wo), n_trials=50, rng=rng)
            snapshots[gen] = {'fitness': sr['fitness'], 'accuracy': sr['accuracy']}

        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            print(f"Gen {gen:4d} | mean={all_fitness.mean():+.4f} "
                  f"best={best_fitness:+.4f} acc={all_acc.mean():.1%}")

    Wr_f, Wi_f, Wo_f = unflatten(best_params)
    return {
        'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
        'W_rec_final': Wr_f, 'W_in_final': Wi_f, 'W_out_final': Wo_f,
        'best_fitness': best_fitness,
        'history': history,
        'snapshots': snapshots,
        'snapshot_gens': list(snapshots.keys()),
    }


# ============================================================================
# BPTT
# ============================================================================

def train_bptt(conf: Config) -> dict | None:
    if not TORCH_AVAILABLE:
        return None

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    N = conf.n_neurons
    model = RNNPolicy(N, conf.obs_dim, conf.action_dim).to("cpu")
    task_torch = make_task_torch(conf)

    with torch.no_grad():
        W_rec_init = model.W_rec.cpu().numpy().copy()
        W_in_init = model.W_in.cpu().numpy().copy()
        W_out_init = model.W_out.cpu().numpy().copy()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.bptt_lr)
    history = {'loss': [], 'fitness': [], 'accuracy': [], 'sparsity_rec': []}
    snapshot_iters = sorted(set(
        [0, 50, 100, 200, 500, 750] + [conf.bptt_iterations - 1]
    ))
    snapshots = {}

    print(f"BPTT: {N} neurons, {count_parameters(model)} params")
    print(f"Task: {conf.task} | iters={conf.bptt_iterations}, lr={conf.bptt_lr}")

    for it in range(conf.bptt_iterations):
        model.train()
        inputs, targets = task_torch.get_batch(conf.bptt_batch_size)
        outputs = model(inputs)
        loss = task_torch.compute_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            acc = task_torch.compute_accuracy(outputs, targets)
            fit = task_torch.compute_fitness(outputs, targets)
            sp = float((model.W_rec.abs() < conf.sparsity_threshold).float().mean())

        history['loss'].append(float(loss.item()))
        history['fitness'].append(float(fit))
        history['accuracy'].append(float(acc))
        history['sparsity_rec'].append(sp)

        if it in snapshot_iters:
            snapshots[it] = {'fitness': float(fit), 'accuracy': float(acc)}

        if it % (conf.print_every * 4) == 0 or it == conf.bptt_iterations - 1:
            print(f"Iter {it:4d} | loss={loss.item():.4f} "
                  f"fitness={fit:+.4f} acc={acc:.1%}")

    with torch.no_grad():
        W_rec_final = model.W_rec.cpu().numpy().copy()
        W_in_final = model.W_in.cpu().numpy().copy()
        W_out_final = model.W_out.cpu().numpy().copy()

    return {
        'model': model,
        'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
        'W_rec_final': W_rec_final, 'W_in_final': W_in_final, 'W_out_final': W_out_final,
        'history': history,
        'snapshots': snapshots,
        'snapshot_iters': list(snapshots.keys()),
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_learning_curves(ea, bptt, conf: Config):
    if not PLOT_AVAILABLE:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(ea['history']['fitness'], 'b-', alpha=0.4, label='EA mean')
    ax.plot(ea['history']['best_fitness'], 'b-', lw=2, label='EA best')
    if bptt:
        bx = np.linspace(0, conf.ea_generations, len(bptt['history']['fitness']))
        ax.plot(bx, bptt['history']['fitness'], 'r-', lw=2, label='BPTT')
    ax.set_xlabel('Generation'); ax.set_ylabel('Fitness')
    ax.set_title('Fitness'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ea['history']['accuracy'], 'b-', alpha=0.7, label='EA')
    if bptt:
        bx = np.linspace(0, conf.ea_generations, len(bptt['history']['accuracy']))
        ax.plot(bx, bptt['history']['accuracy'], 'r-', lw=2, label='BPTT')
    ax.set_xlabel('Generation'); ax.set_ylabel('Accuracy')
    ax.set_title('Symbol Recall Accuracy'); ax.legend()
    ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.05)

    ax = axes[2]
    if bptt and 'sparsity_rec' in bptt['history']:
        bx = np.linspace(0, conf.ea_generations, len(bptt['history']['sparsity_rec']))
        ax.plot(bx, bptt['history']['sparsity_rec'], 'r-', lw=2, label='BPTT')
    ea_sp = float((np.abs(ea['W_rec_final']) < conf.sparsity_threshold).mean())
    ax.axhline(ea_sp, color='b', ls='--', label=f'EA final ({ea_sp:.2f})')
    ax.set_xlabel('Generation'); ax.set_ylabel('Sparsity')
    ax.set_title('W_rec Sparsity'); ax.legend(); ax.grid(True, alpha=0.3)

    task_label = f"{conf.n_back}-back" if conf.task == "nback" else "WM"
    plt.suptitle(f'{task_label} | {conf.n_neurons}N', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/learning_curves.png")


def plot_weight_comparison(ea, bptt, conf: Config):
    if not PLOT_AVAILABLE:
        return

    ncols = 4 if bptt else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5))

    vmax = max(np.abs(ea['W_rec_init']).max(), np.abs(ea['W_rec_final']).max())
    if bptt:
        vmax = max(vmax, np.abs(bptt['W_rec_final']).max())

    im = axes[0].imshow(ea['W_rec_init'], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0].set_title('EA: Init')
    axes[1].imshow(ea['W_rec_final'], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[1].set_title('EA: Final')
    if bptt:
        axes[2].imshow(bptt['W_rec_init'], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[2].set_title('BPTT: Init')
        axes[3].imshow(bptt['W_rec_final'], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        axes[3].set_title('BPTT: Final')

    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle('W_rec Matrices', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'weight_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/weight_comparison.png")


def plot_sample_trial(ea, bptt, conf: Config):
    if not PLOT_AVAILABLE:
        return

    task = make_task_np(conf)
    rng = np.random.default_rng(conf.seed + 999)

    ea_policy = RSNNPolicy(ea['W_rec_final'], ea['W_in_final'], ea['W_out_final'])

    bptt_policy = None
    if bptt and TORCH_AVAILABLE:
        class _W:
            def __init__(s, m):
                s.m = m; s.m.eval(); s.h = None
            def reset(s):
                with torch.no_grad(): s.h = s.m.h0.detach().clone()
            def act(s, obs):
                with torch.no_grad():
                    o = torch.tensor(obs, dtype=torch.float32)
                    s.h = torch.tanh(s.h @ s.m.W_rec.T + o @ s.m.W_in.T)
                    return torch.tanh(s.h @ s.m.W_out.T).numpy()
        bptt_policy = _W(bptt['model'])

    policies = [('EA', ea_policy)]
    if bptt_policy:
        policies.append(('BPTT', bptt_policy))

    ncols = len(policies)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 6), squeeze=False)

    for col, (label, policy) in enumerate(policies):
        if conf.task == "nback":
            inputs, targets, letters = task.get_trial(rng=np.random.default_rng(conf.seed + 999))
            policy.reset()
            outputs = []
            for t in range(task.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                a = policy.act(obs)
                outputs.append(float(a[0]) if hasattr(a, '__len__') else float(a))
            outputs = np.array(outputs)

            ax = axes[0, col]
            ax.step(range(len(inputs)), inputs, 'k-', lw=1.5, where='mid', label='Input')
            ax.set_ylabel('Letter'); ax.set_title(f'{label}: Input')
            ax.set_yticks(list(SYMBOL_VALUES))
            ax.set_yticklabels(SYMBOL_LABELS)
            ax.set_ylim(0.05, 1.15)
            ax.legend(); ax.grid(True, alpha=0.3)

            ax = axes[1, col]
            ax.plot(targets, 'g--', lw=2, label='Target (recall)')
            ax.plot(outputs, 'b-', lw=1.5, label='Output')
            ax.axvspan(0, task.n_back - 0.5, alpha=0.15, color='gray')
            acc = task.compute_accuracy(outputs, targets)
            ax.set_title(f'{label}: Recall (acc={acc:.0%})')
            ax.set_xlabel('Step'); ax.set_ylabel('Recalled Letter')
            ax.set_yticks(list(SYMBOL_VALUES))
            ax.set_yticklabels(SYMBOL_LABELS)
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.1, 1.2)
        else:
            for cue, ls in [(1.0, '-'), (-1.0, '--')]:
                inp, tgt = task.get_trial(cue=cue, rng=np.random.default_rng(conf.seed + 999))
                policy.reset()
                outs = [float(policy.act(np.array([inp[t]], dtype=np.float32))[0])
                        for t in range(task.total_steps)]
                axes[0, col].plot(inp, f'b{ls}', lw=1.5, label=f'cue={cue:+.0f}')
                axes[1, col].plot(outs, f'b{ls}', lw=1.5, label=f'cue={cue:+.0f}')
                axes[1, col].axhline(cue, color='g', ls=ls, alpha=0.4)
            axes[0, col].set_title(f'{label}: Input'); axes[0, col].legend()
            axes[1, col].set_title(f'{label}: Output'); axes[1, col].legend()
            for r in range(2):
                axes[r, col].grid(True, alpha=0.3)
            axes[1, col].set_ylim(-1.5, 1.5)

    task_label = f"{conf.n_back}-back recall" if conf.task == "nback" else "WM"
    plt.suptitle(f'{task_label} | {conf.n_neurons}N | Sample Trial', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(conf.output_dir, 'sample_trial.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {conf.output_dir}/sample_trial.png")


# ============================================================================
# Save
# ============================================================================

def save_results(ea, bptt, conf: Config):
    os.makedirs(conf.output_dir, exist_ok=True)

    with open(os.path.join(conf.output_dir, 'config.json'), 'w') as f:
        json.dump(asdict(conf), f, indent=2)

    np.savez(os.path.join(conf.output_dir, 'ea_weights.npz'),
             W_rec_init=ea['W_rec_init'], W_in_init=ea['W_in_init'], W_out_init=ea['W_out_init'],
             W_rec_final=ea['W_rec_final'], W_in_final=ea['W_in_final'], W_out_final=ea['W_out_final'])
    with open(os.path.join(conf.output_dir, 'ea_history.json'), 'w') as f:
        json.dump({'history': ea['history'],
                   'snapshots': {str(k): v for k, v in ea['snapshots'].items()}}, f)

    if bptt:
        np.savez(os.path.join(conf.output_dir, 'bptt_weights.npz'),
                 W_rec_init=bptt['W_rec_init'], W_in_init=bptt['W_in_init'], W_out_init=bptt['W_out_init'],
                 W_rec_final=bptt['W_rec_final'], W_in_final=bptt['W_in_final'], W_out_final=bptt['W_out_final'])
        with open(os.path.join(conf.output_dir, 'bptt_history.json'), 'w') as f:
            json.dump({'history': bptt['history'],
                       'snapshots': {str(k): v for k, v in bptt['snapshots'].items()}}, f)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'task': conf.task, 'n_neurons': conf.n_neurons,
        'ea_best_fitness': float(ea['best_fitness']),
        'ea_final_accuracy': float(ea['history']['accuracy'][-1]),
        'bptt_final_fitness': float(bptt['history']['fitness'][-1]) if bptt else None,
        'bptt_final_accuracy': float(bptt['history']['accuracy'][-1]) if bptt else None,
    }
    with open(os.path.join(conf.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {conf.output_dir}/")


# ============================================================================
# Main
# ============================================================================

def run(task="nback", n_neurons=32, output_dir=None,
        method="es", bptt=True, **overrides):
    """
    Run experiment.

    method: "es" (OpenAI-ES), "ga" (genetic algorithm), or "both"
    bptt:   whether to also train BPTT for comparison
    """
    if output_dir is None:
        output_dir = f"results/{task}_{n_neurons}n"

    conf = Config(task=task, n_neurons=n_neurons, output_dir=output_dir, **overrides)
    os.makedirs(conf.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"EXPERIMENT: {conf.task.upper()} | {conf.n_neurons} neurons")
    print("=" * 60)
    if conf.task == "nback":
        print(f"  {conf.n_back}-back recall, seq_len={conf.seq_length}, 5 symbols")
    else:
        print(f"  WM: cue={conf.cue_duration}, delay={conf.delay_duration}, "
              f"response={conf.response_duration}")

    ea_results = None
    ga_results = None

    if method in ("es", "both"):
        print("\n" + "-" * 60)
        print("Training ES (OpenAI Evolution Strategy)...")
        print("-" * 60)
        t0 = time.time()
        ea_results = train_ea(conf)
        print(f"ES time: {time.time() - t0:.1f}s\n")

    if method in ("ga", "both"):
        print("-" * 60)
        print("Training GA (Genetic Algorithm)...")
        print("-" * 60)
        from ea_genetic import train_ga
        t0 = time.time()
        ga_results = train_ga(conf)
        print(f"GA time: {time.time() - t0:.1f}s\n")

    bptt_results = None
    if bptt:
        print("-" * 60)
        print("Training BPTT...")
        print("-" * 60)
        t0 = time.time()
        bptt_results = train_bptt(conf)
        if bptt_results:
            print(f"BPTT time: {time.time() - t0:.1f}s\n")

    # Use whichever EA ran (prefer GA if both)
    primary_ea = ga_results if ga_results else ea_results

    if primary_ea:
        save_results(primary_ea, bptt_results, conf)

    # Generate all figures
    try:
        from visualize_outputs import generate_all_figures
        generate_all_figures(primary_ea, bptt_results, conf)
    except ImportError:
        if primary_ea:
            plot_learning_curves(primary_ea, bptt_results, conf)
            plot_weight_comparison(primary_ea, bptt_results, conf)
            plot_sample_trial(primary_ea, bptt_results, conf)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    if ea_results:
        print(f"ES:   fitness={ea_results['best_fitness']:+.4f}  "
              f"acc={ea_results['history']['accuracy'][-1]:.1%}")
    if ga_results:
        print(f"GA:   fitness={ga_results['best_fitness']:+.4f}  "
              f"acc={ga_results['history']['accuracy'][-1]:.1%}")
    if bptt_results:
        print(f"BPTT: fitness={bptt_results['history']['fitness'][-1]:+.4f}  "
              f"acc={bptt_results['history']['accuracy'][-1]:.1%}")
    print(f"Output: {conf.output_dir}/")

    return {'es': ea_results, 'ga': ga_results, 'bptt': bptt_results, 'config': conf}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['nback', 'wm'], default='nback')
    p.add_argument('--neurons', type=int, default=32)
    p.add_argument('--n-back', type=int, default=2)
    p.add_argument('--method', choices=['es', 'ga', 'both'], default='ga')
    p.add_argument('--no-bptt', action='store_true')
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--ea-gens', type=int, default=300)
    p.add_argument('--bptt-iters', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    run(task=args.task, n_neurons=args.neurons, output_dir=args.output,
        method=args.method, bptt=not args.no_bptt,
        n_back=args.n_back,
        ea_generations=args.ea_gens, bptt_iterations=args.bptt_iters,
        seed=args.seed)