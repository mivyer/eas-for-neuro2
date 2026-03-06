# trainers/train_es.py
"""OpenAI Evolution Strategy — estimates gradient from population statistics."""

import numpy as np
from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import LetterNBackTask
from envs.working_memory import WorkingMemoryTask


def make_task(conf):
    if conf.task == "nback":
        return LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)
    if conf.task == "evidence":
        from envs.evidence_accumulation import EvidenceAccumulationTask
        return EvidenceAccumulationTask(
            evidence_strength=conf.evidence_strength,
            noise_std=conf.noise_std,
            trial_length=conf.trial_length,
            response_length=conf.response_length,
        )
    if conf.task == "robot":
        from envs.robot_arm import RobotArmTask
        return RobotArmTask(seq_length=conf.seq_length)
    return WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
    )


def train_es(conf) -> dict:
    """Evolve W_rec + W_in + W_out via OpenAI-ES with mirrored sampling."""
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    task = make_task(conf)

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
    snapshots = {}
    best_fitness  = -np.inf
    best_accuracy = 0.0   # all-time best, updated when best_params changes
    best_params = params.copy()
    half_pop = conf.ea_pop_size // 2

    print(f"ES (direct weights): {N} neurons, {n_params} params")
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
            best_fitness  = all_fitness[idx]
            best_accuracy = all_acc[idx]
            sign = +1 if idx < half_pop else -1
            noise_idx = idx if idx < half_pop else idx - half_pop
            best_params = params + sign * conf.ea_sigma * noise[noise_idx]

        grad = np.mean((fitness_pos - fitness_neg)[:, None] * noise, axis=0) / conf.ea_sigma
        params += conf.ea_lr * grad

        history['fitness'].append(float(all_fitness.mean()))
        history['accuracy'].append(float(all_acc.mean()))
        history['best_fitness'].append(float(best_fitness))

        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            print(f"Gen {gen:4d} | mean={all_acc.mean():.1%} "
                  f"best={best_accuracy:.1%}")

    Wr_f, Wi_f, Wo_f = unflatten(best_params)

    # Stable final estimate: re-evaluate best individual with more trials
    final_eval    = task.evaluate_policy(RSNNPolicy(Wr_f, Wi_f, Wo_f),
                                         n_trials=50, rng=rng)
    best_accuracy = final_eval['accuracy']

    return {
        'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
        'W_rec_final': Wr_f, 'W_in_final': Wi_f, 'W_out_final': Wo_f,
        'best_fitness':  best_fitness,
        'best_accuracy': best_accuracy,   # stable 50-trial estimate
        'history': history,
        'snapshots': snapshots,
        'best_gene': best_params,   # flat parameter vector for the best individual
    }
