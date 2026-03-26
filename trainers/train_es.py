"""OpenAI Evolution Strategy with mirrored sampling and 1/5-success-rule sigma adaptation."""

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
    rng = np.random.default_rng(conf.seed)
    N = conf.n_neurons
    task = make_task(conf)

    scale = np.sqrt(2.0 / N)
    W_rec = scale * rng.standard_normal((N, N)).astype(np.float32)
    W_in  = scale * rng.standard_normal((N, conf.obs_dim)).astype(np.float32)
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
    l2_coef = getattr(conf, 'ea_l2_coef', 0.0)

    sigma = conf.ea_sigma
    if getattr(conf, 'ea_sigma_scaling', False):
        baseline = getattr(conf, 'ea_baseline_params', 1344)
        sigma = conf.ea_sigma / np.sqrt(n_params / baseline)

    pop_size = conf.ea_pop_size
    if getattr(conf, 'ea_auto_pop', False):
        pop_size = max(conf.ea_pop_size, int(4 * np.sqrt(n_params)))
    half_pop = pop_size // 2

    # 1/5-success-rule (Rechenberg 1973)
    SUCCESS_WIN = 20
    SIGMA_UP    = 1.10
    SIGMA_DN    = 0.95
    SIGMA_MIN   = max(sigma * 0.25, 0.01)
    SIGMA_MAX   = sigma * 4.0
    success_hist = []

    history = {'fitness': [], 'accuracy': [], 'best_fitness': [], 'sigma': []}
    snapshots = {}
    best_fitness  = -np.inf
    best_accuracy = 0.0
    best_params   = params.copy()

    print(f"ES: {N} neurons, {n_params} params, pop={pop_size}, gens={conf.ea_generations}")
    print(f"sigma={sigma:.5f}, l2_coef={l2_coef}")

    for gen in range(conf.ea_generations):
        noise = rng.standard_normal((half_pop, n_params)).astype(np.float32)

        fitness_pos = np.zeros(half_pop)
        fitness_neg = np.zeros(half_pop)
        acc_pos     = np.zeros(half_pop)
        acc_neg     = np.zeros(half_pop)

        for i in range(half_pop):
            for sign, fit_arr, acc_arr in [(+1, fitness_pos, acc_pos),
                                           (-1, fitness_neg, acc_neg)]:
                candidate = params + sign * sigma * noise[i]
                Wr, Wi, Wo = unflatten(candidate)
                r = task.evaluate_policy(RSNNPolicy(Wr, Wi, Wo),
                                         n_trials=conf.ea_n_eval_trials, rng=rng)
                penalty = l2_coef * float(np.mean(candidate ** 2)) if l2_coef > 0.0 else 0.0
                fit_arr[i] = r['fitness'] - penalty
                acc_arr[i] = r['accuracy']

        all_fitness = np.concatenate([fitness_pos, fitness_neg])
        all_acc     = np.concatenate([acc_pos, acc_neg])

        idx = np.argmax(all_fitness)
        improved = all_fitness[idx] > best_fitness
        if improved:
            best_fitness  = all_fitness[idx]
            best_accuracy = all_acc[idx]
            sign      = +1 if idx < half_pop else -1
            noise_idx = idx if idx < half_pop else idx - half_pop
            best_params = params + sign * sigma * noise[noise_idx]

        success_hist.append(1 if improved else 0)
        if len(success_hist) > SUCCESS_WIN:
            success_hist.pop(0)
        if len(success_hist) == SUCCESS_WIN:
            rate = sum(success_hist) / SUCCESS_WIN
            if rate > 0.2:
                sigma = min(sigma * SIGMA_UP, SIGMA_MAX)
            elif rate < 0.2:
                sigma = max(sigma * SIGMA_DN, SIGMA_MIN)

        grad = np.mean((fitness_pos - fitness_neg)[:, None] * noise, axis=0) / sigma
        params += conf.ea_lr * grad

        history['fitness'].append(float(all_fitness.mean()))
        history['accuracy'].append(float(all_acc.mean()))
        history['best_fitness'].append(float(best_fitness))
        history['sigma'].append(float(sigma))

        if gen % conf.print_every == 0 or gen == conf.ea_generations - 1:
            print(f"Gen {gen:4d} | mean={all_acc.mean():.1%} "
                  f"best={best_accuracy:.1%} σ={sigma:.5f}")

    Wr_f, Wi_f, Wo_f = unflatten(best_params)
    final_eval    = task.evaluate_policy(RSNNPolicy(Wr_f, Wi_f, Wo_f), n_trials=50, rng=rng)
    best_accuracy = final_eval['accuracy']

    return {
        'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
        'W_rec_final': Wr_f, 'W_in_final': Wi_f, 'W_out_final': Wo_f,
        'best_fitness':  best_fitness,
        'best_accuracy': best_accuracy,
        'history': history,
        'snapshots': snapshots,
        'best_gene': best_params,
    }
