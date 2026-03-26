"""
GA + Oja's rule (Hebbian plasticity on W_rec).

Same GA mechanics as train_ga.py. Each fitness evaluation runs the network
step-by-step and applies Oja's rule to a copy of W_rec; modifications are
discarded after the trial (no Lamarckian inheritance). The GA evolves
initial weights plus plasticity parameters (eta, w_max).

Genotype:  [W_rec | W_in | W_out | log_eta | log_wmax | sigma_0..N-1]
Oja rule:  dW = eta * (outer(h, h) - (h**2)[:, None] * W_rec)
"""

import numpy as np
from envs.letter_nback import LetterNBackTask


class GeneticAlgorithmOja:

    def __init__(self, n_neurons=32, obs_dim=5, action_dim=5,
                 pop_size=128, n_elite=4, tournament_k=5,
                 crossover_rate=0.7, mutation_rate=0.05,
                 mutation_std=0.3, n_eval_trials=20, seed=42,
                 l2_coef=0.0):
        self.n_neurons    = n_neurons
        self.obs_dim      = obs_dim
        self.action_dim   = action_dim
        self.pop_size     = pop_size
        self.n_elite      = n_elite
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate  = mutation_rate
        self.mutation_std   = mutation_std
        self.n_eval_trials  = n_eval_trials
        self.l2_coef        = l2_coef
        self.rng = np.random.default_rng(seed)

        N = n_neurons
        self.weight_length   = N * N + N * obs_dim + action_dim * N
        self.oja_gene_length = self.weight_length + 2
        self.tau = 1.0 / np.sqrt(2.0 * N)

    def _encode(self, W_rec, W_in, W_out, log_eta, log_wmax):
        return np.concatenate([
            W_rec.ravel(), W_in.ravel(), W_out.ravel(),
            np.array([log_eta, log_wmax], dtype=np.float32),
        ])

    def _decode(self, gene):
        N  = self.n_neurons
        nr = N * N
        ni = N * self.obs_dim
        no = self.action_dim * N
        w  = gene[:self.oja_gene_length]

        W_rec = w[:nr].reshape(N, N).astype(np.float32)
        W_in  = w[nr:nr + ni].reshape(N, self.obs_dim).astype(np.float32)
        W_out = w[nr + ni:nr + ni + no].reshape(self.action_dim, N).astype(np.float32)
        eta   = float(np.clip(np.exp(float(w[-2])), 1e-6, 1.0))
        w_max = float(np.clip(np.exp(float(w[-1])), 0.1,  20.0))
        return W_rec, W_in, W_out, eta, w_max

    def _run_oja_trial(self, W_rec_init, W_in, W_out, eta, w_max, inputs, targets):
        T     = inputs.shape[0]
        W_rec = W_rec_init.copy()
        h     = np.zeros(self.n_neurons, dtype=np.float32)
        outputs = np.zeros((T, self.action_dim), dtype=np.float32)

        for t in range(T):
            h = np.tanh(W_rec @ h + W_in @ inputs[t])
            outputs[t] = np.tanh(W_out @ h)
            dW    = eta * (np.outer(h, h) - (h ** 2)[:, None] * W_rec)
            W_rec = np.clip(W_rec + dW, -w_max, w_max)

        if targets.ndim == 2:
            acc = -float(np.mean((outputs - targets) ** 2))
        else:
            mask = targets >= 0
            acc  = float((np.argmax(outputs[mask], axis=-1) == targets[mask]).mean()) \
                   if mask.any() else 0.0
        return acc, W_rec

    def init_population(self):
        N = self.n_neurons
        scale = np.sqrt(2.0 / N)
        population = []
        for _ in range(self.pop_size):
            weights  = (scale * self.rng.standard_normal(self.weight_length)).astype(np.float32)
            log_eta  = np.float32(np.log(0.01))
            log_wmax = np.float32(np.log(2.0))
            sigmas   = np.full(N, self.mutation_rate, dtype=np.float32)
            population.append(np.concatenate([weights, [log_eta, log_wmax], sigmas]))
        return population

    def _evaluate_one(self, gene, task):
        W_rec, W_in, W_out, eta, w_max = self._decode(gene)
        accs = []
        for _ in range(self.n_eval_trials):
            inputs, targets, _ = task.get_trial(rng=self.rng)
            acc, _ = self._run_oja_trial(W_rec, W_in, W_out, eta, w_max, inputs, targets)
            accs.append(acc)
        mean_acc = float(np.mean(accs))
        display_acc = float(np.exp(mean_acc)) if mean_acc < 0 else mean_acc
        fitness = mean_acc
        if self.l2_coef > 0.0:
            fitness -= self.l2_coef * float(np.mean(gene[:self.weight_length] ** 2))
        return {'fitness': fitness, 'accuracy': display_acc}

    def evaluate(self, population, task):
        return [self._evaluate_one(g, task) for g in population]

    def fitness_sharing(self, population, raw_fitnesses):
        genes = np.array([g[:self.oja_gene_length] for g in population], dtype=np.float64)
        sq      = np.sum(genes ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (genes @ genes.T)
        dists   = np.sqrt(np.maximum(dist_sq, 0.0))

        sigma_share = dists.max() / 10.0
        if sigma_share < 1e-8:
            return list(raw_fitnesses)

        sh           = np.maximum(0.0, 1.0 - (dists / sigma_share) ** 2)
        niche_counts = sh.sum(axis=1).clip(min=1.0)
        return [float(f) / nc for f, nc in zip(raw_fitnesses, niche_counts)]

    def tournament_select(self, population, fitnesses):
        indices  = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy(), int(best_idx)

    def crossover(self, parent1, parent2):
        N = self.n_neurons
        W_rec1, W_in1, W_out1, _, _ = self._decode(parent1)
        W_rec2, W_in2, W_out2, _, _ = self._decode(parent2)
        oja1    = parent1[self.weight_length:self.oja_gene_length]
        oja2    = parent2[self.weight_length:self.oja_gene_length]
        sigmas1 = parent1[self.oja_gene_length:]
        sigmas2 = parent2[self.oja_gene_length:]

        alpha    = self.rng.uniform(0.3, 0.7, size=N).astype(np.float32)
        W_rec_c  = alpha[:, None] * W_rec1 + (1.0 - alpha[:, None]) * W_rec2
        W_in_c   = alpha[:, None] * W_in1  + (1.0 - alpha[:, None]) * W_in2
        W_out_c  = alpha[None, :] * W_out1 + (1.0 - alpha[None, :]) * W_out2
        sigmas_c = alpha * sigmas1 + (1.0 - alpha) * sigmas2

        alpha_oja = float(self.rng.uniform(0.3, 0.7))
        oja_c     = alpha_oja * oja1 + (1.0 - alpha_oja) * oja2

        weights_c = self._encode(W_rec_c, W_in_c, W_out_c, oja_c[0], oja_c[1])
        return np.concatenate([weights_c, sigmas_c.astype(np.float32)])

    def mutate(self, gene, sigma_scale=1.0):
        N = self.n_neurons

        log_eta  = float(gene[self.weight_length])
        log_wmax = float(gene[self.weight_length + 1])
        sigmas   = gene[self.oja_gene_length:].copy()

        sigmas *= np.exp(self.tau * self.rng.standard_normal(N)).astype(np.float32)
        sigmas  = np.clip(sigmas, 0.005, 0.15)
        eff_sigmas = np.clip(sigmas * sigma_scale, 0.0, 1.0)

        nr = N * N; ni = N * self.obs_dim; no = self.action_dim * N
        w     = gene[:self.weight_length].copy()
        W_rec = w[:nr].reshape(N, N)
        W_in  = w[nr:nr + ni].reshape(N, self.obs_dim)
        W_out = w[nr + ni:nr + ni + no].reshape(self.action_dim, N)

        mask = self.rng.random((N, N)) < eff_sigmas[:, None]
        if mask.any():
            W_rec[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        mask = self.rng.random((N, self.obs_dim)) < eff_sigmas[:, None]
        if mask.any():
            W_in[mask] += (self.mutation_std *
                           self.rng.standard_normal(mask.sum())).astype(np.float32)

        mask = self.rng.random((self.action_dim, N)) < eff_sigmas[None, :]
        if mask.any():
            W_out[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        log_eta  += self.mutation_std * float(self.rng.standard_normal())
        log_wmax += self.mutation_std * float(self.rng.standard_normal())

        return np.concatenate([self._encode(W_rec, W_in, W_out, log_eta, log_wmax), sigmas])

    def evolve(self, task, n_generations=300, print_every=25, patience=100):
        population = self.init_population()

        _decoded   = [self._decode(g) for g in population]
        W_rec_init = np.mean([d[0] for d in _decoded], axis=0).astype(np.float32)
        W_in_init  = np.mean([d[1] for d in _decoded], axis=0).astype(np.float32)
        W_out_init = np.mean([d[2] for d in _decoded], axis=0).astype(np.float32)

        history = {'fitness': [], 'accuracy': [], 'best_fitness': [], 'mean_sigma': []}

        best_gene       = None
        best_fitness    = -np.inf
        best_accuracy   = 0.0
        gens_no_improve = 0

        snapshot_gens = sorted(set([0, 25, 50, 100, 150, 200, 250] + [n_generations - 1]))
        snapshots = {}

        for gen in range(n_generations):
            results       = self.evaluate(population, task)
            raw_fitnesses = [r['fitness']  for r in results]
            accuracies    = [r['accuracy'] for r in results]

            shared_fitnesses = self.fitness_sharing(population, raw_fitnesses)

            gen_mean_acc = float(np.mean(accuracies))
            mean_sigma   = float(np.mean([g[self.oja_gene_length:] for g in population]))

            idx_best = int(np.argmax(raw_fitnesses))
            if raw_fitnesses[idx_best] > best_fitness:
                best_fitness    = raw_fitnesses[idx_best]
                best_gene       = population[idx_best].copy()
                best_accuracy   = accuracies[idx_best]
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            history['fitness'].append(float(np.mean(raw_fitnesses)))
            history['accuracy'].append(gen_mean_acc)
            history['best_fitness'].append(float(best_fitness))
            history['mean_sigma'].append(mean_sigma)

            if gen in snapshot_gens:
                Wr, Wi, Wo, eta_s, wmax_s = self._decode(best_gene)
                snap_accs = []
                for _ in range(50):
                    inp, tgt, _ = task.get_trial(rng=self.rng)
                    acc, _ = self._run_oja_trial(Wr, Wi, Wo, eta_s, wmax_s, inp, tgt)
                    snap_accs.append(acc)
                snapshots[gen] = {
                    'fitness':  float(np.mean(snap_accs)),
                    'accuracy': float(np.mean(snap_accs)),
                }

            if gen % print_every == 0 or gen == n_generations - 1:
                _, _, _, eta_p, wmax_p = self._decode(best_gene)
                print(f"Gen {gen:4d} | mean={gen_mean_acc:.1%} "
                      f"best={best_accuracy:.1%} σ̄={mean_sigma:.4f} "
                      f"η={eta_p:.5f} w_max={wmax_p:.2f}")

            if gens_no_improve >= patience:
                print(f"  Early stop at gen {gen} (no improvement for {patience} gens)")
                break

            sorted_by_raw = np.argsort(raw_fitnesses)[::-1]

            n = self.pop_size
            rank_scale = np.empty(n, dtype=np.float32)
            for rank, idx in enumerate(sorted_by_raw):
                if rank < n // 4:
                    rank_scale[idx] = 0.5
                elif rank < 3 * n // 4:
                    rank_scale[idx] = 1.0
                else:
                    rank_scale[idx] = 2.0

            new_population = [population[sorted_by_raw[i]].copy()
                              for i in range(self.n_elite)]

            while len(new_population) < self.pop_size:
                parent1, p1_idx = self.tournament_select(population, shared_fitnesses)
                parent2, _      = self.tournament_select(population, shared_fitnesses)
                child = (self.crossover(parent1, parent2)
                         if self.rng.random() < self.crossover_rate
                         else parent1.copy())
                new_population.append(self.mutate(child, sigma_scale=rank_scale[p1_idx]))

            population = new_population[:self.pop_size]

        W_rec_geno, W_in_f, W_out_f, eta_f, wmax_f = self._decode(best_gene)

        final_accs = []
        W_rec_post = None
        for _ in range(50):
            inp, tgt, _ = task.get_trial(rng=self.rng)
            acc, W_rec_post = self._run_oja_trial(
                W_rec_geno, W_in_f, W_out_f, eta_f, wmax_f, inp, tgt)
            final_accs.append(acc)
        mean_final    = float(np.mean(final_accs))
        best_accuracy = float(np.exp(mean_final)) if mean_final < 0 else mean_final

        return {
            'W_rec_init':     W_rec_init, 'W_in_init':     W_in_init, 'W_out_init':     W_out_init,
            'W_rec_final':    W_rec_geno, 'W_in_final':    W_in_f,    'W_out_final':    W_out_f,
            'W_rec_post_oja': W_rec_post, 'W_in_post_oja': W_in_f,    'W_out_post_oja': W_out_f,
            'W_rec_genotype': W_rec_geno,
            'best_fitness':   best_fitness,
            'best_accuracy':  best_accuracy,
            'eta':            eta_f,
            'w_max':          wmax_f,
            'history':        history,
            'snapshots':      snapshots,
            'snapshot_gens':  list(snapshots.keys()),
            'best_gene':      best_gene,
        }


def train_ga_oja(conf) -> dict:
    task_name = getattr(conf, 'task', 'nback')
    if task_name == 'nback':
        task = LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)
    elif task_name == 'evidence':
        from envs.evidence_accumulation import EvidenceAccumulationTask
        task = EvidenceAccumulationTask(
            evidence_strength=conf.evidence_strength,
            noise_std=conf.noise_std,
            trial_length=conf.trial_length,
            response_length=conf.response_length,
        )
    elif task_name == 'robot':
        from envs.robot_arm import RobotArmTask
        task = RobotArmTask(seq_length=conf.seq_length)
    else:
        from envs.working_memory import WorkingMemoryTask
        task = WorkingMemoryTask(
            cue_duration=conf.cue_duration,
            delay_duration=conf.delay_duration,
            response_duration=conf.response_duration,
        )

    N               = conf.n_neurons
    n_params        = N * N + conf.obs_dim * N + conf.action_dim * N
    baseline_params = getattr(conf, 'ea_baseline_params', 1344)
    mut_std_raw     = getattr(conf, 'ga_mutation_std', 0.3)

    if getattr(conf, 'ea_sigma_scaling', False):
        mutation_std = mut_std_raw / np.sqrt(n_params / baseline_params)
    else:
        mutation_std = mut_std_raw * np.sqrt(32 / N)

    pop_size = conf.ea_pop_size
    if getattr(conf, 'ea_auto_pop', False):
        pop_size = max(conf.ea_pop_size,
                       int(conf.ea_pop_size * np.sqrt(n_params / baseline_params)))

    l2_coef = getattr(conf, 'ea_l2_coef', 0.0)

    ga = GeneticAlgorithmOja(
        n_neurons=N,
        obs_dim=conf.obs_dim,
        action_dim=conf.action_dim,
        pop_size=pop_size,
        n_elite=max(2, pop_size // 32),
        tournament_k=5,
        crossover_rate=0.7,
        mutation_rate=getattr(conf, 'ga_mutation_rate', 0.05),
        mutation_std=mutation_std,
        n_eval_trials=conf.ea_n_eval_trials,
        seed=conf.seed,
        l2_coef=l2_coef,
    )

    print(f"GA-Oja: {N} neurons, {ga.oja_gene_length} params + {N} sigma genes")
    print(f"pop={pop_size}, gens={conf.ea_generations}, mut_std={ga.mutation_std:.4f}, "
          f"l2_coef={l2_coef}")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every, patience=conf.ea_patience)
    print(f"\nGA-Oja final: best_acc={result['best_accuracy']:.1%}  "
          f"η={result['eta']:.5f}  w_max={result['w_max']:.2f}")
    return result


if __name__ == "__main__":
    task = LetterNBackTask(n_back=1, seq_length=20)
    ga = GeneticAlgorithmOja(n_neurons=32, obs_dim=5, action_dim=5,
                             pop_size=64, n_elite=2, tournament_k=5,
                             crossover_rate=0.7, mutation_rate=0.05,
                             mutation_std=0.3, n_eval_trials=10, seed=42)
    result = ga.evolve(task, n_generations=50, print_every=10)
    print(f"\nbest_acc={result['best_accuracy']:.1%}  "
          f"η={result['eta']:.5f}  w_max={result['w_max']:.2f}")
