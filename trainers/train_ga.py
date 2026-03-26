"""
Genetic algorithm with tournament selection, neuron-level blend crossover,
self-adaptive mutation (Schwefel's 1/sqrt(2N) rule), fitness sharing, and elitism.

Genotype per individual:
  [W_rec.ravel() | W_in.ravel() | W_out.ravel() | sigma_0..sigma_{N-1}]
"""

import numpy as np
from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import LetterNBackTask


class GeneticAlgorithm:

    def __init__(self, n_neurons=32, obs_dim=1, action_dim=1,
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
        self.gene_length = N * N + N * obs_dim + action_dim * N
        self.tau = 1.0 / np.sqrt(2.0 * N)  # Schwefel's lognormal learning rate

    def _encode(self, W_rec, W_in, W_out):
        return np.concatenate([W_rec.ravel(), W_in.ravel(), W_out.ravel()])

    def _decode(self, gene):
        N  = self.n_neurons
        nr = N * N
        ni = N * self.obs_dim
        w  = gene[:self.gene_length]
        W_rec = w[:nr].reshape(N, N)
        W_in  = w[nr:nr + ni].reshape(N, self.obs_dim)
        W_out = w[nr + ni:].reshape(self.action_dim, N)
        return W_rec.astype(np.float32), W_in.astype(np.float32), W_out.astype(np.float32)

    def _make_policy(self, gene):
        return RSNNPolicy(*self._decode(gene))

    def init_population(self):
        N = self.n_neurons
        scale = np.sqrt(2.0 / N)
        population = []
        for _ in range(self.pop_size):
            weights = scale * self.rng.standard_normal(self.gene_length).astype(np.float32)
            sigmas  = np.full(N, self.mutation_rate, dtype=np.float32)
            population.append(np.concatenate([weights, sigmas]))
        return population

    def evaluate(self, population, task):
        results = []
        for gene in population:
            policy = self._make_policy(gene)
            r = task.evaluate_policy(policy, n_trials=self.n_eval_trials, rng=self.rng)
            if self.l2_coef > 0.0:
                w = gene[:self.gene_length]
                r = dict(r, fitness=r['fitness'] - self.l2_coef * float(np.mean(w ** 2)))
            results.append(r)
        return results

    def fitness_sharing(self, population, raw_fitnesses):
        """Divide fitness by niche count to discourage population collapse."""
        genes = np.array([g[:self.gene_length] for g in population], dtype=np.float64)
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
        """Neuron-level blend crossover: alpha_i ~ Uniform(0.3, 0.7) per neuron."""
        N = self.n_neurons
        W_rec1, W_in1, W_out1 = self._decode(parent1)
        W_rec2, W_in2, W_out2 = self._decode(parent2)
        sigmas1 = parent1[self.gene_length:]
        sigmas2 = parent2[self.gene_length:]

        alpha = self.rng.uniform(0.3, 0.7, size=N).astype(np.float32)
        W_rec_c  = alpha[:, None] * W_rec1 + (1.0 - alpha[:, None]) * W_rec2
        W_in_c   = alpha[:, None] * W_in1  + (1.0 - alpha[:, None]) * W_in2
        W_out_c  = alpha[None, :] * W_out1 + (1.0 - alpha[None, :]) * W_out2
        sigmas_c = alpha * sigmas1 + (1.0 - alpha) * sigmas2

        return np.concatenate([self._encode(W_rec_c, W_in_c, W_out_c),
                                sigmas_c.astype(np.float32)])

    def mutate(self, gene, sigma_scale=1.0):
        """Self-adaptive mutation; sigma_scale is rank-based (0.5/1.0/2.0), not stored."""
        N       = self.n_neurons
        weights = gene[:self.gene_length].copy()
        sigmas  = gene[self.gene_length:].copy()

        sigmas *= np.exp(self.tau * self.rng.standard_normal(N)).astype(np.float32)
        sigmas  = np.clip(sigmas, 0.005, 0.15)
        eff_sigmas = np.clip(sigmas * sigma_scale, 0.0, 1.0)

        W_rec, W_in, W_out = self._decode(weights)

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

        return np.concatenate([self._encode(W_rec, W_in, W_out), sigmas])

    def evolve(self, task, n_generations=300, print_every=25, patience=100):
        population = self.init_population()

        _decoded   = [self._decode(g) for g in population]
        W_rec_init = np.mean([d[0] for d in _decoded], axis=0).astype(np.float32)
        W_in_init  = np.mean([d[1] for d in _decoded], axis=0).astype(np.float32)
        W_out_init = np.mean([d[2] for d in _decoded], axis=0).astype(np.float32)

        history = {
            'fitness': [], 'accuracy': [], 'best_fitness': [],
            'mean_sigma': [], 'mutation_std': [],
        }

        best_gene       = None
        best_fitness    = -np.inf
        best_accuracy   = 0.0
        gens_no_improve = 0

        # 1/5-success-rule on global mutation_std (Rechenberg 1973)
        SUCCESS_WIN  = 20
        MUT_UP       = 1.05
        MUT_DN       = 0.97
        MUT_MIN      = self.mutation_std * 0.25
        MUT_MAX      = self.mutation_std * 4.0
        success_hist = []

        snapshot_gens = sorted(set([0, 25, 50, 100, 150, 200, 250] + [n_generations - 1]))
        snapshots = {}

        for gen in range(n_generations):
            results       = self.evaluate(population, task)
            raw_fitnesses = [r['fitness']  for r in results]
            accuracies    = [r['accuracy'] for r in results]

            shared_fitnesses = self.fitness_sharing(population, raw_fitnesses)

            gen_mean_acc = float(np.mean(accuracies))
            mean_sigma   = float(np.mean([g[self.gene_length:] for g in population]))

            idx_best = int(np.argmax(raw_fitnesses))
            improved = raw_fitnesses[idx_best] > best_fitness
            if improved:
                best_fitness    = raw_fitnesses[idx_best]
                best_gene       = population[idx_best].copy()
                best_accuracy   = accuracies[idx_best]
                gens_no_improve = 0
            else:
                gens_no_improve += 1

            success_hist.append(1 if improved else 0)
            if len(success_hist) > SUCCESS_WIN:
                success_hist.pop(0)
            if len(success_hist) == SUCCESS_WIN:
                rate = sum(success_hist) / SUCCESS_WIN
                if rate > 0.2:
                    self.mutation_std = min(self.mutation_std * MUT_UP, MUT_MAX)
                elif rate < 0.2:
                    self.mutation_std = max(self.mutation_std * MUT_DN, MUT_MIN)

            history['fitness'].append(float(np.mean(raw_fitnesses)))
            history['accuracy'].append(gen_mean_acc)
            history['best_fitness'].append(float(best_fitness))
            history['mean_sigma'].append(mean_sigma)
            history['mutation_std'].append(float(self.mutation_std))

            if gen in snapshot_gens:
                sr = task.evaluate_policy(self._make_policy(best_gene),
                                          n_trials=50, rng=self.rng)
                snapshots[gen] = {'fitness': sr['fitness'], 'accuracy': sr['accuracy']}

            if gen % print_every == 0 or gen == n_generations - 1:
                print(f"Gen {gen:4d} | mean={gen_mean_acc:.1%} "
                      f"best={best_accuracy:.1%} σ̄={mean_sigma:.4f} "
                      f"mut_std={self.mutation_std:.4f}")

            if gens_no_improve >= patience:
                print(f"  Early stop at gen {gen} (no improvement for {patience} gens)")
                break

            sorted_by_shared = np.argsort(shared_fitnesses)[::-1]
            sorted_by_raw    = np.argsort(raw_fitnesses)[::-1]

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

        W_rec_f, W_in_f, W_out_f = self._decode(best_gene)
        final_eval    = task.evaluate_policy(self._make_policy(best_gene),
                                             n_trials=50, rng=self.rng)
        best_accuracy = final_eval['accuracy']

        return {
            'W_rec_init':  W_rec_init, 'W_in_init':  W_in_init, 'W_out_init':  W_out_init,
            'W_rec_final': W_rec_f,    'W_in_final': W_in_f,    'W_out_final': W_out_f,
            'best_fitness':  best_fitness,
            'best_accuracy': best_accuracy,
            'history':       history,
            'snapshots':     snapshots,
            'snapshot_gens': list(snapshots.keys()),
            'best_gene':     best_gene,
        }


def train_ga(conf) -> dict:
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

    N = conf.n_neurons
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

    ga = GeneticAlgorithm(
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

    print(f"GA: {N} neurons, {ga.gene_length} weight params + {N} sigma genes")
    print(f"pop={pop_size}, gens={conf.ea_generations}, elite={ga.n_elite}, "
          f"mut_std={ga.mutation_std:.4f}, l2_coef={l2_coef}")

    return ga.evolve(task, n_generations=conf.ea_generations,
                     print_every=conf.print_every, patience=conf.ea_patience)


if __name__ == "__main__":
    task = LetterNBackTask(n_back=1, seq_length=20)
    ga = GeneticAlgorithm(n_neurons=32, obs_dim=5, action_dim=5,
                          pop_size=64, n_elite=2, tournament_k=5,
                          crossover_rate=0.7, mutation_rate=0.05,
                          mutation_std=0.3, n_eval_trials=10, seed=42)
    result = ga.evolve(task, n_generations=50, print_every=10)
    print(f"\nbest_fitness={result['best_fitness']:+.4f}  "
          f"accuracy={result['history']['accuracy'][-1]:.1%}")
