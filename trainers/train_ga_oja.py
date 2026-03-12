# trainers/train_ga_oja.py
"""
Genetic Algorithm + Oja's Rule (Hebbian plasticity on W_rec)

Same GA mechanics as train_ga.py (neuron-level blend crossover, self-adaptive
mutation, fitness sharing, elitism, rank-based mutation scaling).

Key difference: each fitness evaluation runs the network step-by-step and
applies Oja's rule to a COPY of W_rec after every timestep.  The modified
W_rec is discarded after the trial — no Lamarckian inheritance.  The GA
evolves the initial weights AND the plasticity parameters (η, w_max).

Genotype layout:
  [W_rec flat | W_in flat | W_out flat | log_eta | log_wmax | sigma_0..N-1]
   ←──────────────── oja_gene_length = weight_length + 2 ─────────────────→
                                                           ←── N sigma genes ──→

  weight_length = N*N + N*obs_dim + action_dim*N
  oja_gene_length = weight_length + 2            (e.g. 1346 for N=32)

  eta   = exp(log_eta)   clipped to [1e-6, 1.0]
  w_max = exp(log_wmax)  clipped to [0.1,  20.0]

Oja's rule (applied each timestep, W_rec copy only):
  h = tanh activations (N,)
  dW = eta * (outer(h, h) - (h**2)[:, None] * W_rec)
  W_rec = clip(W_rec + dW, -w_max, w_max)

Saved results:
  W_rec_init     — population centroid at gen 0 (random initialisation baseline)
  W_rec_final    — best individual's evolved weights from the genotype (pre-Oja)
                    → W_rec_final - W_rec_init = evolutionary ΔW
  W_rec_post_oja — W_rec after Oja plasticity ran for one representative trial
                    → W_rec_post_oja - W_rec_final = within-trial Oja ΔW

  The separation of evolutionary ΔW vs Oja ΔW is the thesis point:
    evolution sets up the initial connectivity, Oja adapts it during the task.
"""

import numpy as np
from envs.letter_nback import LetterNBackTask


class GeneticAlgorithmOja:
    """
    (μ+λ) GA with Oja Hebbian plasticity.

    Args:
        n_neurons:      network size
        obs_dim:        input dimension
        action_dim:     output dimension
        pop_size:       population size (μ)
        n_elite:        elites preserved each generation
        tournament_k:   tournament size
        crossover_rate: probability of crossover vs cloning
        mutation_rate:  initial per-neuron sigma (evolves)
        mutation_std:   std of Gaussian weight perturbation
        n_eval_trials:  trials per fitness evaluation
        seed:           random seed
    """

    def __init__(self, n_neurons=32, obs_dim=5, action_dim=5,
                 pop_size=128, n_elite=4, tournament_k=5,
                 crossover_rate=0.7, mutation_rate=0.05,
                 mutation_std=0.3, n_eval_trials=20, seed=42):

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
        self.rng = np.random.default_rng(seed)

        N = n_neurons
        self.weight_length   = N * N + N * obs_dim + action_dim * N
        self.oja_gene_length = self.weight_length + 2   # + log_eta, log_wmax
        self.tau = 1.0 / np.sqrt(2.0 * N)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _encode(self, W_rec, W_in, W_out, log_eta, log_wmax):
        return np.concatenate([
            W_rec.ravel(), W_in.ravel(), W_out.ravel(),
            np.array([log_eta, log_wmax], dtype=np.float32),
        ])

    def _decode(self, gene):
        N   = self.n_neurons
        nr  = N * N
        ni  = N * self.obs_dim
        no  = self.action_dim * N
        w   = gene[:self.oja_gene_length]

        W_rec = w[:nr].reshape(N, N).astype(np.float32)
        W_in  = w[nr:nr + ni].reshape(N, self.obs_dim).astype(np.float32)
        W_out = w[nr + ni:nr + ni + no].reshape(self.action_dim, N).astype(np.float32)

        log_eta  = float(w[-2])
        log_wmax = float(w[-1])
        eta   = float(np.clip(np.exp(log_eta),  1e-6, 1.0))
        w_max = float(np.clip(np.exp(log_wmax), 0.1,  20.0))

        return W_rec, W_in, W_out, eta, w_max

    # ------------------------------------------------------------------
    # Oja trial
    # ------------------------------------------------------------------

    def _run_oja_trial(self, W_rec_init, W_in, W_out, eta, w_max, inputs, targets):
        """
        Run one trial step-by-step with Oja's rule on a W_rec copy.

        Args:
            W_rec_init: (N, N) initial recurrent weights (not modified)
            inputs:     (T, obs_dim) one-hot observations
            targets:    (T,) class indices, -1 = no response

        Returns:
            accuracy:   float, fraction correct on response steps
            W_rec_post: (N, N) W_rec after Oja plasticity
        """
        T     = inputs.shape[0]
        W_rec = W_rec_init.copy()
        h     = np.zeros(self.n_neurons, dtype=np.float32)
        outputs = np.zeros((T, self.action_dim), dtype=np.float32)

        for t in range(T):
            h = np.tanh(W_rec @ h + W_in @ inputs[t])
            outputs[t] = np.tanh(W_out @ h)

            # Oja's rule: dW_ij = eta * (h_i * h_j - h_i^2 * W_ij)
            dW    = eta * (np.outer(h, h) - (h ** 2)[:, None] * W_rec)
            W_rec = np.clip(W_rec + dW, -w_max, w_max)

        if targets.ndim == 2:
            # Regression task (e.g. robot arm): fitness = -MSE
            acc = -float(np.mean((outputs - targets) ** 2))
        else:
            mask = targets >= 0
            acc  = float((np.argmax(outputs[mask], axis=-1) == targets[mask]).mean()) \
                   if mask.any() else 0.0
        return acc, W_rec

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def init_population(self):
        N     = self.n_neurons
        scale = np.sqrt(2.0 / N)
        population = []
        for _ in range(self.pop_size):
            weights  = (scale * self.rng.standard_normal(self.weight_length)
                        ).astype(np.float32)
            log_eta  = np.log(0.01).astype(np.float32)   # initial eta  ≈ 0.01
            log_wmax = np.log(2.0).astype(np.float32)    # initial w_max ≈ 2.0
            sigmas   = np.full(N, self.mutation_rate, dtype=np.float32)
            population.append(
                np.concatenate([weights, [log_eta, log_wmax], sigmas])
            )
        return population

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_one(self, gene, task):
        W_rec, W_in, W_out, eta, w_max = self._decode(gene)
        accs = []
        for _ in range(self.n_eval_trials):
            inputs, targets, _ = task.get_trial(rng=self.rng)
            acc, _ = self._run_oja_trial(W_rec, W_in, W_out, eta, w_max,
                                         inputs, targets)
            accs.append(acc)
        mean_acc = float(np.mean(accs))
        # For regression tasks, accs = -MSE (negative). Convert to exp(-MSE) for display.
        display_acc = float(np.exp(mean_acc)) if mean_acc < 0 else mean_acc
        return {'fitness': mean_acc, 'accuracy': display_acc}

    def evaluate(self, population, task):
        return [self._evaluate_one(g, task) for g in population]

    # ------------------------------------------------------------------
    # Fitness sharing
    # ------------------------------------------------------------------

    def fitness_sharing(self, population, raw_fitnesses):
        genes = np.array([g[:self.oja_gene_length] for g in population],
                         dtype=np.float64)
        sq      = np.sum(genes ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (genes @ genes.T)
        dists   = np.sqrt(np.maximum(dist_sq, 0.0))

        sigma_share = dists.max() / 10.0
        if sigma_share < 1e-8:
            return list(raw_fitnesses)

        sh           = np.maximum(0.0, 1.0 - (dists / sigma_share) ** 2)
        niche_counts = sh.sum(axis=1).clip(min=1.0)
        return [float(f) / nc for f, nc in zip(raw_fitnesses, niche_counts)]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(self, population, fitnesses):
        """Returns (gene, index)."""
        indices  = self.rng.choice(len(population), size=self.tournament_k,
                                   replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy(), int(best_idx)

    # ------------------------------------------------------------------
    # Crossover: neuron-level blend + global blend for Oja params
    # ------------------------------------------------------------------

    def crossover(self, parent1, parent2):
        N = self.n_neurons
        W_rec1, W_in1, W_out1, _, _ = self._decode(parent1)
        W_rec2, W_in2, W_out2, _, _ = self._decode(parent2)

        oja1    = parent1[self.weight_length:self.oja_gene_length]   # [log_eta, log_wmax]
        oja2    = parent2[self.weight_length:self.oja_gene_length]
        sigmas1 = parent1[self.oja_gene_length:]
        sigmas2 = parent2[self.oja_gene_length:]

        # Per-neuron blend for weight matrices
        alpha   = self.rng.uniform(0.3, 0.7, size=N).astype(np.float32)
        W_rec_c = alpha[:, None] * W_rec1 + (1.0 - alpha[:, None]) * W_rec2
        W_in_c  = alpha[:, None] * W_in1  + (1.0 - alpha[:, None]) * W_in2
        W_out_c = alpha[None, :] * W_out1 + (1.0 - alpha[None, :]) * W_out2
        sigmas_c = alpha * sigmas1 + (1.0 - alpha) * sigmas2

        # Global blend for Oja params (not neuron-specific)
        alpha_oja = float(self.rng.uniform(0.3, 0.7))
        oja_c     = alpha_oja * oja1 + (1.0 - alpha_oja) * oja2

        weights_c = self._encode(W_rec_c, W_in_c, W_out_c, oja_c[0], oja_c[1])
        return np.concatenate([weights_c, sigmas_c.astype(np.float32)])

    # ------------------------------------------------------------------
    # Mutation: self-adaptive sigmas (weights) + log-space Gaussian (Oja)
    # ------------------------------------------------------------------

    def mutate(self, gene, sigma_scale=1.0):
        """
        Self-adaptive mutation for weight matrices (same as train_ga.py).
        Oja params mutated in log-space with fixed std (mutation_std).
        sigma_scale: rank-based multiplier (0.5/1.0/2.0) — not stored.
        """
        N = self.n_neurons

        log_eta  = float(gene[self.weight_length])
        log_wmax = float(gene[self.weight_length + 1])
        sigmas   = gene[self.oja_gene_length:].copy()

        # 1. Lognormal update of sigma genes
        sigmas *= np.exp(self.tau * self.rng.standard_normal(N)).astype(np.float32)
        sigmas  = np.clip(sigmas, 0.005, 0.15)  # cap: prevent runaway / dead exploration
        eff_sigmas = np.clip(sigmas * sigma_scale, 0.0, 1.0)

        # 2. Decode and mutate weight matrices
        nr = N * N;  ni = N * self.obs_dim;  no = self.action_dim * N
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

        # 3. Mutate Oja params in log-space (Gaussian, fixed std)
        log_eta  += self.mutation_std * float(self.rng.standard_normal())
        log_wmax += self.mutation_std * float(self.rng.standard_normal())

        weights_new = self._encode(W_rec, W_in, W_out, log_eta, log_wmax)
        return np.concatenate([weights_new, sigmas])

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self, task, n_generations=300, print_every=25, patience=100):
        """
        Run the full GA with Oja plasticity.

        Returns:
            dict with keys:
              W_rec_init, W_in_init, W_out_init       ← population centroid at gen 0
              W_rec_final, W_in_final, W_out_final    ← best genotype's evolved weights (pre-Oja)
              W_rec_post_oja, W_in_post_oja,
                W_out_post_oja                        ← post-Oja weights (W_in/out unchanged)
              W_rec_genotype                          ← alias for W_rec_final
              best_fitness, best_accuracy
              eta, w_max                              ← evolved plasticity parameters
              history, snapshots, best_gene
        """
        population = self.init_population()

        # Population centroid as init reference (mean across all individuals).
        # ΔW = best_final - centroid_init, comparable to BPTT's init→final delta.
        _decoded   = [self._decode(g) for g in population]
        W_rec_init = np.mean([d[0] for d in _decoded], axis=0).astype(np.float32)
        W_in_init  = np.mean([d[1] for d in _decoded], axis=0).astype(np.float32)
        W_out_init = np.mean([d[2] for d in _decoded], axis=0).astype(np.float32)

        history = {
            'fitness':      [],
            'accuracy':     [],
            'best_fitness': [],
            'mean_sigma':   [],
        }

        best_gene       = None
        best_fitness    = -np.inf
        best_accuracy   = 0.0
        gens_no_improve = 0

        snapshot_gens = sorted(set(
            [0, 25, 50, 100, 150, 200, 250] + [n_generations - 1]
        ))
        snapshots = {}

        for gen in range(n_generations):
            # 1. Evaluate
            results       = self.evaluate(population, task)
            raw_fitnesses = [r['fitness']  for r in results]
            accuracies    = [r['accuracy'] for r in results]

            # 2. Fitness sharing
            shared_fitnesses = self.fitness_sharing(population, raw_fitnesses)

            # 3. Track stats
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

            # 4. Snapshot (50-trial stable eval of best)
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

            # 5. Print
            if gen % print_every == 0 or gen == n_generations - 1:
                _, _, _, eta_p, wmax_p = self._decode(best_gene)
                print(f"Gen {gen:4d} | mean={gen_mean_acc:.1%} "
                      f"best={best_accuracy:.1%} σ̄={mean_sigma:.4f} "
                      f"η={eta_p:.5f} w_max={wmax_p:.2f}")

            # Early stopping
            if gens_no_improve >= patience:
                print(f"  Early stop at gen {gen} (no improvement for {patience} gens)")
                break

            # 6. Build next generation
            sorted_by_raw = np.argsort(raw_fitnesses)[::-1]

            # Rank-based mutation scale: top 25% → 0.5×, mid 50% → 1×, bot 25% → 2×
            n = self.pop_size
            rank_scale = np.empty(n, dtype=np.float32)
            for rank, idx in enumerate(sorted_by_raw):
                if rank < n // 4:
                    rank_scale[idx] = 0.5
                elif rank < 3 * n // 4:
                    rank_scale[idx] = 1.0
                else:
                    rank_scale[idx] = 2.0

            new_population = []
            for i in range(self.n_elite):
                new_population.append(population[sorted_by_raw[i]].copy())

            while len(new_population) < self.pop_size:
                parent1, p1_idx = self.tournament_select(population, shared_fitnesses)
                parent2, _      = self.tournament_select(population, shared_fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self.mutate(child, sigma_scale=rank_scale[p1_idx])
                new_population.append(child)

            population = new_population[:self.pop_size]

        # ── Post-evolution: decode best gene ──────────────────────────
        W_rec_geno, W_in_f, W_out_f, eta_f, wmax_f = self._decode(best_gene)

        # Stable final accuracy: 50 trials with Oja; keep last trial's post-Oja W_rec
        final_accs  = []
        W_rec_post  = None
        for _ in range(50):
            inp, tgt, _ = task.get_trial(rng=self.rng)
            acc, W_rec_post = self._run_oja_trial(
                W_rec_geno, W_in_f, W_out_f, eta_f, wmax_f, inp, tgt)
            final_accs.append(acc)
        mean_final = float(np.mean(final_accs))
        # For regression tasks, acc = -MSE (negative); convert to exp(-MSE) for display
        best_accuracy = float(np.exp(mean_final)) if mean_final < 0 else mean_final

        return {
            # Evolutionary ΔW: W_rec_final - W_rec_init  (what evolution changed)
            'W_rec_init':      W_rec_init,   # population centroid at gen 0
            'W_in_init':       W_in_init,
            'W_out_init':      W_out_init,
            'W_rec_final':     W_rec_geno,   # best genotype's evolved weights (pre-Oja)
            'W_in_final':      W_in_f,
            'W_out_final':     W_out_f,
            # Oja ΔW: W_rec_post_oja - W_rec_final  (what plasticity changed within a trial)
            'W_rec_post_oja':  W_rec_post,   # W_rec after Oja plasticity
            'W_in_post_oja':   W_in_f,       # unchanged by Oja
            'W_out_post_oja':  W_out_f,      # unchanged by Oja
            'W_rec_genotype':  W_rec_geno,   # alias for W_rec_final
            'best_fitness':    best_fitness,
            'best_accuracy':   best_accuracy,
            'eta':             eta_f,
            'w_max':           wmax_f,
            'history':         history,
            'snapshots':       snapshots,
            'snapshot_gens':   list(snapshots.keys()),
            'best_gene':       best_gene,
        }


# ============================================================================
# Convenience: train_ga_oja() — same interface as train_ga()
# ============================================================================

def train_ga_oja(conf) -> dict:
    """Train using the GA with Oja plasticity. Drop-in for train_ga()."""
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
    # Same sqrt(32/N) scaling as train_ga — keeps per-mutation amplitude
    # proportional to weight init scale sqrt(2/N) across network sizes.
    mut_std_raw = getattr(conf, 'ga_mutation_std', 0.3)
    mutation_std = mut_std_raw * np.sqrt(32 / N)

    pop_size = conf.ea_pop_size
    if getattr(conf, 'ea_auto_pop', False):
        n_params_est = N * N + conf.obs_dim * N + conf.action_dim * N
        pop_size = max(pop_size, int(4 * np.sqrt(n_params_est)))

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
    )

    n_params = ga.oja_gene_length
    print(f"GA-Oja: {N} neurons, {n_params} params "
          f"({ga.weight_length} weights + 2 Oja) + {N} sigma genes")
    print(f"Task: {task_name} | pop={pop_size}, gens={conf.ea_generations}")
    print(f"elite={ga.n_elite}, tournament_k={ga.tournament_k}, "
          f"crossover=blend, mut_std={ga.mutation_std:.4f} (raw={mut_std_raw}, N-scaled), "
          f"sigma0={ga.mutation_rate}, tau={ga.tau:.4f}, sigma_cap=[0.005,0.15]"
          f", eta_clip=[1e-6,1.0], wmax_clip=[0.1,20.0]")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every,
                       patience=conf.ea_patience)

    print(f"\nGA-Oja final: best_acc={result['best_accuracy']:.1%}  "
          f"η={result['eta']:.5f}  w_max={result['w_max']:.2f}")
    return result


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    task = LetterNBackTask(n_back=1, seq_length=20)

    ga = GeneticAlgorithmOja(
        n_neurons=32,
        obs_dim=5,
        action_dim=5,
        pop_size=64,
        n_elite=2,
        tournament_k=5,
        crossover_rate=0.7,
        mutation_rate=0.05,
        mutation_std=0.3,
        n_eval_trials=10,
        seed=42,
    )

    print(f"GA-Oja standalone: 1-back, 32 neurons, 64 pop, 50 gens")
    print(f"Gene length: {ga.oja_gene_length} + {ga.n_neurons} sigma = "
          f"{ga.oja_gene_length + ga.n_neurons} total")
    result = ga.evolve(task, n_generations=50, print_every=10)
    print(f"\nFinal: best_acc={result['best_accuracy']:.1%}  "
          f"η={result['eta']:.5f}  w_max={result['w_max']:.2f}")
    print(f"       |ΔW_rec evol| mean = "
          f"{float(np.abs(result['W_rec_final'] - result['W_rec_init']).mean()):.4f}")
    print(f"       |ΔW_rec oja|  mean = "
          f"{float(np.abs(result['W_rec_post_oja'] - result['W_rec_final']).mean()):.4f}")
