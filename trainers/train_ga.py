# trainers/train_ga.py
"""
Genetic Algorithm for Evolving RNN Weights

True evolutionary algorithm — NO gradient estimation.
Selection + crossover + mutation only.

Inspired by:
  - Zhu Spike Lab esnn-ver2 (robot_engine): tournament select → crossover → mutate
  - Such et al. 2017: "Deep Neuroevolution" — GAs competitive with RL at scale
  - Schuman et al. 2018: diversity-aware selection for spiking RNNs
  - SpiFoG (Sci Reports 2020): elitist GA with hybrid crossover for SNNs
  - Stanley & Miikkulainen 2002 (NEAT): elitism preserves best solutions

Key design choices:
  - Tournament selection (pressure without ranking the whole pop)
  - Neuron-level crossover: for each neuron i, take W_rec[i,:], W_in[i,:],
    W_out[:,i] all from the same parent — preserves each neuron's identity
  - Self-adaptive mutation: each individual carries N per-neuron mutation-rate
    genes that evolve via lognormal perturbation (tau = 1/sqrt(2N))
  - Fitness sharing: shared_fitness = raw / niche_count to prevent collapse
  - Elitism (top-k survive unchanged — prevents losing good solutions)
  - No gradient signal whatsoever — pure selection pressure

This is the clean contrast to BPTT:
  BPTT = gradient through time (knows which input→output mapping matters)
  GA   = blind search with selection (only knows "this network scored X")
"""

import numpy as np
from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import LetterNBackTask


class GeneticAlgorithm:
    """
    (μ+λ) genetic algorithm for RNN weight evolution.

    Population of μ parents. Each generation:
      1. Evaluate all individuals → raw fitnesses
      2. Apply fitness sharing → shared fitnesses (for selection)
      3. Select parents via tournament selection (uses shared fitness)
      4. Generate λ offspring via neuron-level crossover + self-adaptive mutation
      5. Elitism: top-k (by raw fitness) survive unchanged
      6. New population = elites + offspring (trimmed to pop_size)

    Genotype layout (per individual):
      [weight genes | sigma genes]
      weight genes: W_rec.ravel() | W_in.ravel() | W_out.ravel()  (gene_length)
      sigma genes:  per-neuron mutation rates σ_0..σ_{N-1}         (n_neurons)

    Args:
        n_neurons:      network size
        obs_dim:        input dimension
        action_dim:     output dimension
        pop_size:       population size (μ)
        n_elite:        number of elites preserved each generation
        tournament_k:   tournament size for selection
        crossover_rate: probability of crossover vs cloning
        mutation_rate:  initial per-neuron mutation rate (evolves)
        mutation_std:   std of Gaussian mutation noise
        n_eval_trials:  trials per fitness evaluation
        seed:           random seed
    """

    def __init__(self, n_neurons=32, obs_dim=1, action_dim=1,
                 pop_size=128, n_elite=4, tournament_k=5,
                 crossover_rate=0.7, mutation_rate=0.05,
                 mutation_std=0.3, n_eval_trials=20, seed=42):

        self.n_neurons = n_neurons
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.pop_size = pop_size
        self.n_elite = n_elite
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate  # initial value for sigma genes
        self.mutation_std = mutation_std
        self.n_eval_trials = n_eval_trials
        self.rng = np.random.default_rng(seed)

        N = n_neurons
        self.gene_length = N * N + N * obs_dim + action_dim * N
        # Self-adaptation learning rate (Schwefel's 1/sqrt(2N) rule)
        self.tau = 1.0 / np.sqrt(2.0 * N)

    # ------------------------------------------------------------------
    # Encode / Decode  (operates on weight genes only)
    # ------------------------------------------------------------------

    def _encode(self, W_rec, W_in, W_out):
        """Flatten weight matrices into a weight gene vector (gene_length,)."""
        return np.concatenate([W_rec.ravel(), W_in.ravel(), W_out.ravel()])

    def _decode(self, gene):
        """Weight gene vector (or full gene) → weight matrices.

        Slices only the first gene_length elements so it works with both
        legacy weight-only genes and new extended genes.
        """
        N = self.n_neurons
        nr = N * N
        ni = N * self.obs_dim
        w = gene[:self.gene_length]
        W_rec = w[:nr].reshape(N, N)
        W_in  = w[nr:nr + ni].reshape(N, self.obs_dim)
        W_out = w[nr + ni:].reshape(self.action_dim, N)
        return W_rec.astype(np.float32), W_in.astype(np.float32), W_out.astype(np.float32)

    def _make_policy(self, gene):
        """Gene → RSNNPolicy (uses only weight genes)."""
        W_rec, W_in, W_out = self._decode(gene)
        return RSNNPolicy(W_rec, W_in, W_out)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def init_population(self):
        """Create initial population with Xavier-like random weights + initial sigmas."""
        N = self.n_neurons
        scale = np.sqrt(2.0 / N)
        population = []
        for _ in range(self.pop_size):
            weights = scale * self.rng.standard_normal(self.gene_length).astype(np.float32)
            sigmas  = np.full(N, self.mutation_rate, dtype=np.float32)
            population.append(np.concatenate([weights, sigmas]))
        return population

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, population, task):
        """Evaluate all individuals. Returns list of fitness dicts."""
        results = []
        for gene in population:
            policy = self._make_policy(gene)
            r = task.evaluate_policy(policy, n_trials=self.n_eval_trials, rng=self.rng)
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Fitness sharing for diversity
    # ------------------------------------------------------------------

    def fitness_sharing(self, population, raw_fitnesses):
        """
        Divide each individual's fitness by its niche count to discourage
        the population from collapsing to a single solution.

        sh(d) = 1 - (d/sigma_share)^2   if d < sigma_share
              = 0                         otherwise

        sigma_share = max pairwise distance / 10

        Returns shared_fitnesses (same length as raw_fitnesses).
        """
        # Extract only weight genes for distance computation
        genes = np.array([g[:self.gene_length] for g in population], dtype=np.float64)

        # Pairwise Euclidean distances via norm trick: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        sq = np.sum(genes ** 2, axis=1)
        dist_sq = sq[:, None] + sq[None, :] - 2.0 * (genes @ genes.T)
        dists = np.sqrt(np.maximum(dist_sq, 0.0))

        sigma_share = dists.max() / 10.0
        if sigma_share < 1e-8:
            return list(raw_fitnesses)  # population degenerate — skip sharing

        sh = np.maximum(0.0, 1.0 - (dists / sigma_share) ** 2)
        niche_counts = sh.sum(axis=1).clip(min=1.0)

        return [float(f) / nc for f, nc in zip(raw_fitnesses, niche_counts)]

    # ------------------------------------------------------------------
    # Selection: tournament (uses shared fitness)
    # ------------------------------------------------------------------

    def tournament_select(self, population, fitnesses):
        """Select one parent via tournament selection. Returns (gene, index)."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy(), int(best_idx)

    # ------------------------------------------------------------------
    # Crossover: neuron-level
    # ------------------------------------------------------------------

    def crossover(self, parent1, parent2):
        """
        Neuron-level blend crossover: for each neuron i, draw
        alpha_i ~ Uniform(0.3, 0.7) and interpolate all weights
        associated with that neuron:
          - W_rec[i, :] — outgoing connections
          - W_in[i, :]  — input weights
          - W_out[:, i] — output contribution
          - sigma[i]    — mutation rate

        Blending keeps neuron identity coherent while allowing partial
        contributions from both parents (unlike hard binary swaps).
        """
        N = self.n_neurons
        W_rec1, W_in1, W_out1 = self._decode(parent1)
        W_rec2, W_in2, W_out2 = self._decode(parent2)
        sigmas1 = parent1[self.gene_length:]
        sigmas2 = parent2[self.gene_length:]

        # Per-neuron blend coefficient: alpha_i ~ Uniform(0.3, 0.7)
        alpha = self.rng.uniform(0.3, 0.7, size=N).astype(np.float32)

        # W_rec (N, N): row i = outgoing edges of neuron i
        W_rec_c = alpha[:, None] * W_rec1 + (1.0 - alpha[:, None]) * W_rec2
        # W_in  (N, obs_dim): row i = input weights of neuron i
        W_in_c  = alpha[:, None] * W_in1  + (1.0 - alpha[:, None]) * W_in2
        # W_out (action_dim, N): col i = readout contribution of neuron i
        W_out_c = alpha[None, :] * W_out1 + (1.0 - alpha[None, :]) * W_out2
        # Sigma: blend with same per-neuron alpha
        sigmas_c = alpha * sigmas1 + (1.0 - alpha) * sigmas2

        weights_c = self._encode(W_rec_c, W_in_c, W_out_c)
        return np.concatenate([weights_c, sigmas_c.astype(np.float32)])

    # ------------------------------------------------------------------
    # Mutation: self-adaptive per-neuron rates
    # ------------------------------------------------------------------

    def mutate(self, gene, sigma_scale=1.0):
        """
        Self-adaptive mutation:
          1. Perturb each neuron's sigma via lognormal:
               sigma_i *= exp(tau * N(0,1)),  tau = 1/sqrt(2N)
             then clip to [1e-4, 1.0].
          2. Compute effective mutation rates:
               eff_sigma_i = clip(sigma_i * sigma_scale, 0, 1)
             where sigma_scale is a rank-based multiplier (0.5 / 1.0 / 2.0).
          3. For neuron i, mutate its weights (W_rec[i,:], W_in[i,:], W_out[:,i])
             with probability eff_sigma_i and std mutation_std.

        sigma_scale does NOT modify the stored sigma genes — it only adjusts
        the mutation probability for this offspring, so self-adaptation is
        unaffected.
        """
        N = self.n_neurons
        weights = gene[:self.gene_length].copy()
        sigmas  = gene[self.gene_length:].copy()

        # 1. Lognormal update of sigma genes (rank scale not applied here)
        sigmas *= np.exp(self.tau * self.rng.standard_normal(N)).astype(np.float32)
        sigmas  = np.clip(sigmas, 0.005, 0.15)  # cap: prevent runaway / dead exploration

        # 2. Rank-based effective mutation rates (not stored back)
        eff_sigmas = np.clip(sigmas * sigma_scale, 0.0, 1.0)

        # 3. Decode and mutate weights
        W_rec, W_in, W_out = self._decode(weights)

        # W_rec (N, N): row i uses eff_sigmas[i]
        mask = self.rng.random((N, N)) < eff_sigmas[:, None]
        if mask.any():
            W_rec[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        # W_in (N, obs_dim): row i uses eff_sigmas[i]
        mask = self.rng.random((N, self.obs_dim)) < eff_sigmas[:, None]
        if mask.any():
            W_in[mask] += (self.mutation_std *
                           self.rng.standard_normal(mask.sum())).astype(np.float32)

        # W_out (action_dim, N): col i uses eff_sigmas[i]
        mask = self.rng.random((self.action_dim, N)) < eff_sigmas[None, :]
        if mask.any():
            W_out[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        weights = self._encode(W_rec, W_in, W_out)
        return np.concatenate([weights, sigmas])

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self, task, n_generations=300, print_every=25, patience=100):
        """
        Run the full GA.

        Returns:
            dict with keys:
              W_rec_init, W_in_init, W_out_init
              W_rec_final, W_in_final, W_out_final
              best_fitness, history, snapshots, best_gene
        """
        population = self.init_population()

        # Save population centroid as init reference (mean across all individuals).
        # Comparable to BPTT's single-model init: ΔW measures best vs. starting centroid.
        _all_wr, _all_wi, _all_wo = zip(*[self._decode(g) for g in population])
        W_rec_init = np.mean(_all_wr, axis=0).astype(np.float32)
        W_in_init  = np.mean(_all_wi, axis=0).astype(np.float32)
        W_out_init = np.mean(_all_wo, axis=0).astype(np.float32)

        history = {
            'fitness':      [],   # pop mean raw fitness per gen
            'accuracy':     [],   # pop mean accuracy per gen
            'best_fitness': [],   # best-so-far raw fitness
            'mean_sigma':   [],   # mean per-neuron mutation rate across pop
            'mutation_std': [],   # global mutation amplitude (1/5-rule adapted)
        }

        best_gene       = None
        best_fitness    = -np.inf
        best_accuracy   = 0.0   # all-time best, updated when best_gene changes
        gens_no_improve = 0

        # 1/5-success-rule on global mutation_std (Rechenberg 1973).
        # When the population finds improvements > 1-in-5 gens → bolder mutations;
        # when stuck < 1-in-5 → finer mutations.  Analogous to ES sigma adaptation.
        _SUCCESS_WIN  = 20
        _MUT_UP       = 1.05   # grow mutation_std when improving frequently
        _MUT_DN       = 0.97   # shrink when stuck
        _MUT_MIN      = self.mutation_std * 0.25
        _MUT_MAX      = self.mutation_std * 4.0
        _success_hist = []

        snapshot_gens = sorted(set(
            [0, 25, 50, 100, 150, 200, 250] + [n_generations - 1]
        ))
        snapshots = {}

        for gen in range(n_generations):
            # 1. Evaluate (raw fitness)
            results        = self.evaluate(population, task)
            raw_fitnesses  = [r['fitness']  for r in results]
            accuracies     = [r['accuracy'] for r in results]

            # 2. Fitness sharing → selection pressure
            shared_fitnesses = self.fitness_sharing(population, raw_fitnesses)

            # 3. Track stats (raw fitness, raw accuracy)
            gen_mean_fit = float(np.mean(raw_fitnesses))
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

            # 1/5-success-rule: adapt global mutation_std each generation
            _success_hist.append(1 if improved else 0)
            if len(_success_hist) > _SUCCESS_WIN:
                _success_hist.pop(0)
            if len(_success_hist) == _SUCCESS_WIN:
                rate = sum(_success_hist) / _SUCCESS_WIN
                if rate > 0.2:
                    self.mutation_std = min(self.mutation_std * _MUT_UP, _MUT_MAX)
                elif rate < 0.2:
                    self.mutation_std = max(self.mutation_std * _MUT_DN, _MUT_MIN)

            history['fitness'].append(gen_mean_fit)
            history['accuracy'].append(gen_mean_acc)
            history['best_fitness'].append(float(best_fitness))
            history['mean_sigma'].append(mean_sigma)
            history['mutation_std'].append(float(self.mutation_std))

            # 4. Snapshot
            if gen in snapshot_gens:
                policy = self._make_policy(best_gene)
                sr = task.evaluate_policy(policy, n_trials=50, rng=self.rng)
                snapshots[gen] = {'fitness': sr['fitness'], 'accuracy': sr['accuracy']}

            # 5. Print
            if gen % print_every == 0 or gen == n_generations - 1:
                print(f"Gen {gen:4d} | mean={gen_mean_acc:.1%} "
                      f"best={best_accuracy:.1%} σ̄={mean_sigma:.4f} mut_std={self.mutation_std:.4f}")

            # Early stopping
            if gens_no_improve >= patience:
                print(f"  Early stop at gen {gen} (no improvement for {patience} gens)")
                break

            # 6. Build next generation
            # Sort by SHARED fitness for elitism (shared = selection pressure)
            sorted_by_shared = np.argsort(shared_fitnesses)[::-1]
            # But track elites by RAW fitness to keep the actual best performers
            sorted_by_raw    = np.argsort(raw_fitnesses)[::-1]

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

            # Elitism: top-k by raw fitness survive unchanged
            for i in range(self.n_elite):
                new_population.append(population[sorted_by_raw[i]].copy())

            # Fill rest with offspring (tournament uses shared fitness)
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

        # Decode best individual's weights
        W_rec_f, W_in_f, W_out_f = self._decode(best_gene)

        # Stable final estimate: re-evaluate best gene with more trials
        final_eval    = task.evaluate_policy(self._make_policy(best_gene),
                                             n_trials=50, rng=self.rng)
        best_accuracy = final_eval['accuracy']

        return {
            'W_rec_init':  W_rec_init,  'W_in_init':  W_in_init,  'W_out_init':  W_out_init,
            'W_rec_final': W_rec_f,     'W_in_final': W_in_f,     'W_out_final': W_out_f,
            'best_fitness':  best_fitness,
            'best_accuracy': best_accuracy,   # stable 50-trial estimate
            'history':       history,
            'snapshots':     snapshots,
            'snapshot_gens': list(snapshots.keys()),
            'best_gene':     best_gene,   # full gene (weights + sigmas)
        }


# ============================================================================
# Convenience: train_ga() matching train_ea() interface
# ============================================================================

def train_ga(conf) -> dict:
    """
    Train using the genetic algorithm. Drop-in replacement for train_ea().
    Uses Config from run_full_analysis.
    """
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
    n_params = N * N + conf.obs_dim * N + conf.action_dim * N

    # ── Dimension-aware scaling ───────────────────────────────────────────
    # mutation_std: scale with sqrt(baseline_N / N) so each mutation's
    # amplitude stays proportional to the weight initialisation scale
    # sqrt(2/N).  Without this, mut_std=0.3 is 2.4× the init scale at 128n
    # vs 1.2× at 32n — mutations are twice as disruptive per weight, and
    # the total disruption per individual is 11× larger (883 vs 80 weights).
    # Scaling gives ~equal disruption at all network sizes.
    #   N=32:  mut_std = 0.30  (unchanged, baseline)
    #   N=64:  mut_std = 0.21
    #   N=128: mut_std = 0.15
    baseline_N = 32
    mut_std_raw = getattr(conf, 'ga_mutation_std', 0.3)
    mutation_std = mut_std_raw * np.sqrt(baseline_N / N)

    # sigma0 (per-neuron mutation rate gene initial value): left unscaled —
    # the self-adaptive mechanism and the 1/5 rule on mut_std handle this.
    sigma0 = getattr(conf, 'ga_mutation_rate', 0.05)

    pop_size = conf.ea_pop_size
    if getattr(conf, 'ea_auto_pop', False):
        pop_size = max(conf.ea_pop_size, int(4 * np.sqrt(n_params)))
    # ─────────────────────────────────────────────────────────────────────

    ga = GeneticAlgorithm(
        n_neurons=N,
        obs_dim=conf.obs_dim,
        action_dim=conf.action_dim,
        pop_size=pop_size,
        n_elite=max(2, pop_size // 32),  # ~3% elitism
        tournament_k=5,
        crossover_rate=0.7,
        mutation_rate=sigma0,
        mutation_std=mutation_std,
        n_eval_trials=conf.ea_n_eval_trials,
        seed=conf.seed,
    )

    print(f"GA: {N} neurons, {ga.gene_length} weight params + {N} sigma params")
    print(f"Task: {task_name} | pop={pop_size}, gens={conf.ea_generations}")
    print(f"elite={ga.n_elite}, tournament_k={ga.tournament_k}, "
          f"crossover=neuron-level, mut_std={ga.mutation_std:.4f} (raw={mut_std_raw}, N-scaled), "
          f"sigma0={ga.mutation_rate:.4f}, tau={ga.tau:.4f}, sigma_cap=[0.005,0.15], fitness=shared")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every,
                       patience=conf.ea_patience)
    return result


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    task = LetterNBackTask(n_back=1, seq_length=20)

    ga = GeneticAlgorithm(
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

    print("GA standalone test: 1-back, 32 neurons, 64 pop, 50 gens")
    print(f"Neuron crossover + self-adaptive mutation + fitness sharing")
    result = ga.evolve(task, n_generations=50, print_every=10)
    print(f"\nFinal: best_fitness={result['best_fitness']:+.4f}")
    print(f"       accuracy={result['history']['accuracy'][-1]:.1%}")
    print(f"       mean_sigma_final={result['history']['mean_sigma'][-1]:.4f}")
