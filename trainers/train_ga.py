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
                 pop_size=128, n_elite=4, tournament_k=3,
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
        """Select one parent via tournament selection."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    # ------------------------------------------------------------------
    # Crossover: neuron-level
    # ------------------------------------------------------------------

    def crossover(self, parent1, parent2):
        """
        Neuron-level crossover: for each neuron i, pick one parent and
        inherit that neuron's entire identity:
          - W_rec[i, :] — outgoing connections
          - W_in[i, :]  — input weights
          - W_out[:, i] — output contribution
          - sigma[i]    — mutation rate

        This keeps each neuron's weight profile coherent rather than
        mixing individual weights from different evolutionary lineages.
        """
        N = self.n_neurons
        W_rec1, W_in1, W_out1 = self._decode(parent1)
        W_rec2, W_in2, W_out2 = self._decode(parent2)
        sigmas1 = parent1[self.gene_length:]
        sigmas2 = parent2[self.gene_length:]

        # Per-neuron binary choice: True → take from parent2
        from_p2 = self.rng.integers(0, 2, size=N).astype(bool)

        # W_rec shape (N, N): row i = outgoing edges of neuron i
        W_rec_c = np.where(from_p2[:, None], W_rec2, W_rec1)
        # W_in  shape (N, obs_dim): row i = input weights of neuron i
        W_in_c  = np.where(from_p2[:, None], W_in2,  W_in1)
        # W_out shape (action_dim, N): col i = readout contribution of neuron i
        W_out_c = np.where(from_p2[None, :], W_out2, W_out1)
        # Sigma: same choice as the neuron's weights
        sigmas_c = np.where(from_p2, sigmas2, sigmas1)

        weights_c = self._encode(W_rec_c, W_in_c, W_out_c)
        return np.concatenate([weights_c, sigmas_c.astype(np.float32)])

    # ------------------------------------------------------------------
    # Mutation: self-adaptive per-neuron rates
    # ------------------------------------------------------------------

    def mutate(self, gene):
        """
        Self-adaptive mutation:
          1. Perturb each neuron's sigma via lognormal:
               sigma_i *= exp(tau * N(0,1)),  tau = 1/sqrt(2N)
             then clip to [1e-4, 1.0].
          2. For neuron i, mutate its weights (W_rec[i,:], W_in[i,:], W_out[:,i])
             with probability sigma_i and std mutation_std.

        Sigma genes evolve alongside weights, allowing evolution to discover
        which neurons benefit from more or less exploration.
        """
        N = self.n_neurons
        weights = gene[:self.gene_length].copy()
        sigmas  = gene[self.gene_length:].copy()

        # 1. Lognormal update of sigma genes
        sigmas *= np.exp(self.tau * self.rng.standard_normal(N)).astype(np.float32)
        sigmas  = np.clip(sigmas, 1e-4, 1.0)

        # 2. Decode and mutate weights
        W_rec, W_in, W_out = self._decode(weights)

        # W_rec (N, N): row i uses sigma[i]
        mask = self.rng.random((N, N)) < sigmas[:, None]
        if mask.any():
            W_rec[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        # W_in (N, obs_dim): row i uses sigma[i]
        mask = self.rng.random((N, self.obs_dim)) < sigmas[:, None]
        if mask.any():
            W_in[mask] += (self.mutation_std *
                           self.rng.standard_normal(mask.sum())).astype(np.float32)

        # W_out (action_dim, N): col i uses sigma[i]
        mask = self.rng.random((self.action_dim, N)) < sigmas[None, :]
        if mask.any():
            W_out[mask] += (self.mutation_std *
                            self.rng.standard_normal(mask.sum())).astype(np.float32)

        weights = self._encode(W_rec, W_in, W_out)
        return np.concatenate([weights, sigmas])

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self, task, n_generations=300, print_every=25):
        """
        Run the full GA.

        Returns:
            dict with keys:
              W_rec_init, W_in_init, W_out_init
              W_rec_final, W_in_final, W_out_final
              best_fitness, history, snapshots, best_gene
        """
        population = self.init_population()

        # Save initial best weights for comparison
        W_rec_init, W_in_init, W_out_init = self._decode(population[0])

        history = {
            'fitness':      [],   # pop mean raw fitness per gen
            'accuracy':     [],   # pop mean accuracy per gen
            'best_fitness': [],   # best-so-far raw fitness
            'mean_sigma':   [],   # mean per-neuron mutation rate across pop
        }

        best_gene    = None
        best_fitness = -np.inf

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
            if raw_fitnesses[idx_best] > best_fitness:
                best_fitness = raw_fitnesses[idx_best]
                best_gene    = population[idx_best].copy()

            history['fitness'].append(gen_mean_fit)
            history['accuracy'].append(gen_mean_acc)
            history['best_fitness'].append(float(best_fitness))
            history['mean_sigma'].append(mean_sigma)

            # 4. Snapshot
            if gen in snapshot_gens:
                policy = self._make_policy(best_gene)
                sr = task.evaluate_policy(policy, n_trials=50, rng=self.rng)
                snapshots[gen] = {'fitness': sr['fitness'], 'accuracy': sr['accuracy']}

            # 5. Print
            if gen % print_every == 0 or gen == n_generations - 1:
                print(f"Gen {gen:4d} | mean={gen_mean_fit:+.4f} "
                      f"best={best_fitness:+.4f} acc={gen_mean_acc:.1%} "
                      f"σ̄={mean_sigma:.4f}")

            # 6. Build next generation
            # Sort by SHARED fitness for elitism (shared = selection pressure)
            sorted_by_shared = np.argsort(shared_fitnesses)[::-1]
            # But track elites by RAW fitness to keep the actual best performers
            sorted_by_raw    = np.argsort(raw_fitnesses)[::-1]

            new_population = []

            # Elitism: top-k by raw fitness survive unchanged
            for i in range(self.n_elite):
                new_population.append(population[sorted_by_raw[i]].copy())

            # Fill rest with offspring (tournament uses shared fitness)
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_select(population, shared_fitnesses)
                parent2 = self.tournament_select(population, shared_fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self.mutate(child)
                new_population.append(child)

            population = new_population[:self.pop_size]

        # Decode best individual's weights
        W_rec_f, W_in_f, W_out_f = self._decode(best_gene)

        return {
            'W_rec_init':  W_rec_init,  'W_in_init':  W_in_init,  'W_out_init':  W_out_init,
            'W_rec_final': W_rec_f,     'W_in_final': W_in_f,     'W_out_final': W_out_f,
            'best_fitness': best_fitness,
            'history':      history,
            'snapshots':    snapshots,
            'snapshot_gens': list(snapshots.keys()),
            'best_gene':    best_gene,   # full gene (weights + sigmas)
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
    else:
        from envs.working_memory import WorkingMemoryTask
        task = WorkingMemoryTask(
            cue_duration=conf.cue_duration,
            delay_duration=conf.delay_duration,
            response_duration=conf.response_duration,
        )

    ga = GeneticAlgorithm(
        n_neurons=conf.n_neurons,
        obs_dim=conf.obs_dim,
        action_dim=conf.action_dim,
        pop_size=conf.ea_pop_size,
        n_elite=max(2, conf.ea_pop_size // 32),  # ~3% elitism
        tournament_k=3,
        crossover_rate=0.7,
        mutation_rate=getattr(conf, 'ga_mutation_rate', 0.05),
        mutation_std=getattr(conf, 'ga_mutation_std', 0.3),
        n_eval_trials=conf.ea_n_eval_trials,
        seed=conf.seed,
    )

    print(f"GA: {conf.n_neurons} neurons, {ga.gene_length} weight params + {conf.n_neurons} sigma params")
    print(f"Task: {task_name} | pop={conf.ea_pop_size}, gens={conf.ea_generations}")
    print(f"elite={ga.n_elite}, tournament_k={ga.tournament_k}, "
          f"crossover=neuron-level, mut_std={ga.mutation_std}, "
          f"sigma0={ga.mutation_rate}, tau={ga.tau:.4f}, fitness=shared")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every)
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
        tournament_k=3,
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
