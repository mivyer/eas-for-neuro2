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

Key design choices from literature:
  - Tournament selection (pressure without ranking the whole pop)
  - Uniform crossover (each weight 50/50 from either parent)
  - Gaussian mutation (small perturbations, not full replacement)
  - Elitism (top-k survive unchanged — prevents losing good solutions)
  - No gradient signal whatsoever — pure selection pressure

This is the clean contrast to BPTT:
  BPTT = gradient through time (knows which input→output mapping matters)
  GA   = blind search with selection (only knows "this network scored X")
"""

import numpy as np
from copy import deepcopy
from models.rsnn_policy import RSNNPolicy
from envs.letter_nback import LetterNBackTask


class GeneticAlgorithm:
    """
    (μ+λ) genetic algorithm for RNN weight evolution.

    Population of μ parents. Each generation:
      1. Evaluate all individuals
      2. Select parents via tournament selection
      3. Generate λ offspring via crossover + mutation
      4. Elitism: top-k survive unchanged
      5. New population = elites + offspring (trimmed to pop_size)

    Args:
        n_neurons:      network size
        obs_dim:        input dimension
        action_dim:     output dimension
        pop_size:       population size (μ)
        n_elite:        number of elites preserved each generation
        tournament_k:   tournament size for selection
        crossover_rate: probability of crossover vs cloning
        mutation_rate:  per-weight probability of mutation
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
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.n_eval_trials = n_eval_trials
        self.rng = np.random.default_rng(seed)

        N = n_neurons
        self.gene_length = N * N + N * obs_dim + action_dim * N

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _encode(self, W_rec, W_in, W_out):
        """Flatten weight matrices into a gene vector."""
        return np.concatenate([W_rec.ravel(), W_in.ravel(), W_out.ravel()])

    def _decode(self, gene):
        """Gene vector → weight matrices."""
        N = self.n_neurons
        nr = N * N
        ni = N * self.obs_dim
        W_rec = gene[:nr].reshape(N, N)
        W_in = gene[nr:nr + ni].reshape(N, self.obs_dim)
        W_out = gene[nr + ni:].reshape(self.action_dim, N)
        return W_rec.astype(np.float32), W_in.astype(np.float32), W_out.astype(np.float32)

    def _make_policy(self, gene):
        """Gene → RSNNPolicy."""
        W_rec, W_in, W_out = self._decode(gene)
        return RSNNPolicy(W_rec, W_in, W_out)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def init_population(self):
        """Create initial population with Xavier-like random weights."""
        scale = np.sqrt(2.0 / self.n_neurons)
        population = []
        for _ in range(self.pop_size):
            gene = scale * self.rng.standard_normal(self.gene_length).astype(np.float32)
            population.append(gene)
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
    # Selection: tournament
    # ------------------------------------------------------------------

    def tournament_select(self, population, fitnesses):
        """Select one parent via tournament selection."""
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    # ------------------------------------------------------------------
    # Crossover: uniform
    # ------------------------------------------------------------------

    def crossover(self, parent1, parent2):
        """Uniform crossover: each gene from parent1 or parent2 with 50% prob."""
        child = parent1.copy()
        mask = self.rng.random(len(child)) < 0.5
        child[mask] = parent2[mask]
        return child

    # ------------------------------------------------------------------
    # Mutation: Gaussian perturbation
    # ------------------------------------------------------------------

    def mutate(self, gene):
        """Each weight mutated with probability mutation_rate."""
        mask = self.rng.random(len(gene)) < self.mutation_rate
        gene[mask] += self.mutation_std * self.rng.standard_normal(mask.sum()).astype(np.float32)
        return gene

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
              best_fitness, history, snapshots
        """
        population = self.init_population()

        # Save initial best for comparison
        W_rec_init, W_in_init, W_out_init = self._decode(population[0])

        history = {
            'fitness': [],       # pop mean fitness per gen
            'accuracy': [],      # pop mean accuracy per gen
            'best_fitness': [],  # best-so-far fitness
        }

        best_gene = None
        best_fitness = -np.inf

        snapshot_gens = sorted(set(
            [0, 25, 50, 100, 150, 200, 250] + [n_generations - 1]
        ))
        snapshots = {}

        for gen in range(n_generations):
            # 1. Evaluate
            results = self.evaluate(population, task)
            fitnesses = [r['fitness'] for r in results]
            accuracies = [r['accuracy'] for r in results]

            # 2. Track stats
            gen_mean_fit = np.mean(fitnesses)
            gen_mean_acc = np.mean(accuracies)

            idx_best = np.argmax(fitnesses)
            if fitnesses[idx_best] > best_fitness:
                best_fitness = fitnesses[idx_best]
                best_gene = population[idx_best].copy()

            history['fitness'].append(float(gen_mean_fit))
            history['accuracy'].append(float(gen_mean_acc))
            history['best_fitness'].append(float(best_fitness))

            # 3. Snapshot
            if gen in snapshot_gens:
                policy = self._make_policy(best_gene)
                sr = task.evaluate_policy(policy, n_trials=50, rng=self.rng)
                snapshots[gen] = {'fitness': sr['fitness'], 'accuracy': sr['accuracy']}

            # 4. Print
            if gen % print_every == 0 or gen == n_generations - 1:
                print(f"Gen {gen:4d} | mean={gen_mean_fit:+.4f} "
                      f"best={best_fitness:+.4f} acc={gen_mean_acc:.1%}")

            # 5. Build next generation
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]

            new_population = []

            # Elitism: top-k survive unchanged
            for i in range(self.n_elite):
                new_population.append(population[sorted_indices[i]].copy())

            # Fill rest with offspring
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_select(population, fitnesses)
                parent2 = self.tournament_select(population, fitnesses)

                if self.rng.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                child = self.mutate(child)
                new_population.append(child)

            population = new_population[:self.pop_size]

        # Decode best
        W_rec_f, W_in_f, W_out_f = self._decode(best_gene)

        return {
            'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
            'W_rec_final': W_rec_f, 'W_in_final': W_in_f, 'W_out_final': W_out_f,
            'best_fitness': best_fitness,
            'history': history,
            'snapshots': snapshots,
            'snapshot_gens': list(snapshots.keys()),
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

    print(f"GA: {conf.n_neurons} neurons, {ga.gene_length} params")
    print(f"Task: {task_name} | pop={conf.ea_pop_size}, gens={conf.ea_generations}")
    print(f"elite={ga.n_elite}, tournament_k={ga.tournament_k}, "
          f"crossover={ga.crossover_rate}, mut_rate={ga.mutation_rate}, "
          f"mut_std={ga.mutation_std}")

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
    result = ga.evolve(task, n_generations=50, print_every=10)
    print(f"\nFinal: best_fitness={result['best_fitness']:+.4f}")
    print(f"       accuracy={result['history']['accuracy'][-1]:.1%}")
