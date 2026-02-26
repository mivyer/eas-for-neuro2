# trainers/train_ga_stdp.py
"""
GA + STDP Hybrid: Evolutionary Algorithm with Lifetime Plasticity

The Baldwin Effect applied to spiking neural networks:
  - GA evolves: initial weights (W_rec, W_in, W_out) + STDP hyperparameters
  - STDP learns: within each trial, synapses update based on spike timing
  - Fitness is measured AFTER STDP has modified the network during the trial
  - STDP changes are DISCARDED after fitness evaluation (no Lamarckian inheritance)

The GA doesn't need to solve temporal credit assignment itself — STDP does that.
GA just needs to find initial conditions + plasticity rules that enable good STDP learning.

This is the key thesis condition: does giving evolution access to local
plasticity close the gap with BPTT?

Architecture:
  - LIF neurons with E/I split and Dale's law
  - GA evolves: W_rec, W_in, W_out (1088 params for 32N) + 6 STDP params = 1094 total
  - During each fitness evaluation trial:
    1. Copy initial weights
    2. Run LIF network on trial with STDP active
    3. Measure accuracy/fitness on the trial output
    4. Discard STDP-modified weights
"""

import numpy as np
from copy import deepcopy
from models.lif_rsnn import LIF_RSNN_NP, make_dale_mask, enforce_dale_weights
from models.stdp import STDP_Rule
from envs.letter_nback import LetterNBackTask


class GA_STDP:
    """
    Genetic Algorithm that evolves LIF network weights + STDP parameters.

    Each individual's gene = [W_rec flat | W_in flat | W_out flat | STDP params (6)]

    During fitness evaluation:
      1. Decode gene → initial weights + STDP rule
      2. For each trial:
         a. Create LIF network with initial weights (copied, not shared)
         b. Run network step-by-step with STDP active
         c. Record outputs
      3. Fitness = mean performance across trials
    """

    def __init__(self, n_neurons=32, obs_dim=1, action_dim=1,
                 pop_size=128, n_elite=4, tournament_k=3,
                 crossover_rate=0.7, mutation_rate=0.05,
                 mutation_std=0.3, n_eval_trials=20,
                 ei_ratio=0.8, beta=0.85, threshold=1.0,
                 seed=42):

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
        self.beta = beta
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

        # E/I
        self.ei_ratio = ei_ratio
        self.dale_mask = make_dale_mask(n_neurons, ei_ratio)

        # Gene layout
        N = n_neurons
        self.n_weight_params = N * N + N * obs_dim + action_dim * N
        self.n_stdp_params = STDP_Rule.N_PARAMS
        self.gene_length = self.n_weight_params + self.n_stdp_params

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def _decode_gene(self, gene):
        """Gene → (W_rec, W_in, W_out, STDP_Rule)"""
        N = self.n_neurons
        nr = N * N
        ni = N * self.obs_dim

        W_rec = gene[:nr].reshape(N, N).astype(np.float32)
        W_in = gene[nr:nr + ni].reshape(N, self.obs_dim).astype(np.float32)
        W_out = gene[nr + ni:nr + ni + self.action_dim * N].reshape(
            self.action_dim, N).astype(np.float32)

        stdp_params = gene[self.n_weight_params:]
        stdp = STDP_Rule.from_params(N, stdp_params, dale_mask=self.dale_mask)

        # Enforce Dale's law on initial weights
        W_rec = enforce_dale_weights(W_rec, self.dale_mask)

        return W_rec, W_in, W_out, stdp

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def init_population(self):
        """Create initial population."""
        scale = np.sqrt(2.0 / self.n_neurons)
        population = []
        for _ in range(self.pop_size):
            # Random weights
            w_gene = scale * self.rng.standard_normal(self.n_weight_params).astype(np.float32)
            # STDP params: reasonable defaults + small noise
            stdp_defaults = np.array([0.01, 0.012, 20.0, 20.0, 3.0, -3.0], dtype=np.float32)
            stdp_gene = stdp_defaults + 0.1 * self.rng.standard_normal(self.n_stdp_params).astype(np.float32)
            gene = np.concatenate([w_gene, stdp_gene])
            population.append(gene)
        return population

    # ------------------------------------------------------------------
    # Fitness evaluation with STDP
    # ------------------------------------------------------------------

    def evaluate_one(self, gene, task):
        """
        Evaluate one individual on multiple trials WITH STDP active.

        For each trial:
          1. Copy initial weights (don't modify the gene's weights)
          2. Build LIF network
          3. Run trial step-by-step, applying STDP after each step
          4. Collect outputs
        """
        W_rec_init, W_in, W_out, stdp = self._decode_gene(gene)

        total_fitness = 0.0
        total_accuracy = 0.0

        for trial_idx in range(self.n_eval_trials):
            # Fresh copy of weights for this trial
            W_rec = W_rec_init.copy()

            # Build LIF
            net = LIF_RSNN_NP(W_rec, W_in, W_out,
                              beta=self.beta, threshold=self.threshold,
                              dale_mask=self.dale_mask)

            # Reset STDP traces
            stdp.reset()

            # Get trial data
            inputs, targets, letters = task.get_trial(rng=self.rng)

            outputs = []
            for t in range(task.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                y = net.step(obs)
                outputs.append(float(y[0]) if hasattr(y, '__len__') else float(y))

                # Apply STDP to recurrent weights
                # In recurrent network, every neuron is both pre and post
                spikes = net.s
                net.W_rec = stdp.update(net.W_rec, spikes, spikes)

            outputs = np.array(outputs)
            total_fitness += task.evaluate_outputs(outputs, targets)
            total_accuracy += task.compute_accuracy(outputs, targets)

        return {
            'fitness': total_fitness / self.n_eval_trials,
            'accuracy': total_accuracy / self.n_eval_trials,
        }

    def evaluate(self, population, task):
        """Evaluate all individuals."""
        return [self.evaluate_one(gene, task) for gene in population]

    # ------------------------------------------------------------------
    # Selection, crossover, mutation (same as GA)
    # ------------------------------------------------------------------

    def tournament_select(self, population, fitnesses):
        indices = self.rng.choice(len(population), size=self.tournament_k, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        mask = self.rng.random(len(child)) < 0.5
        child[mask] = parent2[mask]
        return child

    def mutate(self, gene):
        mask = self.rng.random(len(gene)) < self.mutation_rate
        gene[mask] += self.mutation_std * self.rng.standard_normal(mask.sum()).astype(np.float32)
        return gene

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def evolve(self, task, n_generations=300, print_every=25):
        """Run GA+STDP evolution."""

        population = self.init_population()

        # Save initial for comparison
        W_rec_init, W_in_init, W_out_init, _ = self._decode_gene(population[0])

        history = {
            'fitness': [],
            'accuracy': [],
            'best_fitness': [],
        }

        best_gene = None
        best_fitness = -np.inf

        for gen in range(n_generations):
            # Evaluate with STDP
            results = self.evaluate(population, task)
            fitnesses = [r['fitness'] for r in results]
            accuracies = [r['accuracy'] for r in results]

            gen_mean_fit = np.mean(fitnesses)
            gen_mean_acc = np.mean(accuracies)

            idx_best = np.argmax(fitnesses)
            if fitnesses[idx_best] > best_fitness:
                best_fitness = fitnesses[idx_best]
                best_gene = population[idx_best].copy()

            history['fitness'].append(float(gen_mean_fit))
            history['accuracy'].append(float(gen_mean_acc))
            history['best_fitness'].append(float(best_fitness))

            if gen % print_every == 0 or gen == n_generations - 1:
                # Show STDP params of best
                _, _, _, best_stdp = self._decode_gene(best_gene)
                print(f"Gen {gen:4d} | mean={gen_mean_fit:+.4f} "
                      f"best={best_fitness:+.4f} acc={gen_mean_acc:.1%} "
                      f"| A+={best_stdp.A_plus:.4f} A-={best_stdp.A_minus:.4f} "
                      f"τ+={best_stdp.tau_plus:.1f} τ-={best_stdp.tau_minus:.1f}")

            # Next generation
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = []

            # Elitism
            for i in range(self.n_elite):
                new_population.append(population[sorted_indices[i]].copy())

            # Offspring
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
        W_rec_f, W_in_f, W_out_f, best_stdp = self._decode_gene(best_gene)

        return {
            'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
            'W_rec_final': W_rec_f, 'W_in_final': W_in_f, 'W_out_final': W_out_f,
            'best_fitness': best_fitness,
            'history': history,
            'stdp_params': best_stdp.get_params(),
            'stdp_rule': best_stdp,
        }


# ============================================================================
# Convenience: train_ga_stdp() matching train_ea() interface
# ============================================================================

def train_ga_stdp(conf) -> dict:
    """Train GA+STDP. Drop-in for run_full_analysis."""
    task = LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)

    ga = GA_STDP(
        n_neurons=conf.n_neurons,
        obs_dim=conf.obs_dim,
        action_dim=conf.action_dim,
        pop_size=conf.ea_pop_size,
        n_elite=max(2, conf.ea_pop_size // 32),
        tournament_k=3,
        crossover_rate=0.7,
        mutation_rate=getattr(conf, 'ga_mutation_rate', 0.05),
        mutation_std=getattr(conf, 'ga_mutation_std', 0.3),
        n_eval_trials=conf.ea_n_eval_trials,
        seed=conf.seed,
    )

    print(f"GA+STDP: {conf.n_neurons} neurons, {ga.gene_length} params "
          f"({ga.n_weight_params} weights + {ga.n_stdp_params} STDP)")
    print(f"Task: nback | pop={conf.ea_pop_size}, gens={conf.ea_generations}")
    print(f"LIF: beta={ga.beta}, threshold={ga.threshold}, "
          f"E/I={ga.ei_ratio:.0%}/{1-ga.ei_ratio:.0%}")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every)
    return result


# ============================================================================
# Standalone test
# ============================================================================

if __name__ == "__main__":
    task = LetterNBackTask(n_back=1, seq_length=20)

    ga = GA_STDP(
        n_neurons=32,
        pop_size=32,     # small for quick test
        n_elite=2,
        n_eval_trials=5,
        seed=42,
    )

    print("GA+STDP standalone test: 1-back, 32 neurons, 32 pop, 30 gens")
    print(f"Gene length: {ga.gene_length} ({ga.n_weight_params} weights + {ga.n_stdp_params} STDP)")
    result = ga.evolve(task, n_generations=30, print_every=5)
    print(f"\nFinal: best_fitness={result['best_fitness']:+.4f}")
    print(f"       accuracy={result['history']['accuracy'][-1]:.1%}")
    print(f"STDP params: A+={result['stdp_params'][0]:.4f} "
          f"A-={result['stdp_params'][1]:.4f} "
          f"τ+={result['stdp_params'][2]:.1f} "
          f"τ-={result['stdp_params'][3]:.1f}")
