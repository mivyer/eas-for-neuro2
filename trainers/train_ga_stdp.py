# trainers/train_ga_stdp.py
"""
GA + R-STDP with Connectivity Evolution

Three key improvements over the old GA+STDP:

1. EVOLVE CONNECTIVITY, NOT RAW WEIGHTS
   Genotype: Bernoulli probability matrix P_ij ∈ [0,1] for connections
   Phenotype: sample binary mask from P, multiply by fixed weight magnitude
   This gives the GA the right abstraction: WHICH neurons connect, not HOW STRONG
   (Wang et al., NeurIPS 2023: "Evolving Connectivity for Recurrent SNNs")

2. REWARD-MODULATED STDP (R-STDP)
   Spike timing creates eligibility traces during the trial.
   After each response timestep, a reward signal (+1 correct, -1 wrong)
   converts eligibility into weight changes.
   (Izhikevich 2007: STDP + dopamine solves distal reward)

3. TWO-PHASE FITNESS
   fitness = post_STDP_accuracy + α * improvement
   This rewards networks that are LEARNABLE, not just accidentally good.
   (Baldwin Effect: evolution selects for plasticity-friendly structure)

Genotype layout:
   [P_rec (N*N) | P_in (N*obs) | W_out (act*N) | R-STDP params (8)]
   - P_rec, P_in: connection probabilities (sigmoid-transformed for [0,1])
   - W_out: output weights (evolved directly, small)
   - R-STDP: A+, A-, τ+, τ-, τ_e, η, w_max, w_min
"""

import numpy as np
from models.lif_rsnn import LIF_RSNN_NP, make_dale_mask, enforce_dale_weights
from models.stdp import RewardSTDP
from envs.letter_nback import LetterNBackTask, decode_output, N_SYMBOLS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


class GA_ConnectivityRSTDP:
    """
    GA that evolves:
      - Connection probability matrix (Bernoulli params for recurrent + input)
      - Output weights (direct)
      - R-STDP hyperparameters

    During fitness evaluation:
      1. Sample binary connectivity from evolved probabilities
      2. Create LIF network with fixed weight magnitude × connectivity mask
      3. Run trial with R-STDP active (eligibility traces + reward modulation)
      4. Measure pre-STDP and post-STDP performance
      5. Fitness = post_performance + bonus for improvement
    """

    def __init__(self, n_neurons=32, obs_dim=1, action_dim=1,
                 pop_size=128, n_elite=4, tournament_k=3,
                 crossover_rate=0.7, mutation_rate=0.05, mutation_std=0.3,
                 n_eval_trials=20, weight_magnitude=0.5,
                 ei_ratio=0.8, beta=0.9, threshold=0.5,
                 improvement_bonus=0.3,
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
        self.weight_magnitude = weight_magnitude
        self.beta = beta
        self.threshold = threshold
        self.improvement_bonus = improvement_bonus
        self.rng = np.random.default_rng(seed)

        # E/I
        self.ei_ratio = ei_ratio
        self.dale_mask = make_dale_mask(n_neurons, ei_ratio)

        # Gene layout: P_rec (logits) + P_in (logits) + W_out + R-STDP params
        N = n_neurons
        self.n_p_rec = N * N
        self.n_p_in = N * obs_dim
        self.n_w_out = action_dim * N
        self.n_stdp = RewardSTDP.N_PARAMS
        self.gene_length = self.n_p_rec + self.n_p_in + self.n_w_out + self.n_stdp

    def _decode_gene(self, gene):
        """Gene → (P_rec, P_in, W_out, RewardSTDP)"""
        N = self.n_neurons
        idx = 0

        # Connection probability logits → probabilities via sigmoid
        p_rec_logits = gene[idx:idx + self.n_p_rec].reshape(N, N)
        P_rec = sigmoid(p_rec_logits)
        idx += self.n_p_rec

        p_in_logits = gene[idx:idx + self.n_p_in].reshape(N, self.obs_dim)
        P_in = sigmoid(p_in_logits)
        idx += self.n_p_in

        # Output weights (direct, not connectivity)
        W_out = gene[idx:idx + self.n_w_out].reshape(self.action_dim, N).astype(np.float32)
        idx += self.n_w_out

        # R-STDP params
        stdp_params = gene[idx:idx + self.n_stdp]
        rstdp = RewardSTDP.from_params(N, stdp_params, dale_mask=self.dale_mask)

        return P_rec.astype(np.float32), P_in.astype(np.float32), W_out, rstdp

    def _sample_network(self, P_rec, P_in, W_out):
        """Sample a concrete network from connection probabilities."""
        N = self.n_neurons

        # Sample binary connectivity
        mask_rec = (self.rng.random(P_rec.shape) < P_rec).astype(np.float32)
        mask_in = (self.rng.random(P_in.shape) < P_in).astype(np.float32)

        # No self-connections
        np.fill_diagonal(mask_rec, 0)

        # Weights = mask × fixed magnitude (sign from Dale's law)
        W_rec = mask_rec * self.weight_magnitude
        W_rec = enforce_dale_weights(W_rec, self.dale_mask)

        W_in = mask_in * self.weight_magnitude * 2.0  # stronger input drive

        return W_rec, W_in, W_out.copy()

    def init_population(self):
        """Create initial population."""
        population = []
        for _ in range(self.pop_size):
            # P_rec logits: start near 0 (50% connectivity)
            p_rec = 0.5 * self.rng.standard_normal(self.n_p_rec).astype(np.float32)
            # P_in logits: start slightly positive (higher input connectivity)
            p_in = 1.0 + 0.3 * self.rng.standard_normal(self.n_p_in).astype(np.float32)
            # W_out: small random
            w_out = 0.25 * self.rng.standard_normal(self.n_w_out).astype(np.float32)
            # R-STDP defaults + noise
            stdp_defaults = np.array([0.005, 0.005, 20.0, 20.0, 25.0, 0.01, 3.0, -3.0],
                                     dtype=np.float32)
            stdp_gene = stdp_defaults + 0.05 * self.rng.standard_normal(self.n_stdp).astype(np.float32)

            gene = np.concatenate([p_rec, p_in, w_out, stdp_gene])
            population.append(gene)
        return population

    def _compute_reward(self, output, target):
        """Per-timestep reward: +1 if output is closest to correct symbol, -1 otherwise."""
        pred = decode_output(output)
        tgt = decode_output(target)
        if pred == tgt:
            return 1.0
        return -0.5  # smaller penalty than reward to encourage exploration

    def evaluate_one(self, gene, task):
        """
        Evaluate one individual with two-phase fitness.

        Phase 1: Run trial WITHOUT R-STDP → pre_accuracy
        Phase 2: Run trial WITH R-STDP → post_accuracy
        Fitness = post_fitness + improvement_bonus * (post_acc - pre_acc)
        """
        P_rec, P_in, W_out, rstdp = self._decode_gene(gene)

        pre_scores = []
        post_scores = []
        post_fitnesses = []

        for trial_idx in range(self.n_eval_trials):
            # Sample same network for both phases
            W_rec, W_in, W_out_copy = self._sample_network(P_rec, P_in, W_out)

            inputs, targets, letters = task.get_trial(rng=self.rng)

            # --- Phase 1 — Pre-STDP (baseline connectivity) ---
            net_pre = LIF_RSNN_NP(W_rec.copy(), W_in.copy(), W_out_copy.copy(),
                                   beta=self.beta, threshold=self.threshold,
                                   dale_mask=self.dale_mask)
            outputs_pre = []
            for t in range(task.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                y = net_pre.step(obs)
                # Clamp to symbol range [0, 1.2] via sigmoid-like squash
                val = float(y[0]) if hasattr(y, '__len__') else float(y)
                val = 1.0 / (1.0 + np.exp(-val))  # squash to (0, 1)
                outputs_pre.append(val)
            pre_acc = task.compute_accuracy(np.array(outputs_pre), targets)
            pre_scores.append(pre_acc)

            # --- Phase 2 — Post-STDP (connectivity + R-STDP) ---
            net_post = LIF_RSNN_NP(W_rec.copy(), W_in.copy(), W_out_copy.copy(),
                                    beta=self.beta, threshold=self.threshold,
                                    dale_mask=self.dale_mask)
            rstdp.reset()
            outputs_post = []

            for t in range(task.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                y = net_post.step(obs)
                val = float(y[0]) if hasattr(y, '__len__') else float(y)
                val = 1.0 / (1.0 + np.exp(-val))  # squash to (0, 1)
                outputs_post.append(val)

                # Update eligibility traces every timestep (spike timing)
                rstdp.update_traces(net_post.s)

                # Apply reward at EVERY response timestep
                if targets[t] != 0:
                    reward = self._compute_reward(val, targets[t])
                    net_post.W_rec = rstdp.apply_reward(net_post.W_rec, reward)

            outputs_post = np.array(outputs_post)
            post_acc = task.compute_accuracy(outputs_post, targets)
            post_fit = task.evaluate_outputs(outputs_post, targets)
            post_scores.append(post_acc)
            post_fitnesses.append(post_fit)

        mean_pre = np.mean(pre_scores)
        mean_post = np.mean(post_scores)
        mean_fit = np.mean(post_fitnesses)
        improvement = mean_post - mean_pre

        # Two-phase fitness
        fitness = mean_fit + self.improvement_bonus * improvement

        return {
            'fitness': fitness,
            'accuracy': mean_post,
            'pre_accuracy': mean_pre,
            'improvement': improvement,
        }

    def evaluate(self, population, task):
        return [self.evaluate_one(gene, task) for gene in population]

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

    def evolve(self, task, n_generations=300, print_every=25):
        population = self.init_population()
        history = {'fitness': [], 'accuracy': [], 'best_fitness': [],
                   'pre_accuracy': [], 'improvement': []}
        best_gene = None
        best_fitness = -np.inf

        for gen in range(n_generations):
            results = self.evaluate(population, task)
            fitnesses = [r['fitness'] for r in results]
            accuracies = [r['accuracy'] for r in results]
            pre_accs = [r['pre_accuracy'] for r in results]
            improvements = [r['improvement'] for r in results]

            idx_best = np.argmax(fitnesses)
            if fitnesses[idx_best] > best_fitness:
                best_fitness = fitnesses[idx_best]
                best_gene = population[idx_best].copy()

            history['fitness'].append(float(np.mean(fitnesses)))
            history['accuracy'].append(float(np.mean(accuracies)))
            history['best_fitness'].append(float(best_fitness))
            history['pre_accuracy'].append(float(np.mean(pre_accs)))
            history['improvement'].append(float(np.mean(improvements)))

            if gen % print_every == 0 or gen == n_generations - 1:
                _, _, _, best_rstdp = self._decode_gene(best_gene)
                P_rec, _, _, _ = self._decode_gene(best_gene)
                sparsity = (sigmoid(best_gene[:self.n_p_rec]) > 0.5).mean()
                print(f"Gen {gen:4d} | fit={np.mean(fitnesses):+.4f} "
                      f"pre={np.mean(pre_accs):.1%} post={np.mean(accuracies):.1%} "
                      f"Δ={np.mean(improvements):+.1%} "
                      f"| η={best_rstdp.eta:.4f} conn={sparsity:.0%}")

            # Next generation
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = []
            for i in range(self.n_elite):
                new_population.append(population[sorted_indices[i]].copy())
            while len(new_population) < self.pop_size:
                p1 = self.tournament_select(population, fitnesses)
                p2 = self.tournament_select(population, fitnesses)
                child = self.crossover(p1, p2) if self.rng.random() < self.crossover_rate else p1.copy()
                child = self.mutate(child)
                new_population.append(child)
            population = new_population[:self.pop_size]

        # Decode best
        P_rec, P_in, W_out, best_rstdp = self._decode_gene(best_gene)
        W_rec, W_in, _ = self._sample_network(P_rec, P_in, W_out)

        return {
            'W_rec_init': np.zeros_like(W_rec),
            'W_in_init': np.zeros_like(W_in),
            'W_out_init': np.zeros_like(W_out),
            'W_rec_final': W_rec, 'W_in_final': W_in, 'W_out_final': W_out,
            'P_rec': sigmoid(best_gene[:self.n_p_rec].reshape(self.n_neurons, self.n_neurons)),
            'best_fitness': best_fitness,
            'history': history,
            'stdp_params': best_rstdp.get_params(),
        }


def train_ga_stdp(conf) -> dict:
    """Train GA+R-STDP with connectivity evolution."""
    task = LetterNBackTask(n_back=conf.n_back, seq_length=conf.seq_length)

    ga = GA_ConnectivityRSTDP(
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
        beta=conf.lif_beta,
        threshold=conf.lif_threshold,
        seed=conf.seed,
    )

    print(f"GA+R-STDP (connectivity): {conf.n_neurons} neurons, {ga.gene_length} params")
    print(f"  {ga.n_p_rec} conn probs + {ga.n_p_in} input probs + "
          f"{ga.n_w_out} output weights + {ga.n_stdp} R-STDP params")
    print(f"Task: nback | pop={conf.ea_pop_size}, gens={conf.ea_generations}")
    print(f"LIF: beta={ga.beta}, threshold={ga.threshold}, E/I={ga.ei_ratio:.0%}/{1-ga.ei_ratio:.0%}")

    result = ga.evolve(task, n_generations=conf.ea_generations,
                       print_every=conf.print_every)
    return result


if __name__ == "__main__":
    from config import Config
    task = LetterNBackTask(n_back=1, seq_length=20)
    ga = GA_ConnectivityRSTDP(n_neurons=32, pop_size=32, n_eval_trials=5, seed=42)
    print(f"Gene: {ga.gene_length} ({ga.n_p_rec} P_rec + {ga.n_p_in} P_in + "
          f"{ga.n_w_out} W_out + {ga.n_stdp} R-STDP)")
    result = ga.evolve(task, n_generations=20, print_every=5)
    print(f"\nFinal: acc={result['history']['accuracy'][-1]:.1%}")