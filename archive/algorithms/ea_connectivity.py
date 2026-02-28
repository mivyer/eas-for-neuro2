# algorithms/ea_connectivity.py

import numpy as np
from config import Config
from models.factory import policy_builder_from_connectivity


def init_params(conf: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Initialize connection probability matrix P.

    P[i, j] is the probability that a connection exists from neuron j to neuron i.
    """
    # Start from a homogeneous, low connection probability.
    P = np.full((conf.n_neurons, conf.n_neurons), 0.1, dtype=np.float32)
    # Clip into the allowed exploration interval to avoid 0/1 extremes.[file:7]
    P = np.clip(P, conf.min_p, conf.max_p)
    return P


def sample_population(P: np.ndarray, conf: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a population of binary connectivity matrices from Bernoulli(P).

    Shape:
        P               : (n_neurons, n_neurons)
        pop_connectivity: (pop_size, n_neurons, n_neurons)
    """
    probs = np.repeat(P[None, :, :], conf.pop_size, axis=0)  # (pop_size, n, n)
    uniforms = rng.random(size=probs.shape)
    return (uniforms < probs).astype(np.float32)


def evaluate_individual(connectivity: np.ndarray, n_steps: int) -> float:
    """
    Evaluate a single connectivity matrix on a task‑specific objective.

    This assumes policy_builder_from_connectivity(connectivity) returns an
    object with a .evaluate_on_oned_task(n_steps) method or a similar
    task‑specific evaluation that returns a scalar fitness.[file:7]
    """
    policy = policy_builder_from_connectivity(connectivity)
    return float(policy.evaluate_on_oned_task(n_steps))


def evaluate_population(pop_connectivity: np.ndarray, conf: Config) -> np.ndarray:
    """
    Evaluate the whole population and return fitness values.

    pop_connectivity: (pop_size, n_neurons, n_neurons)
    returns fitness:  (pop_size,)
    """
    pop_size = pop_connectivity.shape[0]
    fitness = np.zeros(pop_size, dtype=np.float32)
    for i in range(pop_size):
        connectivity = pop_connectivity[i]
        policy = policy_builder_from_connectivity(connectivity)
        # Replace this with any task‑specific fitness function you like.
        fitness[i] = float(policy.evaluate_on_oned_task(conf.n_steps))
    return fitness


def centered_rank(fitness: np.ndarray) -> np.ndarray:
    """
    Map raw fitness values to centered ranks in [-0.5, 0.5].

    This is a standard fitness shaping trick used in evolution strategies.[file:7]
    """
    ranks = np.argsort(np.argsort(fitness))
    ranks = ranks.astype(np.float32) / (len(fitness) - 1)
    return ranks - 0.5


def nes_update(
    P: np.ndarray,
    pop_connectivity: np.ndarray,
    fitness: np.ndarray,
    conf: Config,
) -> np.ndarray:
    """
    NES‑style update for Bernoulli parameters P.

    pop_connectivity: (pop_size, n_neurons, n_neurons) with entries in {0, 1}
    P               : (n_neurons, n_neurons) with entries in (0, 1)
    """
    shaped_fit = centered_rank(fitness)[:, None, None]  # (pop_size, 1, 1)
    eps = pop_connectivity - P  # (pop_size, n, n), since E[eps] = 0 under Bernoulli(P)

    # Variance term for Bernoulli; avoid division by zero near 0/1.[file:7]
    var = P * (1.0 - P)
    grad = np.mean(shaped_fit * eps, axis=0) / (var + 1e-8)

    P_new = P + conf.lr * grad
    # Keep probabilities within a safe exploration range (e.g. [1e-3, 1 - 1e-3]).[file:7]
    P_new = np.clip(P_new, conf.min_p, conf.max_p)
    return P_new


def train_ea(conf: Config) -> np.ndarray:
    """
    Run the evolutionary search over connectivity probabilities.

    This is purely evolutionary: it samples connectivity from P, evaluates each
    sample on your task‑specific objective, and updates P with an NES‑style step.
    No reinforcement‑learning algorithm is used here.[file:8]
    """
    rng = np.random.default_rng(conf.seed if hasattr(conf, "seed") else 0)

    P = init_params(conf, rng)
    for gen in range(conf.n_generations):
        pop_connectivity = sample_population(P, conf, rng)
        fitness = evaluate_population(pop_connectivity, conf)

        print(f"Gen {gen:04d} | mean {fitness.mean():.3f} | best {fitness.max():.3f}")

        P = nes_update(P, pop_connectivity, fitness, conf)

    return P
