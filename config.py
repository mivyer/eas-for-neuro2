# config.py
"""Shared experiment configuration for all trainers and scripts."""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Architecture
    n_neurons: int = 32
    obs_dim: int = 5       # one-hot over 5 symbols
    action_dim: int = 5    # 5-class softmax output

    # Task
    task: str = "nback"         # "nback" or "wm"
    n_back: int = 2
    seq_length: int = 20

    # Working memory params (used when task="wm")
    cue_duration: int = 5
    delay_duration: int = 10
    response_duration: int = 10

    # ES (OpenAI Evolution Strategy)
    ea_pop_size: int = 128
    ea_generations: int = 300
    ea_lr: float = 0.03
    ea_sigma: float = 0.02
    ea_n_eval_trials: int = 20

    # GA (Genetic Algorithm)
    ga_mutation_rate: float = 0.05
    ga_mutation_std: float = 0.3
    ea_patience: int = 999_999     # early-stop patience (default = off; use --patience N to enable)

    # BPTT
    bptt_iterations: int = 1000
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3

    # LIF neuron params
    lif_beta: float = 0.9        # slower leak → voltage accumulates
    lif_threshold: float = 0.5   # lower threshold → neurons actually fire
    lif_refractory: int = 2
    ei_ratio: float = 0.8

    # Analysis
    sparsity_threshold: float = 0.01

    # Misc
    seed: int = 42
    print_every: int = 25
    output_dir: str = "results"

    def to_dict(self):
        return asdict(self)