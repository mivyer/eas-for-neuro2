"""Shared experiment configuration."""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    n_neurons: int = 32
    obs_dim: int = 5
    action_dim: int = 5

    task: str = "nback"   # "nback" | "evidence" | "robot"
    n_back: int = 2
    seq_length: int = 20

    evidence_strength: float = 0.1
    noise_std: float = 0.5
    trial_length: int = 50
    response_length: int = 5

    cue_duration: int = 5
    delay_duration: int = 10
    response_duration: int = 10

    ea_pop_size: int = 128
    ea_generations: int = 300
    ea_lr: float = 0.03
    ea_sigma: float = 0.02
    ea_n_eval_trials: int = 20

    ga_mutation_rate: float = 0.05
    ga_mutation_std: float = 0.3
    ea_patience: int = 999_999

    ea_sigma_scaling: bool = False   # scale sigma by 1/sqrt(n_params/baseline)
    ea_auto_pop: bool = False        # scale pop by sqrt(n_params/baseline)
    ea_baseline_params: int = 1344

    ea_l2_coef: float = 0.0

    bptt_iterations: int = 1000
    bptt_batch_size: int = 64
    bptt_lr: float = 1e-3

    lif_beta: float = 0.9
    lif_threshold: float = 0.5
    lif_refractory: int = 2
    ei_ratio: float = 0.8

    sparsity_threshold: float = 0.01

    seed: int = 42
    print_every: int = 25
    output_dir: str = "results"

    def to_dict(self):
        return asdict(self)
