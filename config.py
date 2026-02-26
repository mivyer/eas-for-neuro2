# config.py

from dataclasses import dataclass


@dataclass
class Config:
    # Network / connectivity settings
    n_neurons: int = 20

    # Evolutionary algorithm settings
    pop_size: int = 32
    n_generations: int = 50
    lr: float = 0.05          # learning rate for probability update
    min_p: float = 0.05       # lower clip for connection probabilities
    max_p: float = 0.95       # upper clip for connection probabilities

    # Task / evaluation settings
    n_steps: int = 50         # steps per episode / trial

    # Misc
    sigma: float = 0.1        # kept for conceptual NES noise scale (not required)
    seed: int = 0             # RNG seed for reproducibility


default_config = Config()
