from brax import envs
from brax.envs import base

def make_hopper(batch_size=512, episode_length=1000):
    """Thesis Week 2: Hopper sanity check (Wang et al. 3-DoF)"""
    return envs.create(
        'hopper',
        episode_length=episode_length,
        action_repeat=1,
        batch_size=batch_size  # Small for sanity check
    )
