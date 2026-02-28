# envs/oned_control.py

import numpy as np

class OneDControlTask:
    """
    Simple 1D control: keep x near 0.
    Agent supplies a sequence of actions; environment returns total reward.
    """
    def __init__(self, noise_scale: float = 0.1, action_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.action_scale = action_scale

    def rollout(self, actions, init_x: float = 1.0) -> float:
        x = init_x
        total_reward = 0.0
        for a in actions:
            x = x + self.noise_scale * np.random.randn() - self.action_scale * float(a)
            reward = -x**2
            total_reward += reward
        return float(total_reward)
