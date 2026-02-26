# models/rsnn_policy.py
"""
RSNN Policy wrapper for EA experiments.
Provides the interface expected by tasks: reset() and act(obs).
"""

import numpy as np
from models.rsnn_core import SimpleRSNNCore, LIFCore


class RSNNPolicy:
    """
    Policy that wraps an RSNN core for use with environments/tasks.
    
    Interface:
        - reset(): reset internal state
        - act(obs) -> action: given observation, return action
    """
    
    def __init__(
        self,
        W_rec: np.ndarray,
        W_in: np.ndarray | None,
        W_out: np.ndarray,
        use_lif: bool = False,
        **lif_kwargs,
    ):
        """
        Args:
            W_rec: (N, N) recurrent weights
            W_in: (N, obs_dim) input weights, or None
            W_out: (action_dim, N) output weights
            use_lif: if True, use LIF neurons; else use rate-based tanh
            **lif_kwargs: passed to LIFCore (beta, v_th, etc.)
        """
        if use_lif:
            self.core = LIFCore(W_rec, W_in, **lif_kwargs)
        else:
            self.core = SimpleRSNNCore(W_rec, W_in)
        
        self.W_out = W_out.astype(np.float32)
        self.use_lif = use_lif
    
    def reset(self):
        """Reset network state."""
        self.core.reset_state()
    
    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Given an observation, update the network and return an action.
        
        Args:
            obs: (obs_dim,) observation
        
        Returns:
            action: (action_dim,) action
        """
        # Step the network
        self.core.step(obs)
        
        # Read out from hidden state
        readout = self.core.get_state()
        
        # Linear readout + tanh to bound actions
        action = np.tanh(self.W_out @ readout)
        
        return action.astype(np.float32)
    
    @property
    def n_neurons(self) -> int:
        return self.core.N
    
    def rollout_actions(self, n_steps: int) -> np.ndarray:
        """
        Generate a sequence of actions using the RSNN (no external input).
        Used for autonomous tasks like OneDControlTask.
        """
        self.core.reset_state()
        actions = []

        for t in range(n_steps):
            readout = self.core.get_readout()
            action = np.tanh(readout)
            actions.append(action)
            self.core.step()

        return np.array(actions, dtype=np.float32)

    def evaluate_on_oned_task(
        self, n_steps: int, noise_scale: float = 0.1, action_scale: float = 0.1
    ) -> float:
        """
        End-to-end: generate actions, roll out in OneDControlTask, return total reward.
        """
        from envs.oned_control import OneDControlTask
        
        actions = self.rollout_actions(n_steps)
        env = OneDControlTask(noise_scale=noise_scale, action_scale=action_scale)
        total_reward = env.rollout(actions, init_x=1.0)
        return total_reward


def make_policy_from_connectivity(
    connectivity: np.ndarray,
    obs_dim: int = 1,
    action_dim: int = 1,
    weight_scale: float = 0.1,
    use_lif: bool = False,
    rng: np.random.Generator | None = None,
    **lif_kwargs,
) -> RSNNPolicy:
    """
    Factory function: create a policy from a binary connectivity matrix.
    
    Args:
        connectivity: (N, N) binary matrix (0/1)
        obs_dim: dimension of observations
        action_dim: dimension of actions
        weight_scale: scale for random weight initialization
        use_lif: whether to use LIF neurons
        rng: random generator for weight init
    
    Returns:
        RSNNPolicy instance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = connectivity.shape[0]
    
    # Recurrent weights: connectivity * random weights
    W_rec = connectivity * weight_scale * rng.standard_normal((N, N)).astype(np.float32)
    
    # Input weights
    W_in = weight_scale * rng.standard_normal((N, obs_dim)).astype(np.float32)
    
    # Output weights
    W_out = weight_scale * rng.standard_normal((action_dim, N)).astype(np.float32)
    
    return RSNNPolicy(W_rec, W_in, W_out, use_lif=use_lif, **lif_kwargs)


def make_policy_from_P(
    P: np.ndarray,
    obs_dim: int = 1,
    action_dim: int = 1,
    weight_scale: float = 0.5,
    use_lif: bool = False,
    rng: np.random.Generator | None = None,
    **lif_kwargs,
) -> RSNNPolicy:
    """
    Factory function: use matrix P directly as recurrent weights.
    
    This is the "P as W" variant from your thesis notes:
    Instead of sampling binary connectivity from P, use P itself as weight magnitudes.
    
    Args:
        P: (N, N) weight matrix (can be any real values)
        obs_dim: dimension of observations
        action_dim: dimension of actions
        weight_scale: scale for input/output weight initialization
        use_lif: whether to use LIF neurons
        rng: random generator
    
    Returns:
        RSNNPolicy instance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = P.shape[0]
    
    # Use P directly as recurrent weights (already evolved)
    W_rec = P.astype(np.float32)
    
    # Input and output weights still random but with consistent scale
    W_in = weight_scale * rng.standard_normal((N, obs_dim)).astype(np.float32)
    W_out = weight_scale * rng.standard_normal((action_dim, N)).astype(np.float32)
    
    return RSNNPolicy(W_rec, W_in, W_out, use_lif=use_lif, **lif_kwargs)
