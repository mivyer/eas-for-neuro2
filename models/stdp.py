# models/stdp.py
"""
Spike-Timing-Dependent Plasticity — Three Variants

1. STDP_Rule        — plain unsupervised STDP (baseline)
2. RewardSTDP       — reward-modulated STDP (R-STDP, Izhikevich 2007)
3. STDP_Rule_Torch  — PyTorch R-STDP for BPTT+STDP

R-STDP (three-factor rule):
  e_ij[t] = γ * e_ij[t-1] + STDP(pre_i, post_j)   (eligibility trace)
  Δw_ij   = η * reward * e_ij                        (reward-gated update)

References:
  - Izhikevich 2007: STDP + dopamine solves distal reward problem
  - Frémaux & Gerstner 2016: three-factor learning rules
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RewardSTDP:
    """
    Three-factor rule: pre × post × reward.
    Spike timing creates eligibility traces; reward converts them to weight changes.

    8 evolvable parameters:
      A_plus, A_minus, tau_plus, tau_minus, tau_e, eta, w_max, w_min
    """

    def __init__(self, n_neurons,
                 A_plus=0.005, A_minus=0.005,
                 tau_plus=20.0, tau_minus=20.0,
                 tau_e=25.0, eta=0.01,
                 w_max=3.0, w_min=-3.0,
                 dale_mask=None):
        self.N = n_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_e = tau_e
        self.eta = eta
        self.w_max = w_max
        self.w_min = w_min
        self.dale_mask = dale_mask
        self.decay_pre = np.exp(-1.0 / max(tau_plus, 1.0))
        self.decay_post = np.exp(-1.0 / max(tau_minus, 1.0))
        self.decay_e = np.exp(-1.0 / max(tau_e, 1.0))
        self.reset()

    def reset(self):
        self.x_pre = np.zeros(self.N, dtype=np.float32)
        self.x_post = np.zeros(self.N, dtype=np.float32)
        self.eligibility = np.zeros((self.N, self.N), dtype=np.float32)

    def update_traces(self, spikes):
        """Update eligibility traces from current spikes. Call every timestep."""
        self.x_pre = self.x_pre * self.decay_pre + spikes
        self.x_post = self.x_post * self.decay_post + spikes
        dE_ltp = self.A_plus * np.outer(spikes, self.x_pre)
        dE_ltd = self.A_minus * np.outer(self.x_post, spikes)
        self.eligibility = self.eligibility * self.decay_e + dE_ltp - dE_ltd

    def apply_reward(self, W_rec, reward):
        """Apply reward-modulated weight update. Call after reward is known."""
        dW = self.eta * reward * self.eligibility
        W_rec = W_rec + dW
        if self.dale_mask is not None:
            for j in range(self.N):
                if self.dale_mask[j] > 0:
                    W_rec[:, j] = np.clip(W_rec[:, j], 0, self.w_max)
                else:
                    W_rec[:, j] = np.clip(W_rec[:, j], self.w_min, 0)
        else:
            np.clip(W_rec, self.w_min, self.w_max, out=W_rec)
        return W_rec

    def get_params(self):
        return np.array([self.A_plus, self.A_minus, self.tau_plus, self.tau_minus,
                         self.tau_e, self.eta, self.w_max, self.w_min], dtype=np.float32)

    @classmethod
    def from_params(cls, n_neurons, params, dale_mask=None):
        return cls(
            n_neurons=n_neurons,
            A_plus=float(np.clip(np.abs(params[0]), 1e-4, 0.05)),
            A_minus=float(np.clip(np.abs(params[1]), 1e-4, 0.05)),
            tau_plus=float(np.clip(params[2], 5.0, 50.0)),
            tau_minus=float(np.clip(params[3], 5.0, 50.0)),
            tau_e=float(np.clip(params[4], 5.0, 100.0)),
            eta=float(np.clip(np.abs(params[5]), 1e-4, 0.1)),
            w_max=float(np.clip(params[6], 0.5, 10.0)),
            w_min=float(np.clip(params[7], -10.0, -0.5)),
            dale_mask=dale_mask,
        )

    N_PARAMS = 8


class STDP_Rule:
    """Plain unsupervised STDP. Baseline only."""

    def __init__(self, n_neurons, A_plus=0.01, A_minus=0.012,
                 tau_plus=20.0, tau_minus=20.0,
                 w_max=3.0, w_min=-3.0, dale_mask=None):
        self.N = n_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.w_max = w_max
        self.w_min = w_min
        self.dale_mask = dale_mask
        self.decay_pre = np.exp(-1.0 / tau_plus)
        self.decay_post = np.exp(-1.0 / tau_minus)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.reset()

    def reset(self):
        self.x_pre = np.zeros(self.N, dtype=np.float32)
        self.x_post = np.zeros(self.N, dtype=np.float32)

    def update(self, W_rec, s_pre, s_post):
        self.x_pre = self.x_pre * self.decay_pre + s_pre
        self.x_post = self.x_post * self.decay_post + s_post
        if np.any(s_post > 0):
            W_rec += self.A_plus * np.outer(s_post, self.x_pre)
        if np.any(s_pre > 0):
            W_rec -= self.A_minus * np.outer(self.x_post, s_pre)
        if self.dale_mask is not None:
            for j in range(self.N):
                if self.dale_mask[j] > 0:
                    W_rec[:, j] = np.clip(W_rec[:, j], 0, self.w_max)
                else:
                    W_rec[:, j] = np.clip(W_rec[:, j], self.w_min, 0)
        else:
            np.clip(W_rec, self.w_min, self.w_max, out=W_rec)
        return W_rec

    def get_params(self):
        return np.array([self.A_plus, self.A_minus, self.tau_plus, self.tau_minus,
                         self.w_max, self.w_min], dtype=np.float32)

    N_PARAMS = 6