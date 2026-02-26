# models/stdp.py
"""
Spike-Timing-Dependent Plasticity (STDP)

Online, within-trial synaptic plasticity based on relative spike timing
of pre- and post-synaptic neurons.

Rule:
  Δw_ij = A+ * exp(-Δt / τ+)   if pre before post  (Δt > 0, LTP)
  Δw_ij = -A- * exp(+Δt / τ-)  if post before pre  (Δt < 0, LTD)

Implementation uses eligibility traces (efficient, no need to store all spike times):
  x_pre[t]  = x_pre[t-1] * exp(-1/τ+) + s_pre[t]    (pre-synaptic trace)
  x_post[t] = x_post[t-1] * exp(-1/τ-) + s_post[t]   (post-synaptic trace)

  Δw[t] = A+ * s_post[t] * x_pre[t]   (post spike → LTP using pre trace)
         - A- * s_pre[t] * x_post[t]   (pre spike → LTD using post trace)

This is applied to W_rec only (recurrent synapses).
Respects Dale's law: weight updates clamped to preserve sign.
Weight bounds prevent runaway growth.

References:
  - Bi & Poo (1998): original STDP measurements
  - Song et al. (2000): competitive STDP dynamics
  - Zenke & Ganguli (2018): SuperSpike, surrogate + plasticity
  - eLife (2021): evolving interpretable plasticity for spiking networks
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class STDP_Rule:
    """
    STDP rule operating on W_rec of a LIF network.
    NumPy implementation for use with GA.

    Parameters can be evolved by the GA:
      A_plus, A_minus: learning rates for potentiation / depression
      tau_plus, tau_minus: time constants (in timesteps)
      w_max, w_min: weight bounds
    """

    def __init__(self, n_neurons,
                 A_plus=0.01, A_minus=0.012,
                 tau_plus=20.0, tau_minus=20.0,
                 w_max=3.0, w_min=-3.0,
                 dale_mask=None):
        """
        Args:
            n_neurons: number of recurrent neurons
            A_plus: LTP amplitude
            A_minus: LTD amplitude (slightly larger → net depression for stability)
            tau_plus: pre-synaptic trace decay (timesteps)
            tau_minus: post-synaptic trace decay (timesteps)
            w_max: upper weight bound
            w_min: lower weight bound
            dale_mask: (N,) +1/-1 for excitatory/inhibitory
        """
        self.N = n_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        self.dale_mask = dale_mask

        # Decay factors
        self.decay_pre = np.exp(-1.0 / tau_plus)
        self.decay_post = np.exp(-1.0 / tau_minus)

        self.reset()

    def reset(self):
        """Reset traces at the start of a trial."""
        self.x_pre = np.zeros(self.N, dtype=np.float32)   # pre-synaptic trace
        self.x_post = np.zeros(self.N, dtype=np.float32)  # post-synaptic trace

    def update(self, W_rec, s_pre, s_post):
        """
        One STDP update step.

        Args:
            W_rec: (N, N) current recurrent weights (modified in-place)
            s_pre:  (N,) pre-synaptic spikes at this timestep
            s_post: (N,) post-synaptic spikes (same as s_pre for recurrent)

        For recurrent networks, s_pre = s_post = s[t] (every neuron
        is both pre- and post-synaptic to other neurons).

        Returns:
            W_rec: updated weights
        """
        # Decay traces
        self.x_pre = self.x_pre * self.decay_pre + s_pre
        self.x_post = self.x_post * self.decay_post + s_post

        # LTP: post-synaptic spike → potentiate based on pre trace
        # Δw_ij += A+ * s_post_i * x_pre_j
        if np.any(s_post > 0):
            dW_ltp = self.A_plus * np.outer(s_post, self.x_pre)
            W_rec += dW_ltp

        # LTD: pre-synaptic spike → depress based on post trace
        # Δw_ij -= A- * s_pre_j * x_post_i
        if np.any(s_pre > 0):
            dW_ltd = self.A_minus * np.outer(self.x_post, s_pre)
            W_rec -= dW_ltd

        # Enforce Dale's law
        if self.dale_mask is not None:
            for j in range(self.N):
                if self.dale_mask[j] > 0:  # excitatory
                    W_rec[:, j] = np.clip(W_rec[:, j], 0, self.w_max)
                else:  # inhibitory
                    W_rec[:, j] = np.clip(W_rec[:, j], self.w_min, 0)
        else:
            np.clip(W_rec, self.w_min, self.w_max, out=W_rec)

        return W_rec

    def get_params(self):
        """Return STDP parameters as a flat array (for GA encoding)."""
        return np.array([
            self.A_plus, self.A_minus,
            self.tau_plus, self.tau_minus,
            self.w_max, self.w_min,
        ], dtype=np.float32)

    @classmethod
    def from_params(cls, n_neurons, params, dale_mask=None):
        """Create STDP rule from a flat parameter array."""
        return cls(
            n_neurons=n_neurons,
            A_plus=float(np.clip(params[0], 1e-5, 0.1)),
            A_minus=float(np.clip(params[1], 1e-5, 0.1)),
            tau_plus=float(np.clip(params[2], 2.0, 50.0)),
            tau_minus=float(np.clip(params[3], 2.0, 50.0)),
            w_max=float(np.clip(params[4], 0.5, 10.0)),
            w_min=float(np.clip(params[5], -10.0, -0.5)),
            dale_mask=dale_mask,
        )

    N_PARAMS = 6  # number of evolvable STDP parameters


# ============================================================================
# PyTorch STDP (for BPTT+STDP condition)
# ============================================================================

if TORCH_AVAILABLE:

    class STDP_Rule_Torch:
        """
        STDP rule in PyTorch. Used during BPTT+STDP condition.
        Applied as a non-gradient online update within episodes.

        During BPTT+STDP:
          - BPTT optimizes base weights between episodes
          - STDP modifies weights within episodes (on top of BPTT weights)
          - STDP changes are discarded after the episode
        """

        def __init__(self, n_neurons,
                     A_plus=0.01, A_minus=0.012,
                     tau_plus=20.0, tau_minus=20.0,
                     w_max=3.0, w_min=-3.0,
                     dale_mask=None):
            self.N = n_neurons
            self.A_plus = A_plus
            self.A_minus = A_minus
            self.decay_pre = np.exp(-1.0 / tau_plus)
            self.decay_post = np.exp(-1.0 / tau_minus)
            self.w_max = w_max
            self.w_min = w_min
            self.dale_mask = dale_mask  # torch tensor or None
            self.reset()

        def reset(self):
            self.x_pre = None
            self.x_post = None

        @torch.no_grad()
        def update(self, W_rec_delta, spikes):
            """
            Update the STDP weight delta.

            Args:
                W_rec_delta: (N, N) tensor — accumulated STDP changes
                spikes: (B, N) or (N,) current spikes

            Returns:
                W_rec_delta: updated delta
            """
            if spikes.dim() > 1:
                s = spikes.mean(dim=0)  # average across batch
            else:
                s = spikes

            if self.x_pre is None:
                self.x_pre = torch.zeros_like(s)
                self.x_post = torch.zeros_like(s)

            # Decay traces
            self.x_pre = self.x_pre * self.decay_pre + s
            self.x_post = self.x_post * self.decay_post + s

            # LTP
            dW_ltp = self.A_plus * torch.outer(s, self.x_pre)
            W_rec_delta += dW_ltp

            # LTD
            dW_ltd = self.A_minus * torch.outer(self.x_post, s)
            W_rec_delta -= dW_ltd

            # Clamp
            W_rec_delta.clamp_(self.w_min, self.w_max)

            return W_rec_delta
