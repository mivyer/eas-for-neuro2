# models/lif_rsnn.py
"""
Leaky Integrate-and-Fire Recurrent Spiking Neural Network

Two implementations:
  1. LIF_RSNN_NP  — NumPy, for GA/EA evaluation (no gradients)
  2. LIF_RSNN_Torch — PyTorch, for BPTT with surrogate gradients

Architecture (matching thesis spec):
  - N recurrent LIF neurons (default 32, spec says 256 for final)
  - 80/20 excitatory/inhibitory split, Dale's law enforced
  - Input: linear projection → synaptic current
  - Output: linear readout from membrane potentials (smoothed spike rates)
  - Refractory period after spiking

LIF dynamics:
  v[t] = β * v[t-1] + W_in @ x[t] + W_rec @ s[t-1] - v_reset * s[t-1]
  s[t] = Θ(v[t] - v_thresh)  (spike if above threshold)
  If s[t]=1: refractory counter set, v reset

Output:
  y[t] = W_out @ v[t]  (readout from membrane potentials)
  Using membrane potential gives a smooth, continuous output.
  Alternative: readout from low-pass filtered spikes.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# NumPy LIF (for GA / EA)
# ============================================================================

class LIF_RSNN_NP:
    """
    LIF spiking RNN in NumPy. No autograd.
    Used by GA and EA for fitness evaluation.
    """

    def __init__(self, W_rec, W_in, W_out,
                 beta=0.85, threshold=1.0, refractory_steps=2,
                 dale_mask=None):
        """
        Args:
            W_rec: (N, N) recurrent weights (pre-masked for Dale's law)
            W_in:  (N, obs_dim)
            W_out: (action_dim, N)
            beta:  membrane decay (0 < beta < 1). Higher = slower leak.
            threshold: spike threshold
            refractory_steps: timesteps of refractoriness after spike
            dale_mask: (N,) array of +1 (excitatory) or -1 (inhibitory).
                       If provided, enforces sign constraints on W_rec columns.
        """
        self.N = W_rec.shape[0]
        self.W_rec = W_rec.astype(np.float32)
        self.W_in = W_in.astype(np.float32)
        self.W_out = W_out.astype(np.float32)
        self.beta = beta
        self.threshold = threshold
        self.refractory_steps = refractory_steps
        self.dale_mask = dale_mask

        if dale_mask is not None:
            self._enforce_dale()

        self.reset()

    def _enforce_dale(self):
        """Enforce Dale's law: excitatory neurons have non-negative outgoing weights,
        inhibitory neurons have non-positive outgoing weights."""
        for j in range(self.N):
            if self.dale_mask[j] > 0:  # excitatory
                self.W_rec[:, j] = np.abs(self.W_rec[:, j])
            else:  # inhibitory
                self.W_rec[:, j] = -np.abs(self.W_rec[:, j])

    def reset(self):
        self.v = np.zeros(self.N, dtype=np.float32)       # membrane potential
        self.s = np.zeros(self.N, dtype=np.float32)        # spikes
        self.refrac = np.zeros(self.N, dtype=np.float32)   # refractory counter
        self.spike_history = []

    def step(self, x):
        """
        Single timestep.
        Args:
            x: (obs_dim,) input
        Returns:
            y: (action_dim,) output
        """
        # Synaptic input
        I_in = self.W_in @ x
        I_rec = self.W_rec @ self.s

        # Membrane update with leak
        self.v = self.beta * self.v + I_in + I_rec

        # Reset where spiked last step
        self.v -= self.threshold * self.s

        # Refractory: decrement counter, mask spikes
        self.refrac = np.maximum(self.refrac - 1, 0)
        can_spike = (self.refrac == 0).astype(np.float32)

        # Spike
        self.s = ((self.v >= self.threshold) * can_spike).astype(np.float32)

        # Set refractory for neurons that spiked
        self.refrac[self.s > 0] = self.refractory_steps

        # Record
        self.spike_history.append(self.s.copy())

        # Output: readout from membrane potential
        y = self.W_out @ self.v
        return y

    def act(self, obs):
        """Compatible with RSNNPolicy interface."""
        y = self.step(obs.flatten())
        return y

    def get_spikes(self):
        """Return spike history as (T, N) array."""
        if len(self.spike_history) == 0:
            return np.zeros((0, self.N))
        return np.stack(self.spike_history, axis=0)


# ============================================================================
# PyTorch LIF (for BPTT with surrogate gradients)
# ============================================================================

if TORCH_AVAILABLE:

    class SurrogateSpike(torch.autograd.Function):
        """
        Surrogate gradient for the Heaviside spike function.
        Forward: Θ(v - threshold)
        Backward: sigmoid derivative scaled by temperature.

        This is the standard approach (Neftci et al. 2019, Zenke & Ganguli 2018).
        """
        @staticmethod
        def forward(ctx, v, threshold, temperature=1.0):
            ctx.save_for_backward(v, torch.tensor(threshold), torch.tensor(temperature))
            return (v >= threshold).float()

        @staticmethod
        def backward(ctx, grad_output):
            v, threshold, temperature = ctx.saved_tensors
            # Fast sigmoid surrogate
            x = (v - threshold) / temperature
            grad = grad_output * (1 / (1 + torch.abs(x))**2) / temperature
            return grad, None, None

    def surrogate_spike(v, threshold=1.0, temperature=0.5):
        return SurrogateSpike.apply(v, threshold, temperature)


    class LIF_RSNN_Torch(nn.Module):
        """
        LIF spiking RNN in PyTorch with surrogate gradients.
        For BPTT training.

        Architecture:
          - N LIF neurons with E/I split
          - Dale's law enforced via parameterizing W_rec = |W_raw| * dale_mask
          - Surrogate gradient on spike function for backprop
        """

        def __init__(self, n_neurons=32, obs_dim=1, action_dim=1,
                     beta=0.85, threshold=1.0, temperature=0.5,
                     refractory_steps=2, ei_ratio=0.8, dale=True):
            super().__init__()

            self.N = n_neurons
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.beta = beta
            self.threshold = threshold
            self.temperature = temperature
            self.refractory_steps = refractory_steps
            self.dale = dale

            # E/I split
            self.n_exc = int(ei_ratio * n_neurons)
            self.n_inh = n_neurons - self.n_exc

            # Dale's law mask: +1 for excitatory columns, -1 for inhibitory
            dale_mask = torch.ones(n_neurons)
            dale_mask[self.n_exc:] = -1.0
            self.register_buffer('dale_mask', dale_mask)

            # Learnable raw weights (sign enforced via dale_mask)
            scale = np.sqrt(2.0 / n_neurons)
            self.W_rec_raw = nn.Parameter(scale * torch.randn(n_neurons, n_neurons))
            self.W_in = nn.Parameter(scale * torch.randn(n_neurons, obs_dim))
            self.W_out = nn.Parameter(scale * torch.randn(action_dim, n_neurons))

            # Initial hidden state
            self.register_buffer('v0', torch.zeros(n_neurons))

        @property
        def W_rec(self):
            """Effective recurrent weights with Dale's law."""
            if self.dale:
                return torch.abs(self.W_rec_raw) * self.dale_mask.unsqueeze(0)
            return self.W_rec_raw

        def forward(self, inputs):
            """
            Args:
                inputs: (T, batch, obs_dim) or (T, obs_dim)
            Returns:
                outputs: (T, batch, action_dim) or (T, action_dim)
                spikes:  (T, batch, N) or (T, N)
            """
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)  # (T, 1, obs_dim)
                squeeze = True
            else:
                squeeze = False

            T, B, _ = inputs.shape
            device = inputs.device

            v = self.v0.unsqueeze(0).expand(B, -1).clone()  # (B, N)
            s = torch.zeros(B, self.N, device=device)
            refrac = torch.zeros(B, self.N, device=device)

            W_rec = self.W_rec

            outputs = []
            spikes = []

            for t in range(T):
                x = inputs[t]  # (B, obs_dim)

                # Synaptic input
                I_in = x @ self.W_in.T        # (B, N)
                I_rec = s @ W_rec.T            # (B, N)

                # Membrane update
                v = self.beta * v + I_in + I_rec

                # Reset spiked neurons
                v = v - self.threshold * s

                # Refractory
                refrac = torch.clamp(refrac - 1, min=0)
                can_spike = (refrac == 0).float()

                # Spike with surrogate gradient
                s = surrogate_spike(v, self.threshold, self.temperature) * can_spike

                # Set refractory
                refrac = refrac + s * self.refractory_steps

                # Output from membrane potential
                y = v @ self.W_out.T  # (B, action_dim)

                outputs.append(y)
                spikes.append(s)

            outputs = torch.stack(outputs, dim=0)  # (T, B, action_dim)
            spikes = torch.stack(spikes, dim=0)    # (T, B, N)

            if squeeze:
                outputs = outputs.squeeze(1)
                spikes = spikes.squeeze(1)

            return outputs, spikes


# ============================================================================
# Helper: create Dale's law mask
# ============================================================================

def make_dale_mask(n_neurons, ei_ratio=0.8):
    """Returns (N,) array: +1 for excitatory, -1 for inhibitory."""
    n_exc = int(ei_ratio * n_neurons)
    mask = np.ones(n_neurons, dtype=np.float32)
    mask[n_exc:] = -1.0
    return mask


def enforce_dale_weights(W_rec, dale_mask):
    """Enforce Dale's law on a weight matrix in-place."""
    W = W_rec.copy()
    for j in range(len(dale_mask)):
        if dale_mask[j] > 0:
            W[:, j] = np.abs(W[:, j])
        else:
            W[:, j] = -np.abs(W[:, j])
    return W
