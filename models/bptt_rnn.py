"""PyTorch RNN and LIF policies for BPTT training."""

import torch
import torch.nn as nn


class RNNPolicy(nn.Module):
    """
    h_t = tanh(W_rec @ h_{t-1} + W_in @ obs_t)
    action_t = W_out @ h_t  (raw logits)
    """

    def __init__(self, n_neurons, obs_dim=1, action_dim=1, weight_scale=0.1):
        super().__init__()
        self.n_neurons = n_neurons
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.W_rec = nn.Parameter(weight_scale * torch.randn(n_neurons, n_neurons))
        self.W_in  = nn.Parameter(weight_scale * torch.randn(n_neurons, obs_dim))
        self.W_out = nn.Parameter(weight_scale * torch.randn(action_dim, n_neurons))
        self.h0    = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)

        batch_size, seq_len, _ = inputs.shape
        h = self.h0.unsqueeze(0).expand(batch_size, -1)

        outputs = []
        for t in range(seq_len):
            h = torch.tanh(h @ self.W_rec.T + inputs[:, t, :] @ self.W_in.T)
            outputs.append(h @ self.W_out.T)

        outputs = torch.stack(outputs, dim=1)
        if self.action_dim == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def get_connectivity_stats(self):
        with torch.no_grad():
            W = self.W_rec.cpu().numpy()
            return {
                'mean': float(W.mean()),
                'std': float(W.std()),
                'sparsity': float((abs(W) < 0.01).mean()),
                'max_abs': float(abs(W).max()),
            }


class LIFPolicy(nn.Module):
    """
    LIF with surrogate gradients (piecewise-linear).
    v_t = beta * v_{t-1} + W_rec @ s_{t-1} + W_in @ obs_t - v_th * s_{t-1}
    s_t = surrogate(v_t - v_th)
    """

    def __init__(self, n_neurons, obs_dim=1, action_dim=1, weight_scale=0.1,
                 beta=0.9, v_th=1.0, surrogate_scale=10.0):
        super().__init__()
        self.n_neurons = n_neurons
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.beta = beta
        self.v_th = v_th
        self.surrogate_scale = surrogate_scale

        self.W_rec = nn.Parameter(weight_scale * torch.randn(n_neurons, n_neurons))
        self.W_in  = nn.Parameter(weight_scale * torch.randn(n_neurons, obs_dim))
        self.W_out = nn.Parameter(weight_scale * torch.randn(action_dim, n_neurons))

    def surrogate_spike(self, v):
        spikes = (v > self.v_th).float()
        surrogate_grad = torch.clamp(
            1.0 - self.surrogate_scale * torch.abs(v - self.v_th), min=0.0
        )
        return spikes + (surrogate_grad - surrogate_grad.detach())

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)

        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        v = torch.zeros(batch_size, self.n_neurons, device=device)
        s = torch.zeros(batch_size, self.n_neurons, device=device)

        outputs = []
        for t in range(seq_len):
            v = self.beta * v + s @ self.W_rec.T + inputs[:, t, :] @ self.W_in.T - self.v_th * s
            s = self.surrogate_spike(v)
            outputs.append(torch.tanh(v @ self.W_out.T))

        outputs = torch.stack(outputs, dim=1)
        if self.action_dim == 1:
            outputs = outputs.squeeze(-1)
        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
