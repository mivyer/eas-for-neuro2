# models/bptt_rnn.py
"""
PyTorch RNN for BPTT training on working memory task.
Mirrors the architecture of the NumPy RSNN for fair comparison.
"""

import torch
import torch.nn as nn


class RNNPolicy(nn.Module):
    """
    Simple RNN policy trainable with BPTT.
    
    Architecture matches the NumPy RSNNPolicy:
        h_t = tanh(W_rec @ h_{t-1} + W_in @ obs_t)
        action_t = tanh(W_out @ h_t)
    """
    
    def __init__(
        self,
        n_neurons: int,
        obs_dim: int = 1,
        action_dim: int = 1,
        weight_scale: float = 0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Learnable weights
        self.W_rec = nn.Parameter(weight_scale * torch.randn(n_neurons, n_neurons))
        self.W_in = nn.Parameter(weight_scale * torch.randn(n_neurons, obs_dim))
        self.W_out = nn.Parameter(weight_scale * torch.randn(action_dim, n_neurons))
        
        # Learnable initial state (optional, can set to zeros)
        self.h0 = nn.Parameter(torch.zeros(n_neurons))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through entire sequence.
        
        Args:
            inputs: (batch_size, seq_len) or (batch_size, seq_len, obs_dim)
        
        Returns:
            outputs: (batch_size, seq_len) actions at each timestep
        """
        # Handle 2D input (batch, seq) -> (batch, seq, 1)
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        # Initialize hidden state
        h = self.h0.unsqueeze(0).expand(batch_size, -1)  # (batch, n_neurons)
        
        outputs = []
        for t in range(seq_len):
            obs = inputs[:, t, :]  # (batch, obs_dim)
            
            # RNN dynamics
            h = torch.tanh(h @ self.W_rec.T + obs @ self.W_in.T)
            
            # Readout
            action = torch.tanh(h @ self.W_out.T)  # (batch, action_dim)
            outputs.append(action)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, action_dim)
        
        # Squeeze if action_dim == 1
        if self.action_dim == 1:
            outputs = outputs.squeeze(-1)  # (batch, seq_len)
        
        return outputs
    
    def get_connectivity_stats(self) -> dict:
        """Return statistics about learned weights for analysis."""
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
    LIF-based policy with surrogate gradients for BPTT.
    
    Membrane dynamics:
        v_t = beta * v_{t-1} + W_rec @ s_{t-1} + W_in @ obs_t - v_th * s_{t-1}
        s_t = surrogate(v_t - v_th)
    
    Uses a piecewise-linear surrogate gradient for the spike function.
    """
    
    def __init__(
        self,
        n_neurons: int,
        obs_dim: int = 1,
        action_dim: int = 1,
        weight_scale: float = 0.1,
        beta: float = 0.9,
        v_th: float = 1.0,
        surrogate_scale: float = 10.0,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.beta = beta
        self.v_th = v_th
        self.surrogate_scale = surrogate_scale
        
        # Learnable weights
        self.W_rec = nn.Parameter(weight_scale * torch.randn(n_neurons, n_neurons))
        self.W_in = nn.Parameter(weight_scale * torch.randn(n_neurons, obs_dim))
        self.W_out = nn.Parameter(weight_scale * torch.randn(action_dim, n_neurons))
    
    def surrogate_spike(self, v: torch.Tensor) -> torch.Tensor:
        """
        Spike function with surrogate gradient.
        
        Forward: hard threshold
        Backward: piecewise linear (triangular) surrogate
        """
        # Forward pass: hard threshold
        spikes = (v > self.v_th).float()
        
        # Surrogate gradient: triangular function centered at threshold
        # d_spike/d_v ≈ max(0, 1 - |v - v_th| * scale)
        surrogate_grad = torch.clamp(
            1.0 - self.surrogate_scale * torch.abs(v - self.v_th),
            min=0.0
        )
        
        # Straight-through estimator: use surrogate grad in backward pass
        return spikes + (surrogate_grad - surrogate_grad.detach())
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through entire sequence.
        
        Args:
            inputs: (batch_size, seq_len) or (batch_size, seq_len, obs_dim)
        
        Returns:
            outputs: (batch_size, seq_len) actions at each timestep
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(-1)
        
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        # Initialize state
        v = torch.zeros(batch_size, self.n_neurons, device=device)
        s = torch.zeros(batch_size, self.n_neurons, device=device)
        
        outputs = []
        for t in range(seq_len):
            obs = inputs[:, t, :]  # (batch, obs_dim)
            
            # LIF dynamics
            v = self.beta * v + s @ self.W_rec.T + obs @ self.W_in.T - self.v_th * s
            s = self.surrogate_spike(v)
            
            # Readout from membrane potential (smoother than spikes)
            action = torch.tanh(v @ self.W_out.T)
            outputs.append(action)
        
        outputs = torch.stack(outputs, dim=1)
        
        if self.action_dim == 1:
            outputs = outputs.squeeze(-1)
        
        return outputs


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
