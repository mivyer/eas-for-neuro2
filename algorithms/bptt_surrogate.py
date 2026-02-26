import torch
import torch.nn as nn


class TinyRNNPolicy(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.n_neurons = n_neurons
        # Recurrent weights
        self.W_rec = nn.Parameter(0.1 * torch.randn(n_neurons, n_neurons))
        # Initial state (learnable parameter)
        self.h0 = nn.Parameter(torch.zeros(n_neurons))

    def forward(self, n_steps):
        """
        Full BPTT-ready forward pass.
        PyTorch autograd automatically builds the computation graph through the entire loop.
        loss.backward() will backprop through all timesteps.
        """
        h = self.h0
        actions = []
        for t in range(n_steps):
            # Recurrent dynamics: h = tanh(W_rec @ h)
            h = torch.tanh(self.W_rec @ h)
            # Simple readout: average first half of neurons
            readout = h[:self.n_neurons // 2].mean()
            action = torch.tanh(readout)  # scalar action in [-1, 1]
            actions.append(action)
        return torch.stack(actions)  # shape: (n_steps,)
