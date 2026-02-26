# models/rsnn_core.py
"""
RSNN cores for EA and BPTT experiments.
Provides both rate-based (SimpleRSNNCore) and spiking (LIFCore) implementations.
"""

import numpy as np


class SimpleRSNNCore:
    """
    Minimal RNN core: h_{t+1} = tanh(W_rec @ h_t + W_in @ obs)
    
    This is a rate-based placeholder. Replace with LIF dynamics for spiking experiments.
    """
    
    def __init__(self, W_rec: np.ndarray, W_in: np.ndarray | None = None):
        """
        Args:
            W_rec: (N, N) recurrent weight matrix
            W_in: (N, obs_dim) input weight matrix, or None for autonomous dynamics
        """
        self.N = W_rec.shape[0]
        self.W_rec = W_rec.astype(np.float32)
        self.W_in = W_in.astype(np.float32) if W_in is not None else None
        self.reset_state()
    
    def reset_state(self):
        """Reset hidden state to zeros."""
        self.h = np.zeros(self.N, dtype=np.float32)
    
    def step(self, obs: np.ndarray | None = None) -> np.ndarray:
        """
        One step of recurrent dynamics.
        
        Args:
            obs: (obs_dim,) observation vector, or None
        
        Returns:
            h: (N,) new hidden state
        """
        # Recurrent input
        rec_input = self.W_rec @ self.h
        
        # External input
        ext_input = np.zeros(self.N, dtype=np.float32)
        if self.W_in is not None and obs is not None:
            ext_input = self.W_in @ obs.astype(np.float32)
        
        # Update state
        self.h = np.tanh(rec_input + ext_input)
        
        return self.h
    
    def get_state(self) -> np.ndarray:
        """Return current hidden state."""
        return self.h.copy()
    
    def get_readout(self) -> float:
        """Simple readout: mean of first half of units."""
        readout = self.h[: self.N // 2].mean()
        return float(readout)


class LIFCore:
    """
    Leaky Integrate-and-Fire RSNN core.
    
    Membrane dynamics: v_{t+1} = beta * v_t + W_rec @ s_t + W_in @ obs - v_th * s_t
    Spike: s_t = (v_t > v_th)
    
    For now, uses a simple threshold. Add surrogate gradients for BPTT version.
    """
    
    def __init__(
        self,
        W_rec: np.ndarray,
        W_in: np.ndarray | None = None,
        beta: float = 0.9,      # Membrane decay
        v_th: float = 1.0,      # Spike threshold
    ):
        self.N = W_rec.shape[0]
        self.W_rec = W_rec.astype(np.float32)
        self.W_in = W_in.astype(np.float32) if W_in is not None else None
        self.beta = beta
        self.v_th = v_th
        self.reset_state()
    
    def reset_state(self):
        """Reset membrane potential and spikes."""
        self.v = np.zeros(self.N, dtype=np.float32)  # Membrane potential
        self.s = np.zeros(self.N, dtype=np.float32)  # Spikes (0 or 1)
        self.h = np.zeros(self.N, dtype=np.float32)  # Compatibility with SimpleRSNNCore
    
    def step(self, obs: np.ndarray | None = None) -> np.ndarray:
        """
        One step of LIF dynamics.
        
        Returns:
            s: (N,) spike vector
        """
        # Recurrent input from previous spikes
        rec_input = self.W_rec @ self.s
        
        # External input
        ext_input = np.zeros(self.N, dtype=np.float32)
        if self.W_in is not None and obs is not None:
            ext_input = self.W_in @ obs.astype(np.float32)
        
        # Membrane update (with reset)
        self.v = self.beta * self.v + rec_input + ext_input - self.v_th * self.s
        
        # Spike generation
        self.s = (self.v > self.v_th).astype(np.float32)
        
        # Update h for compatibility
        self.h = self.v.copy()
        
        return self.s
    
    def get_state(self) -> np.ndarray:
        """Return membrane potential (compatible with SimpleRSNNCore interface)."""
        return self.v.copy()
    
    def get_readout(self) -> float:
        """Simple readout: mean membrane potential of first half of units."""
        readout = self.v[: self.N // 2].mean()
        return float(readout)
    
    def get_spikes(self) -> np.ndarray:
        """Return current spike state."""
        return self.s.copy()
