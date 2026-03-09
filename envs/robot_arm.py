# envs/robot_arm.py
"""
2-Joint Robot Arm Endpoint Prediction Task

A 2-link planar arm with unit-length links. Joint angular velocities (omega0,
omega1) are generated as sums of sinusoids. The network must predict the
endpoint (x, y) position at each timestep.

Unlike classification tasks, this requires continuous regression: the loss is
MSE, not cross-entropy. The task is scored across ALL timesteps (no response
window).

Input:  (T, 2) float32 — normalized angular velocities [omega0, omega1]
Target: (T, 2) float32 — normalized endpoint positions [x, y] in [-1, 1]

Fitness  = -MSE  (range: (-inf, 0], higher is better)
Accuracy = exp(-MSE)  (range: (0, 1], displayed as percentage)

Trajectory generation adapted from:
  Zhu Spike Lab / robot task — RobotTrajectories class
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_SINES = 5


def _generate_single_trajectory(rng, seq_length, n_periods, dt_step):
    """
    Generate one robot arm trial.

    Returns:
        inputs:  (T, 2) float32 — per-trial normalized (omega0, omega1) in [-1, 1]
        targets: (T, 2) float32 — normalized (x, y) endpoint in [-1, 1]
    """
    T_period = seq_length // n_periods
    t = np.linspace(0, 2.0 * np.pi, T_period)

    # --- Joint 0 ---
    periods0 = rng.random(N_SINES) * 0.7 + 0.3          # (n_sines,)
    phase0   = rng.random(N_SINES) * 2.0 * np.pi
    amp0     = rng.random(N_SINES) * 2.5 * 30            # up to ~75 per sine

    # omega0[i] = sum_k sin(t[i] / periods0[k] + phase0[k]) * amp0[k]
    omega0 = (np.sin(t[:, None] / periods0[None, :] + phase0[None, :])
              * amp0[None, :]).sum(-1)                    # (T_period,)

    phi0 = dt_step * np.cumsum(omega0)                   # (T_period,)

    # Rescale so phi0 stays in [-pi/2, pi/2]
    phi0_max, phi0_min = phi0.max(), phi0.min()
    sc = 1.0
    if phi0_max > np.pi / 2:
        sc = min(sc, (np.pi / 2) / phi0_max)
    if phi0_min < -np.pi / 2:
        sc = min(sc, (-np.pi / 2) / phi0_min)
    phi0   *= sc
    omega0 *= sc

    # --- Joint 1 ---
    periods1 = rng.random(N_SINES) * 0.7 + 0.3
    phase1   = rng.random(N_SINES) * 2.0 * np.pi
    amp1     = rng.random(N_SINES) * 10.0

    sines1 = (np.sin(t[:, None] / periods1[None, :] + phase1[None, :])
              * amp1[None, :])                            # (T_period, n_sines)
    # Normalize each sine component by its temporal range, then scale to 20
    sine_range = sines1.max(0) - sines1.min(0)
    omega1 = (sines1 / (sine_range + 1e-8) * 20.0).sum(-1)  # (T_period,)

    phi1_rel = dt_step * np.cumsum(omega1)
    ph1_max, ph1_min = phi1_rel.max(), phi1_rel.min()
    sc1 = 1.0
    if ph1_max > np.pi / 2:
        sc1 = min(sc1, (np.pi / 2) / ph1_max)
    if ph1_min < -np.pi / 2:
        sc1 = min(sc1, (-np.pi / 2) / ph1_min)
    phi1_rel *= sc1
    omega1   *= sc1

    phi1 = phi0 + phi1_rel + np.pi / 2

    # --- Endpoint (unit-length links) ---
    x = np.cos(phi0) + np.cos(phi1)  # (T_period,)
    y = np.sin(phi0) + np.sin(phi1)  # (T_period,)

    # Normalize x, y to [1, 10] then to [-1, 1]
    x = 1.0 + (x - x.min()) * (9.0 / (x.max() - x.min() + 1e-8))
    y = 1.0 + (y - y.min()) * (9.0 / (y.max() - y.min() + 1e-8))

    # Tile n_periods times
    omega0 = np.tile(omega0, n_periods)[:seq_length]
    omega1 = np.tile(omega1, n_periods)[:seq_length]
    x      = np.tile(x,      n_periods)[:seq_length]
    y      = np.tile(y,      n_periods)[:seq_length]

    # Normalize inputs per-trial to [-1, 1]
    def _minmax(v):
        vmin, vmax = v.min(), v.max()
        rng_v = vmax - vmin
        return (v - vmin) / (rng_v + 1e-8) * 2.0 - 1.0

    omega0_n = _minmax(omega0)
    omega1_n = _minmax(omega1)

    # Convert x, y from [1, 10] to [-1, 1]
    x_n = (x - 5.5) / 4.5
    y_n = (y - 5.5) / 4.5

    inputs  = np.stack([omega0_n, omega1_n], axis=-1).astype(np.float32)
    targets = np.stack([x_n, y_n],           axis=-1).astype(np.float32)
    return inputs, targets


class RobotArmTask:
    """
    NumPy version for EA evaluation.

    Interface mirrors LetterNBackTask:
      get_trial(rng)          -> (inputs, targets, {})
      evaluate_outputs(...)   -> float fitness
      run_trial(policy, rng)  -> float fitness
      evaluate_policy(...)    -> dict {fitness, accuracy, fitness_std}
    """

    def __init__(self, seq_length: int = 20, n_periods: int = 2,
                 dt_step: float = 0.001):
        self.seq_length = seq_length
        self.n_periods  = n_periods
        self.dt_step    = dt_step
        self.obs_dim    = 2
        self.action_dim = 2

    def get_trial(self, rng=None):
        """
        Returns:
            inputs:  (T, 2) float32
            targets: (T, 2) float32
            info:    {} (empty, for interface compatibility)
        """
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets = _generate_single_trajectory(
            rng, self.seq_length, self.n_periods, self.dt_step)
        return inputs, targets, {}

    def evaluate_outputs(self, outputs: np.ndarray,
                         targets: np.ndarray) -> float:
        """
        Fitness = -MSE  (higher is better; range: (-inf, 0]).
        """
        return -float(np.mean((outputs - targets) ** 2))

    def compute_accuracy(self, outputs: np.ndarray,
                         targets: np.ndarray) -> float:
        """
        Display metric = exp(-MSE) in (0, 1].
        100% = perfect, ~61% at MSE=0.5, ~37% at MSE=1.
        """
        mse = float(np.mean((outputs - targets) ** 2))
        return float(np.exp(-mse))

    def run_trial(self, policy, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets, _ = self.get_trial(rng=rng)
        policy.reset()
        outputs = np.array(
            [policy.act(inputs[t]) for t in range(self.seq_length)],
            dtype=np.float32,
        )  # (T, 2)
        return self.evaluate_outputs(outputs, targets)

    def evaluate_policy(self, policy, n_trials: int = 10, rng=None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        fitnesses = []
        for _ in range(n_trials):
            inputs, targets, _ = self.get_trial(rng=rng)
            policy.reset()
            outputs = np.array(
                [policy.act(inputs[t]) for t in range(self.seq_length)],
                dtype=np.float32,
            )
            fitnesses.append(self.evaluate_outputs(outputs, targets))
        fit = float(np.mean(fitnesses))
        mse = -fit
        return {
            'fitness':     fit,
            'accuracy':    float(np.exp(-mse)),   # display metric
            'fitness_std': float(np.std(fitnesses)),
        }


class RobotArmTaskTorch:
    """
    PyTorch version for BPTT.

    Interface mirrors LetterNBackTaskTorch:
      get_batch(batch_size)      -> (inputs, targets)  shapes (B,T,2), (B,T,2)
      compute_loss(outputs, tgt) -> scalar MSE tensor
      compute_accuracy(...)      -> float  exp(-MSE)
      compute_fitness(...)       -> float  -MSE
    """

    def __init__(self, seq_length: int = 20, n_periods: int = 2,
                 dt_step: float = 0.001, device: str = "cpu"):
        self.seq_length = seq_length
        self.n_periods  = n_periods
        self.dt_step    = dt_step
        self.device     = device
        self.np_task    = RobotArmTask(seq_length=seq_length,
                                       n_periods=n_periods,
                                       dt_step=dt_step)

    def get_batch(self, batch_size: int):
        """
        Returns:
            inputs:  (B, T, 2) float32
            targets: (B, T, 2) float32
        """
        rng = np.random.default_rng()
        inps, tgts = [], []
        for _ in range(batch_size):
            inp, tgt, _ = self.np_task.get_trial(rng=rng)
            inps.append(inp)
            tgts.append(tgt)
        return (
            torch.tensor(np.array(inps), dtype=torch.float32,
                         device=self.device),
            torch.tensor(np.array(tgts), dtype=torch.float32,
                         device=self.device),
        )

    def compute_loss(self, outputs, targets):
        """MSE loss over all timesteps and dimensions."""
        import torch.nn.functional as F
        return F.mse_loss(outputs, targets)

    def compute_accuracy(self, outputs, targets):
        """Display metric: exp(-MSE)."""
        mse = float(self.compute_loss(outputs, targets).item())
        return float(np.exp(-mse))

    def compute_fitness(self, outputs, targets):
        """Fitness = -MSE."""
        return -float(self.compute_loss(outputs, targets).item())
