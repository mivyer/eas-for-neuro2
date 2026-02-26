# envs/letter_nback.py
"""
Letter N-Back Recall Task

Stream of 5 symbols (A–E), one per timestep. At each step t >= n,
the network must OUTPUT the symbol from n steps back.

Encoding: 5 symbols → evenly spaced in (0, 1]: {0.2, 0.4, 0.6, 0.8, 1.0}
  Well-separated (spacing = 0.2), all positive.

Why recall (not match/non-match):
  - Match/non-match is binary → constant "non-match" gets ~65% free
  - Recall requires maintaining a rich representation of the actual identity
  - No constant-output exploit (constant gets ~20% acc, fitness ≈ -0.05)
  - Rich continuous gradient signal for BPTT (MSE per timestep)
  - Progressive n (1, 2, 3...) = clean difficulty knob

Trial:
  t=0..n-1:  input=letter, target=0  (no response yet)
  t=n..T-1:  input=letter, target=encoding(letter[t-n])
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_SYMBOLS = 5
SYMBOL_VALUES = np.array([i / N_SYMBOLS for i in range(1, N_SYMBOLS + 1)], dtype=np.float32)
# [0.2, 0.4, 0.6, 0.8, 1.0]
SYMBOL_LABELS = ['A', 'B', 'C', 'D', 'E']


def encode_letter(idx: int) -> float:
    return SYMBOL_VALUES[idx]


def decode_output(value: float) -> int:
    return int(np.argmin(np.abs(SYMBOL_VALUES - value)))


class LetterNBackTask:
    """NumPy version for EA evaluation."""

    def __init__(self, n_back: int = 2, seq_length: int = 20):
        self.n_back = n_back
        self.seq_length = seq_length
        self.total_steps = seq_length
        self.obs_dim = 1
        self.action_dim = 1

    def get_trial(self, rng: np.random.Generator | None = None):
        """
        Returns:
            inputs:  (T,) float32 — encoded current letters
            targets: (T,) float32 — encoded letter from n steps back (0 for t < n)
            letters: (T,) int32   — raw letter indices 0–4
        """
        if rng is None:
            rng = np.random.default_rng()

        letters = rng.integers(0, N_SYMBOLS, size=self.seq_length)
        inputs = np.array([encode_letter(l) for l in letters], dtype=np.float32)

        targets = np.zeros(self.seq_length, dtype=np.float32)
        for t in range(self.n_back, self.seq_length):
            targets[t] = encode_letter(letters[t - self.n_back])

        return inputs, targets, letters

    def evaluate_outputs(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Fitness = negative MSE on response steps. Max = 0 (perfect)."""
        mask = targets > 0
        if not mask.any():
            return 0.0
        return -float(np.mean((outputs[mask] - targets[mask]) ** 2))

    def compute_accuracy(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Fraction of response steps where nearest-symbol decode is correct."""
        mask = targets > 0
        if not mask.any():
            return 0.0
        correct = sum(
            decode_output(outputs[t]) == decode_output(targets[t])
            for t in np.where(mask)[0]
        )
        return correct / mask.sum()

    def run_trial(self, policy, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets, _ = self.get_trial(rng=rng)
        policy.reset()
        outputs = np.array([
            float(policy.act(np.array([inputs[t]], dtype=np.float32))[0])
            if hasattr(policy.act(np.array([inputs[t]], dtype=np.float32)), '__len__')
            else float(policy.act(np.array([inputs[t]], dtype=np.float32)))
            for t in range(self.total_steps)
        ], dtype=np.float32)
        return self.evaluate_outputs(outputs, targets)

    def evaluate_policy(self, policy, n_trials: int = 10, rng=None) -> dict:
        if rng is None:
            rng = np.random.default_rng()

        fitnesses, accuracies = [], []
        for _ in range(n_trials):
            inputs, targets, _ = self.get_trial(rng=rng)
            policy.reset()
            outputs = []
            for t in range(self.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                a = policy.act(obs)
                outputs.append(float(a[0]) if hasattr(a, '__len__') else float(a))
            outputs = np.array(outputs, dtype=np.float32)
            fitnesses.append(self.evaluate_outputs(outputs, targets))
            accuracies.append(self.compute_accuracy(outputs, targets))

        return {
            'fitness': float(np.mean(fitnesses)),
            'accuracy': float(np.mean(accuracies)),
            'fitness_std': float(np.std(fitnesses)),
        }

    def print_trial(self, inputs, outputs, targets, letters):
        print(f"\n{'t':>3} | {'In':>6} | {'Ltr':>3} | {'Target':>10} | "
              f"{'Output':>7} | {'Dec':>3} | {'':>2}")
        print("-" * 52)
        for t in range(self.total_steps):
            ltr = SYMBOL_LABELS[letters[t]]
            if t < self.n_back:
                print(f"{t:3d} | {inputs[t]:.3f}  | {ltr:>3} | {'—':>10} | "
                      f"{outputs[t]:+.3f} |     |")
            else:
                tgt_idx = decode_output(targets[t])
                dec_idx = decode_output(outputs[t])
                ok = "✓" if tgt_idx == dec_idx else "✗"
                print(f"{t:3d} | {inputs[t]:.3f}  | {ltr:>3} | "
                      f"{targets[t]:.2f} ({SYMBOL_LABELS[tgt_idx]}) | "
                      f"{outputs[t]:+.3f} | {SYMBOL_LABELS[dec_idx]:>3} | {ok}")


class LetterNBackTaskTorch:
    """PyTorch version for BPTT."""

    def __init__(self, n_back: int = 2, seq_length: int = 20, device: str = "cpu"):
        self.n_back = n_back
        self.seq_length = seq_length
        self.device = device
        self.np_task = LetterNBackTask(n_back, seq_length)

    def get_batch(self, batch_size: int):
        rng = np.random.default_rng()
        inps, tgts = [], []
        for _ in range(batch_size):
            i, t, _ = self.np_task.get_trial(rng=rng)
            inps.append(i); tgts.append(t)
        return (torch.tensor(np.array(inps), dtype=torch.float32, device=self.device),
                torch.tensor(np.array(tgts), dtype=torch.float32, device=self.device))

    def compute_loss(self, outputs, targets):
        mask = (targets > 0).float()
        n = mask.sum()
        if n == 0:
            return torch.tensor(0.0, device=self.device)
        return ((outputs - targets) ** 2 * mask).sum() / n

    def compute_accuracy(self, outputs, targets):
        mask = targets > 0
        if not mask.any():
            return 0.0
        sym = torch.tensor(SYMBOL_VALUES, device=outputs.device)
        pred = (outputs[mask].unsqueeze(-1) - sym).abs().argmin(dim=-1)
        actual = (targets[mask].unsqueeze(-1) - sym).abs().argmin(dim=-1)
        return float((pred == actual).float().mean().item())

    def compute_fitness(self, outputs, targets):
        mask = targets > 0
        if not mask.any():
            return 0.0
        return -float(((outputs[mask] - targets[mask]) ** 2).mean().item())


# ============================================================================
# Progressive difficulty sweep
# ============================================================================

def sweep_nback(policy_factory, n_values=(1, 2, 3, 4, 5),
                seq_length=25, n_trials=50, seed=42):
    """
    Test across n-back levels to find the breakpoint.

    policy_factory: callable(n) -> trained policy, or dict {n: policy}
    """
    rng = np.random.default_rng(seed)
    results = {}
    for n in n_values:
        task = LetterNBackTask(n_back=n, seq_length=seq_length)
        policy = policy_factory[n] if isinstance(policy_factory, dict) else policy_factory(n)
        r = task.evaluate_policy(policy, n_trials=n_trials, rng=rng)
        results[n] = r
        print(f"  {n}-back: fitness={r['fitness']:+.4f}  acc={r['accuracy']:.1%}")
    return results


# ============================================================================
# Demo
# ============================================================================

def demo_task():
    print("=" * 52)
    print("Letter N-Back Recall (5 symbols: A B C D E)")
    print("=" * 52)

    for n in [1, 2, 3]:
        print(f"\n--- {n}-back ---")
        task = LetterNBackTask(n_back=n, seq_length=15)
        rng = np.random.default_rng(42)
        inputs, targets, letters = task.get_trial(rng=rng)

        # Baselines
        for name, out in [
            ("Constant 0.6", np.full(15, 0.6, dtype=np.float32)),
            ("Perfect",      targets.copy()),
            ("Random [0,1]", rng.random(15).astype(np.float32)),
        ]:
            f = task.evaluate_outputs(out, targets)
            a = task.compute_accuracy(out, targets)
            print(f"  {name:<14s}: fitness={f:+.4f}, acc={a:.0%}")

    print("\n\nSample 2-back trial (constant 0.5 output):")
    task = LetterNBackTask(n_back=2, seq_length=15)
    rng = np.random.default_rng(42)
    inputs, targets, letters = task.get_trial(rng=rng)
    task.print_trial(inputs, np.full(15, 0.5, dtype=np.float32), targets, letters)


if __name__ == "__main__":
    demo_task()