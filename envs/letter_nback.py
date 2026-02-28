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
# [0.2, 0.4, 0.6, 0.8, 1.0] — kept for backward compat (used by GA_STDP)
SYMBOL_LABELS = ['A', 'B', 'C', 'D', 'E']


def encode_letter(idx: int) -> float:
    """Scalar encoding — kept for backward compat."""
    return SYMBOL_VALUES[idx]


def decode_output(value: float) -> int:
    """Nearest-symbol decode from scalar — kept for backward compat."""
    return int(np.argmin(np.abs(SYMBOL_VALUES - value)))


class LetterNBackTask:
    """NumPy version for EA evaluation.

    Input:  one-hot vector of length N_SYMBOLS (5,) — no scalar encoding exploit
    Output: 5-dim vector; argmax gives predicted class
    Target: class index 0–4, or -1 for timesteps before the n-back response window
    """

    def __init__(self, n_back: int = 2, seq_length: int = 20):
        self.n_back = n_back
        self.seq_length = seq_length
        self.total_steps = seq_length
        self.obs_dim = N_SYMBOLS   # 5
        self.action_dim = N_SYMBOLS  # 5

    def get_trial(self, rng=None):
        """
        Returns:
            inputs:  (T, N_SYMBOLS) float32 — one-hot current letter
            targets: (T,)           int32   — class index 0–4, -1 = no response
            letters: (T,)           int32   — raw letter indices 0–4
        """
        if rng is None:
            rng = np.random.default_rng()

        letters = rng.integers(0, N_SYMBOLS, size=self.seq_length)

        # One-hot input
        inputs = np.zeros((self.seq_length, N_SYMBOLS), dtype=np.float32)
        for t, l in enumerate(letters):
            inputs[t, l] = 1.0

        # Class-index targets; -1 = no response required
        targets = np.full(self.seq_length, -1, dtype=np.int32)
        for t in range(self.n_back, self.seq_length):
            targets[t] = int(letters[t - self.n_back])

        return inputs, targets, letters

    def evaluate_outputs(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Fitness = fraction correct on response steps (0–1). Max = 1."""
        mask = targets >= 0
        if not mask.any():
            return 0.0
        pred = np.argmax(outputs[mask], axis=-1)
        return float((pred == targets[mask]).mean())

    def compute_accuracy(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Fraction correct — same as evaluate_outputs for classification."""
        return self.evaluate_outputs(outputs, targets)

    def run_trial(self, policy, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets, _ = self.get_trial(rng=rng)
        policy.reset()
        outputs = np.array([
            policy.act(inputs[t]) for t in range(self.total_steps)
        ], dtype=np.float32)  # (T, 5)
        return self.evaluate_outputs(outputs, targets)

    def evaluate_policy(self, policy, n_trials: int = 10, rng=None) -> dict:
        if rng is None:
            rng = np.random.default_rng()

        fitnesses = []
        for _ in range(n_trials):
            inputs, targets, _ = self.get_trial(rng=rng)
            policy.reset()
            outputs = np.array([
                policy.act(inputs[t]) for t in range(self.total_steps)
            ], dtype=np.float32)  # (T, 5)
            fitnesses.append(self.evaluate_outputs(outputs, targets))

        acc = float(np.mean(fitnesses))
        return {
            'fitness': acc,
            'accuracy': acc,
            'fitness_std': float(np.std(fitnesses)),
        }

    def print_trial(self, outputs, targets, letters):
        """outputs: (T,5) logits/scores; targets: (T,) int with -1; letters: (T,) int."""
        print(f"\n{'t':>3} | {'Ltr':>3} | {'Target':>6} | {'Pred':>6} | {'OK':>2}")
        print("-" * 32)
        for t in range(self.total_steps):
            ltr = SYMBOL_LABELS[letters[t]]
            if targets[t] < 0:
                print(f"{t:3d} | {ltr:>3} | {'—':>6} | {'—':>6} |")
            else:
                tgt_ltr = SYMBOL_LABELS[targets[t]]
                pred_idx = int(np.argmax(outputs[t]))
                pred_ltr = SYMBOL_LABELS[pred_idx]
                ok = "✓" if pred_idx == targets[t] else "✗"
                print(f"{t:3d} | {ltr:>3} | {tgt_ltr:>6} | {pred_ltr:>6} | {ok}")


class LetterNBackTaskTorch:
    """PyTorch version for BPTT.

    Outputs 5 logits (one per symbol) + cross-entropy loss.
    Targets: class index (0–4), or -1 for timesteps before n-back response.
    """

    def __init__(self, n_back: int = 2, seq_length: int = 20, device: str = "cpu"):
        self.n_back = n_back
        self.seq_length = seq_length
        self.device = device
        self.n_symbols = N_SYMBOLS
        self.np_task = LetterNBackTask(n_back, seq_length)

    def get_batch(self, batch_size: int):
        """
        Returns:
            inputs:  (B, T, N_SYMBOLS) float32 — one-hot current letter
            targets: (B, T)            int64   — class index 0–4, or -1 (no response)
        """
        rng = np.random.default_rng()
        inps, tgts = [], []
        for _ in range(batch_size):
            inp, tgt, _ = self.np_task.get_trial(rng=rng)
            # inp: (T, 5) one-hot float32; tgt: (T,) int32 with -1 sentinel
            inps.append(inp)
            tgts.append(tgt)
        return (
            torch.tensor(np.array(inps), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(tgts), dtype=torch.long, device=self.device),
        )

    def compute_loss(self, outputs, targets):
        """
        Args:
            outputs: (B, T, 5) logits
            targets: (B, T) long, -1 = no response (ignored)
        """
        import torch.nn.functional as F
        B, T, C = outputs.shape
        return F.cross_entropy(
            outputs.reshape(B * T, C),
            targets.reshape(B * T),
            ignore_index=-1,
        )

    def compute_accuracy(self, outputs, targets):
        """
        Args:
            outputs: (B, T, 5) logits
            targets: (B, T) long, -1 = no response
        """
        mask = targets >= 0
        if not mask.any():
            return 0.0
        pred = outputs.argmax(dim=-1)  # (B, T)
        return float((pred[mask] == targets[mask]).float().mean().item())

    def compute_fitness(self, outputs, targets):
        """Negative cross-entropy (higher = better). Max = 0."""
        return -float(self.compute_loss(outputs, targets).item())


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
    print("One-hot input, 5-class output, cross-entropy loss")
    print("=" * 52)

    rng = np.random.default_rng(42)
    for n in [1, 2, 3]:
        print(f"\n--- {n}-back ---")
        task = LetterNBackTask(n_back=n, seq_length=15)
        _, targets, letters = task.get_trial(rng=rng)

        # Baseline: uniform random 5-class logits
        random_out = rng.standard_normal((15, N_SYMBOLS)).astype(np.float32)
        # Baseline: always predict class 0 (A)
        class0_out = np.zeros((15, N_SYMBOLS), dtype=np.float32)
        class0_out[:, 0] = 1.0
        # Perfect: one-hot at correct class
        perfect_out = np.zeros((15, N_SYMBOLS), dtype=np.float32)
        for t in range(15):
            if targets[t] >= 0:
                perfect_out[t, targets[t]] = 1.0

        for name, out in [
            ("Always-A",  class0_out),
            ("Perfect",   perfect_out),
            ("Random",    random_out),
        ]:
            f = task.evaluate_outputs(out, targets)
            print(f"  {name:<10s}: acc={f:.0%}")

    print("\n\nSample 2-back trial (random outputs):")
    task = LetterNBackTask(n_back=2, seq_length=15)
    rng = np.random.default_rng(42)
    _, targets, letters = task.get_trial(rng=rng)
    random_out = rng.standard_normal((15, N_SYMBOLS)).astype(np.float32)
    task.print_trial(random_out, targets, letters)


if __name__ == "__main__":
    demo_task()