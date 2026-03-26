"""
Letter N-Back recall task (5 symbols A–E, one-hot input, 5-class output).

At each step t >= n, the network must output the symbol shown n steps earlier.
Targets are class indices 0–4; -1 marks timesteps with no required response.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_SYMBOLS     = 5
SYMBOL_VALUES = np.array([i / N_SYMBOLS for i in range(1, N_SYMBOLS + 1)], dtype=np.float32)
SYMBOL_LABELS = ['A', 'B', 'C', 'D', 'E']


def encode_letter(idx: int) -> float:
    """Scalar encoding — kept for backward compat."""
    return SYMBOL_VALUES[idx]


def decode_output(value: float) -> int:
    """Nearest-symbol decode from scalar — kept for backward compat."""
    return int(np.argmin(np.abs(SYMBOL_VALUES - value)))


class LetterNBackTask:
    """NumPy version for EA evaluation."""

    def __init__(self, n_back: int = 2, seq_length: int = 20):
        self.n_back      = n_back
        self.seq_length  = seq_length
        self.total_steps = seq_length
        self.obs_dim     = N_SYMBOLS
        self.action_dim  = N_SYMBOLS

    def get_trial(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        letters = rng.integers(0, N_SYMBOLS, size=self.seq_length)

        inputs = np.zeros((self.seq_length, N_SYMBOLS), dtype=np.float32)
        for t, l in enumerate(letters):
            inputs[t, l] = 1.0

        targets = np.full(self.seq_length, -1, dtype=np.int32)
        for t in range(self.n_back, self.seq_length):
            targets[t] = int(letters[t - self.n_back])

        return inputs, targets, letters

    def evaluate_outputs(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        mask = targets >= 0
        if not mask.any():
            return 0.0
        return float((np.argmax(outputs[mask], axis=-1) == targets[mask]).mean())

    def compute_accuracy(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        return self.evaluate_outputs(outputs, targets)

    def run_trial(self, policy, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets, _ = self.get_trial(rng=rng)
        policy.reset()
        outputs = np.array([policy.act(inputs[t]) for t in range(self.total_steps)],
                           dtype=np.float32)
        return self.evaluate_outputs(outputs, targets)

    def evaluate_policy(self, policy, n_trials: int = 10, rng=None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        fitnesses = []
        for _ in range(n_trials):
            inputs, targets, _ = self.get_trial(rng=rng)
            policy.reset()
            outputs = np.array([policy.act(inputs[t]) for t in range(self.total_steps)],
                               dtype=np.float32)
            fitnesses.append(self.evaluate_outputs(outputs, targets))
        acc = float(np.mean(fitnesses))
        return {'fitness': acc, 'accuracy': acc, 'fitness_std': float(np.std(fitnesses))}

    def print_trial(self, outputs, targets, letters):
        print(f"\n{'t':>3} | {'Ltr':>3} | {'Target':>6} | {'Pred':>6} | {'OK':>2}")
        print("-" * 32)
        for t in range(self.total_steps):
            ltr = SYMBOL_LABELS[letters[t]]
            if targets[t] < 0:
                print(f"{t:3d} | {ltr:>3} | {'—':>6} | {'—':>6} |")
            else:
                tgt_ltr  = SYMBOL_LABELS[targets[t]]
                pred_idx = int(np.argmax(outputs[t]))
                pred_ltr = SYMBOL_LABELS[pred_idx]
                ok = "✓" if pred_idx == targets[t] else "✗"
                print(f"{t:3d} | {ltr:>3} | {tgt_ltr:>6} | {pred_ltr:>6} | {ok}")


class LetterNBackTaskTorch:
    """PyTorch version for BPTT (batched, cross-entropy loss)."""

    def __init__(self, n_back: int = 2, seq_length: int = 20, device: str = "cpu"):
        self.n_back     = n_back
        self.seq_length = seq_length
        self.device     = device
        self.n_symbols  = N_SYMBOLS
        self.np_task    = LetterNBackTask(n_back, seq_length)

    def get_batch(self, batch_size: int):
        rng = np.random.default_rng()
        inps, tgts = [], []
        for _ in range(batch_size):
            inp, tgt, _ = self.np_task.get_trial(rng=rng)
            inps.append(inp)
            tgts.append(tgt)
        return (
            torch.tensor(np.array(inps), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(tgts), dtype=torch.long,    device=self.device),
        )

    def compute_loss(self, outputs, targets):
        import torch.nn.functional as F
        B, T, C = outputs.shape
        return F.cross_entropy(outputs.reshape(B * T, C), targets.reshape(B * T),
                               ignore_index=-1)

    def compute_accuracy(self, outputs, targets):
        mask = targets >= 0
        if not mask.any():
            return 0.0
        pred = outputs.argmax(dim=-1)
        return float((pred[mask] == targets[mask]).float().mean().item())

    def compute_fitness(self, outputs, targets):
        return -float(self.compute_loss(outputs, targets).item())


def sweep_nback(policy_factory, n_values=(1, 2, 3, 4, 5),
                seq_length=25, n_trials=50, seed=42):
    rng = np.random.default_rng(seed)
    results = {}
    for n in n_values:
        task   = LetterNBackTask(n_back=n, seq_length=seq_length)
        policy = policy_factory[n] if isinstance(policy_factory, dict) else policy_factory(n)
        r      = task.evaluate_policy(policy, n_trials=n_trials, rng=rng)
        results[n] = r
        print(f"  {n}-back: acc={r['accuracy']:.1%}")
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    for n in [1, 2, 3]:
        task = LetterNBackTask(n_back=n, seq_length=15)
        _, targets, _ = task.get_trial(rng=rng)
        random_out = rng.standard_normal((15, N_SYMBOLS)).astype(np.float32)
        print(f"{n}-back random baseline: {task.evaluate_outputs(random_out, targets):.0%}")
