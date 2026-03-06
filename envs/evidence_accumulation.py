# envs/evidence_accumulation.py
"""
Evidence Accumulation / Perceptual Decision-Making Task

One category is secretly chosen at the start of each trial.  At every
timestep the network receives a K-dimensional noisy input; the correct
channel carries a small positive bias (evidence_strength) while all
channels are corrupted by independent Gaussian noise (noise_std).

No single timestep is diagnostic — the network must *integrate* evidence
over many timesteps to identify the correct category with confidence.

Default parameters (K=5, strength=0.1, noise=0.5, T=50, response=5):
  SNR per step  ≈ evidence_strength / noise_std = 0.2
  Timesteps for SNR≥1 ≈ (noise_std / evidence_strength)^2 = 25

Timeline:
  t = 0 .. T-response_length-1  : accumulation phase, target = -1
  t = T-response_length .. T-1   : response phase,    target = correct_category

Input:  (T, K) float32  — noisy K-dim vector each step
Target: (T,)   int32    — category index 0..K-1, or -1 (no response)

Same interface as LetterNBackTask / LetterNBackTaskTorch.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_CATEGORIES = 5
CATEGORY_LABELS = ['A', 'B', 'C', 'D', 'E']


class EvidenceAccumulationTask:
    """NumPy version for EA evaluation.

    Interface mirrors LetterNBackTask exactly:
      get_trial(rng)          → (inputs, targets, categories)
      evaluate_outputs(...)   → float accuracy
      run_trial(policy, rng)  → float accuracy
      evaluate_policy(...)    → dict {fitness, accuracy, fitness_std}
    """

    def __init__(self,
                 n_categories: int = N_CATEGORIES,
                 evidence_strength: float = 0.1,
                 noise_std: float = 0.5,
                 trial_length: int = 50,
                 response_length: int = 5):
        self.n_categories     = n_categories
        self.evidence_strength = evidence_strength
        self.noise_std        = noise_std
        self.trial_length     = trial_length
        self.response_length  = response_length
        self.total_steps      = trial_length
        self.obs_dim          = n_categories
        self.action_dim       = n_categories

        self._accum_steps = trial_length - response_length  # first timestep of response

    def get_trial(self, rng=None):
        """
        Returns:
            inputs:     (T, K) float32  — noisy evidence vectors
            targets:    (T,)   int32    — category index or -1 (no response)
            categories: (T,)   int32    — correct category index repeated (for analysis)
        """
        if rng is None:
            rng = np.random.default_rng()

        correct = int(rng.integers(0, self.n_categories))

        # All channels: N(0, noise_std); correct channel additionally gets +evidence_strength
        inputs = (self.noise_std * rng.standard_normal(
            (self.trial_length, self.n_categories)
        )).astype(np.float32)
        inputs[:, correct] += self.evidence_strength

        # Targets: -1 during accumulation, correct index during response window
        targets = np.full(self.trial_length, -1, dtype=np.int32)
        targets[self._accum_steps:] = correct

        categories = np.full(self.trial_length, correct, dtype=np.int32)

        return inputs, targets, categories

    def evaluate_outputs(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """Fitness = fraction correct on response steps (0–1). Max = 1."""
        mask = targets >= 0
        if not mask.any():
            return 0.0
        pred = np.argmax(outputs[mask], axis=-1)
        return float((pred == targets[mask]).mean())

    def compute_accuracy(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        return self.evaluate_outputs(outputs, targets)

    def run_trial(self, policy, rng=None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        inputs, targets, _ = self.get_trial(rng=rng)
        policy.reset()
        outputs = np.array([
            policy.act(inputs[t]) for t in range(self.total_steps)
        ], dtype=np.float32)  # (T, K)
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
            ], dtype=np.float32)
            fitnesses.append(self.evaluate_outputs(outputs, targets))
        acc = float(np.mean(fitnesses))
        return {
            'fitness':     acc,
            'accuracy':    acc,
            'fitness_std': float(np.std(fitnesses)),
        }

    def print_trial(self, outputs, targets, categories):
        """Show response-period predictions.

        outputs:    (T, K) logits/scores
        targets:    (T,) int with -1 sentinel
        categories: (T,) int — correct category repeated
        """
        correct = int(categories[0])
        correct_lbl = CATEGORY_LABELS[correct]
        print(f"\nEvidence trial | correct={correct_lbl} "
              f"| accum={self._accum_steps}t, response={self.response_length}t")
        print(f"{'t':>3} | {'Correct':>7} | {'Pred':>7} | {'OK':>2}")
        print("-" * 30)
        for t in range(self.total_steps):
            if targets[t] < 0:
                continue
            tgt_lbl  = CATEGORY_LABELS[targets[t]]
            pred_idx = int(np.argmax(outputs[t]))
            pred_lbl = CATEGORY_LABELS[pred_idx]
            ok       = "✓" if pred_idx == targets[t] else "✗"
            print(f"{t:3d} | {tgt_lbl:>7} | {pred_lbl:>7} | {ok}")


class EvidenceAccumulationTaskTorch:
    """PyTorch version for BPTT.

    Interface mirrors LetterNBackTaskTorch:
      get_batch(batch_size)      → (inputs, targets)
      compute_loss(outputs, tgt) → scalar tensor
      compute_accuracy(...)      → float
      compute_fitness(...)       → float
    """

    def __init__(self,
                 n_categories: int = N_CATEGORIES,
                 evidence_strength: float = 0.1,
                 noise_std: float = 0.5,
                 trial_length: int = 50,
                 response_length: int = 5,
                 device: str = "cpu"):
        self.n_categories     = n_categories
        self.evidence_strength = evidence_strength
        self.noise_std        = noise_std
        self.trial_length     = trial_length
        self.response_length  = response_length
        self.device           = device
        self.np_task = EvidenceAccumulationTask(
            n_categories=n_categories,
            evidence_strength=evidence_strength,
            noise_std=noise_std,
            trial_length=trial_length,
            response_length=response_length,
        )

    def get_batch(self, batch_size: int):
        """
        Returns:
            inputs:  (B, T, K) float32
            targets: (B, T)    int64, -1 = no response
        """
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
        """
        Args:
            outputs: (B, T, K) logits
            targets: (B, T)    long, -1 = no response (ignored)
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
            outputs: (B, T, K) logits
            targets: (B, T)    long, -1 = no response
        """
        mask = targets >= 0
        if not mask.any():
            return 0.0
        pred = outputs.argmax(dim=-1)
        return float((pred[mask] == targets[mask]).float().mean().item())

    def compute_fitness(self, outputs, targets):
        """Negative cross-entropy (higher = better). Max = 0."""
        return -float(self.compute_loss(outputs, targets).item())


# ============================================================================
# Demo
# ============================================================================

def demo_task():
    print("=" * 60)
    print("Evidence Accumulation / Perceptual Decision-Making")
    print(f"K={N_CATEGORIES} categories, strength=0.1, noise=0.5, T=50, resp=5")
    print("=" * 60)

    rng = np.random.default_rng(42)

    for strength, noise in [(0.1, 0.5), (0.3, 0.5), (0.1, 0.2)]:
        task = EvidenceAccumulationTask(
            evidence_strength=strength, noise_std=noise,
            trial_length=50, response_length=5,
        )
        snr = strength / noise
        print(f"\nstrength={strength}, noise={noise}, SNR/step={snr:.2f}, "
              f"steps_for_SNR1≈{int(1/snr**2)}")

        # Baselines
        accs = {'Random': [], 'Always-A': [], 'Perfect': []}
        for _ in range(200):
            inputs, targets, _ = task.get_trial(rng=rng)
            T, K = inputs.shape

            random_out   = rng.standard_normal((T, K)).astype(np.float32)
            classA_out   = np.zeros((T, K), dtype=np.float32); classA_out[:, 0] = 1.0
            # Perfect: use integrated evidence (mean over all steps, take argmax)
            mean_inp = inputs.mean(axis=0)
            perfect_out = np.tile(mean_inp, (T, 1))

            accs['Random'].append(task.evaluate_outputs(random_out, targets))
            accs['Always-A'].append(task.evaluate_outputs(classA_out, targets))
            accs['Perfect'].append(task.evaluate_outputs(perfect_out, targets))

        for name, vals in accs.items():
            print(f"  {name:<10s}: acc={np.mean(vals):.0%}")

    print("\n\nSample trial (T=50, last 5 are response):")
    task = EvidenceAccumulationTask(trial_length=50, response_length=5)
    inputs, targets, categories = task.get_trial(rng=rng)
    # Use integrated mean as a stand-in for a perfect integrator
    mean_inp = inputs.mean(axis=0)
    fake_outputs = np.tile(mean_inp, (50, 1))
    task.print_trial(fake_outputs, targets, categories)


if __name__ == "__main__":
    demo_task()
