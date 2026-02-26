# envs/working_memory.py
"""
Working Memory Task: Delayed Match-to-Sample (Continuous Cue)

Trial structure:
1. CUE phase: Input is a continuous value in [-1, 1]
2. DELAY phase: Input is 0 (silence)
3. RESPONSE phase: Network must output the cue value

Loss/fitness weighting (same for EA and BPTT):
- Response phase: 0.75 weight (reconstruction error)
- Cue + Delay phases: 0.25 weight (penalize activity)
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WorkingMemoryTask:
    """
    NumPy version for EA evaluation.

    Fitness function with weighted phases:
    - Response phase (75% weight): penalize squared error from cue
    - Other phases (25% weight): penalize squared activity (encourage quiet)
    """

    def __init__(
        self,
        cue_duration: int = 5,
        delay_duration: int = 10,
        response_duration: int = 5,
        noise_scale: float = 0.05,
        response_weight: float = 0.5,  # Weight on response phase
    ):
        self.cue_duration = cue_duration
        self.delay_duration = delay_duration
        self.response_duration = response_duration
        self.noise_scale = noise_scale
        self.response_weight = response_weight
        self.other_weight = 1.0 - response_weight
        self.total_steps = cue_duration + delay_duration + response_duration

    def get_trial(self, cue: float | None = None, rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
        """Generate one trial."""
        if rng is None:
            rng = np.random.default_rng()

        if cue is None:
            # Continuous cue in [-1, 1]
            cue = rng.choice([-1.0, 1.0])

        inputs = np.zeros(self.total_steps, dtype=np.float32)
        inputs[:self.cue_duration] = cue
        inputs += self.noise_scale * rng.standard_normal(self.total_steps).astype(np.float32)

        return inputs, float(cue)
    
    def print_trial(
    self,
    inputs: np.ndarray,
    outputs: np.ndarray,
    target_cue: float,
    title: str = "Trial",):
        """Print one trial as text."""
        T = len(inputs)
        print(f"\n{title}")
        print("Step | Input  | Output | Target")
        print("-----|--------|--------|-------")
        for t in range(T):
            out = outputs[t]
            inp = inputs[t]
            tgt = target_cue if t >= self.cue_duration + self.delay_duration else ""
            print(f"{t:4d} | {inp:6.3f} | {out:6.3f} | {tgt}")


    # def evaluate_outputs(self, outputs: np.ndarray, target_cue: float) -> float:
    #     """
    #     Compute weighted fitness (higher is better).

    #     Response phase: reward matching target, penalize oscillation
    #     Other phases: penalize activity (encourage quiet)
    #     """
    #     response_start = self.cue_duration + self.delay_duration

    #     # Response phase
    #     response_outputs = outputs[response_start:]
    #     mean_response = np.mean(response_outputs)
        
    #     # Reward for correct mean response
    #     response_reward = -((mean_response - target_cue) ** 2)
        
    #     # Penalize oscillation (high variance = bad)
    #     variance_penalty = -np.var(response_outputs)
        
    #     # Other phases: penalize activity
    #     other_outputs = outputs[:response_start]
    #     other_penalty = -np.mean(other_outputs ** 2)

    #     # Weighted combination
    #     # 60% correct response, 20% stability, 20% silence
    #     fitness = (0.6 * response_reward + 
    #             0.2 * variance_penalty + 
    #             0.2 * other_penalty)

    #     return float(fitness)
    def evaluate_outputs(self, outputs: np.ndarray, target_cue: float) -> float:
        """Response-only fitness with stability penalty."""
        response_start = self.cue_duration + self.delay_duration
        response_outputs = outputs[response_start:]
        
        mean_response = np.mean(response_outputs)
        variance = np.var(response_outputs)
        
        # Reward correct sign & magnitude, penalize oscillation
        fitness = mean_response * target_cue - 0.5 * variance
        
        return float(fitness)

    def run_trial(self, policy, cue: float | None = None,
                  rng: np.random.Generator | None = None) -> float:
        """Run a single trial with a policy."""
        if rng is None:
            rng = np.random.default_rng()

        inputs, target_cue = self.get_trial(cue=cue, rng=rng)

        policy.reset()
        outputs = []

        for t in range(self.total_steps):
            obs = np.array([inputs[t]], dtype=np.float32)
            action = policy.act(obs)
            if hasattr(action, '__len__'):
                outputs.append(float(action[0]))
            else:
                outputs.append(float(action))

        outputs = np.array(outputs, dtype=np.float32)
        return self.evaluate_outputs(outputs, target_cue)

    def evaluate_policy(self, policy, n_trials: int = 10,
                        rng: np.random.Generator | None = None) -> dict:
        """
        Evaluate a policy over multiple trials.

        Returns dict with fitness, sign accuracy, and mean absolute error.
        """
        if rng is None:
            rng = np.random.default_rng()

        fitnesses = []
        correct_sign = 0
        abs_errors = []

        for i in range(n_trials):
            inputs, target = self.get_trial(cue=None, rng=rng)

            policy.reset()
            outputs = []
            for t in range(self.total_steps):
                obs = np.array([inputs[t]], dtype=np.float32)
                action = policy.act(obs)
                if hasattr(action, '__len__'):
                    outputs.append(float(action[0]))
                else:
                    outputs.append(float(action))

            outputs = np.array(outputs, dtype=np.float32)
            fitness = self.evaluate_outputs(outputs, target)
            fitnesses.append(fitness)

            response_start = self.cue_duration + self.delay_duration
            mean_response = np.mean(outputs[response_start:])
            if np.sign(mean_response) == np.sign(target):
                correct_sign += 1
            abs_errors.append(abs(mean_response - target))

        return {
            'fitness': float(np.mean(fitnesses)),
            'accuracy': correct_sign / n_trials,
            'mean_abs_error': float(np.mean(abs_errors)),
            'fitness_std': float(np.std(fitnesses)),
        }


# ============================================================================
# PyTorch version for BPTT
# ============================================================================

if TORCH_AVAILABLE:
    class WorkingMemoryTaskTorch:
        """
        PyTorch version with same weighted loss as NumPy version.
        """

        def __init__(
            self,
            cue_duration: int = 5,
            delay_duration: int = 10,
            response_duration: int = 5,
            noise_scale: float = 0.05,
            response_weight: float = 0.5,
            device: str = "cpu",
        ):
            self.cue_duration = cue_duration
            self.delay_duration = delay_duration
            self.response_duration = response_duration
            self.noise_scale = noise_scale
            self.response_weight = response_weight
            self.other_weight = 1.0 - response_weight
            self.total_steps = cue_duration + delay_duration + response_duration
            self.device = device

        def get_batch(self, batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
            """Generate a batch of trials with continuous cues in [-1, 1]."""
            targets = 2.0 * torch.randint(0, 2, (batch_size,), device=self.device).float() - 1.0  # -1 or +1

            inputs = torch.zeros(batch_size, self.total_steps, device=self.device)
            inputs[:, :self.cue_duration] = targets.unsqueeze(1)
            inputs = inputs + self.noise_scale * torch.randn_like(inputs)

            return inputs, targets

        def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Compute weighted loss matching the EA fitness function.

            Loss = -fitness (since we minimize loss but maximize fitness)
            """
            response_start = self.cue_duration + self.delay_duration

            # Response phase loss (75% weight): MSE vs target
            response_outputs = outputs[:, response_start:]
            mean_response = response_outputs.mean(dim=1)
            response_loss = ((mean_response - targets) ** 2).mean()

            # Other phases loss (25% weight) - penalize activity
            other_outputs = outputs[:, :response_start]
            other_loss = (other_outputs ** 2).mean()

            # Weighted combination
            total_loss = (self.response_weight * response_loss +
                          self.other_weight * other_loss)

            return total_loss

        def compute_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
            """Compute sign accuracy."""
            response_start = self.cue_duration + self.delay_duration
            response_outputs = outputs[:, response_start:]
            mean_response = response_outputs.mean(dim=1)
            correct = (torch.sign(mean_response) == torch.sign(targets)).float()
            return float(correct.mean().item())

        def compute_fitness(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
            """
            Compute EA-equivalent fitness for comparison.
            """
            response_start = self.cue_duration + self.delay_duration

            # Response reward: negative MSE
            response_outputs = outputs[:, response_start:]
            mean_response = response_outputs.mean(dim=1)
            response_reward = -((mean_response - targets) ** 2).mean()

            # Other penalty
            other_outputs = outputs[:, :response_start]
            other_penalty = -(other_outputs ** 2).mean()

            fitness = (self.response_weight * response_reward +
                       self.other_weight * other_penalty)

            return float(fitness.item())


def demo_task():
    """Quick demo."""
    print("=" * 50)
    print("Working Memory Task Demo (Continuous Cue, Full Value)")
    print("=" * 50)

    task = WorkingMemoryTask(cue_duration=5, delay_duration=10, response_duration=5,
                             response_weight=0.75)
    rng = np.random.default_rng(42)

    print(f"\nWeighting: {task.response_weight:.0%} response, {task.other_weight:.0%} other")
    print(f"Total steps: {task.total_steps}")

    # Perfect network: reproduces exact cue in response
    cue = 0.7
    perfect_outputs = np.zeros(task.total_steps, dtype=np.float32)
    perfect_outputs[15:] = cue  # only active during response
    reward = task.evaluate_outputs(perfect_outputs, cue)
    print(f"\nPerfect (quiet then respond): {reward:.3f}")

    # Always active at wrong value
    always_on = np.full(task.total_steps, 0.3, dtype=np.float32)
    reward = task.evaluate_outputs(always_on, cue)
    print(f"Always 0.3 (wrong value): {reward:.3f}")

    # Wrong sign and magnitude
    wrong = np.zeros(task.total_steps, dtype=np.float32)
    wrong[15:] = -0.7
    reward = task.evaluate_outputs(wrong, cue)
    print(f"Wrong sign and magnitude: {reward:.3f}")


if __name__ == "__main__":
    demo_task()
