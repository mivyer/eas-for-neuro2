# train_bptt.py
"""
Train RNN on working memory task using BPTT.
Uses same weighted loss as EA for fair comparison.
"""

import torch
import torch.optim as optim
import numpy as np
from dataclasses import dataclass

from models.bptt_rnn import RNNPolicy, LIFPolicy, count_parameters
from envs.working_memory import WorkingMemoryTaskTorch, WorkingMemoryTask


@dataclass
class BPTTConfig:
    n_neurons: int = 64
    obs_dim: int = 1
    action_dim: int = 1
    use_lif: bool = False
    
    n_iterations: int = 500
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    
    cue_duration: int = 5
    delay_duration: int = 15
    response_duration: int = 5
    response_weight: float = 0.75  # Weighted loss
    
    seed: int = 42
    print_every: int = 50
    device: str = "cpu"


def train_bptt(conf: BPTTConfig) -> dict:
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    
    device = torch.device(conf.device)
    
    if conf.use_lif:
        model = LIFPolicy(conf.n_neurons, conf.obs_dim, conf.action_dim).to(device)
    else:
        model = RNNPolicy(conf.n_neurons, conf.obs_dim, conf.action_dim).to(device)
    
    task = WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
        device=conf.device,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    
    print(f"BPTT Training: {conf.n_neurons} neurons, {count_parameters(model)} parameters")
    print(f"Loss weighting: {conf.response_weight:.0%} response")
    
    history = {'loss': [], 'fitness': [], 'accuracy': []}
    
    for iteration in range(conf.n_iterations):
        model.train()
        
        inputs, targets = task.get_batch(conf.batch_size)
        outputs = model(inputs)
        loss = task.compute_loss(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.grad_clip)
        optimizer.step()
        
        with torch.no_grad():
            accuracy = task.compute_accuracy(outputs, targets)
            fitness = task.compute_fitness(outputs, targets)
        
        history['loss'].append(float(loss.item()))
        history['fitness'].append(float(fitness))
        history['accuracy'].append(float(accuracy))
        
        if iteration % conf.print_every == 0 or iteration == conf.n_iterations - 1:
            print(f"Iter {iteration:4d} | loss: {loss.item():.4f} | "
                  f"fitness: {fitness:+.3f} | acc: {accuracy:.1%}")
    
    return {'model': model, 'history': history}


class PyTorchPolicyWrapper:
    """Wrap PyTorch model for NumPy task evaluation."""
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        self.h = None
    
    def reset(self):
        with torch.no_grad():
            self.h = self.model.h0.detach().clone()
    
    def act(self, obs):
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            self.h = torch.tanh(self.h @ self.model.W_rec.T + obs_t @ self.model.W_in.T)
            action = torch.tanh(self.h @ self.model.W_out.T)
            return action.cpu().numpy()


def evaluate_with_ea_metric(model, conf: BPTTConfig, n_trials: int = 100) -> dict:
    """Evaluate BPTT model using EA's fitness function."""
    task = WorkingMemoryTask(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
        response_weight=conf.response_weight,
    )
    wrapper = PyTorchPolicyWrapper(model, conf.device)
    rng = np.random.default_rng(conf.seed + 100)
    return task.evaluate_policy(wrapper, n_trials=n_trials, rng=rng)


def main():
    conf = BPTTConfig(n_neurons=64, n_iterations=500, seed=42)
    results = train_bptt(conf)
    
    # Evaluate with EA metric
    test = evaluate_with_ea_metric(results['model'], conf)
    print(f"\nEA-equivalent: fitness={test['fitness']:.3f}, accuracy={test['accuracy']:.1%}")
    
    return results


if __name__ == "__main__":
    main()
