# trainers/train_bptt.py
"""BPTT training with surrogate gradients (rate-coded RNN for now)."""

import numpy as np

try:
    import torch
    from models.bptt_rnn import RNNPolicy, count_parameters
    from envs.letter_nback import LetterNBackTaskTorch
    from envs.working_memory import WorkingMemoryTaskTorch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def make_task_torch(conf):
    if conf.task == "nback":
        return LetterNBackTaskTorch(n_back=conf.n_back, seq_length=conf.seq_length)
    return WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
    )


def train_bptt(conf) -> dict | None:
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping BPTT")
        return None

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    N = conf.n_neurons
    model = RNNPolicy(N, conf.obs_dim, conf.action_dim).to("cpu")
    task_torch = make_task_torch(conf)

    with torch.no_grad():
        W_rec_init = model.W_rec.cpu().numpy().copy()
        W_in_init = model.W_in.cpu().numpy().copy()
        W_out_init = model.W_out.cpu().numpy().copy()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.bptt_lr)
    history = {'loss': [], 'fitness': [], 'accuracy': [], 'sparsity_rec': []}
    snapshots = {}

    print(f"BPTT: {N} neurons, {count_parameters(model)} params")
    print(f"Task: {conf.task} | iters={conf.bptt_iterations}, lr={conf.bptt_lr}")

    for it in range(conf.bptt_iterations):
        model.train()
        inputs, targets = task_torch.get_batch(conf.bptt_batch_size)
        outputs = model(inputs)
        loss = task_torch.compute_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            acc = task_torch.compute_accuracy(outputs, targets)
            fit = task_torch.compute_fitness(outputs, targets)
            sp = float((model.W_rec.abs() < conf.sparsity_threshold).float().mean())

        history['loss'].append(float(loss.item()))
        history['fitness'].append(float(fit))
        history['accuracy'].append(float(acc))
        history['sparsity_rec'].append(sp)

        if it % (conf.print_every * 4) == 0 or it == conf.bptt_iterations - 1:
            print(f"Iter {it:4d} | loss={loss.item():.4f} "
                  f"fitness={fit:+.4f} acc={acc:.1%}")

    with torch.no_grad():
        W_rec_final = model.W_rec.cpu().numpy().copy()
        W_in_final = model.W_in.cpu().numpy().copy()
        W_out_final = model.W_out.cpu().numpy().copy()

    return {
        'model': model,
        'W_rec_init': W_rec_init, 'W_in_init': W_in_init, 'W_out_init': W_out_init,
        'W_rec_final': W_rec_final, 'W_in_final': W_in_final, 'W_out_final': W_out_final,
        'history': history,
        'snapshots': snapshots,
    }
