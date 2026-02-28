# trainers/train_bptt.py
"""
BPTT training — supports both rate-coded RNN and LIF spiking neurons.

Rate-coded: standard tanh RNN, exact gradients, ~80% on 1-back
LIF spiking: surrogate gradient (fast sigmoid), approximate gradients, expect ~60-75%

Usage via run_experiment.py:
    --method bptt          # rate-coded (default)
    --method bptt_lif      # LIF with surrogate gradients
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from envs.letter_nback import LetterNBackTaskTorch
    from envs.working_memory import WorkingMemoryTaskTorch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_task_torch(conf):
    if conf.task == "nback":
        return LetterNBackTaskTorch(n_back=conf.n_back, seq_length=conf.seq_length)
    return WorkingMemoryTaskTorch(
        cue_duration=conf.cue_duration,
        delay_duration=conf.delay_duration,
        response_duration=conf.response_duration,
    )


def train_bptt(conf, use_lif=False) -> dict | None:
    """
    Train with BPTT.

    Args:
        conf: Config dataclass
        use_lif: if True, use LIF_RSNN_Torch with surrogate gradients
                 if False, use rate-coded RNNPolicy (from bptt_rnn.py)
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping BPTT")
        return None

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)

    N = conf.n_neurons
    task_torch = make_task_torch(conf)

    # n-back uses 5-class softmax readout; other tasks use scalar output
    action_dim = 5 if conf.task == "nback" else conf.action_dim

    if use_lif:
        from models.lif_rsnn import LIF_RSNN_Torch
        model = LIF_RSNN_Torch(
            n_neurons=N,
            obs_dim=conf.obs_dim,
            action_dim=action_dim,
            beta=conf.lif_beta,
            threshold=conf.lif_threshold,
            temperature=2.0,  # wider surrogate → more neurons get gradient
            refractory_steps=conf.lif_refractory,
            ei_ratio=conf.ei_ratio,
            dale=True,
        ).to("cpu")
        label = "BPTT-LIF"
        lr = conf.bptt_lr * 3  # LIF needs higher LR (surrogate gradients are weaker)
    else:
        from models.bptt_rnn import RNNPolicy
        model = RNNPolicy(N, conf.obs_dim, action_dim).to("cpu")
        label = "BPTT-Rate"
        lr = conf.bptt_lr

    # Extract initial weights for analysis
    with torch.no_grad():
        W_rec = model.W_rec
        W_rec_init = W_rec.cpu().numpy().copy()
        W_in_init = model.W_in.cpu().numpy().copy()
        W_out_init = model.W_out.cpu().numpy().copy()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'fitness': [], 'accuracy': [], 'sparsity_rec': []}

    print(f"{label}: {N} neurons, {count_parameters(model)} params")
    if use_lif:
        print(f"  LIF: beta={conf.lif_beta}, threshold={conf.lif_threshold}, "
              f"temp=2.0, lr={lr:.4f}")
        print(f"  E/I={conf.ei_ratio:.0%}/{1-conf.ei_ratio:.0%}, Dale's law")
    print(f"Task: {conf.task} | iters={conf.bptt_iterations}, lr={lr}")

    for it in range(conf.bptt_iterations):
        model.train()
        inputs, targets = task_torch.get_batch(conf.bptt_batch_size)
        # inputs: (batch, T), targets: (batch, T)

        if use_lif:
            # LIF expects (T, batch, obs_dim)
            if inputs.dim() == 2:
                inp_lif = inputs.permute(1, 0).unsqueeze(-1)  # (T, B, 1)
            else:
                inp_lif = inputs.permute(1, 0, 2)  # (T, B, obs_dim)
            model_out = model(inp_lif)  # returns (outputs, spikes)
            outputs_raw = model_out[0]  # (T, B, action_dim)
            if action_dim == 1:
                outputs = outputs_raw.squeeze(-1).permute(1, 0)  # (B, T)
            else:
                outputs = outputs_raw.permute(1, 0, 2)  # (B, T, action_dim)
        else:
            outputs = model(inputs)  # (B, T) or (B, T, action_dim)

        loss = task_torch.compute_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            acc = task_torch.compute_accuracy(outputs, targets)
            fit = task_torch.compute_fitness(outputs, targets)
            W_rec_cur = model.W_rec
            sp = float((W_rec_cur.abs() < conf.sparsity_threshold).float().mean())

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
        'label': label,
    }