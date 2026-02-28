# Training Spiking Neural Networks: Gradients vs Evolution

## What This Is

A thesis project comparing how different learning strategies train the same
spiking neural network to do a working memory task. The core question:

> **What happens when you take away gradients and force a network to learn
> the way biology does — through evolution and local plasticity?**

## The Task

**Letter N-Back Recall.** The network sees a stream of symbols (A–E) and must
recall which symbol appeared N steps ago. This requires holding information in
memory across time — exactly what recurrent networks are for.

```
Input:    B  A  D  C  E  A  B  ...
1-back:   _  B  A  D  C  E  A  ...  (recall the previous symbol)
2-back:   _  _  B  A  D  C  E  ...  (recall two symbols ago)
```

Chance performance is 20% (random guess among 5 symbols). Higher N-back
levels are harder because the memory must persist longer.

## The Network

All conditions use the same architecture: a recurrent spiking neural network
with LIF (Leaky Integrate-and-Fire) neurons.

- **32 neurons** (32×32 recurrent connections = 1024 synapses)
- **80% excitatory, 20% inhibitory** (matching cortical ratios)
- **Dale's law enforced** (each neuron is purely excitatory or inhibitory)
- **LIF dynamics**: membrane potential accumulates input, leaks over time,
  fires a binary spike when crossing threshold, then resets


## Three Training Paradigms

### 1. BPTT with Surrogate Gradients
Backpropagation through time computes
how every weight at every timestep contributed to the error, then adjusts all
weights simultaneously. Since spikes are non-differentiable (all-or-nothing),
we use a smooth approximation ("surrogate gradient") during the backward pass.

**Pros:** Most information per update. Fast convergence.
**Cons:** No known biological mechanism does this. Requires storing the entire
forward pass in memory. The surrogate gradient is an approximation.

### 2. Genetic Algorithm (GA)
Pure evolutionary search. A population of networks compete; the best survive,
recombine, and mutate. No gradients at all — the only signal is "how well did
this network perform overall?"

**Pros:** Truly gradient-free. Works with any network, any task.
**Cons:** No temporal credit assignment. Doesn't know which weight at which
timestep mattered. Has to search blindly over thousands of parameters.

### 3. Evolution Strategy (ES)
A middle ground. Perturbs the current best solution in many random directions,
measures which directions improve performance, and moves that way. Technically
gradient-free, but mathematically estimates a gradient from population statistics.

**Pros:** Smoother than GA, more scalable.
**Cons:** The gradient estimate is very noisy with many parameters.

## Current Results (1-back, 32 neurons)

| Method     | Accuracy | Time   | Gradient Info |
|------------|----------|--------|---------------|
| BPTT (rate)| ~80%     | ~4s    | Full          |
| BPTT (LIF) | testing  | ~6s    | Approximate   |
| ES (tuned) | ~65%     | ~300s  | Estimated     |
| GA         | ~26%     | ~260s  | None          |

The pattern: **more gradient information → better temporal credit assignment
→ higher accuracy.** This is the core thesis finding.

## Project Structure

```
eas-for-neuro-main 2/
├── config.py                  # All hyperparameters in one place
├── models/
│   ├── rsnn_policy.py         # Simple rate-coded RNN (for GA/ES)
│   ├── bptt_rnn.py            # PyTorch rate-coded RNN (for BPTT)
│   ├── lif_rsnn.py            # LIF spiking neurons (NumPy + PyTorch)
│   └── stdp.py                # STDP and R-STDP plasticity rules
├── envs/
│   ├── letter_nback.py        # N-back recall task
│   └── working_memory.py      # Simple cue-delay-response task
├── trainers/
│   ├── train_bptt.py          # BPTT (rate-coded and LIF)
│   ├── train_es.py            # OpenAI Evolution Strategy
│   ├── train_ga.py            # Genetic Algorithm
│   └── train_ga_stdp.py       # GA + R-STDP (experimental)
├── scripts/
│   ├── run_experiment.py       # Main entry point
│   ├── sweep_nback.py          # Test across n-back levels
│   └── visualize.py            # Plot trial outputs
└── archive/                    # Old code from weeks 1-3
```

## Running Experiments

```bash
# Compare BPTT (rate) vs BPTT (LIF) on 1-back
python3 scripts/run_experiment.py --method bptt_lif --n-back 1

# GA on 1-back (skip BPTT)
python3 scripts/run_experiment.py --method ga --n-back 1 --no-bptt --ea-gens 300

# ES on 1-back
python3 scripts/run_experiment.py --method es --n-back 1 --no-bptt --ea-gens 300

# Everything
python3 scripts/run_experiment.py --method all --n-back 1
```

## Key Insight

The performance gap between BPTT and evolution isn't a failure of the
evolutionary approach — it reveals the **fundamental cost of biological
plausibility.** Brains don't have backpropagation. They use local learning
rules (like STDP) modulated by global reward signals (like dopamine), shaped
by millions of years of evolution. Our results quantify what that costs in
terms of task performance, and what each level of gradient information buys.

## References

- Zenke & Ganguli 2018: SuperSpike surrogate gradients
- Neftci et al. 2019: Surrogate gradient learning in SNNs
- Wang et al. NeurIPS 2023: Evolving Connectivity for RSNNs
- Izhikevich 2007: STDP + dopamine (reward-modulated plasticity)
- Such et al. 2017: Deep Neuroevolution
