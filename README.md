# Training Spiking Neural Networks: Gradients vs Evolution

A thesis project comparing how different learning strategies train the same
spiking neural network to do a working memory task.

> **How does learning bias network connectivity?**

## Tasks

**Letter N-Back Recall.** The network sees a stream of symbols (A–E) and must
recall which symbol appeared N steps ago.

```
Input:    B  A  D  C  E  A  B  ...
1-back:   _  B  A  D  C  E  A  ...  (recall the previous symbol)
2-back:   _  _  B  A  D  C  E  ...  (recall two symbols ago)
```

Chance performance is 20% (random guess among 5 symbols). Higher N-back
levels are harder because the memory must persist longer.

## The Network

32, 64, 128 Neurons

## Three Training Paradigms

### 1. BPTT 

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

### 4. GA w Oja

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


- Zenke & Ganguli 2018: SuperSpike surrogate gradients
- Neftci et al. 2019: Surrogate gradient learning in SNNs
- Wang et al. NeurIPS 2023: Evolving Connectivity for RSNNs
- Izhikevich 2007: STDP + dopamine (reward-modulated plasticity)
- Such et al. 2017: Deep Neuroevolution
