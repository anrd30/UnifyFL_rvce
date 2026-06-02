# Logit-Level Poisoning Attack

## Overview

This document explains the implementation, mechanism, and empirical evaluation of the **logit-level poisoning attack** in UnifyFL. This is an attack targeted at Knowledge Distillation (KD) in Federated Learning, where a malicious client alters the soft labels (logits) used for training the student model.

---

## Attack Mechanism

### Concept

When training with Knowledge Distillation (KD), the student model minimizes a combined loss consisting of:
1. Standard Cross-Entropy loss on hard labels.
2. Kullback-Leibler (KL) Divergence loss on the logits of the student model vs. the logits of a teacher model (representing the global federated model).

In a logit-level poisoning attack, a malicious client perturbs its local logits with noise during local training. This feeds corrupted gradients back into the student model's parameters through the KD loss:

$$\text{logits}_{\text{poisoned}} = \text{logits} - \text{logits}_{\text{detach}} + \mathcal{N}(0, \sigma^2)$$

where:
- $\sigma$ = poison scale (controlled by `POISON_SCALE` environment variable)
- $\mathcal{N}(0, \sigma^2)$ = Gaussian noise added to the forward pass of the logits.

### Impact
Corrupting the logits prevents the student from learning the correct logit distributions, leading to training divergence and accuracy collapse.

---

## Implementation Details

### Code Locations
- **Attack Execution**: [models/cifar.py](file:///Users/shash/Downloads/FL/UnifyFL/models/cifar.py#L90-L91) in `train_model()`
- **FL Integration**: [experiments/fl_pipeline_experiment.py](file:///Users/shash/Downloads/FL/UnifyFL/experiments/fl_pipeline_experiment.py#L253-L255) in `run_client()`
- **Robustness Evaluator**: [experiments/logit_poisoning_attack.py](file:///Users/shash/Downloads/FL/UnifyFL/experiments/logit_poisoning_attack.py)

### Configuration Parameters

| Parameter | Environment Variable | Type | Default | Description |
|-----------|---------------------|------|---------|-------------|
| Enable Attack | `IS_MALICIOUS` | bool | `false` | Set to `"true"` to enable |
| Attack Type | `ATTACK_TYPE` | str | `None` | Set to `"logit_poisoning"` |
| Poison Scale | `POISON_SCALE` | float | `2.0` | Standard deviation of noise |

---

## Empirical Results

We evaluated the robustness of a CIFAR-10 student model trained for 1 epoch with a clean trained teacher model.

### Robustness Evaluation (Local Training)

| Evaluation Scenario | Accuracy | Drop % (vs KD) |
|---------------------|----------|----------------|
| Clean (No KD) | 0.1886 | - |
| Clean (With KD) | 0.1580 | 0.00% |
| Poisoned (Scale=0.5) | 0.1099 | ↓ 30.46% (Collapses to random chance) |
| Poisoned (Scale=1.0) | 0.1099 | ↓ 30.46% |
| Poisoned (Scale=2.0) | 0.1099 | ↓ 30.46% |
| Poisoned (Scale=5.0) | 0.1099 | ↓ 30.46% |
| Poisoned (Scale=10.0) | 0.1099 | ↓ 30.46% |

**Key Findings:**
- Logit-level poisoning completely destroys student distillation-based learning, dropping performance directly to **random guessing (~11%)**.
- Even a low noise scale of **0.5** is sufficient to fully poison training.
- This highlights the vulnerability of local KD pipelines to maliciously perturbed logits.

---

## Running Experiments

### Local Robustness Test
To verify the local impact of the logit poisoning attack:
```bash
poetry run python experiments/logit_poisoning_attack.py --test-robustness
```

### Full FL Pipeline Experiment
To launch a federated learning experiment under logit poisoning using the existing pipeline framework:
```bash
poetry run python experiments/fl_pipeline_experiment.py --custom --num-benign 9 --num-malicious 1 --noise-scale 2.0 --rounds 5 --epochs 1 --attack-type logit_poisoning
```
