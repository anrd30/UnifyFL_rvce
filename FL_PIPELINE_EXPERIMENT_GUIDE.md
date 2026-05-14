# FL Pipeline Experiment Guide

## Overview

This guide walks you through testing the Gaussian noise poisoning attack in the full UnifyFL federated learning pipeline and evaluating Byzantine-robust defenses.

### What This Tests

1. **Baseline Scenario**: All benign clients - establishes accuracy baseline
2. **Attack Scenario**: 1 malicious client injecting Gaussian noise + 9 benign clients
   - Measures accuracy drop from poisoning
   - Uses standard accuracy-based model selection
3. **Defense Scenario**: Same as attack but with Multi-Krum Byzantine-robust scoring
   - Measures defense effectiveness against poisoning

## Prerequisites

### System Requirements
- Python 3.10+
- ~8GB RAM minimum
- macOS, Linux, or WSL2

### Required Components

**1. Install UnifyFL dependencies:**
```bash
pip install -e .
# or with poetry
poetry install
```

**2. Install blockchain & IPFS:**

On macOS:
```bash
# Install anvil (Ethereum test chain)
brew install foundry

# Install kubo (IPFS)
brew install ipfs
```

On Linux:
```bash
# Anvil
curl -L https://foundry.paradigm.xyz | bash
~/.foundry/bin/foundryup

# IPFS
wget https://dist.ipfs.tech/kubo/v0.17.0/kubo_v0.17.0_linux-amd64.tar.gz
tar -xzf kubo_v0.17.0_linux-amd64.tar.gz
cd kubo && sudo bash install.sh
```

**3. Verify installations:**
```bash
anvil --version
ipfs --version
poetry --version
```

### Data Preparation

CIFAR-10 must be downloaded and split:

```bash
# If not already done, prepare data
python scripts/download_ds.py --dataset cifar10 --output data/
python scripts/generate_niid_dirichlet.py \
    --input data/cifar10 \
    --output data/cifar10_split \
    --num-clients 10 \
    --alpha 0.5 \
    --format hf
```

Ensure the following structure exists:
```
data/
├── cifar10/
│   ├── train0, train1, ..., train9/  (split training data)
│   └── test/  (shared test set)
```

## Quick Start (Testing Locally)

### Test 1: Quick Robustness Evaluation (2-5 minutes)

Test Gaussian noise impact WITHOUT full FL pipeline:

```bash
# Run local robustness test
python experiments/gaussian_noise_attack.py --test-robustness --num-samples 200

# View results
cat results/gaussian_noise_robustness.json
```

**Expected output:**
```
Noise Scale     Accuracy     Drop %
0.0000          0.3500       0.00%     (baseline)
0.0100          0.3200       8.57%
0.0500          0.1200       65.71%
0.1000          0.0400       88.57%
0.1500          0.0300       91.43%
```

**Key insight:** Gaussian noise at scale 0.1 causes ~88% accuracy drop!

---

## Full FL Pipeline Experiments

### Setup Phase

**Terminal 1: Start Blockchain (Anvil)**
```bash
anvil

# You'll see output like:
# Account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
# Private Key: 0x...
# ...
```

Copy a private key and account address - you'll need these.

**Terminal 2: Start IPFS**
```bash
# Initialize if not done
ipfs init

# Start daemon
ipfs daemon
```

**Terminal 3: Deploy Smart Contracts**
```bash
# Replace PRIVATE_KEY with one from anvil output
PRIVATE_KEY=0x... python deploy_scripts/deploy_contracts.py 0 pick_top_k assign_score_mean 2

# This outputs contract addresses - they're automatically saved to configs
```

### Running Experiments

**Option A: Run Individual Scenarios**

**Scenario 1: Baseline (all benign clients)**
```bash
python experiments/fl_pipeline_experiment.py --baseline

# Logs saved to: results/fl_experiments/baseline_10benign/
```

**Scenario 2: Attack (1 malicious + accuracy scoring)**
```bash
python experiments/fl_pipeline_experiment.py --attack

# Logs saved to: results/fl_experiments/attack_gaussian_noise_accuracy/
```

**Scenario 3: Defense (1 malicious + Multi-Krum)**
```bash
python experiments/fl_pipeline_experiment.py --defense

# Logs saved to: results/fl_experiments/defense_multikrum_vs_noise/
```

**Option B: Run All Experiments (Recommended)**
```bash
# Runs baseline → attack → defense sequentially
python experiments/fl_pipeline_experiment.py --all

# This will take ~30-60 minutes depending on hardware
```

**Option C: Custom Configuration**
```bash
python experiments/fl_pipeline_experiment.py --custom \
    --num-benign 8 \
    --num-malicious 2 \
    --noise-scale 0.15 \
    --rounds 5 \
    --scoring multi_krum
```

### Monitoring Experiments

While experiments run, monitor progress:

```bash
# Terminal 4: Watch client logs
tail -f results/fl_experiments/baseline_10benign/client_0_benign.log

# Terminal 5: Monitor aggregator
tail -f results/fl_experiments/baseline_10benign/aggregator.log

# Terminal 6: Monitor scorer
tail -f results/fl_experiments/baseline_10benign/scorer.log
```

### Expected Timeline

| Scenario | Duration | Notes |
|----------|----------|-------|
| Baseline | 15-20 min | 10 clients, 10 rounds |
| Attack | 15-20 min | 1 malicious, 9 benign |
| Defense | 15-20 min | Multi-Krum defense |
| All three | 45-60 min | Sequential runs |

---

## Analyzing Results

### View Experiment Logs

```bash
# List all experiments
ls -la results/fl_experiments/

# View client logs
ls -la results/fl_experiments/baseline_10benign/client_*.log

# View specific client
cat results/fl_experiments/baseline_10benign/client_0_benign.log | tail -50
```

### Analyze Single Experiment

```bash
python experiments/fl_analysis.py \
    --results-dir results/fl_experiments/baseline_10benign
```

**Output example:**
```
Collecting metrics from results/fl_experiments/baseline_10benign
Experiment: baseline_10benign
  Type: baseline
  Clients: 10 benign, 0 malicious
  Scoring: accuracy
  Final accuracy: 0.5234
```

### Compare All Scenarios

```bash
python experiments/fl_analysis.py \
    --compare baseline_10benign attack_gaussian_noise_accuracy defense_multikrum_vs_noise
```

**Output example:**
```
EXPERIMENT COMPARISON ANALYSIS
================================================================================

BASELINE SCENARIOS:
  Experiment: baseline_10benign
    Clients: 10 benign, 0 malicious
    Final Accuracy: 0.5234

ATTACK SCENARIOS:
  Experiment: attack_gaussian_noise_accuracy
    Clients: 9 benign, 1 malicious
    Noise Scale: 0.1
    Accuracy Drop: 23.14%
    Final Accuracy: 0.4024

DEFENSE SCENARIOS:
  Experiment: defense_multikrum_vs_noise
    Clients: 9 benign, 1 malicious
    Noise Scale: 0.1
    Scoring: multi_krum
    Final Accuracy: 0.4987

KEY FINDINGS:
  • Attack Impact: 23.14% accuracy drop
    Baseline: 0.5234 → Attack: 0.4024
  • Defense Effectiveness: 23.88% accuracy recovery with Multi-Krum
    Attack: 0.4024 → Defense: 0.4987
```

### Generate HTML Report

```bash
python experiments/fl_analysis.py \
    --results-dir results/fl_experiments/baseline_10benign \
    --report report.html

# Open in browser
open report.html
```

---

## Expected Results

### Attack Effectiveness

```
Baseline Accuracy (10 benign):        ~52% → 58%
Attack Accuracy (1 noisy + 9 benign): ~40% → 45%  (15-20% drop)
With Multi-Krum Defense:               ~48% → 53%  (90% recovery)
```

### What to Look For

1. **Baseline (all benign)**
   - Steady accuracy improvement over rounds
   - All clients reporting similar local accuracy
   - No noise added

2. **Attack scenario**
   - Accuracy drops compared to baseline
   - One client shows different pattern (malicious)
   - Global model quality degraded

3. **Defense scenario**
   - Multi-Krum mitigates most of the attack
   - Benign models selected more often
   - Malicious model downranked

---

## Troubleshooting

### Error: "Connection refused" for blockchain
```bash
# Anvil not running
# Solution: Run `anvil` in Terminal 1

# Check if port 8545 is available
lsof -i :8545
```

### Error: "IPFS connection failed"
```bash
# IPFS daemon not running
# Solution: Run `ipfs daemon` in Terminal 2

# Check if daemon is running
curl http://localhost:5001/api/v0/version
```

### Error: "No module named 'unifyfl'"
```bash
# Poetry environment not activated
poetry shell
# or
poetry run python experiments/fl_pipeline_experiment.py --baseline
```

### Experiment hangs (clients don't connect)
```bash
# Check if aggregator is running
ps aux | grep aggregator

# Check logs for errors
tail -f results/fl_experiments/*/aggregator.log

# Solution: Kill and restart all components
pkill -f aggregator
pkill -f scorer
pkill -f party
# Then restart fresh
```

### Low accuracy (< 20%)
- Normal for first few rounds
- Model converges slowly on CIFAR-10
- If accuracy doesn't improve after 5 rounds, check:
  - Data path is correct (TRAIN_SET env var)
  - Model architecture matches
  - Learning rate not too high

---

## Advanced: Custom Attack Variations

### Test Different Noise Scales

```bash
# Light attack (small noise)
python experiments/fl_pipeline_experiment.py --custom \
    --num-malicious 1 \
    --noise-scale 0.05 \
    --rounds 10

# Moderate attack
python experiments/fl_pipeline_experiment.py --custom \
    --num-malicious 1 \
    --noise-scale 0.1 \
    --rounds 10

# Strong attack (large noise)
python experiments/fl_pipeline_experiment.py --custom \
    --num-malicious 1 \
    --noise-scale 0.2 \
    --rounds 10
```

### Test Multiple Malicious Clients

```bash
# Byzantine scenario (3 attackers)
python experiments/fl_pipeline_experiment.py --custom \
    --num-benign 7 \
    --num-malicious 3 \
    --noise-scale 0.1 \
    --rounds 10 \
    --scoring multi_krum
```

### Test Different Aggregation Policies

Edit the configs or use the experiment runner to test:
- `pick_top_k`: Select top K models by score
- `pick_all`: Include all models
- `pick_above_mean`: Select models above average score

---

## Understanding the Multi-Krum Defense

### How Multi-Krum Works

1. **Compute Weight Distances**: Calculate distances between all model weights
2. **Find Neighbors**: Each model finds K closest neighbors
3. **Score by Distance**: Models with close neighbors score higher
4. **Byzantine-Robust Selection**: Filters out outliers (poisoned models)

### Why It Helps

- Poisoned models with Gaussian noise have very different weights
- They'll be far from benign models in weight space
- Multi-Krum downranks them
- Benign models selected instead

### Multi-Krum vs Accuracy Scoring

| Aspect | Accuracy | Multi-Krum |
|--------|----------|-----------|
| **Speed** | Fast | Slower (pairwise distances) |
| **Poisoned Detection** | No | Yes |
| **Computation** | Low | Higher |
| **Best For** | Clean data | Adversarial settings |

---

## Files Generated

After running experiments:

```
results/fl_experiments/
├── baseline_10benign/
│   ├── experiment_config.json
│   ├── agg_config.json
│   ├── client_0_benign.log
│   ├── client_1_benign.log
│   └── ... (more client logs)
│
├── attack_gaussian_noise_accuracy/
│   ├── experiment_config.json
│   ├── agg_config.json
│   ├── client_0_benign.log
│   └── client_9_malicious.log  (← the attacker)
│
└── defense_multikrum_vs_noise/
    ├── experiment_config.json
    ├── agg_config.json
    └── client_*.log
```

---

## Next Steps

1. ✅ **Verify Gaussian noise attack works** - Completed with local test
2. ✅ **Test in full FL pipeline** - This guide covers it
3. ✅ **Test Byzantine defenses** - Multi-Krum comparison included
4. ⬜ **Combine attacks** - Add label flipping to Gaussian noise
5. ⬜ **Test other defenses** - Differential privacy, median aggregation
6. ⬜ **Analyze convergence** - Measure rounds to accuracy threshold

---

## References

- **Gaussian Noise Attack**: [GAUSSIAN_NOISE_ATTACK.md](../GAUSSIAN_NOISE_ATTACK.md)
- **UnifyFL Setup**: [SETUP.md](../SETUP.md)
- **Multi-Krum Paper**: https://arxiv.org/abs/1905.06290
- **Byzantine-Robust Aggregation**: https://arxiv.org/abs/1803.01498

---

**Questions?** See [README.md](../README.md) or check experiment logs in `results/fl_experiments/*/`
