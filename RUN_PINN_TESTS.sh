#!/bin/bash
# PINN Guard Defense - Quick Testing Guide

# ============================================================================
# 1. LOCAL UNIT TESTS (No Infrastructure Required)
# ============================================================================

# Test 1: Basic PINN Guard functionality
echo "Testing PINN Guard core components..."
poetry run python -c "
import torch
from unifyfl.base.pinn import PINNGuard, train_adversarial_pinn_guard, _compute_physics_loss

# Initialize PINN Guard
device = 'cpu'
pinn = PINNGuard(input_dim=10, hidden_dim=64, num_layers=3).to(device)
print(f'✓ PINNGuard initialized with {sum(p.numel() for p in pinn.parameters())} parameters')

# Train on clean logits
clean_logits = torch.randn(100, 10)
pinn_trained, history = train_adversarial_pinn_guard(clean_logits, n_epochs=20, device=device)
print(f'✓ Training completed: {len(history[\"pinn_loss\"])} epochs')
print(f'  Final PINN loss: {history[\"pinn_loss\"][-1]:.6f}')
"

# ============================================================================
# 2. INTEGRATION TEST (CIFAR-10 Model)
# ============================================================================

# Test 2: Full integration with CIFAR-10 model
echo ""
echo "Testing integration with CIFAR-10 model..."
poetry run python -c "
import torch
from unifyfl.base.pinn import PINNGuard, train_adversarial_pinn_guard
from unifyfl.base.model import pinn_guard_scorer
from models.cifar import CIFAR10Model
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

Path('save').mkdir(exist_ok=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Create model and test data
model = CIFAR10Model().to(device)
test_images = torch.randn(128, 3, 32, 32)
test_labels = torch.randint(0, 10, (128,))

class FakeLoader:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __iter__(self):
        for i in range(0, len(self.images), 32):
            yield {'image': self.images[i:i+32], 'label': self.labels[i:i+32]}

loader = FakeLoader(test_images.to(device), test_labels.to(device))

# Extract and train on clean logits
model.eval()
clean_logits = []
with torch.no_grad():
    for batch in loader:
        logits = model(batch['image'].float())
        clean_logits.append(logits.cpu())
clean_logits = torch.cat(clean_logits, dim=0)

# Train PINN Guard
pinn_guard, _ = train_adversarial_pinn_guard(clean_logits, n_epochs=30, device=device)
torch.save(pinn_guard.state_dict(), 'save/pinn_guard.pt')

# Score the model
loss, score = pinn_guard_scorer(model, loader, pinn_path='save/pinn_guard.pt')
print(f'✓ Model scored successfully: {score:.6f}')
"

# ============================================================================
# 3. COMPREHENSIVE FL AGGREGATION TEST
# ============================================================================

# Test 3: Full FL aggregation simulation
echo ""
echo "Running comprehensive FL aggregation test..."
echo "(9 benign clients + 1 malicious client with Gaussian noise attack)"
echo ""

poetry run python test_pinn_defense.py

# ============================================================================
# 4. FULL PIPELINE TEST (Requires Ethereum + IPFS)
# ============================================================================

echo ""
echo "============================================================================"
echo "For full pipeline testing, you need:"
echo "1. Anvil: anvil"
echo "2. IPFS: kubo"
echo "3. Smart contracts deployed: python deploy_scripts/deploy_contracts.py ..."
echo ""
echo "Then run:"
echo "  poetry run python experiments/fl_pipeline_experiment.py --pinn"
echo ""
echo "Or via shell script:"
echo "  bash run_fl_experiment.sh pinn"
echo "============================================================================"
