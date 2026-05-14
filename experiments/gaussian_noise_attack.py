#!/usr/bin/env python3
"""
Gaussian Noise Poisoning Attack Experiment

This script demonstrates the impact of Gaussian noise poisoning on model accuracy
in the UnifyFL federated learning framework.

The attack injects Gaussian noise into model weights after local training,
degrading the global model's performance.

Usage:
    python experiments/gaussian_noise_attack.py --noise-scale 0.1 --rounds 5
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cifar import CIFAR10Model
import torch


def run_fl_round(config_path: str, attack_type: str = None, noise_scale: float = 0.1):
    """
    Run a single FL round with optional attack
    
    Args:
        config_path: Path to FL configuration JSON
        attack_type: Type of attack ('gaussian_noise', 'label_flipping', or None)
        noise_scale: Scale parameter for Gaussian noise (0.0 - 1.0)
    
    Returns:
        Dict with round results
    """
    env = os.environ.copy()
    
    # Set attack parameters
    if attack_type:
        env["IS_MALICIOUS"] = "true"
        env["ATTACK_TYPE"] = attack_type
        if attack_type == "gaussian_noise":
            env["NOISE_SCALE"] = str(noise_scale)
    
    print(f"\n{'='*70}")
    print(f"Running FL round with config: {config_path}")
    if attack_type:
        print(f"Attack: {attack_type} (noise_scale={noise_scale})")
    else:
        print("Attack: NONE (baseline)")
    print(f"{'='*70}\n")
    
    # This is a placeholder - actual execution depends on your FL setup
    # You would typically run: python -m unifyfl.sync.sync <config>
    return {
        "config": config_path,
        "attack_type": attack_type,
        "noise_scale": noise_scale,
    }


def evaluate_model_robustness(noise_scales: list, num_samples: int = 1000):
    """
    Quick local evaluation of model robustness to Gaussian noise
    
    Args:
        noise_scales: List of noise scales to test [0.01, 0.05, 0.1, ...]
        num_samples: Number of test samples to use
    """
    print("\n" + "="*70)
    print("LOCAL ROBUSTNESS EVALUATION: Impact of Gaussian Noise on Accuracy")
    print("="*70 + "\n")
    
    # Load model and test data
    model = CIFAR10Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    trainloader, testloader = CIFAR10Model.load_data()
    optimizer = model.get_optimizer()
    
    # First, train the model briefly to get some accuracy
    print("Training model (this may take a minute)...")
    model.train_model(trainloader, epochs=1, optimizer=optimizer)
    model.eval()
    
    # Get initial (clean) accuracy
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        correct_clean = 0
        total = 0
        loss_clean = 0.0
        
        for batch in testloader:
            images = batch["image"].to(device).float()
            labels = batch["label"].to(device)
            
            outputs = model(images)
            loss_clean += criterion(outputs, labels).item()
            correct_clean += (torch.max(outputs, 1)[1] == labels).sum().item()
            total += labels.size(0)
            
            if total >= num_samples:
                break
    
    clean_accuracy = correct_clean / total
    clean_loss = loss_clean / (total // batch["image"].size(0) + 1)
    
    print(f"Clean Model Performance (no noise):")
    print(f"  Accuracy: {clean_accuracy:.4f}")
    print(f"  Loss:     {clean_loss:.4f}\n")
    
    # Test with increasing noise levels
    results = {
        "clean": {
            "accuracy": clean_accuracy,
            "loss": clean_loss,
            "noise_scale": 0.0,
        }
    }
    
    for noise_scale in noise_scales:
        print(f"Testing with noise scale: {noise_scale}")
        
        # Add Gaussian noise to model weights
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        
        # Evaluate poisoned model
        with torch.no_grad():
            correct_noisy = 0
            total_noisy = 0
            loss_noisy = 0.0
            
            for batch in testloader:
                images = batch["image"].to(device).float()
                labels = batch["label"].to(device)
                
                outputs = model(images)
                loss_noisy += criterion(outputs, labels).item()
                correct_noisy += (torch.max(outputs, 1)[1] == labels).sum().item()
                total_noisy += labels.size(0)
                
                if total_noisy >= num_samples:
                    break
        
        noisy_accuracy = correct_noisy / total_noisy
        noisy_loss = loss_noisy / (total_noisy // batch["image"].size(0) + 1)
        
        # Calculate accuracy drop (handle division by zero)
        if clean_accuracy > 0:
            accuracy_drop = (clean_accuracy - noisy_accuracy) / clean_accuracy * 100
        else:
            accuracy_drop = 0
        
        results[f"noise_{noise_scale}"] = {
            "accuracy": noisy_accuracy,
            "loss": noisy_loss,
            "noise_scale": noise_scale,
            "accuracy_drop_pct": accuracy_drop,
        }
        
        print(f"  Accuracy: {noisy_accuracy:.4f} (↓ {accuracy_drop:.2f}%)")
        print(f"  Loss:     {noisy_loss:.4f}\n")
    
    return results


def print_summary(results: dict):
    """Print formatted summary of robustness evaluation"""
    print("\n" + "="*70)
    print("SUMMARY: Accuracy Drop vs Noise Scale")
    print("="*70 + "\n")
    
    print(f"{'Noise Scale':<15} {'Accuracy':<12} {'Loss':<12} {'Accuracy Drop':<15}")
    print("-" * 70)
    
    for key in sorted(results.keys(), 
                     key=lambda x: results[x]['noise_scale']):
        result = results[key]
        scale = result['noise_scale']
        acc = result['accuracy']
        loss = result['loss']
        drop = result.get('accuracy_drop_pct', 0)
        
        print(f"{scale:<15.4f} {acc:<12.4f} {loss:<12.4f} {drop:<15.2f}%")
    
    print("\nKey Findings:")
    scales = sorted([r['noise_scale'] for r in results.values()])
    if len(scales) > 1:
        clean_acc = results['clean']['accuracy']
        max_drop = max(r.get('accuracy_drop_pct', 0) for r in results.values())
        print(f"  • Clean model accuracy: {clean_acc:.4f}")
        print(f"  • Maximum accuracy drop: {max_drop:.2f}%")
        print(f"  • Tested noise scales: {scales}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Gaussian noise poisoning attack experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick robustness test with default noise scales
  python experiments/gaussian_noise_attack.py --test-robustness
  
  # Test with custom noise scales
  python experiments/gaussian_noise_attack.py --test-robustness --noise-scales 0.01 0.05 0.1 0.2 0.5
  
  # Test with fewer samples (faster)
  python experiments/gaussian_noise_attack.py --test-robustness --num-samples 100
        """
    )
    
    parser.add_argument(
        "--test-robustness",
        action="store_true",
        help="Run local robustness evaluation (fast, no FL needed)"
    )
    parser.add_argument(
        "--noise-scales",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.15, 0.2],
        help="Noise scales to test (default: 0.01 0.05 0.1 0.15 0.2)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of test samples to use (default: 1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sync.json",
        help="FL configuration file (default: configs/sync.json)"
    )
    
    args = parser.parse_args()
    
    if args.test_robustness:
        # Run quick local robustness evaluation
        results = evaluate_model_robustness(args.noise_scales, args.num_samples)
        print_summary(results)
        
        # Save results to file
        results_file = "results/gaussian_noise_robustness.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        print("Use --test-robustness to run the experiment")
        print("Run with --help for more options")


if __name__ == "__main__":
    main()
