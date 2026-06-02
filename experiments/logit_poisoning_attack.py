#!/usr/bin/env python3
"""
Logit-Level Poisoning Attack Experiment

This script demonstrates the impact of logit-level poisoning on model accuracy
during Knowledge Distillation (KD) in the UnifyFL framework.

Usage:
    python experiments/logit_poisoning_attack.py --test-robustness
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cifar import CIFAR10Model
import torch


def evaluate_logit_poisoning(poison_scales: list, num_samples: int = 1000):
    """
    Evaluate student model robustness under logit poisoning with KD.
    
    Args:
        poison_scales: List of poison scales to test [0.5, 1.0, 2.0, 5.0, 10.0]
        num_samples: Number of test samples to use
    """
    print("\n" + "="*70)
    print("LOCAL EVALUATION: Impact of Logit Poisoning Attack on Knowledge Distillation")
    print("="*70 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading data...")
    trainloader, testloader = CIFAR10Model.load_data()
    
    # 1. Train Teacher Model briefly to act as our distillation target
    print("\n--- Phase 1: Training Teacher Model ---")
    teacher = CIFAR10Model().to(device)
    teacher_optimizer = teacher.get_optimizer()
    teacher.train_model(trainloader, epochs=1, optimizer=teacher_optimizer)
    teacher.eval()
    
    # Test teacher accuracy
    _, teacher_acc = teacher.test_model(testloader)
    print(f"Teacher Model Accuracy: {teacher_acc:.4f}")
    
    results = {}
    
    # 2. Train clean Student Model (No KD)
    print("\n--- Phase 2: Training Clean Student Model (No KD) ---")
    os.environ["USE_KD"] = "false"
    os.environ["IS_MALICIOUS"] = "false"
    
    student_clean = CIFAR10Model().to(device)
    optimizer = student_clean.get_optimizer()
    student_clean.train_model(trainloader, epochs=1, optimizer=optimizer)
    _, clean_acc = student_clean.test_model(testloader)
    print(f"Clean Student Model Accuracy (No KD): {clean_acc:.4f}")
    results["clean_no_kd"] = {
        "accuracy": clean_acc,
        "poison_scale": 0.0,
        "kd": False
    }
    
    # 3. Train clean Student Model (With KD)
    print("\n--- Phase 3: Training Clean Student Model (With KD) ---")
    os.environ["USE_KD"] = "true"
    os.environ["IS_MALICIOUS"] = "false"
    
    student_kd = CIFAR10Model().to(device)
    optimizer = student_kd.get_optimizer()
    student_kd.train_model(trainloader, epochs=1, optimizer=optimizer, teacher_model=teacher, alpha=0.5, temperature=2.0)
    _, clean_kd_acc = student_kd.test_model(testloader)
    print(f"Clean Student Model Accuracy (With KD): {clean_kd_acc:.4f}")
    results["clean_with_kd"] = {
        "accuracy": clean_kd_acc,
        "poison_scale": 0.0,
        "kd": True
    }
    
    # 4. Train student models under Logit Poisoning
    print("\n--- Phase 4: Training Student Models with Logit Poisoning ---")
    os.environ["USE_KD"] = "true"
    os.environ["IS_MALICIOUS"] = "true"
    os.environ["ATTACK_TYPE"] = "logit_poisoning"
    
    for scale in poison_scales:
        print(f"\nTraining with Logit Poison Scale: {scale}")
        os.environ["POISON_SCALE"] = str(scale)
        
        student_poisoned = CIFAR10Model().to(device)
        optimizer = student_poisoned.get_optimizer()
        
        student_poisoned.train_model(trainloader, epochs=1, optimizer=optimizer, teacher_model=teacher, alpha=0.5, temperature=2.0)
        _, poisoned_acc = student_poisoned.test_model(testloader)
        
        accuracy_drop = (clean_kd_acc - poisoned_acc) / clean_kd_acc * 100 if clean_kd_acc > 0 else 0
        print(f"Poisoned Student Model Accuracy (Scale={scale}): {poisoned_acc:.4f} (↓ {accuracy_drop:.2f}%)")
        
        results[f"poison_{scale}"] = {
            "accuracy": poisoned_acc,
            "poison_scale": scale,
            "kd": True,
            "accuracy_drop_pct": accuracy_drop
        }
        
    # Clean up environment variables
    for var in ["USE_KD", "IS_MALICIOUS", "ATTACK_TYPE", "POISON_SCALE"]:
        if var in os.environ:
            del os.environ[var]
            
    return results


def print_summary(results: dict):
    print("\n" + "="*70)
    print("SUMMARY: Accuracy Drop vs Logit Poison Scale")
    print("="*70 + "\n")
    
    clean_no_kd_acc = results["clean_no_kd"]["accuracy"]
    clean_kd_acc = results["clean_with_kd"]["accuracy"]
    
    print(f"Clean Student (No KD):   {clean_no_kd_acc:.4f}")
    print(f"Clean Student (With KD): {clean_kd_acc:.4f}")
    print("-" * 70)
    print(f"{'Poison Scale':<15} {'Accuracy':<12} {'Accuracy Drop (vs KD)':<25}")
    print("-" * 70)
    
    for key, result in results.items():
        if not key.startswith("poison_"):
            continue
        scale = result["poison_scale"]
        acc = result["accuracy"]
        drop = result["accuracy_drop_pct"]
        print(f"{scale:<15.2f} {acc:<12.4f} {drop:<25.2f}%")
        

def main():
    parser = argparse.ArgumentParser(description="Run logit level poisoning attack robustness experiment")
    parser.add_argument("--test-robustness", action="store_true", help="Run local robustness evaluation")
    parser.add_argument("--poison-scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0],
                        help="Poison scale standard deviations to test")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to evaluate on")
    
    args = parser.parse_args()
    
    if args.test_robustness:
        results = evaluate_logit_poisoning(args.poison_scales, args.num_samples)
        print_summary(results)
        
        # Save results
        results_file = "results/logit_poisoning_robustness.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    else:
        print("Use --test-robustness to run the experiment")
        
        
if __name__ == "__main__":
    main()
