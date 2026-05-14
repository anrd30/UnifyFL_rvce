#!/usr/bin/env python3
"""
Fast FL pipeline test - 3 rounds only for quick validation
"""

import subprocess
import sys
import json
from pathlib import Path

def run_fast_attack():
    """Run attack scenario with just 3 rounds"""
    
    # Update experiment config for fast run
    config = {
        "name": "attack_fast_3rounds",
        "num_benign_clients": 9,
        "num_malicious_clients": 1,
        "noise_scale": 0.1,
        "num_rounds": 3,  # Fast!
        "epochs_per_round": 1,
        "aggregation_policy": "pick_top_k",
        "scoring_policy": "accuracy",
        "k": 5,
        "workload": "cifar10",
        "batch_size": 32
    }
    
    # Run the main experiment script with modified config
    cmd = [
        sys.executable, "-m", "experiments.fl_pipeline_experiment",
        "--attack"
    ]
    
    print(f"Running fast attack: {config['num_rounds']} rounds")
    print(f"Expected time: ~{config['num_rounds'] * 13} minutes")
    
    subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

if __name__ == "__main__":
    run_fast_attack()
