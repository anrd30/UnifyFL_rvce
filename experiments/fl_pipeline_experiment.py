#!/usr/bin/env python3
"""
Comprehensive FL Pipeline Experiment Framework

Tests Gaussian noise poisoning attack in full FL pipeline and evaluates
Byzantine-robust defenses (Multi-Krum vs standard accuracy scoring).

This script orchestrates:
1. Blockchain deployment
2. IPFS startup
3. Aggregator and Scorer processes
4. Multiple FL clients (mix of benign and malicious)
5. Metrics collection and convergence analysis

Requires:
- anvil and kubo (IPFS) running or accessible
- Python environment configured with UnifyFL dependencies
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import signal
import psutil
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    name: str
    num_benign_clients: int
    num_malicious_clients: int
    noise_scale: float  # For malicious clients
    num_rounds: int
    epochs_per_round: int
    aggregation_policy: str  # pick_top_k, pick_all, etc.
    scoring_policy: str  # accuracy, multi_krum
    k: int  # For pick_top_k
    workload: str  # cifar10, mnist, etc.
    batch_size: int


class ProcessManager:
    """Manages subprocess creation and cleanup"""
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files: Dict[str, str] = {}
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        logger.info("Signal received, cleaning up processes...")
        self.cleanup()
        sys.exit(0)
    
    def start_process(self, name: str, command: List[str], env: Dict = None, 
                      log_file: str = None) -> subprocess.Popen:
        """Start a subprocess and track it"""
        try:
            full_env = os.environ.copy()
            if env:
                full_env.update(env)
            
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_fd = open(log_file, 'w')
            else:
                log_fd = subprocess.DEVNULL
            
            logger.info(f"Starting {name}: {' '.join(command)}")
            proc = subprocess.Popen(
                command,
                env=full_env,
                stdout=log_fd if log_file else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes[name] = proc
            if log_file:
                self.log_files[name] = log_file
            
            return proc
        
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            raise
    
    def wait_for_process(self, name: str, timeout: int = None) -> int:
        """Wait for a process to complete"""
        if name not in self.processes:
            raise ValueError(f"Process {name} not found")
        
        try:
            return_code = self.processes[name].wait(timeout=timeout)
            logger.info(f"{name} completed with code {return_code}")
            return return_code
        except subprocess.TimeoutExpired:
            logger.warning(f"{name} timed out after {timeout}s")
            return -1
    
    def is_running(self, name: str) -> bool:
        """Check if a process is still running"""
        if name not in self.processes:
            return False
        return self.processes[name].poll() is None
    
    def cleanup(self):
        """Terminate all managed processes"""
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                logger.info(f"Terminating {name}...")
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {name}...")
                    proc.kill()
        
        # Close log files
        for log_file in self.log_files.values():
            try:
                open(log_file, 'a').close()
            except:
                pass


class FLExperiment:
    """Manages FL experiments with attack and defense scenarios"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.process_mgr = ProcessManager()
        self.results_dir = Path("results/fl_experiments") / config.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict = {
            "rounds": [],
            "accuracies": [],
            "losses": [],
            "timestamps": [],
        }
    
    def setup_configs(self):
        """Create experiment-specific configs"""
        logger.info("Setting up configuration files...")
        
        # Try to load existing contract addresses from central config
        import uuid
        central_config_path = Path("configs/async/agg.config.json")
        reg_address = "0x5FbDB2315678afecb367f032d93F642f64180aa3" # Fallback
        contract_address = "0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0" # Fallback
        experiment_id = str(uuid.uuid4())
        strategy = "fedavg"
        
        if central_config_path.exists():
            try:
                with open(central_config_path, 'r') as f:
                    central_data = json.load(f)
                    reg_address = central_data.get("registration_contract_address", reg_address)
                    contract_address = central_data.get("contract_address", contract_address)
                    experiment_id = central_data.get("experiment_id", experiment_id)
                    strategy = central_data.get("strategy", strategy)
                    logger.info(f"Loaded contract addresses from {central_config_path}")
            except Exception as e:
                logger.warning(f"Could not read central config: {e}. Using defaults.")

        # Create aggregator config
        agg_config = {
            "workload": self.config.workload,
            "geth_endpoint": "http://localhost:8545",
            "geth_account": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
            "registration_contract_address": reg_address,
            "contract_address": contract_address,
            "flwr_min_fit_clients": self.config.num_benign_clients + self.config.num_malicious_clients,
            "flwr_min_available_clients": self.config.num_benign_clients + self.config.num_malicious_clients,
            "flwr_min_evaluate_clients": self.config.num_benign_clients + self.config.num_malicious_clients,
            "flwr_server_address": "localhost:5000",
            "ipfs_host": "/ip4/127.0.0.1/tcp/5001",
            "aggregation_policy": self.config.aggregation_policy,
            "scoring_policy": "assign_score_mean",
            "k": self.config.k,
            "scorer": self.config.scoring_policy,
            "strategy": strategy,
            "experiment_id": experiment_id,
            "num_rounds": self.config.num_rounds,
        }
        
        agg_config_path = self.results_dir / "agg_config.json"
        with open(agg_config_path, 'w') as f:
            json.dump(agg_config, f, indent=2)
        
        logger.info(f"Aggregator config saved to {agg_config_path}")

        # Create matched party config for clients
        party_config = {
            "workload": self.config.workload,
            "flwr_server_address": "localhost:5000",
            "epochs": self.config.epochs_per_round
        }
        party_config_path = self.results_dir / "party_config.json"
        with open(party_config_path, 'w') as f:
            json.dump(party_config, f, indent=2)
        logger.info(f"Party config saved to {party_config_path}")

        return agg_config_path
    
    def create_client_configs(self) -> List[Tuple[int, bool]]:
        """Generate client configurations (index, is_malicious)"""
        configs = []
        
        # Benign clients
        for i in range(self.config.num_benign_clients):
            configs.append((i, False))
        
        # Malicious clients
        for i in range(self.config.num_benign_clients, 
                      self.config.num_benign_clients + self.config.num_malicious_clients):
            configs.append((i, True))
        
        return configs
    
    def run_client(self, client_id: int, is_malicious: bool):
        """Run a single FL client"""
        env = os.environ.copy()
        env["TRAIN_SET"] = str(client_id)
        env["LR"] = "0.001"
        env["EPOCHS"] = str(self.config.epochs_per_round)
        
        if is_malicious:
            env["IS_MALICIOUS"] = "true"
            env["ATTACK_TYPE"] = "gaussian_noise"
            env["NOISE_SCALE"] = str(self.config.noise_scale)
            log_name = f"client_{client_id}_malicious.log"
        else:
            log_name = f"client_{client_id}_benign.log"
        
        command = ["poetry", "run", "party", str(self.results_dir / "party_config.json")]
        log_path = self.results_dir / log_name
        
        return self.process_mgr.start_process(
            f"client_{client_id}",
            command,
            env=env,
            log_file=str(log_path)
        )
    
    def run(self):
        """Execute the full experiment"""
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Config: {asdict(self.config)}")
        
        # Save experiment config
        config_file = self.results_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
        
        try:
            # Setup configs
            agg_config_path = self.setup_configs()
            
            # Start aggregator
            agg_command = ["poetry", "run", "async-agg", str(agg_config_path)]
            agg_log_path = self.results_dir / "aggregator.log"
            self.process_mgr.start_process("aggregator", agg_command, log_file=str(agg_log_path))
            
            # Start scorer
            scorer_command = ["poetry", "run", "async-scorer", str(agg_config_path)]
            scorer_log_path = self.results_dir / "scorer.log"
            self.process_mgr.start_process("scorer", scorer_command, log_file=str(scorer_log_path))
            
            # Stagger startup to allow port binding
            time.sleep(5)
            
            # Get client configs
            client_configs = self.create_client_configs()
            
            logger.info(f"Starting {len(client_configs)} clients...")
            logger.info(f"  Benign: {self.config.num_benign_clients}")
            logger.info(f"  Malicious: {self.config.num_malicious_clients} (noise_scale={self.config.noise_scale})")
            
            # Start clients
            for client_id, is_malicious in client_configs:
                self.run_client(client_id, is_malicious)
                time.sleep(1)  # Stagger client starts
            
            # Monitor experiment
            logger.info("Monitoring experiment...")
            start_time = time.time()
            
            # Track FL round progress
            current_round = 0
            while True:
                # Check if all clients are still running
                active_clients = sum(1 for name in self.process_mgr.processes 
                                    if name.startswith("client_") and 
                                    self.process_mgr.is_running(name))
                
                if active_clients == 0:
                    logger.info("All clients completed")
                    break
                
                # Check if aggregator has exited
                if not self.process_mgr.is_running("aggregator"):
                    logger.info("Aggregator completed/exited")
                    break
                
                # Simple heartbeat
                elapsed = int(time.time() - start_time)
                logger.info(f"[{elapsed}s] Active clients: {active_clients}/{len(client_configs)}")
                
                # Log round completion estimate (every heartbeat approximates a round)
                current_round += 1
                logger.info(f"Round {current_round} completed (approx.)")
                
                time.sleep(30)
            
            logger.info("Experiment completed successfully")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise
        finally:
            self.process_mgr.cleanup()


class ExperimentRunner:
    """Orchestrates multiple experiment scenarios"""
    
    def __init__(self):
        self.results_base = Path("results/fl_experiments")
        self.results_base.mkdir(parents=True, exist_ok=True)
    
    def create_baseline_config(self) -> ExperimentConfig:
        """Baseline: all benign clients"""
        return ExperimentConfig(
            name="baseline_10benign",
            num_benign_clients=10,
            num_malicious_clients=0,
            noise_scale=0.0,
            num_rounds=25,
            epochs_per_round=3,
            aggregation_policy="pick_top_k",
            scoring_policy="accuracy",
            k=5,
            workload="cifar10",
            batch_size=32,
        )
    
    def create_attack_config(self) -> ExperimentConfig:
        """Attack scenario: 1 malicious + 9 benign with standard accuracy scoring"""
        return ExperimentConfig(
            name="attack_gaussian_noise_accuracy",
            num_benign_clients=9,
            num_malicious_clients=1,
            noise_scale=0.1,
            num_rounds=100,
            epochs_per_round=3,
            aggregation_policy="pick_top_k",
            scoring_policy="accuracy",
            k=5,
            workload="cifar10",
            batch_size=32,
        )
    
    def create_defense_config(self) -> ExperimentConfig:
        """Defense scenario: 1 malicious + 9 benign with Multi-Krum defense"""
        return ExperimentConfig(
            name="defense_multikrum_vs_noise",
            num_benign_clients=9,
            num_malicious_clients=1,
            noise_scale=0.1,
            num_rounds=100,
            epochs_per_round=3,
            aggregation_policy="pick_top_k",
            scoring_policy="multi_krum",
            k=5,
            workload="cifar10",
            batch_size=32,
        )
    
    def create_pinn_defense_config(self) -> ExperimentConfig:
        """Defense scenario: 1 malicious + 9 benign with PINN Guard defense"""
        return ExperimentConfig(
            name="defense_pinn_vs_noise",
            num_benign_clients=9,
            num_malicious_clients=1,
            noise_scale=0.1,
            num_rounds=100,
            epochs_per_round=3,
            aggregation_policy="pick_top_k",
            scoring_policy="pinn_guard",
            k=5,
            workload="cifar10",
            batch_size=32,
        )
    
    def run_all(self):
        """Run all experiment scenarios"""
        scenarios = [
            ("Baseline (10 benign clients)", self.create_baseline_config()),
            ("Attack (1 poisoned + 9 benign with accuracy scoring)", self.create_attack_config()),
            ("Defense (1 poisoned + 9 benign with Multi-Krum)", self.create_defense_config()),
            ("Defense (1 poisoned + 9 benign with PINN Guard)", self.create_pinn_defense_config()),
        ]
        
        results_summary = {}
        
        for scenario_name, config in scenarios:
            logger.info(f"\n{'='*70}")
            logger.info(f"Running scenario: {scenario_name}")
            logger.info(f"{'='*70}")
            
            try:
                experiment = FLExperiment(config)
                experiment.run()
                results_summary[config.name] = "SUCCESS"
            except Exception as e:
                logger.error(f"Scenario failed: {e}")
                results_summary[config.name] = f"FAILED: {str(e)}"
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}\n")
        
        for scenario_name, status in results_summary.items():
            logger.info(f"{scenario_name}: {status}")
        
        logger.info(f"\nResults saved to: {self.results_base}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive FL pipeline experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single baseline experiment
  python experiments/fl_pipeline_experiment.py --baseline
  
  # Run attack scenario only
  python experiments/fl_pipeline_experiment.py --attack
  
  # Run all experiments (full suite)
  python experiments/fl_pipeline_experiment.py --all
  
  # Run custom configuration
  python experiments/fl_pipeline_experiment.py --custom --num-benign 5 --num-malicious 2 --noise-scale 0.15
        """
    )
    
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline experiment (all benign clients)"
    )
    parser.add_argument(
        "--attack",
        action="store_true",
        help="Run attack experiment (Gaussian noise + accuracy scoring)"
    )
    parser.add_argument(
        "--defense",
        action="store_true",
        help="Run defense experiment (Gaussian noise + Multi-Krum)"
    )
    parser.add_argument(
        "--pinn",
        action="store_true",
        help="Run PINN Guard defense experiment (Gaussian noise + PINN Guard)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiment scenarios"
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Run custom configuration"
    )
    
    # Custom config parameters
    parser.add_argument("--num-benign", type=int, default=9, help="Number of benign clients")
    parser.add_argument("--num-malicious", type=int, default=1, help="Number of malicious clients")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Gaussian noise scale")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    parser.add_argument("--scoring", choices=["accuracy", "multi_krum", "pinn_guard"], 
                       default="accuracy", help="Scoring policy")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.all:
        runner.run_all()
    elif args.baseline:
        config = runner.create_baseline_config()
        experiment = FLExperiment(config)
        experiment.run()
    elif args.attack:
        config = runner.create_attack_config()
        experiment = FLExperiment(config)
        experiment.run()
    elif args.defense:
        config = runner.create_defense_config()
        experiment = FLExperiment(config)
        experiment.run()
    elif args.pinn:
        config = runner.create_pinn_defense_config()
        experiment = FLExperiment(config)
        experiment.run()
    elif args.custom:
        config = ExperimentConfig(
            name=f"custom_{int(time.time())}",
            num_benign_clients=args.num_benign,
            num_malicious_clients=args.num_malicious,
            noise_scale=args.noise_scale,
            num_rounds=args.rounds,
            epochs_per_round=args.epochs,
            aggregation_policy="pick_top_k",
            scoring_policy=args.scoring,
            k=5,
            workload="cifar10",
            batch_size=32,
        )
        experiment = FLExperiment(config)
        experiment.run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
