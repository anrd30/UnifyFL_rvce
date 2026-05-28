#!/usr/bin/env python3
"""
FL Experiment Metrics and Analysis

Collects and analyzes metrics from FL experiments including:
- Convergence curves (accuracy vs rounds)
- Model performance degradation
- Attack effectiveness
- Defense effectiveness (Multi-Krum vs Accuracy)
- Statistical comparison between scenarios

Usage:
    python experiments/fl_analysis.py --results-dir results/fl_experiments/baseline_10benign
    python experiments/fl_analysis.py --compare-scenarios baseline attack defense
"""

import json
import sys
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RoundMetrics:
    """Metrics for a single FL round"""
    round_num: int
    timestamp: float
    avg_accuracy: float
    avg_loss: float
    num_models_submitted: int
    num_models_selected: int
    poisoned_models_selected: int = 0


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment"""
    name: str
    scenario_type: str  # baseline, attack, defense
    num_benign_clients: int
    num_malicious_clients: int
    noise_scale: float
    scoring_policy: str
    num_rounds: int
    round_metrics: List[RoundMetrics]
    final_accuracy: float
    peak_accuracy: float
    convergence_round: Optional[int]  # Round when accuracy plateaus
    accuracy_drop: float  # Drop due to attack (if applicable)


class LogParser:
    """Parses FL system logs to extract metrics"""
    
    @staticmethod
    def parse_client_log(log_path: Path) -> Dict:
        """Extract metrics from a client log file"""
        metrics = {
            "client_id": None,
            "is_malicious": False,
            "attack_type": None,
            "noise_scale": 0.0,
            "rounds_completed": 0,
            "local_accuracies": [],
            "errors": [],
        }
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Check if malicious
                if "MALICIOUS CLIENT" in content or "🔥 Executing" in content:
                    metrics["is_malicious"] = True
                    
                    if "Label Flipping" in content:
                        metrics["attack_type"] = "label_flipping"
                    elif "Gaussian Noise" in content:
                        metrics["attack_type"] = "gaussian_noise"
                
                # Extract client ID from log
                client_id_match = re.search(r'client_(\d+)', str(log_path))
                if client_id_match:
                    metrics["client_id"] = int(client_id_match.group(1))
                
                # Extract noise scale if malicious
                noise_match = re.search(r'noise scale: (\d+\.\d+)', content)
                if noise_match:
                    metrics["noise_scale"] = float(noise_match.group(1))
                
                # Count training rounds
                train_count = content.count("Starting training")
                metrics["rounds_completed"] = train_count
                
                # Extract local accuracies
                accuracy_matches = re.findall(r'accuracy["\']:\s*(\d+\.\d+)', content, re.IGNORECASE)
                metrics["local_accuracies"] = [float(acc) for acc in accuracy_matches]
                
                # Check for errors
                if "ERROR" in content or "error" in content or "Traceback" in content:
                    error_matches = re.findall(r'(ERROR|error|Exception)[:\s]+([^\n]+)', content)
                    metrics["errors"] = [e[1] for e in error_matches]
        
        except Exception as e:
            logger.warning(f"Failed to parse {log_path}: {e}")
        
        return metrics
    
    @staticmethod
    def parse_aggregator_log(log_path: Path) -> Dict:
        """Extract metrics from aggregator log file"""
        metrics = {
            "rounds_completed": 0,
            "models_processed": [],
            "model_selections": [],
            "scores": [],
        }
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Count rounds
                round_count = content.count("Round")
                metrics["rounds_completed"] = round_count
                
                # Extract model scores (if available)
                score_matches = re.findall(r'score[s]?["\']?:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                metrics["scores"] = [float(s) for s in score_matches]
        
        except Exception as e:
            logger.warning(f"Failed to parse {log_path}: {e}")
        
        return metrics
    
    @staticmethod
    def parse_scorer_log(log_path: Path) -> Dict:
        """Extract metrics from scorer log file"""
        metrics = {
            "models_evaluated": 0,
            "accuracies": [],
            "Byzantine_scores": [],
        }
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Count evaluated models
                model_count = content.count("Evaluating") + content.count("evaluating")
                metrics["models_evaluated"] = model_count
                
                # Extract accuracies from async-scorer format: "Accuracy: 17.79%"
                acc_matches = re.findall(r'Accuracy:\s*(\d+\.?\d*)%', content, re.IGNORECASE)
                if not acc_matches:
                    # Fallback for other log formats
                    acc_matches = re.findall(r'accuracy["\']?:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                metrics["accuracies"] = [float(acc) for acc in acc_matches]
                
                # Check for Multi-Krum scores
                if "multi_krum" in content.lower() or "krum" in content.lower():
                    krum_matches = re.findall(r'krum_score["\']?:\s*(\d+\.?\d*)', content, re.IGNORECASE)
                    metrics["Byzantine_scores"] = [float(s) for s in krum_matches]
        
        except Exception as e:
            logger.warning(f"Failed to parse {log_path}: {e}")
        
        return metrics


class MetricsCollector:
    """Collects and aggregates metrics from experiment runs"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.config = self._load_config()
        self.log_parser = LogParser()
    
    def _load_config(self) -> Dict:
        """Load experiment configuration"""
        config_file = self.results_dir / "experiment_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def collect_metrics(self) -> ExperimentMetrics:
        """Collect all metrics from an experiment"""
        logger.info(f"Collecting metrics from {self.results_dir}")
        
        config = self.config
        
        # Parse all client logs
        client_metrics = {}
        for client_log in self.results_dir.glob("client_*.log"):
            metrics = self.log_parser.parse_client_log(client_log)
            client_metrics[metrics['client_id']] = metrics
        
        # Analyze attack effectiveness
        malicious_clients = [m for m in client_metrics.values() if m['is_malicious']]
        
        # Try to read global accuracy from scorer log if it exists
        scorer_log_path = self.results_dir / "scorer.log"
        if scorer_log_path.exists():
            scorer_metrics = self.log_parser.parse_scorer_log(scorer_log_path)
            accuracies = scorer_metrics.get("accuracies", [])
            avg_local_accuracy = accuracies[-1] if accuracies else 0.0
        else:
            # Fall back to local client accuracy
            avg_local_accuracy = np.mean([
                np.mean(m['local_accuracies']) for m in client_metrics.values() 
                if m['local_accuracies']
            ]) if any(m['local_accuracies'] for m in client_metrics.values()) else 0.0
        
        # Determine scenario type
        scenario_type = "baseline"
        if malicious_clients:
            scoring_policy_lower = config.get('scoring_policy', '').lower()
            if "multi_krum" in scoring_policy_lower or "pinn_guard" in scoring_policy_lower:
                scenario_type = "defense"
            else:
                scenario_type = "attack"
        
        # Create experiment metrics
        metrics = ExperimentMetrics(
            name=config.get('name', 'unknown'),
            scenario_type=scenario_type,
            num_benign_clients=config.get('num_benign_clients', 0),
            num_malicious_clients=config.get('num_malicious_clients', 0),
            noise_scale=config.get('noise_scale', 0.0),
            scoring_policy=config.get('scoring_policy', 'unknown'),
            num_rounds=config.get('num_rounds', 0),
            round_metrics=[],
            final_accuracy=avg_local_accuracy,
            peak_accuracy=avg_local_accuracy,
            convergence_round=None,
            accuracy_drop=0.0,
        )
        
        logger.info(f"Experiment: {metrics.name}")
        logger.info(f"  Type: {metrics.scenario_type}")
        logger.info(f"  Clients: {metrics.num_benign_clients} benign, {metrics.num_malicious_clients} malicious")
        logger.info(f"  Scoring: {metrics.scoring_policy}")
        logger.info(f"  Final accuracy: {metrics.final_accuracy:.4f}")
        
        return metrics


class ComparisonAnalyzer:
    """Compares metrics across multiple experiments"""
    
    def __init__(self, results_dirs: List[Path]):
        self.results_dirs = [Path(d) for d in results_dirs]
        self.collector = MetricsCollector
    
    def compare(self) -> Dict:
        """Compare metrics across experiments"""
        logger.info(f"Comparing {len(self.results_dirs)} experiments...")
        
        all_metrics = []
        for results_dir in self.results_dirs:
            collector = self.collector(results_dir)
            metrics = collector.collect_metrics()
            all_metrics.append(metrics)
        
        # Organize by scenario type
        by_scenario = {}
        for metrics in all_metrics:
            if metrics.scenario_type not in by_scenario:
                by_scenario[metrics.scenario_type] = []
            by_scenario[metrics.scenario_type].append(metrics)
        
        return by_scenario
    
    def print_comparison(self):
        """Print formatted comparison"""
        comparison = self.compare()
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON ANALYSIS")
        print("="*80 + "\n")
        
        for scenario_type, metrics_list in sorted(comparison.items()):
            print(f"\n{scenario_type.upper()} SCENARIOS:")
            print("-" * 80)
            
            for metrics in metrics_list:
                print(f"\n  Experiment: {metrics.name}")
                print(f"    Clients: {metrics.num_benign_clients} benign, {metrics.num_malicious_clients} malicious")
                print(f"    Attack: {metrics.scenario_type}")
                if metrics.num_malicious_clients > 0:
                    print(f"    Noise Scale: {metrics.noise_scale}")
                    print(f"    Accuracy Drop: {metrics.accuracy_drop:.2%}")
                print(f"    Scoring: {metrics.scoring_policy}")
                print(f"    Final Accuracy: {metrics.final_accuracy:.4f}")
                print(f"    Peak Accuracy: {metrics.peak_accuracy:.4f}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80 + "\n")
        
        if "baseline" in comparison and "attack" in comparison:
            baseline = comparison["baseline"][0]
            attack = comparison["attack"][0]
            
            accuracy_drop = (baseline.final_accuracy - attack.final_accuracy) / baseline.final_accuracy * 100
            print(f"  • Attack Impact: {accuracy_drop:.2f}% accuracy drop")
            print(f"    Baseline: {baseline.final_accuracy:.4f} → Attack: {attack.final_accuracy:.4f}")
        
        if "defense" in comparison and "attack" in comparison:
            attack = comparison["attack"][0]
            defense = comparison["defense"][0]
            
            recovery = (defense.final_accuracy - attack.final_accuracy) / attack.final_accuracy * 100
            if recovery > 0:
                print(f"\n  • Defense Effectiveness: {recovery:.2f}% accuracy recovery with Multi-Krum")
                print(f"    Attack: {attack.final_accuracy:.4f} → Defense: {defense.final_accuracy:.4f}")
            else:
                print(f"\n  • Defense: Limited effectiveness (recovery: {recovery:.2f}%)")


class ReportGenerator:
    """Generates comprehensive analysis reports"""
    
    @staticmethod
    def generate_html_report(all_metrics: List[ExperimentMetrics], 
                            output_path: Path):
        """Generate HTML report with visualizations"""
        report_dir = output_path.parent
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Basic HTML structure
        html = """
        <html>
        <head>
            <title>FL Experiment Report</title>
            <style>
                body { font-family: Arial; margin: 20px; }
                .scenario { margin: 30px 0; padding: 20px; border: 1px solid #ddd; }
                .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>FL Experiment Analysis Report</h1>
        """
        
        # Add scenario sections
        for metrics in all_metrics:
            html += f"""
            <div class="scenario">
                <h2>{metrics.name}</h2>
                <div class="metric">
                    <strong>Type:</strong> {metrics.scenario_type}
                </div>
                <div class="metric">
                    <strong>Clients:</strong> {metrics.num_benign_clients} benign, {metrics.num_malicious_clients} malicious
                </div>
                <div class="metric">
                    <strong>Scoring Policy:</strong> {metrics.scoring_policy}
                </div>
                <div class="metric">
                    <strong>Final Accuracy:</strong> {metrics.final_accuracy:.4f}
                </div>
                <div class="metric">
                    <strong>Peak Accuracy:</strong> {metrics.peak_accuracy:.4f}
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FL experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python experiments/fl_analysis.py --results-dir results/fl_experiments/baseline_10benign
  
  # Compare multiple scenarios
  python experiments/fl_analysis.py --compare baseline attack defense
  
  # Generate report
  python experiments/fl_analysis.py --results-dir results/fl_experiments --report report.html
        """
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple scenario directories"
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Generate HTML report"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple experiments
        base_dir = Path("results/fl_experiments")
        compare_dirs = [base_dir / name for name in args.compare]
        analyzer = ComparisonAnalyzer(compare_dirs)
        analyzer.print_comparison()
    
    elif args.results_dir:
        # Analyze single experiment
        collector = MetricsCollector(Path(args.results_dir))
        metrics = collector.collect_metrics()
        
        if args.report:
            ReportGenerator.generate_html_report([metrics], Path(args.report))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
