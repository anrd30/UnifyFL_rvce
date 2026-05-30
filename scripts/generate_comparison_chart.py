import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_accuracies(log_path: Path):
    if not log_path.exists():
        print(f"Log path does not exist: {log_path}")
        return []
    
    accuracies = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Accuracy:\s*(\d+\.?\d*)%', line, re.IGNORECASE)
            if match:
                accuracies.append(float(match.group(1)))
    return accuracies

def main():
    results_base = Path("results/fl_experiments")
    
    scenarios = {
        "Baseline (Clean)": results_base / "baseline_10benign" / "scorer.log",
        "Undefended Attack": results_base / "attack_gaussian_noise_accuracy" / "scorer.log",
        "PINN Guard Defense": results_base / "defense_pinn_vs_noise" / "scorer.log",
    }
    
    plt.figure(figsize=(10, 6))
    
    # Modern professional colors
    colors = {
        "Baseline (Clean)": "#2ecc71",       # Emerald Green
        "Undefended Attack": "#e74c3c",      # Crimson Red
        "PINN Guard Defense": "#3498db",     # Ocean Blue
    }
    
    markers = {
        "Baseline (Clean)": "o",
        "Undefended Attack": "x",
        "PINN Guard Defense": "s",
    }

    for name, path in scenarios.items():
        accs = parse_accuracies(path)
        if accs:
            # Only plot up to 100 rounds
            accs = accs[:100]
            rounds = list(range(1, len(accs) + 1))
            plt.plot(
                rounds, accs, 
                label=f"{name} (Final: {accs[-1]:.2f}%)", 
                color=colors[name], 
                marker=markers[name], 
                linewidth=2, 
                markersize=6
            )
            print(f"Parsed {len(accs)} rounds for {name}. Final: {accs[-1]:.2f}%")
        else:
            print(f"No accuracy data found for {name}")
            
    plt.title("Federated Learning Accuracy on CIFAR-10 (100 Rounds)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("FL Round", fontsize=12, labelpad=10)
    plt.ylabel("Global Classification Accuracy (%)", fontsize=12, labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    
    output_path = results_base / "cifar10_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved comparison chart to: {output_path}")

if __name__ == "__main__":
    main()
