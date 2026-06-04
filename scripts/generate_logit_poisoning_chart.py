import re
from pathlib import Path
import matplotlib.pyplot as plt

def parse_global_accuracies(log_path: Path):
    if not log_path.exists():
        print(f"Log path does not exist: {log_path}")
        return []
    
    global_accs = []
    current_trainer = None
    with open(log_path, 'r') as f:
        for line in f:
            if "trainer" in line:
                m = re.search(r"'(?:trainer)':\s*'(0x[a-fA-F0-9]+)'", line)
                if not m:
                    m = re.search(r"'(?:trainer)':\s*\"(0x[a-fA-F0-9]+)\"", line)
                if m:
                    current_trainer = m.group(1)
            elif "Accuracy:" in line and current_trainer == "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266":
                m_acc = re.search(r"Accuracy:\s*(\d+\.?\d*)%", line)
                if m_acc:
                    global_accs.append(float(m_acc.group(1)))
                current_trainer = None
    
    # Since logs can append across runs, take the last 101 entries (round 0 to 100)
    return global_accs[-101:]

def main():
    results_base = Path("results/fl_experiments")
    
    scenarios = {
        "Clean Baseline (No Attack)": results_base / "baseline_10benign" / "scorer.log",
        "Undefended Logit Poisoning Attack": results_base / "custom_1780467607" / "scorer_0.log",
        "PINN Guard Defended Logit Poisoning": results_base / "custom_1780485497" / "scorer_0.log",
    }
    
    plt.figure(figsize=(10, 6))
    
    # Modern professional colors
    colors = {
        "Clean Baseline (No Attack)": "#2ecc71",       # Emerald Green
        "Undefended Logit Poisoning Attack": "#e74c3c",      # Crimson Red
        "PINN Guard Defended Logit Poisoning": "#3498db",     # Ocean Blue
    }
    
    markers = {
        "Clean Baseline (No Attack)": "o",
        "Undefended Logit Poisoning Attack": "x",
        "PINN Guard Defended Logit Poisoning": "s",
    }

    for name, path in scenarios.items():
        accs = parse_global_accuracies(path)
        if accs:
            rounds = list(range(len(accs)))
            plt.plot(
                rounds, accs, 
                label=f"{name} (Final: {accs[-1]:.2f}%)", 
                color=colors[name], 
                marker=markers[name], 
                linewidth=2, 
                markersize=4,
                markevery=5 # show marker every 5 rounds to reduce clutter
            )
            print(f"Parsed {len(accs)} rounds for {name}. Final: {accs[-1]:.2f}%")
        else:
            print(f"No accuracy data found for {name}")
            
    plt.title("FL Robustness to Logit Poisoning on CIFAR-10 (100 Rounds)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("FL Round", fontsize=12, labelpad=10)
    plt.ylabel("Global Classification Accuracy (%)", fontsize=12, labelpad=10)
    plt.ylim(0, 55)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    
    output_dir = Path("assets/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "logit_poisoning_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved comparison chart to: {output_path}")

if __name__ == "__main__":
    main()
