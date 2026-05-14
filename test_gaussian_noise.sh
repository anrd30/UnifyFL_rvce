#!/bin/bash
# Quick-start script for testing Gaussian noise poisoning attack

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_header "Gaussian Noise Poisoning Attack - Quick Start"

echo "This script helps you test the Gaussian noise poisoning attack on the CIFAR-10 model."
echo ""
echo "Choose what you want to do:"
echo ""
echo "1) Test attack robustness (local evaluation, fast)"
echo "2) Test with default noise scales (0.01, 0.05, 0.1, 0.15, 0.2)"
echo "3) Test with custom noise scales"
echo "4) View attack documentation"
echo "5) Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        print_header "Running Local Robustness Evaluation"
        print_info "Testing model accuracy with increasing noise levels..."
        print_info "This takes ~2-5 minutes depending on your hardware\n"
        
        python experiments/gaussian_noise_attack.py --test-robustness
        
        print_info "Results saved to: results/gaussian_noise_robustness.json"
        ;;
        
    2)
        print_header "Running Robustness Test (Custom Samples)"
        
        read -p "Number of test samples (default 1000): " num_samples
        num_samples=${num_samples:-1000}
        
        print_info "Testing with $num_samples samples..."
        python experiments/gaussian_noise_attack.py --test-robustness --num-samples $num_samples
        
        print_info "Results saved to: results/gaussian_noise_robustness.json"
        ;;
        
    3)
        print_header "Running with Custom Noise Scales"
        
        echo "Enter noise scales separated by spaces (e.g., 0.01 0.05 0.1):"
        read -p "Noise scales: " noise_scales
        
        python experiments/gaussian_noise_attack.py --test-robustness --noise-scales $noise_scales
        
        print_info "Results saved to: results/gaussian_noise_robustness.json"
        ;;
        
    4)
        print_header "Gaussian Noise Poisoning Attack Documentation"
        
        if command -v less &> /dev/null; then
            less GAUSSIAN_NOISE_ATTACK.md
        else
            cat GAUSSIAN_NOISE_ATTACK.md
        fi
        ;;
        
    5)
        print_info "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Show next steps
print_header "Next Steps"
echo "1. Review results: cat results/gaussian_noise_robustness.json"
echo "2. For FL deployment, use:"
echo "   IS_MALICIOUS=true ATTACK_TYPE=gaussian_noise NOISE_SCALE=0.1 \\"
echo "   python -m unifyfl.sync.client configs/party.json"
echo ""
echo "3. Read full documentation:"
echo "   cat GAUSSIAN_NOISE_ATTACK.md"
echo ""
