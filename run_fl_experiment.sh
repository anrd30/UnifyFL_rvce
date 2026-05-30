#!/bin/bash
# FL Pipeline Orchestrator - All-in-one experiment runner
# 
# Orchestrates the complete FL pipeline:
# - Blockchain (Anvil)
# - IPFS (Kubo)
# - Smart Contract Deployment
# - Aggregator
# - Scorer
# - Multiple FL Clients (mix of benign and malicious)
#
# Usage:
#   bash run_fl_experiment.sh baseline          # All benign clients
#   bash run_fl_experiment.sh attack            # 1 attacker + 9 benign
#   bash run_fl_experiment.sh defense           # Attack + Multi-Krum defense
#   bash run_fl_experiment.sh all               # Run all scenarios

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$PROJECT_DIR/results/fl_experiments"
LOG_DIR="$PROJECT_DIR/logs/fl_pipeline"
BLOCKCHAIN_PORT=8545
IPFS_PORT=5001
FL_SERVER_PORT=5000

# Create directories
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    local all_ok=true
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_info "Python: $(python3 --version)"
    else
        print_error "Python 3 not found"
        all_ok=false
    fi
    
    # Check Poetry
    if command -v poetry &> /dev/null; then
        print_info "Poetry: installed"
    else
        print_error "Poetry not found (required)"
        all_ok=false
    fi
    
    # Check Anvil
    if command -v anvil &> /dev/null; then
        print_info "Anvil: installed"
    else
        print_warning "Anvil not found (required for blockchain)"
        echo "  Install with: brew install foundry"
        all_ok=false
    fi
    
    # Check IPFS
    if command -v ipfs &> /dev/null; then
        print_info "IPFS: installed"
    else
        print_warning "IPFS not found (required for distributed storage)"
        echo "  Install with: brew install ipfs"
        all_ok=false
    fi
    
    if [ "$all_ok" = false ]; then
        print_error "Missing prerequisites. Please install all required tools."
        exit 1
    fi
}

start_blockchain() {
    print_header "Starting Blockchain (Anvil)"
    
    if lsof -Pi :$BLOCKCHAIN_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $BLOCKCHAIN_PORT already in use (blockchain might be running)"
    else
        print_info "Starting Anvil on port $BLOCKCHAIN_PORT..."
        nohup anvil > "$LOG_DIR/anvil.log" 2>&1 &
        sleep 3
        
        if lsof -Pi :$BLOCKCHAIN_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_info "Anvil started successfully"
        else
            print_error "Failed to start Anvil"
            exit 1
        fi
    fi
}

start_ipfs() {
    print_header "Starting IPFS Daemon"
    
    if curl -s http://localhost:$IPFS_PORT/api/v0/version &> /dev/null; then
        print_warning "IPFS already running"
    else
        print_info "Starting IPFS daemon..."
        
        # Initialize if needed
        if [ ! -d ~/.ipfs ]; then
            print_info "Initializing IPFS..."
            ipfs init > /dev/null 2>&1
        fi
        
        nohup ipfs daemon --offline > "$LOG_DIR/ipfs.log" 2>&1 &
        sleep 3
        
        if curl -s http://localhost:$IPFS_PORT/api/v0/version &> /dev/null; then
            print_info "IPFS daemon started successfully"
        else
            print_error "Failed to start IPFS daemon"
            exit 1
        fi
    fi
}

deploy_contracts() {
    print_header "Deploying Smart Contracts"
    
    local mode=$1  # 0 for sync, 1 for async
    local agg_policy=$2
    local score_policy=$3
    local k=$4
    
    print_info "Deploying contracts..."
    print_info "  Mode: $([ $mode -eq 0 ] && echo 'Synchronous' || echo 'Asynchronous')"
    print_info "  Aggregation: $agg_policy"
    print_info "  Scoring: $score_policy"
    print_info "  K: $k"
    
    cd "$PROJECT_DIR"
    poetry run python deploy_scripts/deploy_contracts.py $mode $agg_policy $score_policy $k
    
    print_info "Contracts deployed successfully"
}

run_experiment_scenario() {
    local scenario=$1  # baseline, attack, defense
    local config_name=$2
    local num_benign=$3
    local num_malicious=$4
    local noise_scale=$5
    local scoring_policy=$6
    
    print_header "Running Experiment: $scenario"
    
    local scenario_dir="$RESULTS_DIR/${config_name}"
    mkdir -p "$scenario_dir"
    
    print_info "Configuration:"
    print_info "  Scenario: $scenario"
    print_info "  Benign clients: $num_benign"
    print_info "  Malicious clients: $num_malicious"
    if [ "$num_malicious" -gt 0 ]; then
        print_info "  Noise scale: $noise_scale"
    fi
    print_info "  Scoring policy: $scoring_policy"
    print_info "  Output: $scenario_dir"
    
    # Run the Python experiment
    cd "$PROJECT_DIR"
    
    if [ "$scenario" = "baseline" ]; then
        poetry run python experiments/fl_pipeline_experiment.py --baseline
    elif [ "$scenario" = "attack" ]; then
        poetry run python experiments/fl_pipeline_experiment.py --attack
    elif [ "$scenario" = "defense" ]; then
        poetry run python experiments/fl_pipeline_experiment.py --defense
    elif [ "$scenario" = "pinn" ]; then
        poetry run python experiments/fl_pipeline_experiment.py --pinn
    else
        print_error "Unknown scenario: $scenario"
        return 1
    fi
    
    print_info "Experiment completed: $scenario"
}

analyze_results() {
    print_header "Analyzing Results"
    
    cd "$PROJECT_DIR"
    
    local compare_args=""
    for name in baseline_10benign attack_gaussian_noise_accuracy defense_multikrum_vs_noise defense_pinn_vs_noise; do
        if [ -d "$RESULTS_DIR/$name" ]; then
            compare_args="$compare_args $name"
        fi
    done
    
    if [ -n "$compare_args" ]; then
        echo "Comparing scenarios: $compare_args"
        poetry run python experiments/fl_analysis.py --compare $compare_args
    else
        print_warning "No experiment result directories found to analyze."
    fi
}

cleanup() {
    print_header "Cleanup"
    
    print_warning "Terminating processes..."
    pkill -f anvil 2>/dev/null || true
    pkill -f "ipfs daemon" 2>/dev/null || true
    pkill -f "python.*experiment" 2>/dev/null || true
    
    print_info "Cleanup complete"
}

# Trap Ctrl+C and cleanup
trap cleanup EXIT INT TERM

# Main execution
case "${1:-all}" in
    check)
        check_prerequisites
        ;;
    
    setup)
        check_prerequisites
        start_blockchain
        start_ipfs
        print_info "Infrastructure ready!"
        print_info "Components running:"
        print_info "  - Blockchain: http://localhost:$BLOCKCHAIN_PORT"
        print_info "  - IPFS: http://localhost:$IPFS_PORT"
        print_info ""
        print_info "Next steps:"
        print_info "  1. Deploy contracts: bash run_fl_experiment.sh deploy"
        print_info "  2. Run experiments: bash run_fl_experiment.sh baseline|attack|defense|all"
        ;;
    
    deploy)
        deploy_contracts 0 pick_top_k assign_score_mean 5
        ;;
    
    baseline)
        check_prerequisites
        start_blockchain
        start_ipfs
        deploy_contracts 0 pick_top_k assign_score_mean 5
        run_experiment_scenario "baseline" "baseline_10benign" 10 0 0.0 "accuracy"
        analyze_results
        ;;
    
    attack)
        check_prerequisites
        start_blockchain
        start_ipfs
        deploy_contracts 0 pick_top_k assign_score_mean 5
        run_experiment_scenario "attack" "attack_gaussian_noise_accuracy" 9 1 0.1 "accuracy"
        analyze_results
        ;;
    
    defense)
        check_prerequisites
        start_blockchain
        start_ipfs
        deploy_contracts 0 pick_top_k assign_score_mean 5
        run_experiment_scenario "defense" "defense_multikrum_vs_noise" 9 1 0.1 "multi_krum"
        analyze_results
        ;;
    
    pinn)
        check_prerequisites
        start_blockchain
        start_ipfs
        deploy_contracts 0 pick_top_k assign_score_mean 5
        run_experiment_scenario "pinn" "defense_pinn_vs_noise" 9 1 0.1 "pinn_guard"
        analyze_results
        ;;
    
    all)
        check_prerequisites
        start_blockchain
        start_ipfs
        deploy_contracts 0 pick_top_k assign_score_mean 5
        
        run_experiment_scenario "baseline" "baseline_10benign" 10 0 0.0 "accuracy"
        sleep 30
        
        run_experiment_scenario "attack" "attack_gaussian_noise_accuracy" 9 1 0.1 "accuracy"
        sleep 30
        
        run_experiment_scenario "defense" "defense_multikrum_vs_noise" 9 1 0.1 "multi_krum"
        sleep 30
        
        run_experiment_scenario "pinn" "defense_pinn_vs_noise" 9 1 0.1 "pinn_guard"
        
        analyze_results
        ;;
    
    *)
        echo "FL Pipeline Orchestrator"
        echo ""
        echo "Usage: bash run_fl_experiment.sh <command>"
        echo ""
        echo "Commands:"
        echo "  check              - Check if prerequisites are installed"
        echo "  setup              - Start blockchain and IPFS (keeps running)"
        echo "  deploy             - Deploy smart contracts"
        echo "  baseline           - Run baseline experiment (all benign)"
        echo "  attack             - Run attack experiment (1 attacker + accuracy scoring)"
        echo "  defense            - Run defense experiment (Multi-Krum scoring)"
        echo "  pinn               - Run PINN Guard defense experiment"
        echo "  all                - Run all experiments in sequence"
        echo ""
        echo "Examples:"
        echo "  bash run_fl_experiment.sh check"
        echo "  bash run_fl_experiment.sh setup    # Terminal 1"
        echo "  bash run_fl_experiment.sh baseline # Terminal 2"
        echo "  bash run_fl_experiment.sh pinn     # Run PINN Guard defense"
        echo "  bash run_fl_experiment.sh all      # Runs all scenarios"
        echo ""
        exit 1
        ;;
esac

print_info "Done!"
