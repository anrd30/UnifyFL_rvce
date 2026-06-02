#!/bin/bash
# 100-Round FL Pipeline Logit Poisoning Experiment Runner
#
# Orchestrates the complete FL pipeline for logit level poisoning:
# - Blockchain (Anvil)
# - IPFS (Kubo)
# - Smart Contract Deployment
# - Custom FL Pipeline Experiment with logit poisoning

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$PROJECT_DIR/results/fl_experiments"
LOG_DIR="$PROJECT_DIR/logs/fl_pipeline"
BLOCKCHAIN_PORT=8545
IPFS_PORT=5001

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

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

start_blockchain() {
    print_header "Starting Blockchain (Anvil)"
    if lsof -Pi :$BLOCKCHAIN_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port $BLOCKCHAIN_PORT already in use (blockchain might be running)"
    else
        print_info "Starting Anvil on port $BLOCKCHAIN_PORT..."
        nohup anvil > "$LOG_DIR/anvil.log" 2>&1 &
        sleep 3
    fi
}

start_ipfs() {
    print_header "Starting IPFS Daemon"
    if curl -s http://localhost:$IPFS_PORT/api/v0/version &> /dev/null; then
        print_warning "IPFS already running"
    else
        print_info "Starting IPFS daemon..."
        if [ ! -d ~/.ipfs ]; then
            ipfs init > /dev/null 2>&1
        fi
        nohup ipfs daemon --offline > "$LOG_DIR/ipfs.log" 2>&1 &
        sleep 3
    fi
}

deploy_contracts() {
    print_header "Deploying Smart Contracts"
    cd "$PROJECT_DIR"
    poetry run python deploy_scripts/deploy_contracts.py 0 pick_top_k assign_score_mean 5
    print_info "Contracts deployed successfully"
}

run_experiment() {
    print_header "Running 100-Round Logit Poisoning Experiment"
    cd "$PROJECT_DIR"
    poetry run python experiments/fl_pipeline_experiment.py --custom \
        --num-benign 9 \
        --num-malicious 1 \
        --noise-scale 2.0 \
        --rounds 100 \
        --epochs 3 \
        --attack-type logit_poisoning
}

cleanup() {
    print_header "Cleanup"
    print_warning "Terminating background processes..."
    pkill -f anvil 2>/dev/null || true
    pkill -f "ipfs daemon" 2>/dev/null || true
    pkill -f "python.*experiment" 2>/dev/null || true
    print_info "Cleanup complete"
}

# Trap Ctrl+C and exit/cleanup
trap cleanup EXIT INT TERM

# Execute
start_blockchain
start_ipfs
deploy_contracts
run_experiment

print_info "Experiment Run Completed successfully!"
