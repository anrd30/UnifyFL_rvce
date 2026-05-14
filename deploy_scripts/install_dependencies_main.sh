#!/usr/bin/env bash
set -euo pipefail
# ========================
# Install Dependencies
# ========================

echo ">>> Installing dependencies for UnifyFL experiments..."
echo ">>> Detecting platform and installing prerequisites..."

OS="$(uname -s)"

require_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo ">>> Missing required command: $1"
        exit 1
    fi
}

download_file() {
    local url="$1"
    local out="$2"
    if command -v wget &> /dev/null; then
        wget "$url" -O "$out"
    elif command -v curl &> /dev/null; then
        curl -L "$url" -o "$out"
    else
        echo ">>> Neither wget nor curl is available. Install one of them first."
        exit 1
    fi
}

# --- Install IPFS v0.17 (if not already installed) ---
if ! command -v ipfs &> /dev/null; then
    echo ">>> Installing IPFS..."
    if [[ "$OS" == "Linux" ]]; then
        require_cmd tar
        require_cmd sudo
        download_file "https://dist.ipfs.tech/kubo/v0.17.0/kubo_v0.17.0_linux-amd64.tar.gz" "/tmp/kubo.tar.gz"
        tar -xzf /tmp/kubo.tar.gz -C /tmp
        sudo bash /tmp/kubo/install.sh
        rm -rf /tmp/kubo /tmp/kubo.tar.gz
    elif [[ "$OS" == "Darwin" ]]; then
        require_cmd tar
        download_file "https://dist.ipfs.tech/kubo/v0.17.0/kubo_v0.17.0_darwin-arm64.tar.gz" "/tmp/kubo.tar.gz"
        tar -xzf /tmp/kubo.tar.gz -C /tmp
        bash /tmp/kubo/install.sh
        rm -rf /tmp/kubo /tmp/kubo.tar.gz
    else
        echo ">>> Unsupported OS: $OS"
        exit 1
    fi

    # ipfs init fails if repo is already initialized.
    ipfs init || true
else
    echo ">>> IPFS already installed."
fi

# --- Install Foundry (Forge + Anvil) ---
if ! command -v forge &> /dev/null; then
    echo ">>> Installing Foundry (forge + anvil)..."
    require_cmd curl
    curl -L https://foundry.paradigm.xyz | bash
    export PATH="$HOME/.foundry/bin:$PATH"
    foundryup
else
    echo ">>> Foundry (forge) already installed."
fi


# ========================
# Done
# ========================
echo ">>> Setup complete!"
