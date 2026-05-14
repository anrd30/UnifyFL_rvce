#!/usr/bin/env bash
set -euo pipefail


# ========================
# Python & Poetry Setup
# ========================

echo "Warning poetry and python will be installed for this user"
echo "Run this in a virtualenv to prevent this"

OS="$(uname -s)"
PYTHON_BIN=""

require_cmd() {
    if ! command -v "$1" &> /dev/null; then
        echo ">>> Missing required command: $1"
        exit 1
    fi
}

detect_python_310() {
    if command -v python3.10 &> /dev/null; then
        PYTHON_BIN="python3.10"
        return 0
    fi

    if command -v python3 &> /dev/null; then
        local pyv
        pyv="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        if [[ "$pyv" == "3.10" ]]; then
            PYTHON_BIN="python3"
            return 0
        fi
    fi

    return 1
}

# --- Check Python 3.10 ---
if ! detect_python_310; then
    echo ">>> Installing Python 3.10..."
    if [[ "$OS" == "Linux" ]]; then
        require_cmd sudo
        require_cmd apt-get
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
        PYTHON_BIN="python3.10"
    elif [[ "$OS" == "Darwin" ]]; then
        require_cmd brew
        HOMEBREW_NO_AUTO_UPDATE=1 brew install python@3.10
        PYTHON_BIN="$(brew --prefix python@3.10)/bin/python3.10"
    else
        echo ">>> Unsupported OS: $OS"
        exit 1
    fi
else
    echo ">>> Python 3.10 already installed."
fi

# --- Install Poetry v1.4 ---
if ! command -v poetry &> /dev/null; then
    echo ">>> Installing Poetry 1.4..."
    require_cmd curl
    curl -sSL https://install.python-poetry.org | "$PYTHON_BIN" - --version 1.4.0
    export PATH="$HOME/.local/bin:$PATH"
else
    echo ">>> Poetry already installed."
fi

# ========================
# Install Python Project Dependencies
# ========================

echo ">>> Installing project dependencies with Poetry..."
poetry install

# Creating folders for models
mkdir -p upload
mkdir -p download
echo ">>> Setup complete!"


