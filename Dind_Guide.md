# Docker-in-Docker (DinD) Setup Guide for UnifyFL

## Overview

This guide explains how to set up Docker-in-Docker (DinD) for UnifyFL to simulate a realistic distributed federated learning environment with isolated nodes on a single host.

**Why DinD for UnifyFL?**
- **Node Isolation**: Each simulated "node" runs its own nested Docker daemon, providing realistic network and process isolation.
- **Cross-Platform Compatibility**: Enables Windows/Mac/Linux users to run a realistic Linux-based multi-node deployment locally.
- **Topology Simulation**: Runs multiple aggregators on separate nodes, colocated FL clients, and dedicated infrastructure (Blockchain and IPFS storage).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Host Machine (Windows/Mac/Linux)                            │
│ Running Docker Desktop / Engine                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐
│  │  Node 1 (DinD)   │  │  Node 2 (DinD)   │  │ Node 3 (DinD)│
│  │ (Aggregator 1)   │  │ (Aggregator 2)   │  │(Aggregator 3)│
│  │                  │  │                  │  │              │
│  │ Docker daemon    │  │ Docker daemon    │  │Docker daemon │
│  │ └─ FL clients    │  │ └─ FL clients    │  │└─FL clients  │
│  │ └─ Aggregator    │  │ └─ Aggregator    │  │└─Aggregator  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘
│                                                             │
│  ┌─────────────────────────────────────────┐                │
│  │  Node 4 (DinD) - Infrastructure         │                │
│  │  (Blockchain + Storage)                 │                │
│  │                                         │                │
│  │  Docker daemon                          │                │
│  │  ├─ Anvil (blockchain)                  │                │
│  │  └─ Kubo/IPFS (distributed storage)     │                │
│  └─────────────────────────────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Each "Node X (DinD)" is a Docker container running a Docker daemon inside, capable of spinning up its own nested services.

---

## Prerequisites

### 1. Install Docker Desktop
- **Windows/Mac**: Download from [docker.com](https://www.docker.com/products/docker-desktop)
- **Linux**: Install Docker Engine via your package manager

### 2. Minimum System Requirements
- **CPU**: 4+ cores (DinD requires additional overhead)
- **RAM**: 16GB+ (8GB minimum)
- **Disk**: 30GB+ free space

### 3. Git & Bash
- Git for cloning the repository
- Bash shell (native on Linux/Mac, or WSL on Windows)

---

## Setup Instructions

### Step 1: Clone the Repository

```bash
cd ~
git clone https://github.com/DaSH-Lab-CSIS/UnifyFL
cd UnifyFL
```

### Step 2: Create the DinD Dockerfile (`Dockerfile.dind`)

Create a `Dockerfile.dind` in the repository root. We use a **Debian-based image** (not Alpine) to ensure proper compatibility with python dependencies (e.g., PyTorch, cryptography) and pin specific docker-py versions to avoid client errors:

```dockerfile
# Dockerfile.dind - DinD image for UnifyFL nodes
FROM debian:bookworm

# Install Docker and required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    docker-compose \
    python3 python3-pip python3-venv \
    curl bash git \
    build-essential python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and compatible Docker/urllib3 versions
RUN pip install --no-cache-dir --break-system-packages \
    poetry \
    docker==6.1.3 \
    urllib3==1.26.18 \
    requests==2.31.0 \
    certifi==2023.7.22

# Clone UnifyFL into the container
RUN git clone https://github.com/DaSH-Lab-CSIS/UnifyFL /unifyfl
WORKDIR /unifyfl

# Install dependencies (skip NVIDIA for DinD environment)
RUN poetry install --no-directory

# Install Foundry tools (Anvil, etc.)
RUN curl -L https://foundry.paradigm.xyz | bash && \
    /root/.foundry/bin/foundryup

# Install IPFS/Kubo (latest via package or direct download)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ipfs-go \
    && rm -rf /var/lib/apt/lists/* || \
    (curl -L https://github.com/ipfs/kubo/releases/download/v0.24.0/kubo_v0.24.0_linux-amd64.tar.gz | tar xz && \
    mv kubo/ipfs /usr/local/bin/ && \
    rm -rf kubo)

# Expose necessary ports
EXPOSE 8545 5001 8080 8000 8001 8002

# Start Docker daemon and keep container running
CMD ["/bin/sh", "-c", "service docker start && sleep infinity"]
```

### Step 3: Build the DinD Image

From the repository root, build the DinD base image:

```bash
docker build -f Dockerfile.dind -t unifyfl-dind:latest .
```

### Step 4: Configure the Node Orchestra (`docker-compose.dind.yaml`)

Create `docker-compose.dind.yaml` in the root directory. To enable Docker-in-Docker functionality without permissions issues, we specify `security_opt: ["apparmor=unconfined"]` and `cap_add: [SYS_ADMIN]`. We also bind-mount the host codebase `./:/unifyfl` so any edits sync automatically:

```yaml
version: '3.8'

services:
  # Node 1: Aggregator 1
  node1:
    image: unifyfl-dind:latest
    container_name: unifyfl-node1
    privileged: true
    security_opt:
      - "apparmor=unconfined"
    cap_add:
      - SYS_ADMIN
    ports:
      - "2375:2375"  # Docker daemon port
      - "8000:8000"  # Aggregator 1 port
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - node1-data:/var/lib/docker
      - ./:/unifyfl # Bind mount host repo for real-time syncing
    networks:
      - unifyfl-net

  # Node 2: Aggregator 2
  node2:
    image: unifyfl-dind:latest
    container_name: unifyfl-node2
    privileged: true
    security_opt:
      - "apparmor=unconfined"
    cap_add:
      - SYS_ADMIN
    ports:
      - "2376:2375"
      - "8001:8001"
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - node2-data:/var/lib/docker
      - ./:/unifyfl
    networks:
      - unifyfl-net

  # Node 3: Aggregator 3
  node3:
    image: unifyfl-dind:latest
    container_name: unifyfl-node3
    privileged: true
    security_opt:
      - "apparmor=unconfined"
    cap_add:
      - SYS_ADMIN
    ports:
      - "2377:2375"
      - "8002:8002"
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - node3-data:/var/lib/docker
      - ./:/unifyfl
    networks:
      - unifyfl-net

  # Node 4: Infrastructure (Blockchain + Storage)
  node4-infra:
    image: unifyfl-dind:latest
    container_name: unifyfl-node4
    privileged: true
    security_opt:
      - "apparmor=unconfined"
    cap_add:
      - SYS_ADMIN
    ports:
      - "2378:2375"
      - "8545:8545"   # Anvil blockchain
      - "5001:5001"   # IPFS API
      - "8080:8080"   # IPFS HTTP gateway
    environment:
      - DOCKER_HOST=unix:///var/run/docker.sock
    volumes:
      - node4-data:/var/lib/docker
      - ./:/unifyfl
    networks:
      - unifyfl-net

volumes:
  node1-data:
  node2-data:
  node3-data:
  node4-data:

networks:
  unifyfl-net:
    driver: bridge
```

### Step 5: Start the DinD Nodes

Run the orchestration compose file to spawn the four nested nodes:

```bash
docker-compose -f docker-compose.dind.yaml up -d
```

Verify that all four containers are running successfully:

```bash
docker-compose -f docker-compose.dind.yaml ps
```

---

## Workflow: Running Federated Learning

### 1. Launch Blockchain Infrastructure (Node 4)

Access the infrastructure container:
```bash
docker exec -it unifyfl-node4 bash
```
Inside the container, spin up Anvil. It is **critical** to specify `--host 0.0.0.0` so other node containers can connect to it:
```bash
anvil --host 0.0.0.0
```
Note the account public keys and private keys output by the terminal.

### 2. Configure Aggregators and RPC Endpoint
Update the JSON configuration files (e.g. `configs/async/agg1.config.json` and `configs/async/agg2.config.json`) with your blockchain account details and appropriate values. Ensure the blockchain RPC endpoint points to the container name:
```json
"blockchain_rpc": "http://unifyfl-node4:8545"
```

### 3. Deploy Smart Contracts and Register Node Roles
Before running FL training, the smart contracts must register each aggregator's account address as **both** `trainer` and `scorer`. Further, inside every node, run the deploy_contracts.py file using their individual aggregator configurations:
```bash
docker exec -it unifyfl-node1 bash
# Inside node1, run the contract deployment script
python deploy_scripts/deploy_contracts.py 0 pick_top_k assign_score_mean 2 configs/async/agg1.config.json
# Next, setup the aggregator as trainer + scorer
poetry run python deploy_scripts/register_node.py configs/async/agg1.config.json
```
Repeat for node2 and node3, using agg2.config.json and agg3.config.json respectively.

### 4. Download and Distribute Datasets
Prepare the datasets on the node starting the federated learning process:
```bash
docker exec -it unifyfl-node1 python scripts/download_ds.py
docker exec -it unifyfl-node1 python scripts/generate_niid_dirichlet.py
```

### 5. Launch Federated Learning Services (Inside Nodes)
Enter the individual node containers to start their local inner compose networks:
```bash
docker exec -it unifyfl-node1 bash
# Inside node 1:
docker-compose up -d
```

*(Note: The inner `docker-compose.yaml` relies on `network_mode: host` and includes startup delays—15s for the aggregator and 25s for clients—to ensure IPFS services are fully ready first.)*

---

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"
**Solution**: Make sure Docker Desktop/Engine is running on your host system.

### Issue: "Permission denied while mounting cgroups" or "mount: /sys/fs/cgroup/cpuset: permission denied"
**Solution**: Verify that `security_opt: ["apparmor=unconfined"]` and `cap_add: [SYS_ADMIN]` are present in `docker-compose.dind.yaml` for each service, and recreate the containers.

### Issue: "Not a registered trainer" Transaction Reverts
**Solution**: Check your smart contract deployment flow. Ensure that the aggregator account in use is explicitly registered under both roles using the contract registration script.
