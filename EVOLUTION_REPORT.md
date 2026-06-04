# UnifyFL Architectural Evolution Report: From Centralized Flower to Decentralized P2P

This report documents the architectural transitions, key technical changes, and strategic decisions made in the UnifyFL repository since the original synchronous setup attacked with Gaussian noise to the current decentralized logit-poisoning scenario.

---

## 1. Architectural Transitions

### Phase 1: The Original Hybrid Setup
*   **Core Framework:** Flower (`flwr`) central client-server framework.
*   **Orchestration:** Hybrid model. Model weights were distributed via Flower's central gRPC server. The blockchain and IPFS were used parallelly only to register scorers, evaluate client model updates, and record scorer validations on-chain.
*   **The Conflict:** Flower operates under strict synchronous rounds (blocking the server until a preset client quota reports back). Integrating decentralized blockchain transactions (which have variable validation latencies) within Flower's training loop led to constant timeouts, port binding races, and desynchronization issues.
*   **Attack Vector:** Basic **Gaussian Noise Poisoning**, where malicious clients added random noise to their parameter updates.

### Phase 2: Bypassing Flower (100% P2P Decentralization)
*   **Decision:** Replaced the central Flower server with a fully custom, asynchronous, peer-to-peer (P2P) orchestration loop.
*   **How it Works:** 
    *   **IPFS** acts as the decentralized model repository (storing weights as `.pt` files mapped to CIDs).
    *   **Smart Contracts (`AsyncRound.sol` and `Registration.sol`)** act as the decentralized state machine, managing node roles, logging model submissions via events, and storing verification scores.
    *   **Clients** pull the latest global model from IPFS, train locally, and submit their new model CID to the smart contract.
    *   **Scorers** detect the contract's submission events, download the model, calculate scores locally, and submit them back to the contract.
    *   **Aggregators** query the contract for client updates, select the top updates based on score, compute averages (FedAvg), and submit the new global model CID.
*   **Outcome:** Eliminated the central server bottleneck, resolving the synchronization and timing issues between Flower and the blockchain.

### Phase 3: Transitioning to Logit-Level Attacks
*   **Decision:** Shifted the focus from simple parameter-level Gaussian noise to **Logit-Level Poisoning Attacks** involving **Knowledge Distillation (KD)**.
*   **Implementation:**
    *   **Knowledge Distillation:** Integrated directly into `CIFAR10Model.train_model()` in `models/cifar.py`. Clients train a student model using a combination of standard cross-entropy loss and KL-divergence distillation loss against a teacher model.
    *   **Logit Poisoning:** Hooked into the local client training. Malicious clients intercept output logits during training, corrupting them with scaled noise before applying KD loss computation.
*   **Impact:** Logit poisoning bypasses standard weight-distance defenses (like Multi-Krum) because the network weights remain geometrically close to benign updates, but their classification boundaries are corrupted.

---

## 2. Key Decisions & Structural Refactors

### Decision 1: Robust Account Resolution
*   **Problem:** Early versions of the decentralized script had overlapping accounts. Clients and aggregators attempted to submit transactions using the same default accounts, leading to transaction collisions and contract nonce errors.
*   **Solution:** Rewrote account mapping in `unifyfl/base/client.py` and `unifyfl/async/async.py`:
    *   Clients are mapped dynamically to accounts `1` through `N` (based on their `client_id`).
    *   Aggregators use Account `0` (or `N+1` / `N+2` for backup nodes).
    *   Anvil is initialized with 20 accounts (`anvil -a 20`) to ensure a collision-free pool.

### Decision 2: Selection & Filtering on Decentralized Aggregators
*   **Problem:** In standard Flower, the server aggregates *every* update (causing total model collapse under attack). In the async loop, the aggregator needs to distinguish between client updates and global aggregator models.
*   **Solution:** Updated the aggregator loop in `unifyfl/async/async.py` to filter out non-client trainer submissions, ensuring that aggregators only compute averages on verified client model updates.

### Decision 3: Standardizing Client Counts (Reverting to 12 Clients)
*   **Problem:** Recent local tests were scaled down to 10 clients (9 benign + 1 malicious) to speed up tests. However, the original UnifyFL data splitting scripts (`prepare_cifar10.sh`) are hardcoded to partition CIFAR-10 into 12 non-IID datasets.
*   **Solution:** Reverted the client configuration to **12 total clients** (10 benign + 2 malicious) to align with the original dataset layout and guarantee that data mapping indexes (`train0` to `train11`) align correctly.

### Decision 4: Single-Aggregator Simulation vs. Multi-Aggregator Scale
*   **Problem:** The `integration/triple-threat-dind` branch introduced a complex Docker-in-Docker setup to run 3 aggregators simultaneously to prove P2P validation. However, this introduced major execution delays and debugging complexity.
*   **Solution:** Kept the custom async logic, but configured it to run **1 primary aggregator/scorer** process by default. This preserves the trustless consortium architecture (since the contract supports multiple aggregators seamlessly) while keeping local execution fast and easy to debug.

---

## 3. Comparative Summary of Setup Parameters

| Feature | Original Setup (Flower) | Current Setup (P2P Async) |
|---|---|---|
| **Orchestration** | Centralized Flower Server (gRPC) | Decentralized Smart Contract (`AsyncRound.sol`) + IPFS |
| **Execution Loop** | Synchronous round-by-round | Asynchronous event-driven |
| **Number of Clients** | 12 clients | 12 clients (10 benign + 2 malicious) |
| **Scoring Mechanics** | Labeled Validation Accuracy (Centralized) | PINN Guard Curvature Scoring (Label-free, Decentralized) |
| **Aggregator Count** | 1 central server | 1 local aggregator (extensible to multiple on-chain) |
| **Attack Type** | Parameter Gaussian Noise | Logit Poisoning Attack with Knowledge Distillation |
| **DP Integration** | Not supported | Local Differential Privacy (Opacus) ready |
