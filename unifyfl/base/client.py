import asyncio
import json
import logging
import os
import sys
import time
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from web3 import Web3

from unifyfl.base.contract import create_reg_contract, create_async_contract
from unifyfl.base.ipfs import load_model_ipfs, save_model_ipfs
from unifyfl.base.model import models

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s:     %(message)s - %(asctime)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

async def main_loop():
    # 1. Load config
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    workload = config["workload"]
    geth_endpoint = config.get("geth_endpoint", "http://localhost:8545")
    registration_contract_address = config.get("registration_contract_address")
    async_contract_address = config.get("contract_address")
    ipfs_host = config.get("ipfs_host", "/ip4/127.0.0.1/tcp/5001")
    epochs = int(os.environ.get("EPOCHS") or config.get("epochs", 1))

    # Priority: Env Var > Config File
    is_malicious = os.environ.get("IS_MALICIOUS", "false").lower() == "true" or config.get("is_malicious", False)
    attack_type = os.environ.get("ATTACK_TYPE") or config.get("attack_type", None)

    client_id = int(os.environ.get("TRAIN_SET", "0"))

    # 2. Connect to Web3
    w3 = Web3(Web3.HTTPProvider(geth_endpoint))
    if os.getenv("GETH_POA"):
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    # Assign aggregator and client accounts
    aggregator_address = w3.eth.accounts[0]
    client_address = w3.eth.accounts[client_id + 1]
    w3.eth.default_account = client_address

    logger.info(f"Initialized client {client_id} at address {client_address}")
    if is_malicious:
        logger.warning(f"⚠️ MALICIOUS CLIENT ACTIVE: Attack={attack_type}")

    # 3. Load model and datasets
    model_class = models[workload]
    model_instance = model_class().to(DEVICE)
    trainloader, testloader = model_class.load_data()
    optimizer = model_instance.get_optimizer()

    if os.environ.get("PRIVACY"):
        logger.info("Privacy Enabled")
        privacy_engine = PrivacyEngine()
        model_instance, optimizer, trainloader = privacy_engine.make_private(
            module=model_instance,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    # 4. Connect to contracts
    registration_contract = create_reg_contract(w3, registration_contract_address)
    async_contract = create_async_contract(w3, async_contract_address)

    # Register as trainer
    try:
        registration_contract.functions.registerNode("trainer").transact()
        logger.info(f"Registered address {client_address} as trainer")
    except Exception as e:
        logger.warning(f"Registration failed/already registered: {e}")

    # Helper function for model noise poisoning
    def add_gaussian_noise(model, scale: float):
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * scale
                param.add_(noise)

    # 5. Client P2P Execution Loop
    last_global_cid = None
    rounds_trained = 0
    num_rounds = config.get("num_rounds", 100)
    while True:
        try:
            # Query latest global model from aggregator
            trainers = registration_contract.functions.getTrainers().call()
            models_list, _ = async_contract.functions.getLatestModelsWithScores().call()
            
            client_addresses = [w3.eth.accounts[i] for i in range(1, 13) if i < len(w3.eth.accounts)]
            global_cid = None
            for t, m in zip(trainers, models_list):
                if t not in client_addresses and m != "":
                    global_cid = m
            
            if global_cid is None or global_cid == "":
                logger.info("Waiting for any aggregator to publish initial global model...")
                await asyncio.sleep(5)
                continue
            
            if global_cid != last_global_cid:
                logger.info(f"New global model detected: {global_cid}. Fetching from IPFS...")
                
                # Fetch parameters from IPFS
                state_dict = await load_model_ipfs(global_cid, ipfs_host)
                model_instance.load_state_dict(state_dict)
                
                # Load teacher model (if KD is enabled)
                teacher_model = None
                if os.environ.get("USE_KD", "true").lower() == "true":
                    teacher_model = model_class().to(DEVICE)
                    teacher_model.load_state_dict(state_dict)
                    teacher_model.eval()
                    logger.info("✓ Loaded teacher model for Knowledge Distillation")

                # Prepare training arguments dynamically based on signature
                import inspect
                sig = inspect.signature(model_instance.train_model)
                train_kwargs = {}
                if "teacher_model" in sig.parameters:
                    train_kwargs["teacher_model"] = teacher_model

                logger.info(f"--- Starting local training for {epochs} epochs ---")
                
                # Apply data poisoning if malicious and using label flipping
                if is_malicious and attack_type == "label_flipping":
                    logger.warning("🔥 Executing Label Flipping Attack...")
                    poisoned_loader = []
                    for batch in trainloader:
                        batch["label"] = (batch["label"] + 1) % 10
                        poisoned_loader.append(batch)
                    model_instance.train_model(poisoned_loader, epochs, optimizer, **train_kwargs)
                else:
                    model_instance.train_model(trainloader, epochs, optimizer, **train_kwargs)
                
                logger.info("--- Finished local training ---")
                
                # Apply model poisoning (Gaussian noise) if malicious
                if is_malicious and attack_type == "gaussian_noise":
                    logger.warning("🔥 Executing Gaussian Noise Poisoning Attack...")
                    noise_scale = float(os.environ.get("NOISE_SCALE", "0.1"))
                    add_gaussian_noise(model_instance, noise_scale)
                    logger.info(f"✓ Gaussian noise added with scale: {noise_scale}")
                
                # Save locally trained model to IPFS
                local_cid = await save_model_ipfs(model_instance.state_dict(), ipfs_host)
                logger.info(f"Model saved to IPFS with CID: {local_cid}")
                
                # Submit local CID to the contract
                async_contract.functions.submitModel(local_cid).transact()
                logger.info(f"Successfully submitted client model CID: {local_cid}")
                
                last_global_cid = global_cid
                rounds_trained += 1
                logger.info(f"Completed local training round {rounds_trained}/{num_rounds}")
                
                if rounds_trained >= num_rounds:
                    logger.info(f"Finished all {num_rounds} rounds. Exiting client execution loop.")
                    break
                
        except Exception as e:
            logger.error(f"Error in client loop: {e}", exc_info=True)
            
        await asyncio.sleep(5)

def main():
    asyncio.run(main_loop())

if __name__ == "__main__":
    main()
