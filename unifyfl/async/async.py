import asyncio
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from operator import itemgetter

from web3 import Web3

from unifyfl.base.contract import create_reg_contract, create_async_contract
from unifyfl.base.ipfs import load_models, save_model_ipfs
from unifyfl.base.policies import pick_selected_model
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy.aggregate import aggregate
from unifyfl.base.model import models
import torch

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

    (
        workload,
        geth_endpoint,
        geth_account,
        registration_contract_address,
        async_contract_address,
        flwr_min_fit_clients,
        ipfs_host,
        aggregation_policy,
        scoring_policy,
        k,
        experiment_id,
    ) = itemgetter(
        "workload",
        "geth_endpoint",
        "geth_account",
        "registration_contract_address",
        "contract_address",
        "flwr_min_fit_clients",
        "ipfs_host",
        "aggregation_policy",
        "scoring_policy",
        "k",
        "experiment_id",
    )(config)

    num_rounds = config.get("num_rounds", 100)
    min_fit_clients = int(flwr_min_fit_clients)

    # 2. Connect to Web3
    w3 = Web3(Web3.HTTPProvider(geth_endpoint))
    if os.getenv("GETH_POA"):
        from web3.middleware import geth_poa_middleware
        w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    # Aggregator uses Account 0
    aggregator_address = w3.eth.accounts[0]
    w3.eth.default_account = aggregator_address

    # 3. Connect to contracts
    registration_contract = create_reg_contract(w3, registration_contract_address)
    async_contract = create_async_contract(w3, async_contract_address)

    # Register as trainer
    try:
        registration_contract.functions.registerNode("trainer").transact()
        logger.info(f"Aggregator registered as trainer at address {aggregator_address}")
    except Exception as e:
        logger.warning(f"Aggregator registration failed/already registered: {e}")

    # 4. Initialize model
    model_class = models[workload]
    model_instance = model_class().to(DEVICE)

    # Helper to set parameters
    def set_parameters(model, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    # Helper to get parameters as numpy list
    def get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    os.makedirs(f"save/async/{workload}/{experiment_id}", exist_ok=True)

    # 5. Aggregator Loop
    round_id = 0
    global_cid = None
    
    while round_id <= num_rounds:
        try:
            # Round 0: Initialize initial global model
            if round_id == 0:
                logger.info("Round 0: Initializing global model...")
                global_cid = await save_model_ipfs(model_instance.state_dict(), ipfs_host)
                logger.info(f"Initial global model saved to IPFS with CID: {global_cid}")
                
                async_contract.functions.submitModel(global_cid).transact()
                logger.info("Initial global model CID submitted to contract.")
                
                round_id = 1
                logger.info(f"Round 1 started - waiting for client models...")
                await asyncio.sleep(5)
                continue

            # Query contract for latest models and scores
            models_list, scores_list = async_contract.functions.getLatestModelsWithScores().call()
            
            # Map CIDs to scores, filtering out aggregator's own CID and empty models
            global_models = []
            for cid, scores in zip(models_list, scores_list):
                if cid != "" and cid != global_cid:
                    global_models.append((cid, scores))

            # Count how many of these have at least one score submitted
            scored_models = [(cid, scores) for cid, scores in global_models if len(scores) > 0]
            
            logger.info(f"Round {round_id}: Received {len(global_models)} client updates, {len(scored_models)} are scored (need >= {min_fit_clients})")
            
            if len(scored_models) >= min_fit_clients:
                logger.info(f"Required scores received. Filtering and aggregating...")
                
                # Apply selection policy
                selected_models = pick_selected_model(
                    scored_models, aggregation_policy, scoring_policy, int(k), global_cid
                )
                
                if len(selected_models) > 0:
                    logger.info(f"Aggregating models: {selected_models}")
                    state_dicts = await load_models(selected_models, ipfs_host)
                    
                    param_list = [
                        [val.cpu().numpy() for _, val in state_dict.items()]
                        for state_dict in state_dicts
                    ]
                    
                    # Package for aggregator: equal weights for each client update
                    packaged_models = list(zip(param_list, [1] * len(param_list)))
                    weight_arrays = aggregate(packaged_models)
                    
                    # Load aggregated weights
                    set_parameters(model_instance, weight_arrays)
                    
                    # Save global model to IPFS
                    global_cid = await save_model_ipfs(model_instance.state_dict(), ipfs_host)
                    logger.info(f"Global model aggregated. Saved to IPFS with CID: {global_cid}")
                    
                    # Submit aggregated CID to contract
                    async_contract.functions.submitModel(global_cid).transact()
                    logger.info(f"Submitted new global model CID to contract.")
                    
                    cur_time = str(datetime.now().strftime("%d-%H-%M-%S"))
                    torch.save(
                        model_instance.state_dict(),
                        f"save/async/{workload}/{experiment_id}/{round_id:02d}-{cur_time}-global.pt",
                    )
                    
                    round_id += 1
                    logger.info(f"Round {round_id} started - waiting for client models...")
                else:
                    logger.warning("No models selected for aggregation! Retrying in 10s...")
                    
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error in aggregator loop: {e}", exc_info=True)
            await asyncio.sleep(10)

    logger.info(f"Max rounds reached ({num_rounds}). Aggregator exiting cleanly.")

def main():
    asyncio.run(main_loop())

if __name__ == "__main__":
    main()
