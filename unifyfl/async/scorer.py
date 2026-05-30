import asyncio
import json
import logging
import torch
import sys
from time import sleep
from operator import itemgetter
import os

# import wandb
from torch.utils.data import DataLoader, Subset
from web3 import Web3

from unifyfl.base.contract import create_async_contract, create_reg_contract

from unifyfl.base.ipfs import load_model_ipfs, load_models
from unifyfl.base.model import accuracy_scorer, models, scorers, set_parameters

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(levelname)s:     %(message)s - %(asctime)s",
)
logger = logging.getLogger(__name__)

# wandb.login()

with open(sys.argv[1]) as f:
    config = json.load(f)
    (
        workload,
        scoring,
        geth_endpoint,
        registration_contract_address,
        async_contract_address,
        ipfs_host,
        account,
        experiment_id,
    ) = itemgetter(
        "workload",
        "scorer",
        "geth_endpoint",
        "registration_contract_address",
        "contract_address",
        "ipfs_host",
        "geth_account",
        "experiment_id",
    )(
        config
    )

model = models[workload]
scorer = scorers[scoring]

logger.info(f"Model: {model.__name__}")
logger.info(f"Scorer: {scorer.__name__}")

w3 = Web3(Web3.HTTPProvider(geth_endpoint))
if os.getenv("GETH_POA"):
    from web3.middleware import geth_poa_middleware

    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
w3.eth.default_account = account

testset = model.get_testset()
testloader = DataLoader(
    testset,
    # Subset(testset, torch.randperm(len(testset))[: math.floor(len(testset) / 2)]),
    batch_size=64,
)

registration_contract = create_reg_contract(w3, registration_contract_address)
async_contract = create_async_contract(w3, async_contract_address)


async def score_model(trainer: str, cid: str):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_instance = model().to(DEVICE)
    logger.info(f"Model received to score with CID: {cid}")
    model_instance.load_state_dict(await load_model_ipfs(cid, ipfs_host))
    logger.info("Model pull from IPFS")
    
    if scoring == "pinn_guard":
        pinn_dir = f"save/async/{workload}/{experiment_id}"
        pinn_path = f"{pinn_dir}/pinn_guard.pt"
        os.makedirs(pinn_dir, exist_ok=True)
        if not os.path.exists(pinn_path):
            logger.info("PINN Guard model checkpoint not found. Training PINN Guard on clean logits first...")
            model_instance.eval()
            clean_logits = []
            with torch.no_grad():
                for batch in testloader:
                    images = batch["image"].to(DEVICE).float()
                    logits = model_instance(images)
                    clean_logits.append(logits.cpu())
            clean_logits = torch.cat(clean_logits, dim=0)[:1000]
            
            from unifyfl.base.pinn import train_adversarial_pinn_guard
            pinn_guard_model, _ = train_adversarial_pinn_guard(
                clean_logits, n_epochs=100, device=DEVICE, verbose=False
            )
            torch.save(pinn_guard_model.state_dict(), pinn_path)
            logger.info(f"PINN Guard trained and saved to {pinn_path}")
            
        from unifyfl.base.model import pinn_guard_scorer
        loss, score = pinn_guard_scorer(model_instance, testloader, pinn_path=pinn_path)
        logger.info(f"PINN Guard Anomaly-Based Score: {(score * 100):>0.2f}")
        
        # Calculate and log standard classification accuracy so fl_analysis.py can track it
        acc_loss, accuracy = accuracy_scorer(model_instance, testloader)
        logger.info(f"Accuracy: {(accuracy * 100):>0.2f}%")
        logger.info(f"Loss: {acc_loss:>0.2f}")
    elif scoring == "multi_krum":
        # Fetch latest models from contract to evaluate distances
        models_list, _ = async_contract.functions.getLatestModelsWithScores().call()
        all_cids = [m for m in models_list if m != ""]
        if cid not in all_cids:
            all_cids.append(cid)
            
        if len(all_cids) >= 3:
            logger.info(f"Computing Multi-Krum across {len(all_cids)} models...")
            weights_list = await load_models(all_cids, ipfs_host)
            ndarrays_list = [[val.cpu().numpy() for _, val in w.items()] for w in weights_list]
            scores_list = scorer(ndarrays_list)
            cid_index = all_cids.index(cid)
            score = scores_list[cid_index] / 100.0  # Normalize Multi-Krum score [0, 100] to [0.0, 1.0] for the contract
            logger.info(f"Multi-Krum Score for {cid}: {scores_list[cid_index]:>0.2f}")
        else:
            logger.info("Not enough models for Multi-Krum scoring. Falling back to accuracy.")
            _, acc = accuracy_scorer(model_instance, testloader)
            score = acc
            
        # Calculate and log standard classification accuracy so fl_analysis.py can track it
        loss, accuracy = accuracy_scorer(model_instance, testloader)
        logger.info(f"Accuracy: {(accuracy * 100):>0.2f}%")
        logger.info(f"Loss: {loss:>0.2f}")
    else:
        loss, score = scorer(model_instance, testloader)
        logger.info(f"Accuracy: {(score * 100):>0.2f}%")
        logger.info(f"Loss: {(loss):>0.2f}")
        
    async_contract.functions.submitScore(cid, int(score * 1000)).transact()
    logger.info("Model scores submitted to contract")


def main():
    registration_contract.functions.registerNode("scorer").transact()
    events = set()
    last_seen_block = w3.eth.block_number
    # wandb.init(
    #     project="unifyfl",
    #     config={
    #         "workload": "cifar10",
    #         "scorer": scoring,
    #     },
    #     group=experiment_id,
    #     name=f"{socket.gethostname() if socket.gethostname() != 'raspberrypi' else getpass.getuser()}-async-scorer",
    # )
    while True:
        for event in async_contract.events.StartScoring().get_logs(
            fromBlock=last_seen_block
        ):
            if event not in events:
                events.add(event)
                last_seen_block = event["blockNumber"]
                if w3.eth.default_account in event["args"]["scorers"]:
                    print(event["args"])
                    asyncio.run(
                        score_model(event["args"]["trainer"], event["args"]["model"])
                    )
        sleep(1)


if __name__ == "__main__":
    main()
