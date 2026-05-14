from collections import OrderedDict

import flwr as fl
from flwr.common.typing import NDArray
import torch
from unifyfl.base.model import models
from opacus import PrivacyEngine
import os


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)


# Login to wandb
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, log=False, epochs=1, is_malicious=False, attack_type=None):
        self.model = model().to(DEVICE)
        self.log = log
        self.epochs = epochs
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.trainloader, self.testloader = model.load_data()
        self.optimizer = self.model.get_optimizer()
        
        if self.is_malicious:
            print(f"⚠️ MALICIOUS CLIENT INITIALIZED: Attack={self.attack_type}")

        if os.environ.get("PRIVACY"):
            print("Privacy Enabled")
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.trainloader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.trainloader,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
            )
        super().__init__()

    def get_parameters(self, config):
        print("get param")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: NDArray):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"--- Starting training for {self.epochs} epochs ---", flush=True)
        self.set_parameters(parameters)
        
        # Apply data poisoning if malicious
        if self.is_malicious and self.attack_type == "label_flipping":
            print("🔥 Executing Label Flipping Attack...")
            poisoned_loader = []
            for batch in self.trainloader:
                # Flip labels: (label + 1) % 10
                batch["label"] = (batch["label"] + 1) % 10
                poisoned_loader.append(batch)
            self.model.train_model(poisoned_loader, self.epochs, self.optimizer)
        else:
            self.model.train_model(self.trainloader, self.epochs, self.optimizer)
            
        print("--- Finished training ---", flush=True)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.test_model(self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def main():
    import logging
    from operator import itemgetter
    import sys
    import json

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(levelname)s:     %(message)s - %(asctime)s",
    )
    # TODO: Add logs in agg and super
    # logger = logging.getLogger(__name__)

    with open(sys.argv[1]) as f:
        config = json.load(f)
        (workload, flwr_server_address, epochs) = itemgetter(
            "workload", "flwr_server_address", "epochs"
        )(config)
        
        # Priority: Env Var > Config File
        is_malicious = os.environ.get("IS_MALICIOUS", "false").lower() == "true" or config.get("is_malicious", False)
        attack_type = os.environ.get("ATTACK_TYPE") or config.get("attack_type", None)

    model = models[workload]
    import time
    max_retries = 10
    for i in range(max_retries):
        try:
            fl.client.start_numpy_client(
                server_address=flwr_server_address,
                client=FlowerClient(model, log=True, epochs=epochs, is_malicious=is_malicious, attack_type=attack_type),
                grpc_max_message_length=536870912,
            )
            break
        except Exception as e:
            if i < max_retries - 1:
                print(f"Connection failed, retrying in 5s... ({i+1}/{max_retries})")
                time.sleep(5)
            else:
                raise e



if __name__ == "__main__":
    main()
# Start Flower client
