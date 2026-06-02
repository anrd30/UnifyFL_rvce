#!/usr/bin/env python3
"""Register aggregator as trainer and scorer on the blockchain."""
import sys
import os
import json
from web3 import Web3

# Standard Anvil accounts to private keys mapping for easy setup
ANVIL_KEYS = {
    "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    "0x70997970c51812dc3a010c7d01b50e0d17dc79c8": "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
    "0x3c44cdddb6a900fa2b585dd299e03d12fa4293bc": "0x5de4111e5eb1307650536b44917f8d55269bb61676418f8ecb2e90d112001f05",
    "0x90f79bf6eb2c4f870365e785982e1f101e93b906": "0x7c852118294e51e653712a81e04800d294a12862f2eac93b65500a12e54f0a0d"
}

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/async/agg1.config.json'
    
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    print(f"Reading configuration from: {config_file}")
    with open(config_file) as f:
        config = json.load(f)

    # Initialize Web3
    endpoint = config.get('geth_endpoint', 'http://unifyfl-node4:8545')
    w3 = Web3(Web3.HTTPProvider(endpoint))
    
    if not w3.is_connected():
        print(f"Error: Could not connect to blockchain endpoint at {endpoint}")
        sys.exit(1)

    # Read contract address and target account
    reg_addr = Web3.to_checksum_address(config['registration_contract_address'])
    geth_account = config['geth_account'].lower()
    
    # Resolve private key
    private_key = os.getenv('GETH_PRIVATE_KEY')
    if not private_key:
        private_key = ANVIL_KEYS.get(geth_account)
        
    if not private_key:
        print(f"Error: Private key for account {config['geth_account']} not found.")
        print("Please set the GETH_PRIVATE_KEY environment variable.")
        sys.exit(1)

    # Load ABI
    abi_path = 'contracts/abi/Registration.json'
    if not os.path.exists(abi_path):
        # Check relative to base
        abi_path = os.path.join(os.path.dirname(__file__), '../contracts/abi/Registration.json')
    
    with open(abi_path) as f:
        registration_abi = json.load(f)

    reg_contract = w3.eth.contract(address=reg_addr, abi=registration_abi)
    account = w3.eth.account.from_key(private_key)
    print(f"Using account address: {account.address}")
    print(f"Registration contract: {reg_contract.address}")

    # Register as both trainer and scorer
    for node_role in ["trainer", "scorer"]:
        print(f"Registering as '{node_role}'...")
        tx = reg_contract.functions.registerNode(node_role).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
        })
        
        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        print(f"Sent transaction: {tx_hash.hex()}")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt['status'] == 1:
            print(f"Successfully registered as '{node_role}'!")
        else:
            print(f"Transaction failed for node role '{node_role}'!")
            try:
                # Get revert reason
                w3.eth.call(tx, 'pending')
            except Exception as e:
                print(f"Revert reason: {e}")

    # Verification checks
    trainers = reg_contract.functions.getTrainers().call()
    scorers = reg_contract.functions.getScorers().call()
    
    print("\n--- Blockchain Status ---")
    print(f"Registered Trainers: {trainers}")
    print(f"Registered Scorers: {scorers}")
    print(f"Registration status successful: {account.address in trainers and account.address in scorers}")

if __name__ == "__main__":
    main()
