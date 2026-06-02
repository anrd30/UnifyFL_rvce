#!/usr/bin/env python3
"""Register aggregator as scorer on blockchain."""
from web3 import Web3
import json
import os
import sys

w3 = Web3(Web3.HTTPProvider('http://unifyfl-node4:8545'))

config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/async/agg1.config.json'

with open(config_file) as f:
    config = json.load(f)

with open('contracts/abi/Registration.json') as f:
    registration_abi = json.load(f)

reg_addr = Web3.to_checksum_address(config['registration_contract_address'])
private_key = os.getenv('GETH_PRIVATE_KEY')

if not private_key:
    print("Error: GETH_PRIVATE_KEY environment variable not set")
    sys.exit(1)

reg_contract = w3.eth.contract(address=reg_addr, abi=registration_abi)

account = w3.eth.account.from_key(private_key)
print(f"Registering account: {account.address}")

# registerNode takes "trainer" or "scorer" as node_id
node_id = "scorer"

tx = reg_contract.functions.registerNode(node_id).build_transaction({
    'from': account.address,
    'nonce': w3.eth.get_transaction_count(account.address),
    'gas': 300000,
    'gasPrice': w3.eth.gas_price,
})

signed_tx = account.sign_transaction(tx)
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"Transaction status: {receipt['status']}")
print(f"Transaction hash: {tx_hash.hex()}")
if receipt['status'] == 0:
    print("Transaction reverted!")
    # Try to decode the error
    try:
        result = w3.eth.call(tx, 'pending')
    except Exception as e:
        print(f"Error details: {e}")

# Verify
scorers = reg_contract.functions.getScorers().call()
print(f"Registered scorers: {scorers}")
