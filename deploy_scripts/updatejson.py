# To be run from home, on all the aggregators
import os
import json
import sys

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir("..")
print(os.getcwd())
registration = sys.argv[1]
contract_address = sys.argv[2]
mode = sys.argv[3]  # 0 for async, 1 for sync
aggregation_policy = sys.argv[4]
scoring_policy = sys.argv[5]
k = sys.argv[6]
experiment_id = sys.argv[7]
config_file_path = sys.argv[8]  # e.g., "configs/async/agg1.config.json"

# Extract directory and filename from path
config_dir = os.path.dirname(config_file_path)
config_filename = os.path.basename(config_file_path)

# Change to config directory
os.chdir(config_dir)

files = [config_filename]

for i in files:
    with open(i, "r") as jsonFile:
        data = json.load(jsonFile)
        data["registration_contract_address"] = registration
        data["contract_address"] = contract_address
        data["experiment_id"] = experiment_id
        data["aggregation_policy"] = aggregation_policy
        data["scoring_policy"] = scoring_policy
        data["k"] = k
    with open(i, "w") as jsonFile:
        json.dump(data, jsonFile, indent=4)
    print(f"Updated {i} with {registration} and {contract_address}")
