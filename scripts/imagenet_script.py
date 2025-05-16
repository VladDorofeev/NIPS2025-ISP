import subprocess
import os
import copy
import argparse

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

parser = argparse.ArgumentParser(description="Run federated learning experiments.")
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="GPU device IDx to use (default: 0)",
)
parser.add_argument(
    "--dynamic_clients",
    action="store_true",
    help="set up dynamic number of clients",
)

args = parser.parse_args()

# Configuration parameters
FEDERATED_METHODS = ["fedcor"]
BASE_PARAMS = [
    "models@models_dict.model1=swin_tiny_patch4_window7_224",
    "observed_data_params@dataset=image_net", 
    "dataset.alpha=0.5",
    "observed_data_params@server_test=image_net",
    "observed_data_params@trust_df=image_net_trust",
    "federated_params.amount_of_clients=100",
    "training_params.batch_size=128",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "federated_params.communication_rounds=200",
    "random_state=41",
    "optimizer.lr=0.001",
    "optimizer.weight_decay=0.05",
    "manager.batch_size=5",
    "federated_params.round_epochs=1",
    "optimizer=adamw"
]


def build_command(federated_method):
    params = copy.deepcopy(BASE_PARAMS)

    if args.dynamic_clients:
        params.append("federated_method=ISP")
        params.append("+federated_method.args.num_clients_subset=20")
        # Deterministic sample strategy
        params.append(f"federated_method.base_method=fedcor")
        params.append("federated_method.num_samples=1")
        params.append("+federated_method.args.warmup=10")

    else:
        params.append(f"federated_method={federated_method}")
        params.append("federated_method.args.warmup=10")

    # Build output filename
    output_name = f"{federated_method}_imagenet.txt"
    if args.dynamic_clients:
        output_name = "isp_" + output_name

    return params, f"{output_name}"


# Run experiments
for method in FEDERATED_METHODS:
    # Build command and output path
    params, output_path = build_command(method)

    # Create full command
    cmd = ["nohup", "python", "src/train.py"] + params

    # Convert to string with output redirection
    cmd_str = " ".join(cmd) + f" > {output_path}"

    print(
        f"Running {method} strategy. Dynamic clients {args.dynamic_clients} ",
        flush=True,
    )
    subprocess.run(cmd_str, shell=True, check=True)
