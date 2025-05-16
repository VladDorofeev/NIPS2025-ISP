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
parser.add_argument(
    "--scheduler_clients",
    action="store_true",
    help="set up AdaFL setup",
)

args = parser.parse_args()

# Configuration parameters
FEDERATED_METHODS = ["fedavg", "fedcbs", "delta", "pow", "fedcor"]

BASE_PARAMS = [
    "dataset.alpha=0.1",
    "federated_params.amount_of_clients=100",
    "training_params.batch_size=64",
    f"training_params.device_ids=[{args.device_id}]",
    "federated_params.print_client_metrics=False",
    "federated_params.communication_rounds=1",
    "random_state=41",
]


def build_command(federated_method):
    params = copy.deepcopy(BASE_PARAMS)
    if federated_method == "delta":
        # Set up sgd optimizer, else Adam by default
        params.append("optimizer=sgd")

    if args.dynamic_clients or args.scheduler_clients:
        if args.dynamic_clients:
            params.append("federated_method=ISP")
        else:
            params.append("federated_method=adafl")

        # For python class inherits compability only
        params.append("+federated_method.args.num_clients_subset=20")
        params.append(f"federated_method.base_method={federated_method}")

        if federated_method == "fedcor":
            # Deterministic sample strategy
            params.append("federated_method.num_samples=1")
            params.append("+federated_method.args.warmup=12")

        if federated_method == "delta":
            params.append("+federated_method.args.alpha_1=0.8")
            params.append("+federated_method.args.alpha_2=0.2")

        if federated_method == "pow":
            params.append("+federated_method.args.candidate_set_size=40")

        if federated_method == "fedcbs":
            params.append("+federated_method.args.lambda_=10")
    else:
        params.append(f"federated_method={federated_method}")

    # Build output filename
    output_name = f"{federated_method}_cifar10.txt"
    if args.dynamic_clients:
        output_name = "ISP_" + output_name

    return params, f"outputs/ISP_experiments/{output_name}"


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
    print(f"Command is:\n{cmd_str}\n\n", flush=True)
    subprocess.run(cmd_str, shell=True, check=True)
