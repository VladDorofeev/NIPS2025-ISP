
# Communication-Efficient Federated Learning with Adaptive Number of Participants

## Table of contents
0. [Running Experiments](docs/experiment.md) -- Run the command to get reproducibilit
1. [Quickstart](#-quickstart-guide) -- Follow the instructions and get the result!
2. [C4 notation](docs/C4.md) -- Context Container Component Code scheme.
3. [Federated Method Explaining](docs/method.md) -- Get the basis and write your own method
4. [Config Explaining](docs/config.md) -- See allowed optionalization
5. [Attacks](docs/attacks.md) -- Get the basis and write custom attack

## 🚀 Quickstart Guide
### 📋 Prerequisites
1. **Install dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```
2. **Download CIFAR-10 dataset**
```bash
python src/utils/cifar_download.py --target_dir=cifar10
```
3. **Download TinyImageNet dataset**
```bash
python src/utils/image_net_download.py --target_dir=image_net
```

✅ **Verification**: You should get `"All steps completed successfully!!!"`. 

⚠️ **Important**: Check/update paths in `configs/observed_data_params/`.

### ⚙️ Experiment Setups

#### 🔄 Standard [Federated Averaging](https://arxiv.org/pdf/1602.05629) on CIFAR-10
```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  training_params.device_ids=[0] \
  > test_run_fedavg_cifar.txt
```

`device_ids` controls the GPU number (if there are several GPUs on the machine). 

Additionally, `manager.batch_size` client processes will be created. To forcefully terminate the training, kill any of the processes.

#### 🌪️ Heterogeneous CIFAR10 Experiment

**Dirichlet Partition with $\alpha=0.1$ (strong heterogeneity)**

```bash
python src/train.py \
  training_params.batch_size=32 \
  federated_params.print_client_metrics=False \
  observed_data_params@dataset=cifar10_dirichlet \
  dataset.alpha=0.1 \
  federated_params.amount_of_clients=100 \
  > test_run_fedavg_cifar_dirichlet_strong_heterogeneity_100_clients.txt
```

**Uniform Distribution ($\alpha=1000$) with various amount_of_clients**
```bash
dataset.alpha=1000 \
federated_params.amount_of_clients=42 \
```