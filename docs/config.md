## Key Parameters

All training parameters are in `configs`. 

Main configuration file `config.yaml` contains:

- `defaults`:
    - `models@models_dict.model1`: Model architecture (e.g. `resnet18`). Add custom models in `configs/models`.
    - `observed_data_params@dataset`: Client training data (`cifar10`, or custom datasets in `configs/observed_data_params`).
    - `observed_data_params@server_test`: Server evaluation dataset (doesn't affect training).
    - `federated_method`: Algorithm selection (`fedavg`, `fedprox`). See [method.md](method.md) for details.
    - `losses@loss`: Client loss function, e.g. `ce` (cross-entropy) or custom loss in `configs/losses`
    - `manager`: Type of manager to handle multiple clients. E.g. `sequential` sequentially collects client updates (or custom manager type).
    - `optimizer`: Type of client optimizer, e.g. `adam`, `sgd` or any existing optimization scheme. 

- `random_state`: Sets reproducibility
- `training_params`
    - `batch_size`: client batch size
    - `device`: type of computing device (`cpu` or `cuda`)
    - `device_ids` list of gpu indices (for the case of many GPUs on a machine)
- `federated_params`:
    - `amount_of_clients`: Total number of clients
    - `communication_rounds`: Total number of rounds
    - `round_epochs`: Local client epochs
    - `client_train_val_prop`: Train/validation split on client-side to evaluate server model
    - `print_client_metrics`: Enable client metric logging
    - `server_saving_metrics`: Model selection metrics. Can be one of `"loss", "Specificity", "Sensitivity", "G-mean", "f1-score", "fbeta2-score", "ROC-AUC", "AP", "Precision (PPV)", "NPV"`
    - `server_saving_agg`: Aggregation method (`uniform` average vs `weighted` by dataset size)
    - `clients_attack_types`, `prop_attack_clients`, `attack_scheme`, `prop_attack_rounds`: see [attack.md](attacks.md) for details. 