import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from hydra.utils import instantiate

from utils.utils import create_model_info
from utils.losses import get_loss
from utils.data_utils import read_dataframe_from_cfg, get_dataset_loader
from utils.metrics_utils import (
    calculate_metrics,
    stopping_criterion,
    check_metrics_names,
)


class BaseServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client_gradients = [
            OrderedDict() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.server_metrics = [
            pd.DataFrame() for _ in range(cfg.federated_params.amount_of_clients)
        ]
        self.test_df = read_dataframe_from_cfg(cfg, "test_directories", "server_test")
        self.test_loader = get_dataset_loader(
            self.test_df, cfg, drop_last=False, mode="test"
        )
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )
        self.model_path = self.create_model_path()
        self.best_metrics = {
            metric: 1000 * (metric == "loss")
            for metric in cfg.federated_params.server_saving_metrics
        }
        check_metrics_names(self.best_metrics)
        self.metric_aggregation = cfg.federated_params.server_saving_agg
        assert self.metric_aggregation in [
            "uniform",
            "weighted",
        ], f"federated_params.server_saving_agg can be only ['uniform', 'weighted'], you provide: {self.best_metrics}"
        self.best_round = 0
        self.last_metrics = None

    def eval_fn(self):
        self.global_model.to(self.device)
        self.global_model.eval()
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.test_df,
        )
        self.test_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)
                outputs = self.global_model(inp)

                self.test_loss += self.criterion(outputs, targets)
                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        self.test_loss /= len(self.test_loader)
        return fin_targets, fin_outputs

    def test_global_model(self):
        print(f"\nServer Test Results:")
        fin_targets, fin_outputs = self.eval_fn()
        self.last_metrics = calculate_metrics(
            fin_targets,
            fin_outputs,
            verbose=True,
        )
        print(f"Server Test Loss: {self.test_loss}")

    def set_client_result(self, client_result):
        # Put client information in accordance with his rank
        self.client_gradients[client_result["rank"]] = client_result["grad"]
        self.server_metrics[client_result["rank"]] = client_result["server_metrics"]

    def save_best_model(self, round):
        # Collect metrics from clients
        # server_metrics = (metrics, val_loss, len(val_df))
        server_metrics = [metrics[0] for metrics in self.server_metrics]
        val_losses = [metrics[1] for metrics in self.server_metrics]
        val_len_dfs = [metrics[2] for metrics in self.server_metrics]
        weights = [val_len_df / sum(val_len_dfs) for val_len_df in val_len_dfs]
        metrics_names = server_metrics[0].index

        if self.metric_aggregation == "uniform":
            # Uniform metrics agregation
            val_loss = np.mean(val_losses)
            metrics = pd.concat(server_metrics).groupby(level=0).mean()
        if self.metric_aggregation == "weighted":
            # Weighted metrics aggregation
            val_loss = np.sum(
                [loss * weight for loss, weight in zip(val_losses, weights)]
            )
            metrics = sum(
                weight * metric for weight, metric in zip(weights, server_metrics)
            )
        metrics = metrics.reindex(metrics_names)
        print(f"\nServer Valid Results:\n{metrics}")
        print(f"Server Valid Loss: {val_loss}")
        # Update best metrics
        epochs_no_improve, best_metrics = stopping_criterion(
            val_loss, metrics, self.best_metrics, epochs_no_improve=0
        )
        if epochs_no_improve == 0 and val_loss is not np.nan:
            print("\nServer model saved!")
            prev_model_path = f"{self.model_path}_round_{self.best_round}.pt"
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)
            self.best_metrics = best_metrics
            self.best_round = round
            checkpoint_path = f"{self.model_path}_round_{self.best_round}.pt"
            model_info = create_model_info(
                model_state=self.global_model.state_dict(),
                metrics=self.last_metrics,
                checkpoint_path=checkpoint_path,
                cfg=self.cfg,
            )
            torch.save(model_info, checkpoint_path)
        # Print comparing results
        metrics.loc["loss"] = val_loss
        print(f"\nCriterion metrics:")
        for k, v in self.best_metrics.items():
            print(
                f"Current {k}: {metrics.loc[k].mean()}\nBest {k}: {v}\nBest round: {self.best_round}\n",
            )

    def create_model_path(self):
        self.target_label_names = [self.cfg.dataset.data_name]

        return f"{self.cfg.single_run_dir}/{type(instantiate(self.cfg.federated_method, _recursive_=False)).__name__}_{'_'.join(self.target_label_names)}"

    def send_content_to_client(self, pipe_num, content):
        # Send content to client
        self.pipes[pipe_num].send(content)

    def rcv_content_from_client(self, pipe_num):
        # Get content from current client
        client_content = self.pipes[pipe_num].recv()

        return client_content
