import time
import copy
import torch
import numpy as np
import pandas as pd

from types import MethodType
from hydra.utils import instantiate

from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.data_utils import read_dataframe_from_cfg
from utils.attack_utils import (
    set_client_map_round,
)

from ..fedavg.fedavg import FedAvg
from ..fedcbs.fedcbs import FedCBS
from ..delta.delta import Delta
from ..pow.pow import Pow
from ..fedcor.fedcor import FedCor
from ..compression.compression import Compression


class ISP:
    def __init__(
        self,
        base_method,
        warmup_rounds,
        step_find_optimal,
        num_samples,
        sample_step,
        num_clients_momentum,
        window_sz_loss,
        local_epoch_on_wp,
        args,
    ):
        self.base_method = base_method
        self.warmup_rounds = warmup_rounds
        self.step_find_optimal = step_find_optimal
        self.num_samples = num_samples
        self.sample_step = sample_step
        self.num_clients_momentum = num_clients_momentum
        self.window_sz_loss = window_sz_loss
        self.local_epoch_on_wp = local_epoch_on_wp
        self.args = args

    def _init_federated(self, cfg, df):
        if self.base_method == "fedavg":
            name_method = "FedAvg"
            metaclass = type(FedAvg)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedAvg,), {})

        elif self.base_method == "fedcbs":
            name_method = "FedCBS"
            metaclass = type(FedCBS)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedCBS,), {})

        elif self.base_method == "delta":
            name_method = "Delta"
            metaclass = type(Delta)
            isp_class = metaclass(f"ISPClass_{name_method}", (Delta,), {})

        elif self.base_method == "pow":
            name_method = "Pow"
            metaclass = type(Pow)
            isp_class = metaclass(f"ISPClass_{name_method}", (Pow,), {})

        elif self.base_method == "fedcor":
            name_method = "Fedcor"
            metaclass = type(FedCor)
            isp_class = metaclass(f"ISPClass_{name_method}", (FedCor,), {})

        elif self.base_method == "compression":
            name_method = "Compression"
            metaclass = type(Compression)
            isp_class = metaclass(f"ISPClass_{name_method}", (Compression,), {})

        else:
            raise NotImplementedError(
                f"{self.base_method} not available to ISP version"
            )

        isp_method = isp_class(**self.args)
        isp_method._init_federated(cfg, df)

        isp_method.name_method = name_method
        isp_method.warmup_rounds = self.warmup_rounds
        isp_method.step_find_optimal = self.step_find_optimal
        isp_method.num_samples = self.num_samples
        isp_method.sample_step = self.sample_step
        isp_method.window_sz_loss = self.window_sz_loss
        isp_method.local_epoch_on_wp = self.local_epoch_on_wp
        isp_method.amount_of_clients = cfg.federated_params.amount_of_clients
        isp_method.optimal_amount_clients = 1
        isp_method.borders_of_clients = [1, isp_method.amount_of_clients + 1]
        isp_method.find_optimal_rounds = [
            i
            for i in range(
                isp_method.warmup_rounds,
                isp_method.rounds,
                isp_method.step_find_optimal,
            )
        ]
        print(f"Find optimal rounds at {isp_method.find_optimal_rounds}\n")
        isp_method.wp_amount_cl_history = []

        # Set parametrs to server
        isp_method.server.amount_of_clients = isp_method.amount_of_clients
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        isp_method.server.trust_df = trust_df
        isp_method.server.df = df

        isp_method.server.eval_loader = get_dataset_loader(
            isp_method.server.trust_df,
            isp_method.server.cfg,
            drop_last=False,
            mode="valid",
        )

        isp_method.server.criterion = get_loss(
            loss_cfg=isp_method.server.cfg.loss,
            device=isp_method.server.device,
            df=isp_method.server.trust_df,
            init_pos_weight=None,
        )
        isp_method.server.sample_step = self.sample_step
        isp_method.server.num_samples = self.num_samples
        isp_method.server.window_sz_loss = self.step_find_optimal
        isp_method.server.trust_losses = []
        isp_method.server.clients_history = []
        isp_method.server.saved_weights = None
        isp_method.server.wp_mean_amount_cl = None
        isp_method.server.wp_mean_amount_cl_loss = None
        isp_method.server.cnt_solve_optimal_task = 0
        isp_method.server.num_clients_momentum = self.num_clients_momentum
        isp_method.server.num_classes = trust_df["target"].nunique()
        print(f"Num classes on task is {isp_method.server.num_classes}")

        isp_method.begin_train = MethodType(ISP.begin_train, isp_method)
        isp_method.get_amount_clients = MethodType(
            ISP.get_amount_clients, isp_method
        )

        isp_method.full_method_aggregate = MethodType(
            isp_class.aggregate,
            isp_method,  # Copy default aggregate in another method
        )
        isp_method.aggregate = MethodType(
            ISP.aggregate,
            isp_method,  # ISP aggregate need save weight before aggregate
        )

        isp_method.server.eval_global_model = MethodType(
            ISP.eval_global_model, isp_method.server
        )

        isp_method.full_method_get_communication_content = MethodType(
            isp_class.get_communication_content,
            isp_method,
        )
        isp_method.get_communication_content = MethodType(
            ISP.get_communication_content,
            isp_method,
        )

        # Change methods for client
        isp_method.client_cls.full_method_create_pipe_commands = (
            isp_method.client_cls.create_pipe_commands
        )
        isp_method.client_cls.create_pipe_commands = (
            ISP.client_create_pipe_commands
        )

        isp_method.client_cls.train_fn = ISP.client_train_fn
        isp_method.client_cls.set_local_epoch = ISP.client_set_local_epoch

        # Change methods for solving optimization task
        isp_method.server.find_optimal = MethodType(
            ISP.find_optimal, isp_method.server
        )
        isp_method.server.aggregate_clients = MethodType(
            ISP.aggregate_clients, isp_method.server
        )
        isp_method.server.get_mean_loss = MethodType(
            ISP.get_mean_loss, isp_method.server
        )
        isp_method.server.ema_loss_history = MethodType(
            ISP.ema_loss_history, isp_method.server
        )
        isp_method.server.get_functional_value = MethodType(
            ISP.get_functional_value, isp_method.server
        )
        isp_method.server.choose_optimal = MethodType(
            ISP.choose_optimal, isp_method.server
        )
        isp_method.server.first_positive_functional = MethodType(
            ISP.first_positive_functional, isp_method.server
        )

        return isp_method

    def eval_global_model(self, need_save_loss):
        self.global_model.to(self.device)
        self.global_model.eval()

        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.trust_df,
        )

        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(self.eval_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.global_model(inp)

                val_loss += self.criterion(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        trust_loss = val_loss / len(self.eval_loader)

        if need_save_loss:
            self.trust_losses.append(trust_loss)

        return trust_loss

    def get_amount_clients(self):
        if self.cur_round < self.warmup_rounds:
            print("Warmup round")
            self.num_clients_subset = self.amount_of_clients
            return self.num_clients_subset

        if self.cur_round in self.find_optimal_rounds:
            # We need do full communication round
            # for find optimal amount of clients
            print('Full round for "Optimal" method')
            self.num_clients_subset = self.amount_of_clients
            return self.num_clients_subset

        if self.cur_round - 1 in self.find_optimal_rounds:
            # In prevous round we use all clients, and now
            # we need recalculate optimal amount of clients

            self.optimal_amount_clients = self.server.find_optimal(
                self.borders_of_clients
            )
            self.server.clients_history.append(self.optimal_amount_clients)

        self.num_clients_subset = self.optimal_amount_clients
        return self.num_clients_subset

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        self.server.saved_weights = copy.deepcopy(aggregated_weights)
        self.ts = [1 / len(self.list_clients) for i in self.list_clients]

        return self.full_method_aggregate()

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.clients_loader = self.manager.batches
        self.server.global_model = instantiate(self.cfg.models[0])

        print()
        print(f"ISP version of {self.name_method}")
        print(f"Number of warmup rounds is {self.warmup_rounds}")

        for cur_round in range(self.rounds):
            print(f"\nRound number: {cur_round}")
            begin_round_time = time.time()
            self.cur_round = cur_round
            self.server.cur_round = cur_round

            print("Evaluate started")
            server_val_loss = self.server.eval_global_model(need_save_loss=True)
            print(f"Server Val Loss: {server_val_loss}")

            self.num_clients_subset = self.get_amount_clients()
            self.list_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            self.list_clients.sort()
            print(f"Clients on this communication: {self.list_clients}\n")
            print(
                f"Amount of clients on this communication: {len(self.list_clients)}\n"
            )

            print("\nTraining started\n")
            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                cur_round,
            )

            self.train_round()

            self.server.test_global_model()
            self.server.save_best_model(cur_round)

            self.ts = self.calculate_ts()
            print(f"Client weights for aggregation on this communication {self.ts}")

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.manager.stop_train()

    #
    #
    # Solving optimization problem stuff:
    #
    #

    def aggregate_clients(self, clients):
        # Here we aggregate model with parametr clients

        aggregated_weights = copy.deepcopy(self.saved_weights)

        coef = 1 / len(clients)

        for client in clients:
            for key, weights in self.client_gradients[client].items():
                aggregated_weights[key] = (
                    aggregated_weights[key] + weights.to(self.device) * coef
                )

        return aggregated_weights

    def get_mean_loss(self, amount_of_clients):
        losses = []
        before_sampling = copy.deepcopy(self.global_model.state_dict())
        for _ in range(self.num_samples):
            clients = self.select_clients_to_train(
                amount_of_clients, server_sampling=True
            )
            print(f"Sample of clients: {sorted(clients)}")

            sample_weights = self.aggregate_clients(clients)
            self.global_model.load_state_dict(sample_weights)

            sample_loss = self.eval_global_model(need_save_loss=False)
            self.global_model.load_state_dict(before_sampling)

            print(f"Sample loss : {sample_loss}")
            losses.append(sample_loss)

        print(f"Sample losses = {losses}")

        mean_loss = sum(losses) / self.num_samples
        return mean_loss

    def ema_loss_history(self):
        # Get a trust_losses history until previous full communication round
        trust_losses_series = pd.Series(self.trust_losses[:-1])

        ema_trust_losses = list(
            trust_losses_series.ewm(span=self.window_sz_loss, adjust=False).mean()
        )
        # choose for comparsion last loss from ema history
        return ema_trust_losses[-1]

    def get_functional_value(self, amount_of_clients):
        mean_loss = self.get_mean_loss(amount_of_clients)
        print(f"Mean loss: {mean_loss}")

        func_value = self.first_positive_functional(mean_loss)
        print(f"Functional value = {func_value}")

        return func_value

    def choose_optimal(self, client_dict):
        optimal_amount_cl = None
        # optimal_amount_cl = max(list(client_dict.keys()))

        for k in client_dict.keys():
            if client_dict[k] > 0:
                optimal_amount_cl = k
                break
        if optimal_amount_cl is not None:
            # The optimization task has a solution.
            # Update with momentum
            if len(self.clients_history) == 0:
                optimal_amount_cl = optimal_amount_cl
            else:
                optimal_amount_cl = int(
                    (1 - self.num_clients_momentum) * self.clients_history[-1]
                    + self.num_clients_momentum * optimal_amount_cl
                )
        else:
            # The optimization task has no solution.
            # Take previous (or default=20) num of clients
            if len(self.clients_history) == 0:
                optimal_amount_cl = 20
            else:
                optimal_amount_cl = self.clients_history[-1]

        return optimal_amount_cl

    def find_optimal(self, borders_of_clients):
        # Here we calculate optimal amount of clients
        print(
            f"STARTING FIND OPTIMAL AMOUNT OF CLIENTS #{self.cnt_solve_optimal_task}:\n"
        )
        print(f"Trust loss history: {self.trust_losses}")
        print(f"Borders to solve problem: {borders_of_clients}\n")

        self.global_model.eval()
        saved_after_aggregate = copy.deepcopy(self.global_model.state_dict())

        client_dict = {
            k: 0.0
            for k in range(
                borders_of_clients[0], borders_of_clients[1], self.sample_step
            )
        }

        # self.mean_prev_loss = self.get_mean_prev_loss()
        self.mean_prev_loss = self.ema_loss_history()
        print(f"Mean previously trust loss: {self.mean_prev_loss}")

        for k in range(borders_of_clients[0], borders_of_clients[1], self.sample_step):
            print(f"\nNow clients = {k}")
            # client_dict[k] = self.get_functional_value(k)
            delta_f = self.get_functional_value(k)
            client_dict[k] = delta_f
            print()
            if delta_f > 0:
                print(f"Find positive delta for {k} amount of clients")
                optimal_amount_cl = k
                break

        # update optimal_amount_cl
        optimal_amount_cl = self.choose_optimal(client_dict)

        # print(f"After calculation client_dict: {client_dict}")

        print(f"OPTIMAL AMOUNT OF CLIENTS (optimal_amount_cl): {optimal_amount_cl}\n")

        self.global_model.load_state_dict(saved_after_aggregate)
        self.cnt_solve_optimal_task += 1
        return optimal_amount_cl

    #
    # Functional
    #

    def first_positive_functional(self, mean_loss):
        d_loss = self.mean_prev_loss - mean_loss
        return d_loss

    #
    # For warmup
    #

    def get_communication_content(self, rank):
        content = self.full_method_get_communication_content(rank)
        content["ISP_method_local_epoch"] = (
            self.local_epoch_on_wp
            if (self.cur_round < self.warmup_rounds)
            else self.cfg.federated_params.round_epochs
        )
        return content

    def client_create_pipe_commands(self):
        pipe_commands_map = self.full_method_create_pipe_commands()
        pipe_commands_map["ISP_method_local_epoch"] = self.set_local_epoch
        return pipe_commands_map

    def client_set_local_epoch(self, local_epoch):
        self.local_epoch = local_epoch

    def client_train_fn(self):
        # Diff with original client in 'range(self.local_epoch)'
        self.model.train()

        for _ in range(self.local_epoch):

            for batch in self.train_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(outputs, targets)

                loss.backward()

                self.optimizer.step()

                inp = input[0].to("cpu")
                targets = targets.to("cpu")
