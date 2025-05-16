import sys
import copy
import time
import signal
import torch
import pandas as pd
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate

from utils.losses import get_loss
from utils.data_utils import get_dataset_loader
from utils.metrics_utils import calculate_metrics
from utils.utils import handle_client_process_sigterm
from utils.attack_utils import add_attack_functionality


class BaseClient:
    def __init__(self, *client_args, **client_kwargs):
        self.client_args = client_args
        self.client_kwargs = client_kwargs
        cfg = self.client_args[0]
        df = self.client_args[1]
        self.cfg = cfg
        self.df = df
        self.train_df = df
        self.rank = client_kwargs["rank"]
        self.pipe = client_kwargs["pipe"]
        self.valid_df = None
        self.train_loader = None
        self.valid_loader = None
        self.criterion = None
        self.server_model_state = None
        self.print_metrics = cfg.federated_params.print_client_metrics
        self.train_val_prop = cfg.federated_params.client_train_val_prop
        self.device = (
            "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            if cfg.training_params.device == "cuda"
            else "cpu"
        )

        if self.client_kwargs["first_init"]:
            self.model = instantiate(cfg.models[0])
            self.client_kwargs["first_init"] = False

        self.model.to(self.device)
        self._set_train_df()
        self._init_loaders()
        self._init_optimizer()
        self._init_criterion()
        self.pipe_commands_map = self.create_pipe_commands()

        self.grad = OrderedDict()

    def _init_optimizer(self):
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

    def _init_criterion(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_df,
            init_pos_weight=self.init_pos_weight,
        )

    def _set_train_df(self):
        self.train_df = self.df[self.df["client"] == self.rank]

    def _init_loaders(self):
        n_classes = self.cfg.training_params.num_classes

        # Check can we init pos weight
        self.init_pos_weight = (
            self.train_df["target"]
            .apply(lambda x: x[0] if isinstance(x, list) else x)
            .nunique()
            == n_classes
        )

        minor_classes_ids = (
            self.train_df["target"]
            .value_counts()[self.train_df["target"].value_counts() < 2]
            .index
        )
        major_classes_df = self.train_df[
            ~self.train_df["target"].isin(minor_classes_ids)
        ]
        minor_classes_df = self.train_df[
            self.train_df["target"].isin(minor_classes_ids)
        ]

        if self.train_val_prop * len(major_classes_df) < n_classes:
            self.new_train_val_prop = (
                1 / major_classes_df["target"].value_counts().min()
            )
            self.train_val_prop = self.new_train_val_prop

        self.train_df, self.valid_df = train_test_split(
            major_classes_df,
            test_size=self.train_val_prop,
            stratify=major_classes_df["target"],
            random_state=self.cfg.random_state,
        )
        self.train_df = pd.concat([self.train_df, minor_classes_df]).reset_index(
            drop=True
        )

        self.train_loader = get_dataset_loader(self.train_df, self.cfg, drop_last=False)
        self.valid_loader = get_dataset_loader(
            self.valid_df, self.cfg, drop_last=False, mode="valid"
        )

    def _set_attack_type(self, attack_content):
        self.attack_type = attack_content[0]
        self.attack_config = attack_content[1]

    def reinit_self(self, new_rank):
        self.client_kwargs["rank"] = new_rank
        self.__init__(*self.client_args, **self.client_kwargs)

        # Recive content for local learning
        content = self.pipe.recv()
        self.parse_communication_content(content)

    def shutdown_self(self):
        print(f"Exit child {self.rank} process")
        sys.exit(0)

    def create_pipe_commands(self):
        # define a structure to process pipe values
        pipe_commands_map = {
            "update_model": lambda state_dict: self.model.load_state_dict(
                {k: v.to(self.device) for k, v in state_dict.items()}
            ),
            "attack_type": self._set_attack_type,
            "shutdown": lambda _: self.shutdown_self(),
            "reinit": lambda new_rank: self.reinit_self(new_rank),
        }

        return pipe_commands_map

    def train_fn(self):
        self.model.train()
        for _ in range(self.cfg.federated_params.round_epochs):

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

    def get_loss_value(self, outputs, targets):
        return self.criterion(outputs, targets)

    def eval_fn(self):
        self.model.eval()
        val_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for _, batch in enumerate(self.valid_loader):
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inp)

                val_loss += self.criterion(outputs, targets).detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

                inp = input[0].to("cpu")
                targets = targets.to("cpu")

        client_metrics, _ = calculate_metrics(fin_targets, fin_outputs)

        return val_loss / len(self.valid_loader), client_metrics

    def get_grad(self):
        self.model.eval()
        for key, _ in self.model.state_dict().items():
            self.grad[key] = (
                self.model.state_dict()[key] - self.server_model_state[key]
            ).to("cpu")

    def train(self):
        # Save the server model state to get_grad
        self.server_model_state = copy.deepcopy(self.model).state_dict()
        # Validate server weights before training to set up best model
        start = time.time()
        self.server_val_loss, self.server_metrics = self.eval_fn()
        # Train client
        self.train_fn()
        # Get client metrics
        if self.print_metrics:
            self.client_val_loss, self.client_metrics = self.eval_fn()
        # Calculate client update
        self.get_grad()
        # Save training time
        self.result_time = time.time() - start

    def get_communication_content(self):
        # In fedavg_client we need to send only result of local learning
        result_dict = {
            "grad": self.grad,
            "rank": self.rank,
            "time": self.result_time,
            "server_metrics": (
                self.server_metrics,
                self.server_val_loss,
                len(self.valid_df),
            ),
        }
        if self.print_metrics:
            result_dict["client_metrics"] = (self.client_val_loss, self.client_metrics)

        return result_dict

    def parse_communication_content(self, content):
        # In fedavg_client we need to recive model after aggregate and
        # attack type for this client
        for key, value in content.items():
            if key in self.pipe_commands_map.keys():
                self.pipe_commands_map[key](value)
            else:
                raise ValueError(
                    f"Recieved content in client {self.rank} from server, with unknown key={key}"
                )


def multiprocess_client(*client_args, client_cls, pipe, rank, attack_type):
    # Init client instance
    client_kwargs = {"pipe": pipe, "rank": rank, "first_init": True}
    client = client_cls(*client_args, **client_kwargs)
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: handle_client_process_sigterm(signum, frame, rank),
    )

    # Loop of federated learning
    while True:
        # Wait content from server to start local learning
        content = client.pipe.recv()
        client.parse_communication_content(content)

        # Can be this realization of attack
        if client.attack_type != "no_attack":
            client = add_attack_functionality(
                client, client.attack_type, client.attack_config
            )

        client.train()

        # Send content to server, local learning ended
        content = client.get_communication_content()
        client.pipe.send(content)
