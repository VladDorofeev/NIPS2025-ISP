from collections import OrderedDict
import time
import copy

from ..fedavg.fedavg_client import FedAvgClient
from utils.model_utils import net_dict_weights_norm
from utils.model_utils import summ_dicts
from utils.model_utils import diff_dicts
from utils.model_utils import square_diff_dicts


class DeltaClient(FedAvgClient):
    def __init__(self, *client_args, **client_kwargs):
        super().__init__(*client_args, **client_kwargs)
        self.batch_grads = []

    def get_communication_content(self):
        content = super().get_communication_content()
        content["batch_grads"] = None

        if not self.need_train:
            return content

        content["sigma"] = self.sigma
        return content

    def get_grad_by_batch(self):
        self.model.train()
        for batch in self.train_loader:
            _, (input, targets) = batch

            inp = input[0].to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.get_loss_value(outputs, targets)

            loss.backward()

            inp = input[0].to("cpu")
            targets = targets.to("cpu")

            # Collecting gradients
            self.model.eval()
            batch_grad = OrderedDict()
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    batch_grad[name] = param.grad.to("cpu")

            self.batch_grads.append(batch_grad)
            self.model.train()

    def get_sigma(self):
        self.no_buffer_state_dict = [
            name for name, param in self.model.named_parameters() if param.requires_grad
        ]

        summed_g_hat_b_i = {
            param_name: None for param_name in self.no_buffer_state_dict
        }

        for i in range(len(self.batch_grads)):
            summed_g_hat_b_i = summ_dicts(
                summed_g_hat_b_i, self.batch_grads[i], self.no_buffer_state_dict
            )

        B = len(self.batch_grads)
        for param_name in self.no_buffer_state_dict:
            summed_g_hat_b_i[param_name] = 1 / B * summed_g_hat_b_i[param_name]

        sigma_i = {param_name: None for param_name in self.no_buffer_state_dict}
        for i in range(len(self.batch_grads)):
            sigma_i = summ_dicts(
                sigma_i,
                square_diff_dicts(
                    self.batch_grads[i], summed_g_hat_b_i, self.no_buffer_state_dict
                ),
                self.no_buffer_state_dict,
            )

        for param_name in self.no_buffer_state_dict:
            sigma_i[param_name] = (1 / B * sigma_i[param_name]) ** 0.5

        sigma_i_norm = net_dict_weights_norm(sigma_i)

        return sigma_i_norm

    def train(self):
        if self.need_train:
            self.server_model_state = copy.deepcopy(self.model).state_dict()
            start = time.time()
            self.server_val_loss, self.server_metrics = self.eval_fn()
            self.train_fn()
            if self.print_metrics:
                self.client_val_loss, self.client_metrics = self.eval_fn()
            self.get_grad()
            self.result_time = time.time() - start

            # ------ DELTA ------ #
            self.batch_grads = []
            self.get_grad_by_batch()
            self.sigma = self.get_sigma()
            # ------ DELTA ------ #
        else:
            self.server_val_loss, self.server_metrics = self.eval_fn()
