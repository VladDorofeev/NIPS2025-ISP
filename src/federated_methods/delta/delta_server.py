from ..fedavg.fedavg_server import FedAvgServer
import numpy as np
import copy
from utils.model_utils import net_dict_weights_norm
from utils.model_utils import summ_dicts
from utils.model_utils import diff_dicts
from utils.model_utils import square_diff_dicts


class DeltaServer(FedAvgServer):
    def __init__(self, cfg, alpha_1, alpha_2):
        super().__init__(cfg)

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.client_probs = np.array(
            [1.0 / self.amount_of_clients for _ in range(self.amount_of_clients)]
        )
        print(f"Initial clients probabilities:\n{self.client_probs}")

        self.client_sigmas = [None for _ in range(self.amount_of_clients)]

    def update_probs(self, participated_clients):
        self.no_buffer_state_dict = [
            name
            for name, param in self.global_model.named_parameters()
            if param.requires_grad
        ]

        new_probs = copy.deepcopy(self.client_probs)

        N = len(participated_clients)
        # 1 / n sum_i g_it
        nabla_hat_f = {param_name: None for param_name in self.no_buffer_state_dict}

        # all g_it
        hat_g = [None for i in range(self.amount_of_clients)]

        for rank in participated_clients:
            hat_g[rank] = self.client_gradients[rank]
            nabla_hat_f = summ_dicts(
                nabla_hat_f, hat_g[rank], self.no_buffer_state_dict
            )

        # get 1/n for nabla hat_f
        for param_name in self.no_buffer_state_dict:
            nabla_hat_f[param_name] = 1 / N * nabla_hat_f[param_name]


        sqrts = [None for _ in range(self.amount_of_clients)]
        for rank in participated_clients:
            dzeta_i = net_dict_weights_norm(
                diff_dicts(hat_g[rank], nabla_hat_f, self.no_buffer_state_dict)
            )
            sqrts[rank] = np.sqrt(
                self.alpha_1 * dzeta_i**2 + self.alpha_2 * self.client_sigmas[rank] ** 2
            )
        
        summed_probs = sum([self.client_probs[rank] for rank, _ in enumerate(sqrts) if sqrts[rank] is None])
        for rank in participated_clients:
            new_probs[rank] = (
                sqrts[rank]
                / sum([x for x in sqrts if x is not None])
                * (1 - summed_probs)
            )

        return new_probs

    def set_client_result(self, client_result):
        super().set_client_result(client_result)
        self.client_sigmas[client_result["rank"]] = client_result["sigma"]

    def select_clients_to_train(self, num_clients_subset, server_sampling=False):
        if num_clients_subset == self.amount_of_clients:
            return list(range(self.amount_of_clients))

        clients = list(range(self.amount_of_clients))
        selected_clients = np.random.choice(
            clients, size=num_clients_subset, replace=False, p=self.client_probs
        ).tolist()

        return selected_clients
