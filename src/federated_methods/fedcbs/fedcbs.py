from ..fedavg.fedavg import FedAvg
from .fedcbs_server import FedCBSServer


class FedCBS(FedAvg):
    def __init__(self, num_clients_subset, lambda_):
        super().__init__(num_clients_subset)
        self.lambda_ = lambda_

    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

    def _init_server(self, cfg):
        self.server = FedCBSServer(cfg, self.lambda_, self.df)
    