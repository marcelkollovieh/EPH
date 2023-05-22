import torch
from sacred import Experiment
from torch import nn

from eph_clustering.models import ProbabilisticHierarchy


class EPH(ProbabilisticHierarchy):
    def __init__(
        self,
        internal_nodes,
        num_nodes,
        optimizer_params,
        ReduceLROnPlateau=None,
        loss=None,
        dtype=torch.float64,
        store_on_cpu_process_on_gpu=False,
        initialize_from=None,
        early_stopping=True,
        experiment: Experiment = None,
        **kwargs,
    ):
        """
        :param internal_nodes: Internal nodes, i.e. A.shape[1]
        :param num_nodes: Number of nodes in the dataset, i.e. A.shape[0]
        :param optimizer_params: parameters for the optimizer, see configs for examples
        :param ReduceLROnPlateau: Dict for the LR scheduler, see configs for examples
        :param loss: EXP_DAS or EXP_TSD
        :param initialize_from: Tuple of initial matrices, can be None
        :param experiment: Sacred experiment for logging
        """
        super(EPH, self).__init__(
            num_nodes=num_nodes,
            optimizer_params=optimizer_params,
            ReduceLROnPlateau=ReduceLROnPlateau,
            dtype=dtype,
            loss=loss,
            store_on_cpu_process_on_gpu=store_on_cpu_process_on_gpu,
            early_stopping=early_stopping,
            experiment=experiment,
            **kwargs,
        )
        self.internal_nodes = internal_nodes if internal_nodes != 0 else num_nodes - 1

        if initialize_from is None:
            self.A_u = nn.Embedding(self.num_nodes, embedding_dim=self.internal_nodes)
            self.A_u.weight.data = self.A_u.weight.data.to(self.use_dtype).softmax(-1)

            self.B_u = -1e30 * torch.ones(
                (self.internal_nodes, self.internal_nodes), dtype=self.use_dtype
            )
            triu_ixs = torch.triu_indices(*self.B_u.shape, offset=1)
            self.B_u[tuple(triu_ixs)] = 0
            B_rand = torch.rand(
                self.internal_nodes, self.internal_nodes, dtype=self.use_dtype
            )
            B_rand = torch.triu(B_rand, diagonal=1)
            self.B_u[tuple(triu_ixs)] = B_rand[tuple(triu_ixs)]
            self.B_u = self.B_u.softmax(-1)
            self.B_u = self.B_u.mul(torch.triu(torch.ones_like(self.B_u), diagonal=1))

            self.B_u = nn.Parameter(self.B_u, requires_grad=True)
        elif type(initialize_from) == tuple:
            A, B = initialize_from
            self.A_u = nn.Embedding(self.num_nodes, embedding_dim=self.internal_nodes)
            self.A_u.weight.data = A.to(self.use_dtype)

            self.B_u = nn.Parameter(B.to(self.use_dtype), requires_grad=True)
        elif type(initialize_from) == str and initialize_from == "avg":
            pass
        else:
            raise NotImplementedError("Unknown initialization.")

    def compute_A_B(self, **kwargs):
        """
        Assemble the A and B matrices.
        Parameters
        ----------
        input_nodes: torch.Tensor, shape [V']
        Returns
        -------
        A: torch.Tensor, shape [V', N], dtype: torch.float
            The row-stochastic matrix containing the parent probabilities of the internal nodes
            w.r.t. the graph nodes.
        B: torch.Tensor, shape [N, N], dtype torch.float
            The matrix containing the parent probabilities between the internal nodes.
            Properties:
                - row-stochastic
                - upper-triangular
                - zero diagonal
                :param **kwargs:
        """
        device = self.B_u.device if not self.store_on_cpu_process_on_gpu else "cuda"

        A_u = self.A_u.weight
        A = A_u.to(device)
        B = self.B_u.to(device)
        B = torch.cat([B[:-1], torch.zeros_like(B[-1])[None, :]], dim=0)
        B = B.mul(torch.triu(torch.ones_like(B), diagonal=1))
        return A, B
