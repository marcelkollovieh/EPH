import torch
import pytorch_lightning as pl
from sacred import Experiment
from collections import namedtuple

from eph_clustering.algorithms.shared import (
    PreprocessedGraph,
    compute_dasgupta,
    compute_TSD,
)
from eph_clustering.optimizers.padamax import configure_optimizers
from eph_clustering.util import Losses
from eph_clustering.util.utils import tree_to_A_B, best_tree

Result = namedtuple("Result", field_names=["A", "B", "loss", "tree", "metric"])


class ProbabilisticHierarchy(pl.LightningModule):
    def __init__(
        self,
        num_nodes,
        optimizer_params=None,
        ReduceLROnPlateau=None,
        loss=None,
        dtype=torch.float64,
        store_on_cpu_process_on_gpu=False,
        early_stopping=True,
        experiment: Experiment = None,
    ):
        super(ProbabilisticHierarchy, self).__init__()
        if isinstance(dtype, str):
            if dtype == "float32":
                dtype = torch.float32
            elif dtype == "float64":
                dtype = torch.float64
            elif dtype == "double":
                dtype = torch.float64
            elif dtype == "float16":
                dtype = torch.float16
            else:
                raise NotImplementedError("unknown dtype")

        self.num_nodes = num_nodes
        self.use_dtype = dtype
        torch.set_default_dtype(dtype)
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.ReduceLROnPlateau = ReduceLROnPlateau
        self.iteration_losses = []
        self.epoch_losses = []
        self.val_scores = []
        self.lowest_score = float("inf")
        self.early_stopping = early_stopping
        self.store_on_cpu_process_on_gpu = store_on_cpu_process_on_gpu
        self.best_A = None
        self.best_B = None
        self.best_A_u_cont = None
        self.best_B_u_cont = None
        self.best_A_u = None
        self.best_B_u = None

        if loss is None:
            loss = "EXP_DAS"
        self.loss = Losses[loss]
        self.experiment = experiment

    def sacred_log(self, key, val):
        if self.experiment is not None:
            if key not in self.experiment.current_run.info:
                self.experiment.current_run.info[key] = []
            self.experiment.current_run.info[key].append(val)

    def compute_A_B(self, input_nodes=None, adj=None, **kwargs):
        raise NotImplementedError()

    def forward(
        self,
        graph: PreprocessedGraph,
        A: torch.tensor = None,
        B: torch.tensor = None,
        sample=False,
    ) -> Result:
        node_ids = graph.node_ids
        if len(node_ids) == self.num_nodes:
            node_ids = None

        T = None
        if A is None or B is None:
            A, B = self.compute_A_B(input_nodes=node_ids)
        device = A.device
        if sample:
            A_log = (A + 1e-9).log()
            A = torch.nn.functional.gumbel_softmax(A_log, hard=True, tau=1)
            B_log = (B + 1e-9).log()
            B_log = B_log * torch.triu(torch.ones_like(B_log), diagonal=1) - torch.tril(
                1e16 * torch.ones_like(B_log), diagonal=0
            )
            B = torch.cat(
                [
                    torch.nn.functional.gumbel_softmax(B_log[:-1], hard=True, tau=1),
                    B[-1:],
                ]
            )
        if not self.training:
            T = best_tree(A.detach().cpu().numpy(), B.detach().cpu().numpy())
            A, B = tree_to_A_B(T, A.shape[0], B.shape[0])
            A = A.to(device)
            B = B.to(device)

        if self.loss == Losses.EXP_DAS:
            DAS = compute_dasgupta(A, B, adj=None, preprocessed_adj=graph)
            return Result(A=A, B=B, loss=DAS, tree=T, metric=DAS)
        elif self.loss == Losses.EXP_TSD:
            TSD = compute_TSD(A=A, B=B, preprocessed_adj=graph)
            return Result(
                A=A,
                B=B,
                loss=-TSD,
                tree=T,
                metric=-100 * (TSD / graph.mutual_information),
            )

    def compute_TSD(
        self,
        graph: PreprocessedGraph,
        A: torch.tensor = None,
        B: torch.tensor = None,
        sample=False,
    ) -> Result:
        return self(graph, A, B, sample=sample)

    def compute_dasgupta(
        self,
        graph: PreprocessedGraph,
        A: torch.tensor = None,
        B: torch.tensor = None,
        sample=False,
    ) -> Result:
        return self(graph, A, B, sample=sample)

    def compute_loss(self, batch: PreprocessedGraph):
        if self.loss == Losses.EXP_DAS:
            res = self.compute_dasgupta(batch, sample=True)
            metric = res.metric
            loss = res.loss
        elif self.loss == Losses.EXP_TSD:
            res = self.compute_TSD(batch, sample=True)
            metric = res.metric
            loss = res.loss
        else:
            raise NotImplementedError("unknown loss.")
        return loss, metric

    def training_step(self, batch: PreprocessedGraph):
        loss, metric = self.compute_loss(batch=batch)
        self.iteration_losses.append(loss.item())
        self.sacred_log("train_loss_step", loss.item())
        return dict(loss=loss, DASGUPTA=metric.detach())

    def on_train_epoch_end(self) -> None:
        loss = torch.tensor(self.iteration_losses).mean().item()
        self.iteration_losses = []
        self.epoch_losses.append(loss)
        self.sacred_log("train_loss", loss)

    def validation_step(self, batch: PreprocessedGraph, batch_idx):
        if self.loss == Losses.EXP_DAS:
            res_das = self.compute_dasgupta(batch)
        elif self.loss == Losses.EXP_TSD:
            res_das = self.compute_TSD(batch)
        else:
            raise NotImplementedError
        if self.early_stopping and res_das.metric < self.lowest_score:
            self.lowest_score = res_das.metric.detach()
            self.best_A = res_das.A.clone()
            self.best_B = res_das.B.clone()
            A, B = self.compute_A_B()
            self.best_A_u_cont = A.clone()
            self.best_B_u_cont = B.clone()
            self.best_A_u = torch.nn.functional.one_hot(
                A.argmax(dim=1), num_classes=A.shape[1]
            )
            self.best_B_u = torch.nn.functional.one_hot(
                B.argmax(dim=1), num_classes=B.shape[1]
            )
        self.sacred_log("val_score", res_das.metric.detach().item())
        self.val_scores.append(res_das.metric.detach().item())
        if (
            self.ReduceLROnPlateau is not None
            and self.current_epoch
            and ((self.current_epoch + 1) % self.ReduceLROnPlateau["reset"] == 0)
        ):
            self.trainer.optimizers[0].param_groups[0]["lr"] *= self.ReduceLROnPlateau[
                "factorA"
            ]
            self.trainer.optimizers[0].param_groups[1]["lr"] *= self.ReduceLROnPlateau[
                "factorB"
            ]
            # discrete
            self.A_u.weight.data = self.best_A_u.to(self.use_dtype)
            self.B_u.data = self.best_B_u.to(self.use_dtype)

    def configure_optimizers(self):
        return configure_optimizers(self)
