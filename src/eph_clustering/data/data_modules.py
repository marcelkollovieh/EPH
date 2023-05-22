import gzip
import pickle
from collections import namedtuple

import numpy as np
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import to_undirected

import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from eph_clustering.algorithms.shared import preprocess_graph, torch_coo_eliminate_zeros

GraphBatch = namedtuple("GraphBatch", field_names=["adjacency"])
ProbabilisticGraphBatch = namedtuple(
    "ProbabilisticGraphBatch", field_names=["adjacency", "sample_adjacency"]
)


class EPHDataset(Dataset):
    def __init__(
        self,
        adjacency: torch.sparse_coo_tensor,
        make_undirected=True,
        select_lcc=True,
        remove_selfloops=False,
        make_unweighted=True,
        dtype=torch.float32,
        batches_per_epoch=1,
        normalize_graph=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.preprocessed_train_cache = None
        self.preprocessed_val_cache = None
        adjacency = adjacency.coalesce()
        num_nodes = adjacency.shape[0]

        if remove_selfloops:
            adjacency = (
                adjacency
                - torch.sparse_coo_tensor(
                    torch.arange(num_nodes).tile([2, 1]),
                    torch.ones(num_nodes),
                    size=adjacency.shape,
                )
                * adjacency
            )
            adjacency = torch_coo_eliminate_zeros(adjacency).coalesce()

        edge_index = adjacency.indices()
        values = adjacency.values().to(dtype)

        if make_unweighted:
            values = torch.ones_like(values)

        if make_undirected:
            edge_index, values = to_undirected(edge_index, values, adjacency.shape[0])

        if select_lcc:
            A_sp = sp.csr_matrix((values, edge_index), shape=adjacency.shape)
            n_components, labels = connected_components(
                csgraph=A_sp, directed=False, return_labels=True
            )
            unq_labels, counts = np.unique(labels, return_counts=True)
            asort = np.argsort(counts)[::-1]
            lcc_nodes = (labels == unq_labels[asort[0]]).nonzero()[0]
            A_sp = A_sp[np.ix_(lcc_nodes, lcc_nodes)]
            r, c = A_sp.nonzero()
            edge_index = torch.tensor(np.array((r, c)))
            values = torch.tensor(A_sp[r, c].A1, dtype=dtype)
            num_nodes = A_sp.shape[0]
        self.adjacency = torch.sparse_coo_tensor(
            edge_index, values, size=[num_nodes, num_nodes], dtype=dtype
        ).coalesce()
        self.batches_per_epoch = batches_per_epoch
        self.normalize_graph = normalize_graph

    @classmethod
    def from_pickle(
        self,
        path: str,
        make_undirected=True,
        make_unweighted=True,
        remove_selfloops=False,
        select_lcc=True,
        dtype=torch.float32,
        batches_per_epoch=1,
        normalize_graph=True,
    ):
        if path.endswith(".gzip"):
            with gzip.open(path, "rb") as f:
                loader = pickle.load(f)
        else:
            with open(path, "rb") as f:
                loader = pickle.load(f)
        assert "adjacency" in loader
        adjacency = loader["adjacency"]
        if type(adjacency).__module__ == np.__name__:
            adjacency = sp.csr_matrix(adjacency)
        if sp.isspmatrix(adjacency):
            r, c = adjacency.nonzero()
            adjacency = torch.sparse_coo_tensor(
                torch.tensor(np.array((r, c))),
                torch.tensor(adjacency[r, c].A1, dtype=dtype),
                adjacency.shape,
            )
        return EPHDataset(
            adjacency,
            make_undirected=make_undirected,
            make_unweighted=make_unweighted,
            remove_selfloops=remove_selfloops,
            select_lcc=select_lcc,
            batches_per_epoch=batches_per_epoch,
            normalize_graph=normalize_graph,
        )

    def __len__(self):
        return self.batches_per_epoch
        # return 1   # only one sample (which is the whole graph)

    def __getitem__(self, item):
        return GraphBatch(
            adjacency=self.adjacency,
        )

    def collate_fn_train(self, batch: GraphBatch):
        if self.preprocessed_train_cache is not None:
            preprocessed = self.preprocessed_train_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_train_cache = preprocessed
        return preprocessed

    def collate_fn_val(self, batch: GraphBatch):
        # no node dropout
        if self.preprocessed_val_cache is not None:
            preprocessed = self.preprocessed_val_cache
        else:
            preprocessed = preprocess_graph(
                batch[0].adjacency, normalize=self.normalize_graph
            )
            self.preprocessed_val_cache = preprocessed
        return preprocessed

    def collate_fn_test(self, batch: GraphBatch):
        # no node dropout
        if self.preprocessed_val_cache is not None:
            preprocessed = self.preprocessed_val_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_val_cache = preprocessed
        return preprocessed


class EPHDataModule(pl.LightningDataModule):
    """
    Wrapper class around a EPHDataset to work with Pytorch Lightning.
    **Note**: All data loaders simply return the whole graph.
    """

    def __init__(self, dataset: EPHDataset, val_dataset: EPHDataset = None):
        super().__init__()
        self.dataset = dataset
        self.val_dataset = dataset if val_dataset is None else val_dataset
        self.num_nodes = self.dataset.adjacency.shape[0]

    @classmethod
    def from_pickle(
        self,
        path: str,
        make_undirected=True,
        make_unweighted=True,
        remove_selfloops=False,
        select_lcc=True,
        dtype=torch.float32,
        gumbel_samples=1,
        normalize_graph=True,
        **kwargs,
    ):
        dataset = EPHDataset.from_pickle(
            path,
            make_undirected=make_undirected,
            make_unweighted=make_unweighted,
            remove_selfloops=remove_selfloops,
            select_lcc=select_lcc,
            dtype=dtype,
            batches_per_epoch=gumbel_samples,
            normalize_graph=normalize_graph,
        )

        if gumbel_samples != 1:
            val_dataset = EPHDataset.from_pickle(
                path,
                make_undirected=make_undirected,
                make_unweighted=make_unweighted,
                remove_selfloops=remove_selfloops,
                select_lcc=select_lcc,
                normalize_graph=normalize_graph,
                dtype=dtype,
            )
            return EPHDataModule(dataset, val_dataset)
        else:
            return EPHDataModule(dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_train,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_val,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_test,
            num_workers=0,
        )


class ProbabilisticEPHDataset(Dataset):
    def __init__(
        self,
        adjacency: torch.sparse_coo_tensor,
        make_undirected=True,
        select_lcc=True,
        remove_selfloops=False,
        make_unweighted=True,
        dtype=torch.float32,
        batches_per_epoch=1,
        normalize_graph=True,
        percent=0.01,
        path=None,
        **kwargs,
    ) -> None:
        super().__init__()
        num_nodes = adjacency.shape[0]
        adjacency = adjacency.coalesce()
        self.preprocessed_val_cache = None
        self.path = path
        if remove_selfloops:
            adjacency = (
                adjacency
                - torch.sparse_coo_tensor(
                    torch.arange(num_nodes).tile([2, 1]),
                    torch.ones(num_nodes),
                    size=adjacency.shape,
                )
                * adjacency
            )
            adjacency = torch_coo_eliminate_zeros(adjacency).coalesce()

        edge_index = adjacency.indices()
        values = adjacency.values().to(dtype)

        if make_unweighted:
            values = torch.ones_like(values)

        if make_undirected:
            edge_index, values = to_undirected(edge_index, values, adjacency.shape[0])

        if select_lcc:
            A_sp = sp.csr_matrix((values, edge_index), shape=adjacency.shape)
            n_components, labels = connected_components(
                csgraph=A_sp, directed=False, return_labels=True
            )
            unq_labels, counts = np.unique(labels, return_counts=True)
            asort = np.argsort(counts)[::-1]
            lcc_nodes = (labels == unq_labels[asort[0]]).nonzero()[0]
            A_sp = A_sp[np.ix_(lcc_nodes, lcc_nodes)]
            r, c = A_sp.nonzero()
            edge_index = torch.tensor(np.array((r, c)))
            values = torch.tensor(A_sp[r, c].A1, dtype=dtype)
            num_nodes = A_sp.shape[0]
        self.adjacency = torch.sparse_coo_tensor(
            edge_index, values, size=[num_nodes, num_nodes], dtype=dtype
        ).coalesce()
        self.sample_adjacency = self.adjacency.to_dense().numpy().astype("float64")
        self.sample_adjacency /= self.sample_adjacency.sum()
        self.batches_per_epoch = batches_per_epoch
        self.normalize_graph = normalize_graph
        if percent == "root_n":
            self.num_sampled_edges = int(np.sqrt(adjacency.shape[0]))
        elif percent == "n":
            self.num_sampled_edges = int(adjacency.shape[0])
        elif percent == "n_root_n":
            self.num_sampled_edges = int(
                adjacency.shape[0] * np.sqrt(adjacency.shape[0])
            )
        elif percent == "n_qroot_n":
            self.num_sampled_edges = int(adjacency.shape[0] ** 1.25)
        else:
            self.num_sampled_edges = int(adjacency.shape[0] ** 2 * percent)
        self.all_edges = np.arange(int(adjacency.shape[0] ** 2))

    @classmethod
    def from_pickle(
        self,
        path: str,
        make_undirected=True,
        make_unweighted=True,
        remove_selfloops=False,
        select_lcc=True,
        dtype=torch.float32,
        batches_per_epoch=1,
        normalize_graph=True,
        percent=0.01,
    ):
        if path.endswith(".gzip"):
            with gzip.open(path, "rb") as f:
                loader = pickle.load(f)
        else:
            with open(path, "rb") as f:
                loader = pickle.load(f)
        assert "adjacency" in loader
        adjacency = loader["adjacency"]
        if type(adjacency).__module__ == np.__name__:
            adjacency = sp.csr_matrix(adjacency)
        if sp.isspmatrix(adjacency):
            r, c = adjacency.nonzero()
            adjacency = torch.sparse_coo_tensor(
                torch.tensor(np.array((r, c))),
                torch.tensor(adjacency[r, c].A1, dtype=dtype),
                adjacency.shape,
            )
        return ProbabilisticEPHDataset(
            adjacency,
            make_undirected=make_undirected,
            make_unweighted=make_unweighted,
            remove_selfloops=remove_selfloops,
            select_lcc=select_lcc,
            batches_per_epoch=batches_per_epoch,
            normalize_graph=normalize_graph,
            percent=percent,
            path=path,
        )

    def __len__(self):
        return self.batches_per_epoch
        # return 1   # only one sample (which is the whole graph)

    def __getitem__(self, item):
        return ProbabilisticGraphBatch(
            adjacency=self.adjacency, sample_adjacency=self.sample_adjacency
        )

    def collate_fn_train(self, batch: ProbabilisticGraphBatch):
        edges = np.random.choice(
            batch[0].sample_adjacency.shape[0] ** 2,
            int(self.num_sampled_edges),
            p=batch[0].sample_adjacency.reshape(-1),
            replace=True,
        )
        u, v = torch.from_numpy(
            np.array(np.unravel_index(edges, batch[0].sample_adjacency.shape))
        )
        return preprocess_graph(
            torch.sparse_coo_tensor(
                torch.stack([u, v], 0),
                torch.ones_like(u) / int(self.num_sampled_edges),
                batch[0].sample_adjacency.shape,
            ),
            normalize=True,
        )

    def collate_fn_val(self, batch: ProbabilisticGraphBatch):
        # no node dropout
        if self.preprocessed_val_cache is not None:
            preprocessed = self.preprocessed_val_cache
        else:
            if "cifar100" in self.path:
                edges = np.random.choice(
                    batch[0].sample_adjacency.shape[0] ** 2,
                    int(self.num_sampled_edges),
                    p=batch[0].sample_adjacency.reshape(-1),
                    replace=True,
                )
                u, v = torch.from_numpy(
                    np.array(np.unravel_index(edges, batch[0].sample_adjacency.shape))
                )
                preprocessed = preprocess_graph(
                    torch.sparse_coo_tensor(
                        torch.stack([u, v], 0),
                        torch.ones_like(u) / int(self.num_sampled_edges),
                        batch[0].sample_adjacency.shape,
                    ),
                    normalize=True,
                )
            else:
                preprocessed = preprocess_graph(
                    batch[0].adjacency, normalize=self.normalize_graph
                )
            self.preprocessed_val_cache = preprocessed
        return preprocessed

    def collate_fn_test(self, batch: ProbabilisticGraphBatch):
        # no node dropout
        if self.preprocessed_val_cache is not None:
            preprocessed = self.preprocessed_val_cache
        else:
            preprocessed = preprocess_graph(batch[0].adjacency)
            self.preprocessed_val_cache = preprocessed
        return preprocessed


class ProbabilisticEPHDataModule(pl.LightningDataModule):
    """
    Wrapper class around a EPHDataset to work with Pytorch Lightning.
    **Note**: All data loaders simply return the whole graph.
    """

    def __init__(
        self,
        dataset: ProbabilisticEPHDataset,
        val_dataset: ProbabilisticEPHDataset = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.val_dataset = dataset if val_dataset is None else val_dataset
        self.num_nodes = self.dataset.adjacency.shape[0]

    @classmethod
    def from_pickle(
        self,
        path: str,
        make_undirected=True,
        make_unweighted=True,
        remove_selfloops=False,
        select_lcc=False,
        dtype=torch.float32,
        batches_per_epoch=1,
        normalize_graph=True,
        percent=0.01,
        **kwargs,
    ):
        dataset = ProbabilisticEPHDataset.from_pickle(
            path,
            make_undirected=make_undirected,
            make_unweighted=make_unweighted,
            remove_selfloops=remove_selfloops,
            select_lcc=select_lcc,
            dtype=dtype,
            batches_per_epoch=batches_per_epoch,
            normalize_graph=normalize_graph,
            percent=percent,
        )

        if batches_per_epoch != 1:
            val_dataset = ProbabilisticEPHDataset.from_pickle(
                path,
                make_undirected=make_undirected,
                make_unweighted=make_unweighted,
                remove_selfloops=remove_selfloops,
                select_lcc=select_lcc,
                normalize_graph=normalize_graph,
                percent=percent,
                dtype=dtype,
            )
            return ProbabilisticEPHDataModule(dataset, val_dataset)
        else:
            return ProbabilisticEPHDataModule(dataset)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_train,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_val,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=self.dataset.collate_fn_test,
            num_workers=0,
        )
