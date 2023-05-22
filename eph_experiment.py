import argparse
import os
from pathlib import Path
import warnings
import yaml

warnings.filterwarnings("ignore")

import numpy as np
import torch
import pytorch_lightning as pl

from eph_clustering.models import EPH
from eph_clustering.data import EPHDataModule, ProbabilisticEPHDataModule
from eph_clustering.util import (
    get_initial_hierarchy,
    best_tree,
    Losses,
    compute_skn_tsd,
)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument("--out_dir", type=str, default="./", help="Path to results dir")
    args, _ = parser.parse_known_args()
    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)
    pl.seed_everything(config["seed"])
    gumbel_samples = config["model_params"].pop("gumbel_samples")
    config["dataset_params"]["gumbel_samples"] = gumbel_samples
    config["training_params"]["gumbel_samples"] = gumbel_samples
    return config


def get_dataloader(config):
    file_path = (
        Path(config["dataset_params"]["dataset_path"])
        / f"{config['dataset_params']['dataset_name']}.pkl.gzip"
    )
    if "percent" in config["dataset_params"].keys():
        return ProbabilisticEPHDataModule.from_pickle(
            str(file_path),
            dtype=torch.float32,
            **config["dataset_params"],
        )
    else:
        return EPHDataModule.from_pickle(
            str(file_path),
            dtype=torch.float32,
            **config["dataset_params"],
        )


if __name__ == "__main__":
    config = parse_config()
    data_module = get_dataloader(config)

    trainer = pl.Trainer(
        gpus=1 if config["training_params"]["use_gpu"] else 0,
        max_epochs=config["training_params"]["max_epochs"],
        accumulate_grad_batches=config["training_params"]["gumbel_samples"],
        check_val_every_n_epoch=config["training_params"]["val_every"],
        log_every_n_steps=1,
    )

    if "out_dir" in config.keys():
        out_dir = config["out_dir"]
    else:
        out_dir = trainer.log_dir

    init_from = get_initial_hierarchy(data_module, **config["model_params"])
    shape = init_from[0].shape
    model = EPH(num_nodes=shape[1], initialize_from=init_from, **config["model_params"])

    trainer.fit(
        model,
        data_module,
    )
    loss_function = Losses[config["model_params"]["loss"]]

    A = model.best_A_u_cont.detach().cpu()
    B = model.best_B_u_cont.detach().cpu()
    A_cont = model.best_A_u_cont.detach().cpu()
    B_cont = model.best_B_u_cont.detach().cpu()
    T = best_tree(A_cont.numpy(), B_cont.numpy())

    best_A = os.path.join(out_dir, "best_A.npy")
    best_B = os.path.join(out_dir, "best_B.npy")
    np.savez(best_A, A)
    np.savez(best_B, B)
    print(f"Saved best A in {best_A}")
    print(f"Saved best B in {best_B}")

    graph = next(iter(data_module.test_dataloader()))
    model.eval()
    with torch.no_grad():
        print("Dasgupta cost: ", model.compute_dasgupta(graph, A, B).metric)
        print("TSD: ", compute_skn_tsd(graph, T))
