import shutil

import numpy as np
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.tools.files import json_dump


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    fabric = instantiate(cfg.trainer.fabric)
    fabric.launch()

    if fabric.global_rank == 0:
        json_dump(OmegaConf.to_container(cfg, resolve=True), "hydra.json")

    model = instantiate(cfg.model)
    model = fabric.setup(model)

    for dataset in cfg.test:
        columns = shutil.get_terminal_size().columns
        fabric.print("-" * columns)
        fabric.print(f"Testing {cfg.test[dataset].dataname}".center(columns))

        data = instantiate(cfg.test[dataset])
        test_loader = fabric.setup_dataloaders(data.test_dataloader())

        test = instantiate(cfg.test[dataset].test)
        recalls, scores_q2t = test(model, test_loader, fabric=fabric)

        suffix = "txt_only" if cfg.test[dataset].test["_target_"] == "src.test.webvid_covr_exp.TestWebVidCoVRTextOnly" else ""
        print("Recalls: ")
        print(recalls)

        # Save scores
        np.save(f'scores_q2t_{suffix}.npy', scores_q2t)
        print(f"Query to Target Scores saved in {Path.cwd()} as scores_q2t_{suffix}.npy")

        # Save results
        json_dump(recalls, f"recalls_covr_{suffix}.json")
        print(f"Recalls saved in {Path.cwd()} as recalls_covr_{suffix}.json")


if __name__ == "__main__":
    main()
