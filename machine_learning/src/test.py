import logging
from typing import List, Optional, OrderedDict
import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint

from src.callbacks.log_best_metrics_callback import LogBestMetrics
from src.callbacks.user_halt_callback import UserHaltCallback
from src.helpers import cached_load_and_setup_datamodule, load_and_setup_datamodule
from src.utils import utils
from src.utils.custom_batch_size_tuner import custom_batch_size_tuner
from src.wandb.helpers import upload_config_and_checkpoints_to_wandb, is_wandb_logger_enabled
from src.custom_metrics.accuracy_calculator import MotionAccuracyCalculator

log = utils.get_logger(__name__)


def test(config: DictConfig, path_to_model: str) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    """
    # Init lightning loggers
    loggers = initialize_loggers(config)

    config = _set_seeds(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"seeding everything to {config.seed}")
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    if config.get("cache_datamodule", False):
        log.info(f"Instantiating datamodule (caching) <{config.datamodule._target_}>")
        # this caches the datamodule, which saves some time during stage 1 of the hp search
        datamodule = cached_load_and_setup_datamodule(config.datamodule)
    else:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule = load_and_setup_datamodule(config.datamodule)
    datamodule.setup(stage="test")

    log.info("saving data stats")
    datamodule.safe_train_data_stats_settings()

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")

    if config.model.num_out_classes == "auto":
        config.model.num_out_classes = datamodule.train_dataset.num_classes
    model = hydra.utils.instantiate(config.model, num_features=datamodule.train_dataset.num_features).to("cuda")
    log.info(str(model))

    checkpoint_path = path_to_model
   
    # explicitly set checkpoint directory
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)

    state_dict = OrderedDict(
        (k.replace("model.", ""), v)
        for k, v in checkpoint["state_dict"].items()
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Metric computation
    test_dataloader = datamodule.test_dataloader()
    top_k = 5

    all_knn_labels = []
    all_query_labels = []

    log.info("Running inference on test set")

    with torch.no_grad():
        for batch in test_dataloader:
            x = batch["data"].to("cuda").float()
            y = batch["targets"].to("cuda")

            logits = model(x)                       # [batch, num_classes]
            _, knn = torch.topk(logits, k=top_k, dim=1)

            all_knn_labels.append(knn.cpu())
            all_query_labels.append(y.cpu())

    knn_labels = torch.cat(all_knn_labels, dim=0)       # [N, K] (k number of candidates)
    query_labels = torch.cat(all_query_labels, dim=0)   # [N]

    log.info(f"KNN labels shape: {knn_labels.shape}")
    log.info(f"Query labels shape: {query_labels.shape}")

    log.info("MotionAccuracyCalculator metrics")

    sequence_lengths_minutes = [1, 2, 5, 10, 15, 20, 25]
    metric = MotionAccuracyCalculator(
        sequence_lengths_minutes=sequence_lengths_minutes,
        sliding_window_step_size_seconds=5,
    )

    # This populates internal caches
    metric._get_accuracy(
        function_dict={},
        knn_labels=knn_labels,
        query_labels=query_labels,
    )

    # Path to store results
    ckpt_path = Path(path_to_model)
    base_dir = ckpt_path.parent.parent

    # Start log
    log_file = base_dir / "test_log.txt"
    file_handler = logging.FileHandler(log_file, mode="a")
    log.addHandler(file_handler)
    
    # Print results
    log.info("===== Motion Accuracy Results =====")

    # Results for CSV
    results = []

    # Sequence metrics
    for mins in sequence_lengths_minutes:
        mrr_fn = getattr(metric, f"calculate_sequence_mrr_at_{mins}_mins")
        top1_fn = getattr(metric, f"calculate_sequence_top_1_accuracy_{mins}_mins")
        top2_fn = getattr(metric, f"calculate_sequence_top_2_accuracy_{mins}_mins")
        top3_fn = getattr(metric, f"calculate_sequence_top_3_accuracy_{mins}_mins")

        mrr = mrr_fn(knn_labels, query_labels)
        top1 = top1_fn(knn_labels, query_labels)
        top2 = top2_fn(knn_labels, query_labels)
        top3 = top3_fn(knn_labels, query_labels)

        log.info(
            f"Sequence {mins} min | "
            f"MRR: {mrr:.4f}, "
            f"Top-1: {top1:.4f}, "
            f"Top-2: {top2:.4f}, "
            f"Top-3: {top3:.4f}"
        )

        for name, val in {
            "mrr": mrr,
            "top1": top1,
            "top2": top2,
            "top3": top3,
        }.items():
            results.append({
                "model_path": config.model._target_,
                "category": "sequence",
                "time_window": f"{mins}_mins",
                "metric": name,
                "value": float(val)
            })

    # Path to CSV
    csv_path = base_dir / "motion_accuracy_results.csv"
    
    # Save results to CSV
    csv_path = Path(csv_path)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_path", "category", "time_window", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(results)

    log.info(f"Structured CSV saved to: {csv_path}")
    
    log.info("==================================")
    

    


def initialize_callbacks(config, datamodule):
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    if "monitored_callbacks" in config:
        callbacks += build_checkpoint_callback_list(config.monitored_metrics.metrics, store_path="early_stopping")

    callbacks += [LogBestMetrics(), UserHaltCallback()]
    return callbacks


def build_checkpoint_callback_list(checkpoint_metrics, store_path):
    checkpoint_metric_callbacks = []
    for metric in checkpoint_metrics:
        metric_id = "{name}_{dataset}".format(**metric)

        checkpoint_metric_callbacks.append(ModelCheckpoint(monitor=metric_id,
                                                           save_top_k=1,
                                                           mode=metric['mode'],
                                                           dirpath=store_path,
                                                           filename=f"{metric['mode']}_{metric_id}"
                                                           ))
    return checkpoint_metric_callbacks


def initialize_loggers(config):
    loggers: List[Logger] = []
    if "logger" in config:
        for logger_name, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating loggers <{lg_conf._target_}>")
                logger = hydra.utils.instantiate(lg_conf)
                loggers.append(logger)
    return loggers


def finalize_train(loggers):
    # tell wandb to finish to avoid problems during hp search
    if is_wandb_logger_enabled(loggers):
        import wandb
        wandb.finish()


def _set_seeds(config):
    if config.get("seed") == "random":
        config.seed = np.random.randint(0, 1000)
    if config.datamodule.get("seed") == "random":
        config.datamodule.seed = np.random.randint(0, 1000)
    return config
