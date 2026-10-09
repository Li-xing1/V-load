# -*- coding: utf-8 -*-
import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.V_S_dataset import V_S_dataset
from dataset.V_W_dataset import AxisWeightStandardizer, ContextFeatureStandardizer
from models.V_S import V_S
from utils.Auxiliary import create_exp_dir, move2device
from utils.DistributionEvaluation import (
    continuous_metrics,
    crps_ensemble,
    quantile_errors,
    threshold_probability_errors,
    update_evaluation_report,
)
from utils.Experiment import (
    ExperimentPaths,
    get_experiment_logger,
    get_experiment_writer,
    log_training_start,
    save_array_xlsx,
    save_checkpoint_manifest_entry,
    save_training_curves,
)
from utils.Train import Average, group_balanced_mean, primary_device
from utils.TrainingConfig import load_merged_args, resolve_work_path

torch.set_default_dtype(torch.float32)


class V_S_Train:
    def __init__(self, args, work_path, device="cuda:0"):
        self.args = args
        self.debug = self.args.get("debug", False)
        self.batch_size = self.args["batch_size"]
        self.epoch = 2 if self.debug else self.args["epoch"]
        self.device = primary_device(device)
        self.target_normalizer = None
        self.context_normalizer = None
        self.model = V_S(self.args["model"]).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.args["lr"], eps=self.args["eps"],
            weight_decay=self.args["weight_decay"], foreach=False,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[int(value * self.epoch) for value in self.args["milestones"]],
            gamma=self.args["gamma"],
        )

        self.work_path = work_path
        self.paths = ExperimentPaths(work_path).create()
        self.save_dir = self.paths.model_data_dir("V_S")
        self.figure_dir = self.paths.model_figure_dir("V_S")
        self.checkpoint_path = self.paths.checkpoint_path("V_S")
        self.log_file = self.paths.model_log_file("V_S")
        self.tensorboard_dir = self.paths.tensorboard_dir("V_S")

    def _make_dataset(self, split):
        dataset = V_S_dataset(
            self.args["dataset"],
            debug=self.debug,
            split=split,
            target_normalizer=self.target_normalizer,
            context_normalizer=self.context_normalizer,
        )
        if split == "Train":
            self.target_normalizer = dataset.target_normalizer
            self.context_normalizer = dataset.context_normalizer
        return dataset

    def _normalization_state_dict(self):
        return {
            "target": self.target_normalizer.state_dict(),
            "context": self.context_normalizer.state_dict(),
        }

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        normalization = checkpoint["normalization"]
        self.target_normalizer = AxisWeightStandardizer.from_state_dict(normalization["target"])
        self.context_normalizer = ContextFeatureStandardizer.from_state_dict(normalization["context"])
        self.model.eval()
        return checkpoint

    def train_epoch(self, loader, group_weight_lookup=None):
        average = Average()
        for target, context, bridge, lane, _ in tqdm(loader, unit="batch", desc="Train"):
            self.optimizer.zero_grad()
            target, context, bridge, lane = move2device([target, context, bridge, lane], self.device)
            log_prob = self.model.log_prob(target, context=context)
            loss = -(
                group_balanced_mean(log_prob, bridge, lane, group_weight_lookup)
                if group_weight_lookup is not None else log_prob.mean()
            )
            average.add(loss.item(), len(target))
            loss.backward()
            if self.args["clip_grad_norm"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args["max_grad_norm"], foreach=False)
            self.optimizer.step()
        self.scheduler.step()
        return average.average()

    def val(self, loader, mode="val"):
        average = Average()
        with torch.no_grad():
            for target, context, _, _, _ in tqdm(loader, unit="batch", desc=mode):
                target, context = move2device([target, context], self.device)
                log_prob = self.model.log_prob(target, context=context).mean()
                average.add(log_prob.item(), len(target))
        return average.average()

    def _save_scope(self, variant, samples_by_condition, flow):
        save_dir = self.paths.model_data_dir("V_S", variant) if variant else self.save_dir
        create_exp_dir(save_dir)
        samples = samples_by_condition.reshape(-1)
        flow = flow.reshape(-1)
        np.save(os.path.join(save_dir, "samples_by_condition.npy"), samples_by_condition)
        np.save(os.path.join(save_dir, "samples.npy"), samples)
        np.save(os.path.join(save_dir, "flow.npy"), flow)
        save_array_xlsx(os.path.join(save_dir, "sample_values.xlsx"), {"samples": samples, "flow": flow})

        metrics = continuous_metrics(flow, samples)
        metrics.update(quantile_errors(flow, samples))
        metrics.update(threshold_probability_errors(flow, samples))
        metrics["crps"] = crps_ensemble(samples_by_condition[..., 0], flow)
        update_evaluation_report(
            os.path.join(self.paths.data_dir, "distribution_evaluation.xlsx"),
            "V_S", variant or "All", metrics,
            units={key: "m" for key in metrics if key in {"wasserstein_1", "crps"} or key.endswith("_error")},
        )

    def sample(self, checkpoint_path=None, split="Test", eval_num_samples=5, logger=None):
        self._load_checkpoint(checkpoint_path or self.checkpoint_path)
        dataset = self._make_dataset(split)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        samples_chunks = []
        target_chunks = []
        bridge_chunks = []
        lane_chunks = []
        timeline_chunks = []
        with torch.no_grad():
            for target, context, bridge, lane, timeline in tqdm(loader, unit="batch", desc="Sample"):
                context = context.to(self.device)
                samples = self.model.sample(eval_num_samples, context=context).cpu().numpy()
                samples_chunks.append(self.target_normalizer.inverse_transform(samples))
                target_chunks.append(self.target_normalizer.inverse_transform(target.numpy()))
                bridge_chunks.append(bridge.numpy())
                lane_chunks.append(lane.numpy())
                timeline_chunks.append(timeline.numpy())

        samples_by_condition = np.concatenate(samples_chunks)
        flow = np.concatenate(target_chunks)
        bridge_index = np.concatenate(bridge_chunks)
        standard_lane = np.concatenate(lane_chunks)
        timeline_index = np.concatenate(timeline_chunks)
        np.save(os.path.join(self.save_dir, "bridge_index.npy"), bridge_index)
        np.save(os.path.join(self.save_dir, "standard_lane.npy"), standard_lane)
        np.save(os.path.join(self.save_dir, "timeline_index.npy"), timeline_index)
        self._save_scope(None, samples_by_condition, flow)

        for bridge in np.unique(bridge_index):
            selected_bridge = bridge_index == bridge
            self._save_scope(
                f"Bridge_{bridge}", samples_by_condition[selected_bridge], flow[selected_bridge]
            )
            for lane in np.unique(standard_lane[selected_bridge]):
                selected = selected_bridge & (standard_lane == lane)
                self._save_scope(
                    f"Bridge_{bridge}_Line_{lane + 1}", samples_by_condition[selected], flow[selected]
                )

        if logger is not None:
            logger.info(
                "[%s] sampled %d/%d spacing records from %d bridges",
                split, len(flow), dataset.original_size, len(np.unique(bridge_index)),
            )
        return samples_by_condition, flow

    def run(self):
        create_exp_dir(self.save_dir)
        create_exp_dir(self.figure_dir)
        logger = get_experiment_logger(self.log_file, name=f"V_S.{self.work_path}")
        writer = get_experiment_writer(self.tensorboard_dir)
        log_training_start(logger, "V_S", self.args, self.paths, device=self.device)
        generator = torch.Generator().manual_seed(int(self.args.get("seed", 2026)))

        start = time.time()
        train_set = self._make_dataset("Train")
        val_set = self._make_dataset("Val")
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True, drop_last=False, generator=generator
        )
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        group_weight_lookup = (
            train_set.group_weight_lookup.to(self.device)
            if self.args["dataset"].get("balance_bridge_lanes", False) else None
        )
        logger.info(
            "datasets time: %.6fs; train=%d/%d, val=%d/%d",
            time.time() - start,
            len(train_set), train_set.original_size,
            len(val_set), val_set.original_size,
        )

        losses = []
        validation_values = []
        best_log_prob = -np.inf
        validations_without_improvement = 0
        for epoch in range(1, self.epoch + 1):
            logger.info(f'-- Epoch:{epoch}/{self.epoch} --')
            self.model.train()
            loss = self.train_epoch(train_loader, group_weight_lookup)
            losses.append(loss)
            writer.add_scalar("loss", loss, epoch)
            logger.info("[epoch %d/%d] loss: %.6f", epoch, self.epoch, loss)

            if epoch % self.args["val_freq"] == 0:
                self.model.eval()
                log_prob = self.val(val_loader)
                validation_values.append(log_prob)
                writer.add_scalar("val_log_p", log_prob, epoch)
                logger.info("[epoch %d/%d] validation log_p: %.6f", epoch, self.epoch, log_prob)
                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    validations_without_improvement = 0
                    torch.save({
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "val_log_p": log_prob,
                        "normalization": self._normalization_state_dict(),
                    }, self.checkpoint_path)
                    save_checkpoint_manifest_entry(
                        self.paths, "V_S", None, os.path.basename(self.checkpoint_path),
                        epoch, "val_log_p", log_prob,
                        metadata={"normalization": "training_known_bridges"},
                    )
                else:
                    validations_without_improvement += 1
                if validations_without_improvement >= self.args.get("patience_epochs", 15):
                    break

        save_training_curves(self.save_dir, losses, validation_values)
        self._load_checkpoint(self.checkpoint_path)
        test_set = self._make_dataset("Test")
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        logger.info("[test] log_p: %.6f", self.val(test_loader, mode="test"))
        self.sample(split="Test", eval_num_samples=5, logger=logger)
        writer.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        torch.cuda.empty_cache()


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train the unified full-bridge V_S model.")
    parser.add_argument("--args-dir", default="args")
    parser.add_argument("--dataset-args", default="args/dataset_args.yaml")
    parser.add_argument("--work-path", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_cli_args()
    all_args = load_merged_args(cli_args.args_dir, cli_args.dataset_args, debug=cli_args.debug)
    work_path = cli_args.work_path or resolve_work_path(all_args, "V_S")
    device = cli_args.device or all_args.get("device", "cuda:0")
    V_S_Train(all_args["V_S"], work_path, device=device).run()
