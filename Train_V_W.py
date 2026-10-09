# -*- coding: utf-8 -*-
import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.V_W_dataset import AxisWeightStandardizer, ContextFeatureStandardizer, V_W_dataset, axis_count
from models.V_W import V_W
from utils.Auxiliary import create_exp_dir, move2device
from utils.DistributionEvaluation import continuous_metrics, correlation_mae, energy_score, update_evaluation_report
from utils.EvaluationPlots import (
    add_figure_args_argument,
    create_bridge_figure_scopes,
    load_figure_args,
    plot_axis_weight_marginals,
    plot_axis_weight_pearson,
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
from utils.TrainingConfig import discover_v_w_vehicle_types, load_merged_args, resolve_work_path

torch.set_default_dtype(torch.float32)


class V_W_Train:
    def __init__(self, args, vehicle_class, work_path, device="cuda:0"):
        self.vehicle_class = vehicle_class
        self.args = copy.deepcopy(args)
        self.args["model"]["num_features"] = axis_count(vehicle_class)
        self.figure_args = self.args.get("figure")
        self.debug = self.args.get("debug", False)
        self.batch_size = self.args["batch_size"]
        self.epoch = 2 if self.debug else self.args["epoch"]
        self.device = primary_device(device)
        self.target_normalizer = None
        self.context_normalizer = None
        self.model = V_W(self.args["model"]).to(self.device)
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
        self.variant = vehicle_class
        self.save_dir = self.paths.model_data_dir("V_W", self.variant)
        self.figure_dir = self.paths.model_figure_dir("V_W", "All")
        self.checkpoint_path = self.paths.checkpoint_path("V_W", self.variant)
        self.log_file = self.paths.model_log_file("V_W", self.variant)
        self.tensorboard_dir = self.paths.tensorboard_dir("V_W", self.variant)

    def _make_dataset(self, split):
        dataset = V_W_dataset(
            self.vehicle_class,
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

    def _save_scope(self, variant, samples_axis, flow_axis, figure_dir):
        save_dir = self.paths.model_data_dir("V_W", variant)
        create_exp_dir(save_dir)
        create_exp_dir(figure_dir)
        samples_total = samples_axis.sum(axis=-1)
        flow_total = flow_axis.sum(axis=-1)
        np.save(os.path.join(save_dir, "samples_axis.npy"), samples_axis)
        np.save(os.path.join(save_dir, "flow_axis.npy"), flow_axis)
        np.save(os.path.join(save_dir, "samples_total_weight_by_condition.npy"), samples_total)
        np.save(os.path.join(save_dir, "samples_total_weight.npy"), samples_total.reshape(-1))
        np.save(os.path.join(save_dir, "flow_total_weight.npy"), flow_total)
        save_array_xlsx(
            os.path.join(save_dir, "total_weight_values.xlsx"),
            {"samples_total_weight": samples_total.reshape(-1), "flow_total_weight": flow_total},
        )
        plot_axis_weight_marginals(
            flow_axis, samples_axis, self.vehicle_class,
            os.path.join(figure_dir, f"{self.vehicle_class}_axis_weight_marginals.png"),
            figure_args=self.figure_args,
            data_path=os.path.join(save_dir, "axis_weight_marginals.xlsx"),
        )
        plot_axis_weight_pearson(
            flow_axis, samples_axis, self.vehicle_class,
            os.path.join(figure_dir, f"{self.vehicle_class}_axis_weight_pearson.png"),
            figure_args=self.figure_args,
            data_path=os.path.join(save_dir, "axis_weight_pearson.xlsx"),
        )

        metrics = {}
        for axis_index in range(flow_axis.shape[-1]):
            axis_metrics = continuous_metrics(flow_axis[:, axis_index], samples_axis[..., axis_index].reshape(-1))
            metrics.update({f"axis_{axis_index + 1}_{key}": value for key, value in axis_metrics.items()})
        total_metrics = continuous_metrics(flow_total, samples_total.reshape(-1))
        metrics.update({f"total_weight_{key}": value for key, value in total_metrics.items()})
        metrics["energy_score"] = energy_score(samples_axis, flow_axis)
        metrics["correlation_matrix_mae"] = correlation_mae(
            flow_axis, samples_axis.reshape(-1, samples_axis.shape[-1])
        )
        update_evaluation_report(
            os.path.join(self.paths.data_dir, "distribution_evaluation.xlsx"),
            "V_W", variant, metrics,
            units={key: "kg" for key in metrics if "wasserstein" in key or key == "energy_score"},
        )

    def sample(self, checkpoint_path=None, split="Test", eval_num_samples=5, logger=None):
        create_bridge_figure_scopes(self.work_path, model_names=("V_W",))
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

        samples_axis = np.concatenate(samples_chunks)
        flow_axis = np.concatenate(target_chunks)
        bridge_index = np.concatenate(bridge_chunks)
        standard_lane = np.concatenate(lane_chunks)
        timeline_index = np.concatenate(timeline_chunks)
        np.save(os.path.join(self.save_dir, "bridge_index.npy"), bridge_index)
        np.save(os.path.join(self.save_dir, "standard_lane.npy"), standard_lane)
        np.save(os.path.join(self.save_dir, "timeline_index.npy"), timeline_index)
        self._save_scope(self.variant, samples_axis, flow_axis, self.figure_dir)

        for bridge in np.unique(bridge_index):
            selected_bridge = bridge_index == bridge
            bridge_variant = f"Bridge_{bridge}_{self.vehicle_class}"
            self._save_scope(
                bridge_variant,
                samples_axis[selected_bridge],
                flow_axis[selected_bridge],
                self.paths.model_figure_dir("V_W", f"Bridge_{bridge}"),
            )

        if logger is not None:
            logger.info(
                "[%s] sampled %d/%d records from %d bridges for vehicle class %s",
                split, len(flow_axis), dataset.original_size,
                len(np.unique(bridge_index)), self.vehicle_class,
            )
        return samples_axis, flow_axis

    def run(self):
        create_exp_dir(self.save_dir)
        create_exp_dir(self.figure_dir)
        logger = get_experiment_logger(self.log_file, name=f"V_W.{self.work_path}.{self.variant}")
        writer = get_experiment_writer(self.tensorboard_dir)
        log_training_start(logger, "V_W", self.args, self.paths, variant=self.variant, device=self.device)
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
                        self.paths, "V_W", self.variant, os.path.basename(self.checkpoint_path),
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
    parser = argparse.ArgumentParser(description="Train full-bridge V_W models by vehicle class.")
    parser.add_argument("--args-dir", default="args")
    parser.add_argument("--dataset-args", default="args/dataset_args.yaml")
    add_figure_args_argument(parser)
    parser.add_argument("--work-path", default=None)
    parser.add_argument("--vehicle-class", nargs="*", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_cli_args()
    figure_args = load_figure_args(cli_args.figure_args)
    all_args = load_merged_args(
        cli_args.args_dir, cli_args.dataset_args, figure_args=figure_args, debug=cli_args.debug
    )
    work_path = cli_args.work_path or resolve_work_path(all_args, "V_W")
    device = cli_args.device or all_args.get("device", "cuda:0")
    vehicle_classes = cli_args.vehicle_class or discover_v_w_vehicle_types(all_args["V_W"])
    for vehicle_class in vehicle_classes:
        V_W_Train(all_args["V_W"], vehicle_class, work_path, device=device).run()
