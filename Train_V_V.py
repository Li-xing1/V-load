# -*- coding: utf-8 -*-
import argparse
import logging
import os
import os.path as osp
import time
from datetime import datetime

import matplotlib
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use('Agg')
torch.set_default_dtype(torch.float32)

from dataset.V_V_dataset import V_V_dataset, V_v_handle
from models.V_V import V_V
from utils.EvaluationPlots import (
    plot_v_v_marginal_distributions,
    plot_vehicle_flow_timeseries,
    plot_vehicle_speed_timeseries,
)
from utils.Auxiliary import *
from utils.Train import *
from utils.Experiment import (
    ExperimentPaths,
    get_experiment_logger,
    get_experiment_writer,
    log_training_start,
    save_array_xlsx,
    save_checkpoint_manifest_entry,
    save_training_curves,
)
from utils.TrainingConfig import load_merged_args, move_tensors_to_cpu, move_tensors_to_device, resolve_work_path
from utils.DistributionEvaluation import (
    continuous_metrics,
    crps_ensemble,
    quantile_errors,
    range_violation_rate,
    reset_model_evaluation,
    threshold_probability_errors,
    update_evaluation_report,
)


class V_V_Train():
    def __init__(self, args, work_path, device='cuda:0'):
        self.work_path = work_path
        self.args = args
        self.debug = self.args.get('debug', False)
        self.batch_size = args['batch_size']
        self.epoch = 2 if self.debug else args['epoch']
        self.device = primary_device(device)
        self.parallel_device = device
        self.statics = None
        self.normalizer = None
        self.normalization = None
        self.model = self.build_model(self.args['model'], mode='train', device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'],
                                    eps=self.args['eps'], weight_decay=self.args['weight_decay'],
                                    foreach=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=[int(i * self.epoch) for i in
                                                                    self.args['milestones']],
                                                        gamma=self.args['gamma'])
        self.paths = ExperimentPaths(work_path).create()
        self.save_dir = self.paths.model_data_dir('V_V')
        self.figure_dir = self.paths.model_figure_dir('V_V')
        self.checkpoint_path = self.paths.checkpoint_path('V_V')
        self.log_file = self.paths.model_log_file('V_V')
        self.tensorboard_dir = self.paths.tensorboard_dir('V_V')

    def build_model(self, args, mode, device, state_dict=None, **kwargs):
        model = V_V(args)
        if state_dict is not None:
            model.load_state_dict(normalize_state_dict(state_dict))
        exec(f'model.{mode}()')
        model.to(device)
        return build_data_parallel_model(model, self.parallel_device)

    def _inverse_transform(self, values):
        from dataset.V_V_dataset import MaskedLogStandardizer
        state = self.statics['normalization']['target']
        state = {
            key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in state.items()
        }
        return MaskedLogStandardizer.from_state_dict(state).inverse_transform(values)

    @property
    def observation_low(self):
        if self.normalizer is None:
            raise ValueError("V_V target normalization must be fitted or loaded before use.")
        return -self.normalizer.mean / self.normalizer.std

    @property
    def observation_high(self):
        if self.normalizer is None:
            raise ValueError("V_V target normalization must be fitted or loaded before use.")
        max_speed = float(self.normalization['max_speed_mps'])
        return np.asarray([
            (np.log1p(max_speed) - self.normalizer.mean[0]) / self.normalizer.std[0],
            np.inf,
        ], dtype=np.float32)

    @staticmethod
    def _sampling_time_steps(figure_args=None, default=2016):
        if not isinstance(figure_args, dict):
            return default
        plots = figure_args.get('plots', {})
        section = plots.get('vehicle_speed_timeseries', {}) if isinstance(plots, dict) else {}
        time_steps = section.get('time_steps', default) if isinstance(section, dict) else default
        if time_steps is None:
            return default
        return int(time_steps)

    @staticmethod
    def _prepare_sampling_sequence(flow, tf, time_steps):
        time_steps = min(int(time_steps), flow.shape[0], tf.shape[0])
        return flow[-time_steps:], tf[-time_steps:]

    @staticmethod
    def _format_sample_output(sample):
        sample = sample.detach().cpu()
        if sample.ndim != 5:
            raise ValueError(f"Unexpected V_V sample rank: {sample.ndim}")
        if sample.shape[-1] != 2:
            raise ValueError(f"Unexpected V_V sample channel count: {sample.shape[-1]}")
        return sample.numpy()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        parallel_model_module(self.model).load_state_dict(normalize_state_dict(checkpoint['model']))
        static = checkpoint['static']
        self.normalization = move_tensors_to_cpu(static['normalization'])
        self.statics = move_tensors_to_device(static, self.device)
        from dataset.V_V_dataset import MaskedLogStandardizer
        self.normalizer = MaskedLogStandardizer.from_state_dict(self.normalization['target'])
        self.model.eval()
        return checkpoint

    def _make_dataset(self, split):
        dataset = V_V_dataset(
            self.args['dataset'], debug=self.debug, split=split,
            normalization=self.normalization,
        )
        if split == 'Train':
            self.normalizer = dataset.normalizer
            self.normalization = dataset.normalization
        return dataset

    def train_epoch(self, train_loader, statics):
        ave = Average()
        for batch in tqdm(train_loader, unit='batch', desc='Train'):
            self.optimizer.zero_grad()
            flow, tf, mask = batch
            flow, tf, mask = move2device([flow, tf, mask], device=self.device)
            outputs = self.model(
                flow, tf, statics, mask,
                observation_low=self.observation_low,
                observation_high=self.observation_high,
            )
            valid_count = mask.sum(dim=(1, 2, 3, 4)).clamp_min(1).to(outputs.dtype)
            loss = -torch.mean(outputs / valid_count)
            ave.add(loss.item(), flow.shape[0])
            loss.backward()
            if self.args['clip_grad_norm']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'], foreach=False)
            self.optimizer.step()
        self.scheduler.step()
        return ave.average()

    def val(self, val_loader, statics, mode='val'):
        ave = Average()
        for batch in tqdm(val_loader, unit='batch', desc=mode):
            flow, tf, mask = batch
            flow, tf, mask = move2device([flow, tf, mask], device=self.device)
            with torch.no_grad():
                log_p = self.model(
                    flow, tf, statics, mask,
                    observation_low=self.observation_low,
                    observation_high=self.observation_high,
                )
                valid_count = mask.sum(dim=(1, 2, 3, 4)).clamp_min(1).to(log_p.dtype)
                log_p1 = torch.mean(log_p / valid_count)
                ave.add(log_p1.item(), flow.shape[0])
        return ave.average()

    def run(self):
        create_exp_dir(self.save_dir)
        create_exp_dir(self.figure_dir)
        logger = get_experiment_logger(self.log_file, name=f"V_V.{self.work_path}")
        writer = get_experiment_writer(self.tensorboard_dir)
        log_training_start(logger, 'V_V', self.args, self.paths, device=self.device)

        ave_losses = []
        log_p_values = []

        logger.info('--------- Dataset Info ---------')
        time_dataloader = time.time()
        generator = torch.Generator().manual_seed(int(self.args.get('seed', 2026)))
        train_set = self._make_dataset('Train')
        static = move_tensors_to_device(train_set.statics, self.device)
        self.statics = static
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  drop_last=True, generator=generator)

        val_set = self._make_dataset('Val')
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, generator=generator)
        logger.info('datasets time: {:.6f}s'.format(time.time() - time_dataloader))

        best_log_p = -torch.inf
        best_loss = torch.inf
        epochs_without_loss_update = 0
        patience_epochs = self.args.get('patience_epochs', 15)
        logger.info('---------- Training ----------')
        logger.info('num_samples: {}, num_batches: {}'.format(len(train_set), len(train_loader)))
        for epoch in range(1, self.epoch + 1):
            self.model.train()
            logger.info(f'-- Epoch:{epoch}/{self.epoch} --')

            start = time.time()
            ave_loss = self.train_epoch(train_loader, static)
            time_elapsed = time.time() - start
            ave_losses.append(ave_loss)
            logger.info(f'[epoch {epoch}/{self.epoch}] ave_loss: {ave_loss:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
            writer.add_scalar(tag='loss', scalar_value=ave_loss, global_step=epoch)

            if not np.isnan(ave_loss):
                if ave_loss < best_loss:
                    best_loss = ave_loss
                    epochs_without_loss_update = 0
                else:
                    epochs_without_loss_update += 1

                if epoch % self.args['val_freq'] == 0:
                    self.model.eval()
                    logger.info('Validating...')
                    logger.info('num_samples: {}, num_batches: {}'.format(len(val_set), len(val_loader)))

                    start = time.time()
                    log_p = self.val(val_loader, static)
                    time_elapsed = time.time() - start
                    log_p_values.append(log_p)
                    logger.info(f'[epoch {epoch}/{self.epoch}] validation log_p: {log_p:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
                    writer.add_scalar(tag='val_log_p', scalar_value=log_p, global_step=epoch)

                    if log_p > best_log_p:
                        best_log_p = log_p
                        save_dict = {
                            'model': data_parallel_state_dict(self.model),
                            'epoch': epoch,
                            'static': move_tensors_to_cpu(static),
                        }
                        torch.save(save_dict, self.checkpoint_path)
                        save_checkpoint_manifest_entry(
                            self.paths,
                            model_name='V_V',
                            variant=None,
                            filename=os.path.basename(self.checkpoint_path),
                            epoch=epoch,
                            metric_name='val_log_p',
                            metric_value=log_p,
                            metadata={'has_static': True, 'has_normalization': True},
                        )
                        logger.info(f"The best model checkpoint has been updated: {self.checkpoint_path}")

                if epochs_without_loss_update >= patience_epochs:
                    logger.info(
                        f'Early stopping at epoch {epoch}: loss has not improved for {patience_epochs} epochs. '
                        f'best_loss: {best_loss:.6f}'
                    )
                    break
        save_training_curves(self.save_dir, ave_losses, log_p_values)
        self._load_checkpoint(self.checkpoint_path)

        test_set = self._make_dataset('Test')
        test_loader = DataLoader(test_set, batch_size=min(self.batch_size,len(test_set)), shuffle=True,
                                 drop_last=False, generator=generator)
        logger.info('---------- Testing ----------')
        logger.info('num_samples: {}, num_batches: {}'.format(len(test_set), len(test_loader)))
        start = time.time()
        log_p = self.val(test_loader, self.statics, mode='test')
        time_elapsed = time.time() - start
        logger.info(f'[test] validation log_p: {log_p:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')

        logger.info('---------- Samping ----------')
        self._sample_loader(test_loader, self.save_dir, figure_dir=self.figure_dir, eval_num_samples=1)

        writer.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        torch.cuda.empty_cache()

    def _sample_loader(self, loader, save_dir, figure_dir=None, eval_num_samples=5):
        if self.statics is None:
            raise KeyError("Checkpoint must contain 'static' for V_V sampling.")
        figure_dir = figure_dir or self.figure_dir
        create_exp_dir(save_dir)
        create_exp_dir(figure_dir)
        self.model.eval()
        with torch.no_grad():
            dataset = loader.dataset
            targets, tf, mask = dataset.raw_data, dataset.raw_tf, dataset.raw_mask
            targets, tf = move2device([targets, tf], device=self.device)
            sample_repeats = []
            for _ in range(eval_num_samples):
                sample_repeats.append(
                    self._format_sample_output(
                        parallel_model_module(self.model).sample(
                            tf,
                            self.statics,
                            observation_low=self.observation_low,
                            observation_high=self.observation_high,
                        )
                    )
                )
            samples_by_condition = self._inverse_transform(np.stack(sample_repeats, axis=1))
            observations_by_condition = self._inverse_transform(
                targets.cpu().detach().numpy()[None, ...]
            )
        mask_np = mask.numpy()
        mask_by_condition = mask_np[None, ...]
        observations_by_condition = np.where(
            mask_by_condition, observations_by_condition, np.nan
        )
        samples = samples_by_condition.reshape(-1, 2)
        observations = observations_by_condition.reshape(-1, 2)
        np.save(os.path.join(save_dir, 'samples_by_condition.npy'), samples_by_condition)
        np.save(os.path.join(save_dir, 'observations_by_condition.npy'), observations_by_condition)
        np.save(os.path.join(save_dir, 'valid_mask.npy'), mask_by_condition)
        np.save(os.path.join(save_dir, 'samples.npy'), samples)
        np.save(os.path.join(save_dir, 'observations.npy'), observations)
        exported = {}
        for channel_index, channel_name in ((0, 'speed'), (1, 'minute_flow')):
            channel_samples = samples_by_condition[..., channel_index].reshape(-1)
            channel_observations = observations_by_condition[..., channel_index].reshape(-1)
            np.save(os.path.join(save_dir, f'{channel_name}_samples.npy'), channel_samples)
            np.save(os.path.join(save_dir, f'{channel_name}_observations.npy'), channel_observations)
            exported[f'{channel_name}_samples'] = channel_samples
            exported[f'{channel_name}_observations'] = channel_observations
        save_array_xlsx(os.path.join(save_dir, 'sample_values.xlsx'), exported)
        report_path = os.path.join(self.paths.data_dir, 'distribution_evaluation.xlsx')
        reset_model_evaluation(report_path, 'V_V')
        channel_specs = (
            (0, 'speed', 'm/s', float(self.args.get('max_speed_mps', 50.0)), (5.0, 20.0, 50.0)),
            (1, 'minute_flow', 'vehicles/min', None, None),
        )
        for bridge_node, lanes in self.args['known_lanes'].items():
            bridge_index = int(bridge_node)
            lane_indices = [lane - 1 for lane in lanes]
            bridge_samples = samples_by_condition[:, :, :, bridge_index:bridge_index + 1, :, :][..., lane_indices, :]
            bridge_observations = observations_by_condition[:, :, bridge_index:bridge_index + 1, :, :][..., lane_indices, :]
            bridge_mask = mask_by_condition[:, :, bridge_index:bridge_index + 1, :, :][..., lane_indices, :]
            if not bridge_mask.any():
                logging.getLogger(__name__).warning(
                    "Skipping V_V bridge %s metrics: no observed speed-flow pairs in this split.", bridge_node
                )
            bridge_data_dir = os.path.join(save_dir, f'Bridge_{bridge_node}')
            bridge_figure_dir = os.path.join(figure_dir, f'Bridge_{bridge_node}')
            create_exp_dir(bridge_data_dir)
            create_exp_dir(bridge_figure_dir)
            self._save_line_distribution_files(
                save_dir, bridge_node, lanes, bridge_samples, bridge_observations, bridge_mask
            )
            plot_vehicle_speed_timeseries(
                bridge_samples[..., 0:1],
                bridge_observations[..., 0:1],
                os.path.join(bridge_figure_dir, 'fig_speed_timeseries.png'),
                figure_args=self.args.get('figure'),
                data_path=os.path.join(bridge_data_dir, 'fig_speed_timeseries.xlsx'),
                lane_labels=lanes,
            )
            plot_vehicle_flow_timeseries(
                bridge_samples[..., 1:2],
                bridge_observations[..., 1:2],
                os.path.join(bridge_figure_dir, 'fig_minute_flow_timeseries.png'),
                figure_args=self.args.get('figure'),
                data_path=os.path.join(bridge_data_dir, 'fig_minute_flow_timeseries.xlsx'),
                lane_labels=lanes,
            )
            plot_v_v_marginal_distributions(
                bridge_samples,
                bridge_observations,
                bridge_mask,
                lanes,
                bridge_figure_dir,
                bridge_data_dir,
                figure_args=self.args.get('figure'),
            )

            pair_mask = bridge_mask[0, ..., 0]
            for channel_index, channel_name, unit, upper_bound, thresholds in channel_specs:
                bridge_real = bridge_observations[0, ..., channel_index][pair_mask]
                bridge_matrix = np.moveaxis(
                    bridge_samples[0, ..., channel_index], 0, -1
                )[pair_mask]
                paired_valid = np.isfinite(bridge_real) & np.isfinite(bridge_matrix).all(axis=1)
                if paired_valid.any():
                    bridge_real = bridge_real[paired_valid]
                    bridge_matrix = bridge_matrix[paired_valid]
                    metrics = continuous_metrics(bridge_real, bridge_matrix.reshape(-1))
                    metrics.update(quantile_errors(bridge_real, bridge_matrix.reshape(-1)))
                    if thresholds is not None:
                        metrics.update(threshold_probability_errors(
                            bridge_real, bridge_matrix.reshape(-1), thresholds=thresholds
                        ))
                    metrics['crps'] = crps_ensemble(bridge_matrix, bridge_real)
                    metrics['out_of_range_rate'] = range_violation_rate(
                        bridge_matrix, low=0.0, high=upper_bound
                    )
                    update_evaluation_report(
                        report_path, 'V_V', f'Bridge_{bridge_node}_{channel_name}', metrics,
                        units={
                            key: unit for key in metrics
                            if key in {'wasserstein_1', 'crps'}
                            or (key.startswith('p') and key[1:3].isdigit())
                        },
                    )
                else:
                    logging.getLogger(__name__).warning(
                        "Skipping V_V bridge %s %s metrics: no finite observation/sample pairs.",
                        bridge_node, channel_name,
                    )

                for line_offset, lane in enumerate(lanes):
                    line_mask = bridge_mask[0, ..., line_offset, 0]
                    line_real = bridge_observations[0, ..., line_offset, channel_index][line_mask]
                    line_matrix = np.moveaxis(
                        bridge_samples[0, :, ..., line_offset, channel_index], 0, -1
                    )[line_mask]
                    paired_valid = np.isfinite(line_real) & np.isfinite(line_matrix).all(axis=1)
                    if not paired_valid.any():
                        logging.getLogger(__name__).warning(
                            "Skipping V_V bridge %s lane %d %s metrics: no finite pairs.",
                            bridge_node, lane, channel_name,
                        )
                        continue
                    line_real = line_real[paired_valid]
                    line_matrix = line_matrix[paired_valid]
                    metrics = continuous_metrics(line_real, line_matrix.reshape(-1))
                    metrics.update(quantile_errors(line_real, line_matrix.reshape(-1)))
                    if thresholds is not None:
                        metrics.update(threshold_probability_errors(
                            line_real, line_matrix.reshape(-1), thresholds=thresholds
                        ))
                    metrics['crps'] = crps_ensemble(line_matrix, line_real)
                    metrics['out_of_range_rate'] = range_violation_rate(
                        line_matrix, low=0.0, high=upper_bound
                    )
                    update_evaluation_report(
                        report_path, 'V_V', f'Bridge_{bridge_node}_Line_{lane}_{channel_name}', metrics,
                        units={
                            key: unit for key in metrics
                            if key in {'wasserstein_1', 'crps'}
                            or (key.startswith('p') and key[1:3].isdigit())
                        },
                    )
        return samples, observations

    @staticmethod
    def _save_line_distribution_files(save_dir, bridge_node, lanes, samples_by_condition,
                                      observations_by_condition, mask_by_condition):
        channels = ((0, 'speed'), (1, 'minute_flow'))
        for line_offset, lane in enumerate(lanes):
            line_dir = os.path.join(save_dir, f"Bridge_{bridge_node}_Line_{lane}")
            os.makedirs(line_dir, exist_ok=True)
            line_mask = mask_by_condition[0, ..., line_offset, 0]
            exported = {}
            for channel_index, channel_name in channels:
                observations = np.where(
                    line_mask,
                    observations_by_condition[0, ..., line_offset, channel_index],
                    np.nan,
                ).reshape(-1)
                sample_values = samples_by_condition[0, :, ..., line_offset, channel_index]
                samples = sample_values[:, line_mask].reshape(-1)
                np.save(os.path.join(line_dir, f'{channel_name}_samples.npy'), samples)
                np.save(os.path.join(line_dir, f'{channel_name}_observations.npy'), observations)
                exported[f'{channel_name}_samples'] = samples
                exported[f'{channel_name}_observations'] = observations
            save_array_xlsx(os.path.join(line_dir, 'sample_values.xlsx'), exported)

    def sample(self, best_exp=None, eval_num_samples=5, checkpoint_path=None, save_dir=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(best_exp, 'best.pth') if best_exp is not None else self.checkpoint_path
        if save_dir is None:
            save_dir = best_exp if best_exp is not None else self.save_dir
        self._load_checkpoint(checkpoint_path)
        generator = torch.Generator().manual_seed(int(self.args.get('seed', 2026)))
        test_set = self._make_dataset('Test')
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                 drop_last=False, generator=generator)
        return self._sample_loader(test_loader, save_dir, figure_dir=self.figure_dir, eval_num_samples=eval_num_samples)


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_args(args_path='args.yaml/args.yaml.yaml', dataset_args_path='args.yaml/dataset_args.yaml'):
    with open(args_path, encoding='utf-8') as args_file:
        args = yaml.safe_load(args_file)

    try:
        with open(dataset_args_path, encoding='utf-8') as dataset_args_file:
            dataset_args = yaml.safe_load(dataset_args_file) or {}
    except FileNotFoundError:
        dataset_args = {}

    return deep_merge(args, dataset_args)


def apply_runtime_options(args, debug=False):
    data_root = args.get('data_root', 'data/jsq')
    npy_dir = args.get('npy_dir', 'npy')
    debug_npy_dir = args.get('debug_npy_dir', 'npy_debug')

    args['V_V']['debug'] = debug
    if debug:
        args['V_V']['epoch'] = 2
        args['V_V']['batch_size'] = 32
    args['V_V']['dataset']['data_root'] = data_root
    args['V_V']['dataset']['npy_dir'] = npy_dir
    args['V_V']['dataset']['debug_npy_dir'] = debug_npy_dir
    return args


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train or sample the V_V model.")
    parser.add_argument("--args-dir", default="args", help="Directory containing args.yaml and V_V.yaml")
    parser.add_argument("--dataset-args", default="args/dataset_args.yaml", help="Dataset args yaml path")
    parser.add_argument("--work-path", default=None, help="Experiment folder name; defaults to args.yaml save_name or timestamp")
    parser.add_argument("--debug", action="store_true", help="Use debug npy datasets and train for 2 epochs")
    parser.add_argument("--sample-exp", default=None, help="Legacy sample directory containing best.pth")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path; defaults to exp/<work-path>/checkpoint")
    parser.add_argument("--device", default=None, help="Torch device")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_cli_args()
    all_args = load_merged_args(cli_args.args_dir, cli_args.dataset_args, debug=cli_args.debug)
    work_path = cli_args.work_path or resolve_work_path(all_args, 'V_V')

    device = cli_args.device if cli_args.device is not None else all_args.get('device', 'cuda:0')
    train = V_V_Train(all_args['V_V'], work_path, device=device)
    if cli_args.sample_exp or cli_args.checkpoint:
        train.sample(cli_args.sample_exp, checkpoint_path=cli_args.checkpoint)
    else:
        train.run()
