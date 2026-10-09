# -*- coding: utf-8 -*-
# @Time : 2026/3/9 19:34
# @Author : Xing Li
# @Email : 2434147605@qq.com
# @File : Train_V_C.py
# @Project : V_load
import datetime
import logging
import os
import os.path as osp
import time
import argparse



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import yaml
from torch import nn
from tqdm import tqdm
from datetime import datetime
torch.set_default_dtype(torch.float32)
from utils.Auxiliary import *
from models.V_C import V_C
from dataset.V_C_dataset import V_C_dataset, V_c_handle
from utils.Train import *
from torch.utils.data import DataLoader
from utils.EvaluationPlots import plot_vehicle_class_ratio_timeseries
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
from utils.DistributionEvaluation import categorical_metrics, reset_model_evaluation, update_evaluation_report


class V_C_Train():

    def __init__(self, args, work_path, device='cuda:0'):
        self.work_path = work_path
        self.args = args
        self.figure_args = self.args.get('figure')
        self.debug = self.args.get('debug', False)
        self.batch_size = args['batch_size']
        self.epoch = 2 if self.debug else args['epoch']
        self.device = primary_device(device)
        self.parallel_device = device
        self.statics = None
        self.class_frequencies = None
        self.normalization = None
        self.balance_classes = bool(self.args['dataset'].get('balance_classes', False))
        self.model = self.build_model(self.args['model'], mode='train', device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'],
                                    eps=self.args['eps'], weight_decay=self.args['weight_decay'],
                                    foreach=False)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=[int(i * self.epoch) for i in
                                                                    self.args['milestones']],
                                                        gamma=self.args['gamma'])

        self.paths = ExperimentPaths(work_path).create()
        self.save_dir = self.paths.model_data_dir('V_C')
        self.figure_dir = self.paths.model_figure_dir('V_C')
        self.checkpoint_path = self.paths.checkpoint_path('V_C')
        self.log_file = self.paths.model_log_file('V_C')
        self.tensorboard_dir = self.paths.tensorboard_dir('V_C')

    @staticmethod
    def plot_sample_evaluation(sample, flow, save_path, figure_args=None, data_path=None, lane_labels=None):
        plot_vehicle_class_ratio_timeseries(
            sample, flow, save_path, figure_args=figure_args, data_path=data_path, lane_labels=lane_labels
        )

    @staticmethod
    def _sampling_time_steps(figure_args=None, default=2016):
        if not isinstance(figure_args, dict):
            return default
        plots = figure_args.get('plots', {})
        section = plots.get('vehicle_class_ratio_timeseries', {}) if isinstance(plots, dict) else {}
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
            raise ValueError(f"Unexpected V_C sample rank: {sample.ndim}")
        if sample.shape[-1] != 6:
            raise ValueError(f"Unexpected V_C sample class count: {sample.shape[-1]}")
        return sample.unsqueeze(0)

    # build_model
    def build_model(self, args, mode, device, state_dict=None, **kwargs):
        '''

        :param args:
        :param mode:
        :param device:
        :param state_dict: 历史参数
        :param kwargs:
        :return:
        '''

        model = V_C(args)
        if state_dict is not None:
            model.load_state_dict(normalize_state_dict(state_dict))
        exec(f'model.{mode}()')  # net.train /net.eval
        model.to(device)
        return build_data_parallel_model(model, self.parallel_device, static_arg_index=3)

    def train_epoch(self, train_loader, statics):
        ave = Average()
        for batch in tqdm(train_loader, unit='batch', desc='Train'):
            self.optimizer.zero_grad()
            flow, tf, v_c_v, mask = move2device(batch, device=self.device)
            outputs = self.model(flow, tf, v_c_v, statics, mask)
            positions = mask.sum(dim=(1, 2, 3, 4)).clamp_min(1).to(outputs.dtype)
            loss = -torch.mean(outputs / positions)
            ave.add(loss.item(), flow.shape[0])
            loss.backward()
            if self.args['clip_grad_norm']:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'], foreach=False)
            self.optimizer.step()
        self.scheduler.step()
        ave_loss = ave.average()
        return ave_loss

    def val(self, val_loader, statics, mode='val'):
        ave = Average()
        for batch in tqdm(val_loader, unit='batch', desc=mode):
            flow, tf, v_c_v, mask = move2device(batch, device=self.device)
            with torch.no_grad():
                log_p = self.model(flow, tf, v_c_v, statics, mask)
                positions = mask.sum(dim=(1, 2, 3, 4)).clamp_min(1).to(log_p.dtype)
                log_p1 = torch.mean(log_p / positions)
                # if log_p1 < -10000:
                #     indices = torch.nonzero(log_p < -10000,
                #                             as_tuple=False)  # Get the indices of log_p where condition is met
                #
                #     # Print the flow associated with each of these indices
                #     for idx in indices:
                #         print("Flow at index:", idx)
                #         print(flow[idx])
                #     ave.add(log_p1.item(), flow.shape[0])
                #     continue
                ave.add(log_p1.item(), flow.shape[0])

        ave_loss = ave.average()
        return ave_loss

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model']
        parallel_model_module(self.model).load_state_dict(normalize_state_dict(state_dict))
        static = checkpoint['static']
        self.normalization = move_tensors_to_cpu(static['normalization'])
        self.statics = move_tensors_to_device(static, self.device)
        self.class_frequencies = np.asarray(self.normalization['vehicle_class_frequencies'])
        self.balance_classes = bool(self.normalization['balance_classes'])
        self.model.eval()
        return checkpoint

    def sample(self, val_set=None, checkpoint_path=None, split='Test', save_dir=None, figure_dir=None, logger=None):
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        statics = self.statics
        if statics is None:
            raise KeyError("Checkpoint must contain 'static' for V_C sampling.")

        # ave = Average()
        save_dir = save_dir or self.save_dir
        figure_dir = figure_dir or self.figure_dir
        create_exp_dir(save_dir)
        create_exp_dir(figure_dir)
        val_set = val_set or V_C_dataset(
            self.args['dataset'], debug=self.debug, split=split, normalization=self.normalization
        )
        flow, tf, v_c_v = val_set.raw_data, val_set.raw_tf, val_set.raw_v
        self.model.eval()
        flow, tf, v_c_v = move2device([flow, tf, v_c_v], device=self.device)
        with torch.no_grad():
            sample = parallel_model_module(self.model).sample(tf, v_c_v, statics)
        sample = self._format_sample_output(sample)
        flow = flow.detach().cpu().unsqueeze(0)
        output = {'sample': sample, 'flow': flow}
        torch.save(output, os.path.join(save_dir, 'Output.pt'))
        save_array_xlsx(
            os.path.join(save_dir, 'sample_values.xlsx'),
            {'sample': sample.numpy(), 'flow': flow.numpy()},
        )
        report_path = os.path.join(self.paths.data_dir, 'distribution_evaluation.xlsx')
        reset_model_evaluation(report_path, 'V_C')
        for bridge_node, lanes in self.args['known_lanes'].items():
            bridge_index = int(bridge_node)
            lane_indices = [lane - 1 for lane in lanes]
            bridge_sample = sample[..., bridge_index:bridge_index + 1, :, :][..., lane_indices, :]
            bridge_flow = flow[..., bridge_index:bridge_index + 1, :, :][..., lane_indices, :]
            bridge_total = bridge_flow[..., -1]
            has_observed_traffic = torch.any(torch.isfinite(bridge_total) & (bridge_total > 0)).item()
            bridge_figure_dir = os.path.join(figure_dir, f'Bridge_{bridge_node}')
            bridge_data_dir = os.path.join(save_dir, f'Bridge_{bridge_node}')
            create_exp_dir(bridge_figure_dir)
            create_exp_dir(bridge_data_dir)
            self.plot_sample_evaluation(
                bridge_sample,
                bridge_flow,
                os.path.join(bridge_figure_dir, 'fig_vehicle_class_ratio_timeseries.png'),
                figure_args=self.figure_args,
                data_path=os.path.join(bridge_data_dir, 'fig_vehicle_class_ratio_timeseries.xlsx'),
                lane_labels=lanes,
            )

            bridge_flow_np = bridge_flow.numpy()
            bridge_sample_np = bridge_sample.numpy()
            if has_observed_traffic:
                real_counts = bridge_flow_np[..., :6].sum(axis=(0, 1, 2, 3))
                generated_counts = (
                    bridge_sample_np * bridge_flow_np[..., -1:][:, None, ...]
                ).sum(axis=(0, 1, 2, 3, 4))
                update_evaluation_report(
                    report_path, 'V_C', f'Bridge_{bridge_node}_overall',
                    categorical_metrics(real_counts, generated_counts),
                )
            else:
                logging.getLogger(__name__).warning(
                    "Skipping V_C bridge %s metrics: no positive observed traffic in this split.",
                    bridge_node,
                )

            for line_offset, lane in enumerate(lanes):
                line_flow = bridge_flow_np[..., line_offset:line_offset + 1, :]
                line_sample = bridge_sample_np[..., line_offset:line_offset + 1, :]
                line_total = line_flow[..., -1]
                if not np.any(np.isfinite(line_total) & (line_total > 0)):
                    logging.getLogger(__name__).warning(
                        "Skipping V_C bridge %s lane %d metrics: no positive observed traffic in this split.",
                        bridge_node, lane,
                    )
                    continue
                real_counts = line_flow[..., :6].sum(axis=(0, 1, 2, 3))
                generated_counts = (
                    line_sample * line_flow[..., -1:][:, None, ...]
                ).sum(axis=(0, 1, 2, 3, 4))
                update_evaluation_report(
                    report_path, 'V_C', f'Bridge_{bridge_node}_Line_{lane}',
                    categorical_metrics(real_counts, generated_counts),
                )
        if logger is not None:
            logger.info(f"[{split}] V_C sampling output saved to {os.path.join(save_dir, 'Output.pt')}")

        return sample, flow

    def run(self):
        create_exp_dir(self.save_dir)
        create_exp_dir(self.figure_dir)
        logger = get_experiment_logger(self.log_file, name=f"V_C.{self.work_path}")
        writer = get_experiment_writer(self.tensorboard_dir)
        log_training_start(logger, 'V_C', self.args, self.paths, device=self.device)

        ave_losses = []
        log_p_values = []

        logger.info('--------- Dataset Info ---------')
        time_dataloader = time.time()
        generator = torch.Generator().manual_seed(int(self.args.get('seed', 2026)))
        train_set = V_C_dataset(self.args['dataset'], debug=self.debug, split='Train')
        self.class_frequencies = train_set.class_frequencies
        self.normalization = train_set.normalization
        static = move_tensors_to_device(train_set.statics, self.device)
        self.statics = static

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                  drop_last=True, generator=generator)

        val_set = V_C_dataset(
            self.args['dataset'], debug=self.debug, split='Val',
            normalization=self.normalization,
        )
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                drop_last=False, generator=generator)

        logger.info('datasets time: {:.6f}s'.format(time.time() - time_dataloader))

        best_log_p = -torch.inf
        logger.info('---------- Training ----------')
        logger.info('num_samples: {}, num_batches: {}'.format(len(train_set), len(train_loader)))
        for epoch in range(1, self.epoch + 1):
            self.model.train()
            logger.info(f'-- Epoch:{epoch}/{self.epoch} --')

            start = time.time()
            ave_loss = self.train_epoch(train_loader, static)
            time_elapsed = time.time() - start
            ave_losses.append(ave_loss)
            logger.info(
                f'[epoch {epoch}/{self.epoch}] ave_loss: {ave_loss:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
            writer.add_scalar(tag='loss', scalar_value=ave_loss, global_step=epoch)

            if not np.isnan(ave_loss):
                if (epoch) % self.args['val_freq'] == 0:
                    self.model.eval()
                    logger.info('Validating...')
                    logger.info('num_samples: {}, num_batches: {}'.format(len(val_set), len(val_loader)))

                    start = time.time()
                    log_p = self.val(val_loader, static)
                    time_elapsed = time.time() - start
                    log_p_values.append(log_p)
                    logger.info(
                        f'[epoch {epoch}/{self.epoch}] validation log_p: {log_p:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')
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
                            model_name='V_C',
                            variant=None,
                            filename=os.path.basename(self.checkpoint_path),
                            epoch=epoch,
                            metric_name='val_log_p',
                            metric_value=log_p,
                            metadata={'has_static': True, 'has_normalization': True},
                        )
                        logger.info(f"The best model checkpoint has been updated: {self.checkpoint_path}")
        save_training_curves(self.save_dir, ave_losses, log_p_values)

        # test
        test_set = V_C_dataset(
            self.args['dataset'], debug=self.debug, split='Test',
            normalization=self.normalization,
        )

        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True,
                                 drop_last=True, generator=generator)
        self._load_checkpoint(self.checkpoint_path)
        logger.info('---------- Testing ----------')
        logger.info('num_samples: {}, num_batches: {}'.format(len(test_set), len(test_loader)))
        start = time.time()
        log_p = self.val(test_loader, self.statics, mode='test')
        time_elapsed = time.time() - start
        logger.info(
            f'[test] validation log_p: {log_p:.6f}, time_elapsed: {time_elapsed:.6f}(sec)')

        logger.info('---------- Sampling ----------')
        logger.info('num_samples: {}'.format(len(test_set)))
        start = time.time()
        log_p = self.sample(test_set)
        time_elapsed = time.time() - start
        logger.info(
            f'[test] time_elapsed: {time_elapsed:.6f}(sec)')
        writer.close()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        torch.cuda.empty_cache()

def parse_cli_args():
    parser = argparse.ArgumentParser(description="Train or sample the V_C model.")
    parser.add_argument("--args-dir", default="args", help="Directory containing args.yaml and V_C.yaml")
    parser.add_argument("--dataset-args", default="args/dataset_args.yaml", help="Dataset args yaml path")
    parser.add_argument("--work-path", default=None, help="Experiment folder name; defaults to args.yaml save_name or timestamp")
    parser.add_argument("--debug", action="store_true", help="Use debug npy datasets and train for 2 epochs")
    parser.add_argument("--sample-exp", default=None, help="Legacy sample directory containing best.pth")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file path; defaults to exp/<work-path>/checkpoint")
    parser.add_argument("--device", default=None, help="Torch device")
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_cli_args()
    args = load_merged_args(cli_args.args_dir, cli_args.dataset_args, debug=cli_args.debug)
    work_path = cli_args.work_path or resolve_work_path(args, 'V_C')
    device = cli_args.device if cli_args.device is not None else args.get('device', 'cuda:0')
    train = V_C_Train(args['V_C'], work_path, device=device)
    if cli_args.sample_exp or cli_args.checkpoint:
        checkpoint_path = cli_args.checkpoint or os.path.join(cli_args.sample_exp, 'best.pth')
        train.sample(checkpoint_path=checkpoint_path, save_dir=cli_args.sample_exp)
    else:
        train.run()
