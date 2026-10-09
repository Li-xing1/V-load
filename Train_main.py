# -*- coding: utf-8 -*-
import os

import numpy as np

from Train_V_C import V_C_Train
from Train_V_S import V_S_Train
from Train_V_V import V_V_Train
from Train_V_W import V_W_Train
from utils.EvaluationPlots import load_figure_args, plot_line_distribution_summary, plot_vehicle_total_weight_summary
from utils.Experiment import (
    ExperimentPaths,
    find_checkpoint,
    get_experiment_logger,
    load_resolved_config,
    save_resolved_config,
)
from utils.Train import primary_device, set_random_seed
from utils.TrainingConfig import discover_v_w_vehicle_types, load_merged_args, resolve_work_path


def _require_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find checkpoint: {path}")
    return path


def _log_main_start(logger, args, paths, args_dir, dataset_args_path, figure_args_path, debug):
    logger.info("========== Experiment Start ==========")
    logger.info("work_path: %s", paths.work_path)
    logger.info("experiment_root: %s", paths.root)
    logger.info("args_dir: %s", args_dir)
    logger.info("dataset_args_path: %s", dataset_args_path)
    logger.info("figure_args_path: %s", figure_args_path)
    logger.info("debug: %s", debug)
    logger.info("dataset_root: %s", args["dataset_args"]["dataset_root"])
    logger.info("train_bridge_nodes: %s", args["dataset_args"]["train_bridge_nodes"])
    logger.info("unknown_bridge_nodes: %s", args["dataset_args"]["unknown_bridge_nodes"])
    logger.info("device: %s", args.get("device", "cuda:0"))


def _plot_bridge_vehicle_summary(work_path, bridge_node, vehicle_types, figure_args):
    data_root = ExperimentPaths(work_path).create().model_data_dir("V_W")
    variants = [
        f"Bridge_{bridge_node}_{vehicle_type}"
        for vehicle_type in vehicle_types
        if os.path.isfile(
            os.path.join(data_root, f"Bridge_{bridge_node}_{vehicle_type}", "flow_total_weight.npy")
        )
    ]
    if not variants:
        return
    plot_vehicle_total_weight_summary(
        work_path, variants, figure_args=figure_args, output_variant=f"Bridge_{bridge_node}"
    )


def _available_line_variants(data_root, bridge_node, lines):
    variants = []
    for line in lines:
        variant = f"Bridge_{bridge_node}_Line_{line}"
        line_dir = os.path.join(data_root, variant)
        flow_path = os.path.join(line_dir, "flow.npy")
        sample_path = os.path.join(line_dir, "samples.npy")
        if (
            os.path.isfile(flow_path)
            and os.path.isfile(sample_path)
            and np.isfinite(np.load(flow_path)).any()
            and np.isfinite(np.load(sample_path)).any()
        ):
            variants.append(variant)
    return variants


def _plot_bridge_line_summary(work_path, bridge_node, lines, model_name, xlabel, title, figure_args):
    paths = ExperimentPaths(work_path).create()
    variants = _available_line_variants(paths.model_data_dir(model_name), bridge_node, lines)
    if variants:
        plot_line_distribution_summary(
            work_path,
            variants,
            model_name=model_name,
            xlabel=xlabel,
            title=title,
            figure_args=figure_args,
            output_variant=f"Bridge_{bridge_node}",
        )


def _plot_all_bridge_lines(
    work_path,
    known_lanes,
    model_name,
    xlabel,
    title_template,
    figure_args,
    include_scope_summaries=False,
):
    paths = ExperimentPaths(work_path).create()
    data_root = paths.model_data_dir(model_name)
    if include_scope_summaries:
        if all(os.path.isfile(os.path.join(data_root, name)) for name in ("flow.npy", "samples.npy")):
            plot_line_distribution_summary(
                work_path,
                ["All"],
                model_name=model_name,
                xlabel=xlabel,
                title=f"Overall {model_name} distribution",
                figure_args=figure_args,
                output_variant="All",
            )
    for bridge_node, lines in known_lanes.items():
        if include_scope_summaries:
            bridge_variant = f"Bridge_{bridge_node}"
            bridge_dir = os.path.join(data_root, bridge_variant)
            if all(os.path.isfile(os.path.join(bridge_dir, name)) for name in ("flow.npy", "samples.npy")):
                plot_line_distribution_summary(
                    work_path,
                    [bridge_variant],
                    model_name=model_name,
                    xlabel=xlabel,
                    title=title_template.format(bridge_node=bridge_node),
                    figure_args=figure_args,
                    output_variant=f"{bridge_variant}_Overall",
                )
        _plot_bridge_line_summary(
            work_path,
            bridge_node,
            lines,
            model_name=model_name,
            xlabel=xlabel,
            title=title_template.format(bridge_node=bridge_node),
            figure_args=figure_args,
        )


def Train_main():
    args_dir = "args"
    dataset_args_path = "args/dataset_args.yaml"
    figure_args_path = "args/figure.yaml"
    figure_args = load_figure_args(figure_args_path)
    args = load_merged_args(args_dir, dataset_args_path, figure_args=figure_args)
    set_random_seed(args.get("seed", 2026))

    work_path = resolve_work_path(args, "Train_main")
    paths = ExperimentPaths(work_path).create()
    save_resolved_config(paths.root, args)
    logger = get_experiment_logger(os.path.join(paths.log_dir, "Train_main.log"), name=f"Train_main.{work_path}")
    _log_main_start(
        logger, args, paths, args_dir, dataset_args_path, figure_args_path, args.get("debug", False)
    )
    run_device = args.get("device", "cuda:0")
    single_device = primary_device(run_device)

    if args["V_C_train"]:
        logger.info("Starting V_C training")
        V_C_Train(args["V_C"], work_path, device=run_device).run()

    if args["V_W_train"]:
        vehicle_types = discover_v_w_vehicle_types(args["V_W"])
        logger.info("Starting full-bridge V_W training: %s", vehicle_types)
        for vehicle_type in vehicle_types:
            V_W_Train(args["V_W"], vehicle_type, work_path, device=single_device).run()
        available_vehicle_types = [
            vehicle_type
            for vehicle_type in vehicle_types
            if os.path.isfile(
                os.path.join(
                    paths.model_data_dir("V_W", vehicle_type), "flow_total_weight.npy"
                )
            )
        ]
        if available_vehicle_types:
            plot_vehicle_total_weight_summary(work_path, available_vehicle_types, figure_args=figure_args)
        for bridge_node in args["bridge_nodes"]:
            _plot_bridge_vehicle_summary(work_path, bridge_node, vehicle_types, figure_args)

    if args["V_S_train"]:
        logger.info("Starting unified full-bridge V_S training")
        V_S_Train(args["V_S"], work_path, device=single_device).run()
        _plot_all_bridge_lines(
            work_path,
            args["known_lanes"],
            model_name="V_S",
            xlabel="Spacing (m)",
            title_template="Bridge {bridge_node}: Observed and Generated Vehicle Spacing Distributions",
            figure_args=figure_args,
            include_scope_summaries=True,
        )

    if args["V_V_train"]:
        logger.info("Starting V_V training")
        V_V_Train(args["V_V"], work_path, device=run_device).run()
        _plot_all_bridge_lines(
            work_path,
            args["known_lanes"],
            model_name="V_V",
            xlabel="Speed (m/s)",
            title_template="Bridge {bridge_node}: Observed and Generated Vehicle Speed Distributions",
            figure_args=figure_args,
        )


def Val_main(
    work_path,
    args_dir="args",
    dataset_args_path="args/dataset_args.yaml",
    figure_args_path="args/figure.yaml",
    debug=False,
    device=None,
    eval_num_samples=5,
):
    figure_args = load_figure_args(figure_args_path)
    args = load_merged_args(args_dir, dataset_args_path, figure_args=figure_args, debug=debug)
    paths = ExperimentPaths(work_path).create()
    saved_args = load_resolved_config(paths.root)
    if saved_args is not None:
        args = saved_args
    set_random_seed(args.get("seed", 2026))
    device = primary_device(device or args.get("device", "cuda:0"))

    if args["V_C_train"]:
        v_c_trainer = V_C_Train(args["V_C"], work_path, device=device)
        v_c_trainer.sample(
            checkpoint_path=_require_checkpoint(find_checkpoint(paths.checkpoint_dir, "V_C")),
            split="Test",
        )

    if args["V_W_train"]:
        vehicle_types = discover_v_w_vehicle_types(args["V_W"])
        for vehicle_type in vehicle_types:
            trainer = V_W_Train(args["V_W"], vehicle_type, work_path, device=device)
            trainer.sample(
                checkpoint_path=_require_checkpoint(
                    find_checkpoint(paths.checkpoint_dir, "V_W", vehicle_type)
                ),
                split="Test",
                eval_num_samples=eval_num_samples,
            )
        plot_vehicle_total_weight_summary(work_path, vehicle_types, figure_args=figure_args)
        for bridge_node in args["bridge_nodes"]:
            _plot_bridge_vehicle_summary(work_path, bridge_node, vehicle_types, figure_args)

    if args["V_S_train"]:
        spacing_trainer = V_S_Train(args["V_S"], work_path, device=device)
        spacing_trainer.sample(
            checkpoint_path=_require_checkpoint(find_checkpoint(paths.checkpoint_dir, "V_S")),
            split="Test",
            eval_num_samples=eval_num_samples,
        )
        _plot_all_bridge_lines(
            work_path,
            args["known_lanes"],
            model_name="V_S",
            xlabel="Spacing (m)",
            title_template="Bridge {bridge_node}: Observed and Generated Vehicle Spacing Distributions",
            figure_args=figure_args,
            include_scope_summaries=True,
        )

    if args["V_V_train"]:
        speed_trainer = V_V_Train(args["V_V"], work_path, device=device)
        speed_trainer.sample(
            eval_num_samples=eval_num_samples,
            checkpoint_path=_require_checkpoint(find_checkpoint(paths.checkpoint_dir, "V_V")),
        )
        _plot_all_bridge_lines(
            work_path,
            args["known_lanes"],
            model_name="V_V",
            xlabel="Speed (m/s)",
            title_template="Bridge {bridge_node}: Observed and Generated Vehicle Speed Distributions",
            figure_args=figure_args,
        )


if __name__ == "__main__":
    Train_main()
