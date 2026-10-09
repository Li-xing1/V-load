import math
import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from utils.Experiment import ExperimentPaths, histogram_frame, save_array_xlsx, save_frames_xlsx


MM_TO_INCH = 1 / 25.4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_ARGS_PATH = PROJECT_ROOT / "args" / "figure.yaml"


def create_bridge_figure_scopes(work_path, model_names=("V_W", "V_S"), exp_root="exp"):
    paths = ExperimentPaths(work_path, exp_root=exp_root).create()
    scope_names = ["All", *(f"Bridge_{index}" for index in range(12))]
    for model_name in model_names:
        for scope_name in scope_names:
            os.makedirs(paths.model_figure_dir(model_name, scope_name), exist_ok=True)

DEFAULT_FIGURE_ARGS = {
    "general": {
        "width_mm": 180,
        "font_size": 10,
        "title_font_size": 10,
        "label_font_size": 10,
        "tick_font_size": 10,
        "legend_font_size": 10,
        "dpi": 900,
        "savefig_dpi": 900,
        "bins": 70,
        "line_width": 2.0,
        "spine_width": 1.0,
        "grid_width": 0.8,
        "real_color": "#c6ddeb",
        "generated_color": "#df2528",
        "real_label": "flow",
        "generated_label": "sample",
        "grid_color": "#e7e7e7",
        "fonts": {
            "chinese": r"C:\Windows\Fonts\simsun.ttc",
            "english": r"C:\Windows\Fonts\times.ttf",
        },
    },
    "plots": {
        "axis_weight_marginals": {
            "subplot_aspect": 0.7,
            "show_suptitle": False,
            "unit_scale": 0.001,
            "xlabel_template": "Axle {index} Weight (T)",
            "xlim": None,
            "xlim_by_vehicle": {},
            "max_cols": 3,
            "margins": {"left": 0.1, "right": 0.98, "bottom": 0.13, "top": 0.98},
            "wspace": 0.4,
            "hspace": 0.5,
            "suptitle_y": 0.98,
        },
        "axis_weight_pearson": {
            "margins_mm": {"left": 14, "right": 13, "bottom": 16, "top": 8},
            "titles": ["flow", "sample"],
            "subplot_gap_mm": 22,
            "colorbar_width_mm": 3,
            "colorbar_gap_mm": 3,
            "spine_width": 0.8,
            "show_suptitle": False,
            "suptitle_y": 0.98,
        },
        "vehicle_total_weight_summary": {
            "subplot_aspect": 0.7,
            "show_suptitle": False,
            "unit_scale": 0.001,
            "xlabel": "Total Weight (T)",
            "title_template": "Vehicle {vehicle_type}",
            "xlim": None,
            "xlim_by_vehicle": {},
            "max_cols": 3,
            "margins": {"left": 0.1, "right": 0.98, "bottom": 0.13, "top": 0.92},
            "wspace": 0.4,
            "hspace": 0.65,
            "suptitle_y": 0.98,
        },
        "line_distribution_summary": {
            "subplot_aspect": 0.7,
            "show_suptitle": False,
            "title_template": "{line} Lane",
            "xlabels": {"V_S": "Vehicle Spacing (m)", "V_V": "Vehicle Speed (m/s)"},
            "xlim": None,
            "xlim_by_model": {"V_S": [0, 2500], "V_V": None},
            "max_cols": 3,
            "margins": {"left": 0.1, "right": 0.98, "bottom": 0.13, "top": 0.92},
            "wspace": 0.4,
            "hspace": 0.65,
            "model_overrides": {
                "V_V": {
                    "margins": {"left": 0.08, "right": 0.99, "bottom": 0.15, "top": 0.90},
                    "wspace": 0.45,
                    "hspace": 0.75,
                    "legend_font_size": 8,
                }
            },
            "suptitle_y": 0.98,
        },
        "vehicle_class_ratio_timeseries": {
            "aspect_ratio": 1 / 3,
            "margins": {"left": 0.075, "right": 0.995, "bottom": 0.18, "top": 0.85},
            "show_title": False,
            "sample_index": 0,
            "time_index": 0,
            "line_index": 0,
            "time_steps": 2016,
            "time_step_minutes": 5,
            "tick_interval_hours": 6,
            "xlabel": "Time",
            "ylabel": "Vehicle class ratio",
            "title": "Observed and generated vehicle class ratios",
            "colors": ["#2f6fbb", "#d14b3a", "#2f8f5b", "#8a5ab8", "#d9902f", "#4f9da6"],
            "prediction_label": "Sample",
            "real_label": "Real",
            "prediction_linestyle": "--",
            "real_linestyle": "-",
            "prediction_line_width": 1.4,
            "real_line_width": 1.8,
            "prediction_alpha": 0.90,
            "real_alpha": 0.95,
            "vehicle_legend_title": "Vehicle Class",
            "series_legend_title": "Series",
            "subplots_save_suffix": "_6subplots",
            "subplots_aspect_ratio": 1.05,
            "subplots_max_cols": 2,
            "subplots_margins": {"left": 0.08, "right": 0.98, "bottom": 0.08, "top": 0.90},
            "subplots_wspace": 0.24,
            "subplots_hspace": 0.42,
        },
        "vehicle_speed_timeseries": {
            "aspect_ratio": 1 / 3,
            "margins": {"left": 0.075, "right": 0.995, "bottom": 0.18, "top": 0.85},
            "show_title": False,
            "sample_index": 0,
            "condition_index": 0,
            "line_index": 0,
            "time_steps": 2016,
            "time_step_minutes": 5,
            "tick_interval_hours": 6,
            "xlabel": "Time",
            "ylabel": "Speed (m/s)",
            "title": "Vehicle speed time series comparison",
            "prediction_label": "Sample",
            "real_label": "Real",
            "prediction_linestyle": "--",
            "real_linestyle": "-",
            "prediction_line_width": 1.4,
            "real_line_width": 1.8,
            "prediction_alpha": 0.90,
            "real_alpha": 0.95,
        },
        "vehicle_flow_timeseries": {
            "aspect_ratio": 1 / 3,
            "margins": {"left": 0.075, "right": 0.995, "bottom": 0.18, "top": 0.85},
            "show_title": False,
            "sample_index": 0,
            "condition_index": 0,
            "time_steps": 2016,
            "time_step_minutes": 5,
            "tick_interval_hours": 6,
            "xlabel": "Time",
            "ylabel": "Minute flow (vehicles/min)",
            "title": "Vehicle flow time series comparison",
            "prediction_label": "Sample",
            "real_label": "Real",
            "prediction_linestyle": "--",
            "real_linestyle": "-",
            "prediction_line_width": 1.4,
            "real_line_width": 1.8,
            "prediction_alpha": 0.90,
            "real_alpha": 0.95,
        },
        "v_v_marginal_distributions": {
            "subplot_aspect": 0.7,
            "show_suptitle": False,
            "title_template": "{line} Lane",
            "max_cols": 3,
            "margins": {"left": 0.08, "right": 0.99, "bottom": 0.15, "top": 0.90},
            "wspace": 0.45,
            "hspace": 0.75,
            "legend_font_size": 8,
            "xlim_by_variable": {"speed": None, "minute_flow": None},
            "suptitle_y": 0.98,
        },
    },
}

VEHICLE_NAMES = {
    "2C": "Two-axle Passenger Car",
    "2F": "Two-axle Truck",
    "3": "Three-axle Truck",
    "4": "Four-axle Truck",
    "5": "Five-axle Truck",
    "6": "Six-axle Truck",
}

LANE_NAMES = {
    "1": "Upstream Right",
    "2": "Upstream Middle",
    "3": "Upstream Left",
    "4": "Downstream Left",
    "5": "Downstream Middle",
    "6": "Downstream Right",
}


def _lane_display_name(lane_label):
    lane_label = str(lane_label)
    return LANE_NAMES.get(lane_label, lane_label)


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_figure_args(figure_args_path=None):
    figure_args = deepcopy(DEFAULT_FIGURE_ARGS)
    figure_args_path = Path(figure_args_path or DEFAULT_FIGURE_ARGS_PATH)
    if not figure_args_path.exists():
        return figure_args

    with figure_args_path.open(encoding="utf-8") as figure_args_file:
        user_figure_args = yaml.safe_load(figure_args_file) or {}
    return deep_merge(figure_args, user_figure_args)


def add_figure_args_argument(parser):
    parser.add_argument(
        "--figure-args", "--figure-args.yaml",
        dest="figure_args",
        default=str(DEFAULT_FIGURE_ARGS_PATH),
        help="Figure args.yaml yaml path",
    )
    return parser


def _resolve_figure_args(figure_args=None):
    if figure_args is None or isinstance(figure_args, (str, os.PathLike)):
        return load_figure_args(figure_args)
    return deep_merge(deepcopy(DEFAULT_FIGURE_ARGS), figure_args)


def _general(figure_args):
    return figure_args["general"]


def _plot_section(figure_args, name):
    return figure_args["plots"].get(name, {})


def _clock_time_ticks(time_steps, time_step_minutes, tick_interval_hours):
    time_step_minutes = float(time_step_minutes)
    tick_interval_hours = float(tick_interval_hours)
    if time_step_minutes <= 0 or tick_interval_hours <= 0:
        raise ValueError("Time step and tick interval must be positive.")

    tick_interval_steps = max(1, int(round(tick_interval_hours * 60 / time_step_minutes)))
    positions = np.arange(0, int(time_steps), tick_interval_steps)
    labels = []
    for position in positions:
        total_minutes = int(round(position * time_step_minutes)) % (24 * 60)
        hours, minutes = divmod(total_minutes, 60)
        labels.append(f"{hours:02d}:{minutes:02d}")
    return positions, labels


def _figure_width(figure_args):
    return _general(figure_args).get("width_mm", 180) * MM_TO_INCH


def _aspect_ratio(section, rows=None):
    aspect_ratio = section.get("aspect_ratio", 0.75)
    if not isinstance(aspect_ratio, dict):
        return aspect_ratio

    for key in (rows, str(rows), f"rows_{rows}"):
        if key in aspect_ratio:
            return aspect_ratio[key]
    return aspect_ratio.get("default", 0.75)


def _figure_size(figure_args, section, rows=None, cols=None):
    width = _figure_width(figure_args)
    if rows is not None and cols is not None and "aspect_ratio" not in section:
        return width, width * rows / cols * section.get("subplot_aspect", 0.7)
    return width, width * _aspect_ratio(section, rows=rows)


def _font(figure_args, family="chinese", size_key="font_size"):
    general = _general(figure_args)
    font_path = general.get("fonts", {}).get("english")
    return FontProperties(fname=font_path, size=general.get(size_key, general.get("font_size", 10)))


def _apply_subplot_layout(fig, section):
    margins = section.get("margins", {})
    margin_scale = 1.0
    if "subplot_aspect" in section and "aspect_ratio" not in section:
        reference_height = fig.get_figwidth() * 2 / 3 * section["subplot_aspect"]
        margin_scale = max(1.0, reference_height / fig.get_figheight())
    fig.subplots_adjust(
        left=margins.get("left", 0.07),
        right=margins.get("right", 0.98),
        bottom=margins.get("bottom", 0.12) * margin_scale,
        top=1 - (1 - margins.get("top", 0.86)) * margin_scale,
        wspace=section.get("wspace", 0.35),
        hspace=section.get("hspace", 0.45),
    )


def _normalize_xlim(xlim):
    if xlim is None:
        return None
    if len(xlim) != 2:
        raise ValueError("xlim must contain exactly two values: [min, max].")
    if xlim[0] is None or xlim[1] is None:
        return None
    return float(xlim[0]), float(xlim[1])


def _mapped_xlim(section, name, key):
    xlim_map = section.get(name, {})
    if key is None or not isinstance(xlim_map, dict):
        return None
    for candidate in (key, str(key)):
        if candidate in xlim_map:
            return _normalize_xlim(xlim_map[candidate])
    if "default" in xlim_map:
        return _normalize_xlim(xlim_map["default"])
    return None


def _section_xlim(section, xlim=None, vehicle_type=None, model_name=None):
    if xlim is not None:
        return _normalize_xlim(xlim)

    mapped_xlim = (
        _mapped_xlim(section, "xlim_by_vehicle", vehicle_type)
        or _mapped_xlim(section, "xlim_by_model", model_name)
    )
    if mapped_xlim is not None:
        return mapped_xlim
    return _normalize_xlim(section.get("xlim"))


def setup_plot_style(figure_args=None):
    figure_args = _resolve_figure_args(figure_args)
    general = _general(figure_args)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": general.get("font_size", 10),
        "axes.titlesize": general.get("title_font_size", 10),
        "axes.labelsize": general.get("label_font_size", 10),
        "xtick.labelsize": general.get("tick_font_size", 10),
        "ytick.labelsize": general.get("tick_font_size", 10),
        "legend.fontsize": general.get("legend_font_size", 10),
        "axes.unicode_minus": False,
        "axes.titlepad": 6,
        "axes.linewidth": general.get("spine_width", 1.0),
        "figure.dpi": general.get("dpi", 900),
        "savefig.dpi": general.get("savefig_dpi", 900),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
    })


def _finite_1d(values):
    if values is None:
        return np.asarray([], dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _as_feature_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    return values.reshape(-1, values.shape[-1])


def _bin_count(bins, default=70, max_bins=1000):
    try:
        count = int(bins)
    except (TypeError, ValueError):
        count = int(default)
    return max(1, min(count, int(max_bins)))


def _hist_edges(real, generated, xlim=None, bins=70, range_percentile=99.5):
    bins = _bin_count(bins)
    if xlim is not None:
        return np.linspace(xlim[0], xlim[1], bins + 1)
    combined = np.concatenate([_finite_1d(real), _finite_1d(generated)])
    if combined.size == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    data_min = float(np.min(combined))
    data_max = float(np.percentile(combined, range_percentile))
    if data_min == data_max:
        padding = max(abs(data_min) * 0.05, 0.5)
        data_min -= padding
        data_max += padding
    return np.linspace(data_min, data_max, bins + 1)


def _hist_density(values, bin_edges):
    values = _finite_1d(values)
    if values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    counts, _ = np.histogram(values, bins=bin_edges)
    widths = np.diff(bin_edges)
    total = values.size
    if total == 0 or np.any(widths <= 0):
        return np.zeros(len(bin_edges) - 1, dtype=float)
    density = counts / (total * widths)
    return np.nan_to_num(density)


def _plot_real_hist_generated_pdf(ax, real, generated, xlabel, xlim=None, bins=None, figure_args=None):
    figure_args = _resolve_figure_args(figure_args)
    general = _general(figure_args)
    bins = bins or general.get("bins", 70)
    real = _finite_1d(real)
    generated = _finite_1d(generated)
    bin_edges = _hist_edges(
        real,
        generated,
        xlim=xlim,
        bins=bins,
        range_percentile=general.get("range_percentile", 99.5),
    )
    real_density = _hist_density(real, bin_edges)
    generated_density = _hist_density(generated, bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if real.size:
        ax.bar(
        bin_edges[:-1],
        real_density,
        width=np.diff(bin_edges),
        align="edge",
        color=general["real_color"],
        alpha=0.85,
        label=general.get("real_label", "flow"),
        )
    ax.plot(
        bin_centers,
        generated_density,
        color=general["generated_color"],
        linewidth=general.get("line_width", 2.0),
        label=general.get("generated_label", "sample"),
    )
    ax.set_xlabel(xlabel, labelpad=3, fontproperties=_font(figure_args, "english", "label_font_size"))
    ax.set_ylabel("Pdf", fontproperties=_font(figure_args, "english", "label_font_size"))
    ax.grid(True, color=general["grid_color"], linestyle="--", linewidth=general.get("grid_width", 0.8))
    ax.set_axisbelow(True)
    if xlim is not None:
        ax.set_xlim(*xlim)
        real_outside = np.mean((real < xlim[0]) | (real > xlim[1])) if real.size else 0.0
        generated_outside = np.mean(
            (generated < xlim[0]) | (generated > xlim[1])
        ) if generated.size else 0.0
        if real_outside > 0 or generated_outside > 0:
            ax.text(
                0.02,
                0.98,
                f"outside x-range: real {real_outside:.1%}, generated {generated_outside:.1%}",
                transform=ax.transAxes,
                va="top",
                fontsize=general.get("tick_font_size", 10),
            )
    for spine in ax.spines.values():
        spine.set_linewidth(general.get("spine_width", 0.8))
    return real, generated, bin_edges


def _hide_unused_axes(axes, used_count):
    for ax in axes.reshape(-1)[used_count:]:
        ax.axis("off")


def _grid_shape(count, max_cols=3):
    cols = min(max_cols, count)
    rows = math.ceil(count / cols)
    return rows, cols


def plot_axis_weight_marginals(real_axis, generated_axis, vehicle_type, save_path, figure_args=None, data_path=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    section = _plot_section(figure_args, "axis_weight_marginals")
    unit_scale = section.get("unit_scale", 0.001)
    generated_axis = _as_feature_matrix(generated_axis) * unit_scale
    has_real = _finite_1d(real_axis).size > 0
    real_axis = (
        _as_feature_matrix(real_axis) * unit_scale
        if has_real
        else np.empty((0, generated_axis.shape[-1]), dtype=float)
    )
    if has_real and real_axis.shape[-1] != generated_axis.shape[-1]:
        raise ValueError(
            f"real_axis and generated_axis feature counts differ: "
            f"{real_axis.shape[-1]} != {generated_axis.shape[-1]}"
        )
    axis_count = real_axis.shape[-1]
    rows, cols = _grid_shape(axis_count, max_cols=section.get("max_cols", 3))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(figure_args, section, rows=rows, cols=cols), squeeze=False)
    vehicle_name = VEHICLE_NAMES.get(str(vehicle_type), f"Vehicle {vehicle_type}")
    xlim = _section_xlim(section, vehicle_type=vehicle_type)
    data_frames = {}

    for axis_idx, ax in enumerate(axes.reshape(-1)[:axis_count]):
        real_values, generated_values, bin_edges = _plot_real_hist_generated_pdf(
            ax,
            real_axis[:, axis_idx],
            generated_axis[:, axis_idx],
            xlabel=section.get("xlabel_template", "Axle {index} Weight (T)").format(index=axis_idx + 1),
            xlim=xlim,
            figure_args=figure_args,
        )
        if data_path is not None:
            data_frames[f"axis_{axis_idx + 1}"] = histogram_frame(real_values, generated_values, bin_edges)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                  prop=_font(figure_args, "english", "legend_font_size"))

    _hide_unused_axes(axes, axis_count)
    if section.get("show_suptitle", False):
        fig.suptitle(
            f"{vehicle_name} Axle Weight Marginal Distributions",
            y=section.get("suptitle_y", 0.98),
            fontproperties=_font(figure_args, "english", "title_font_size"),
        )
    _apply_subplot_layout(fig, section)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    if data_path is not None:
        save_frames_xlsx(data_path, data_frames)


def _corrcoef(data):
    data = _as_feature_matrix(data)
    feature_count = data.shape[-1]
    corr = np.eye(feature_count, dtype=float)
    for row in range(feature_count):
        for col in range(row + 1, feature_count):
            pair = data[:, [row, col]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            if pair.shape[0] < 2:
                value = 0.0
            else:
                row_std = float(np.std(pair[:, 0]))
                col_std = float(np.std(pair[:, 1]))
                if row_std <= np.finfo(float).eps or col_std <= np.finfo(float).eps:
                    value = 0.0
                else:
                    value = float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])
                    if not np.isfinite(value):
                        value = 0.0
            corr[row, col] = value
            corr[col, row] = value
    return corr


def _plot_corr_matrix(ax, corr, labels, figure_args):
    general = _general(figure_args)
    section = _plot_section(figure_args, "axis_weight_pearson")
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontproperties=_font(figure_args, "english", "tick_font_size"))
    ax.set_yticklabels(labels, fontproperties=_font(figure_args, "english", "tick_font_size"))
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            ax.text(
                col,
                row,
                f"{corr[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=general.get("font_size", 10),
            )
    ax.tick_params(axis="both", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(section.get("spine_width", 0.8))
    return im


def plot_axis_weight_pearson(real_axis, generated_axis, vehicle_type, save_path, figure_args=None, data_path=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    section = _plot_section(figure_args, "axis_weight_pearson")
    generated_axis = _as_feature_matrix(generated_axis)
    has_real = _finite_1d(real_axis).size > 0
    if has_real:
        real_axis = _as_feature_matrix(real_axis)
    if has_real and real_axis.shape[-1] != generated_axis.shape[-1]:
        raise ValueError(
            f"real_axis and generated_axis feature counts differ: "
            f"{real_axis.shape[-1]} != {generated_axis.shape[-1]}"
        )
    axis_count = generated_axis.shape[-1]
    labels = [f"Axle {axis_index + 1}" for axis_index in range(axis_count)]
    generated_corr = _corrcoef(generated_axis)
    vehicle_name = VEHICLE_NAMES.get(str(vehicle_type), f"Vehicle {vehicle_type}")

    titles = section.get("titles", ["flow", "sample"])
    correlations = [(generated_corr, titles[1] if len(titles) > 1 else "sample")]
    if has_real:
        real_corr = _corrcoef(real_axis)
        correlations.insert(0, (real_corr, titles[0]))

    width_mm = _figure_width(figure_args) / MM_TO_INCH
    margins = section["margins_mm"]
    subplot_gap = section["subplot_gap_mm"]
    colorbar_width = section["colorbar_width_mm"]
    colorbar_gap = section["colorbar_gap_mm"]
    panel_count = len(correlations)
    matrix_size = (
        width_mm - margins["left"] - margins["right"]
        - subplot_gap * (panel_count - 1) - colorbar_gap - colorbar_width
    ) / panel_count
    if matrix_size <= 0 or colorbar_width <= 0 or min(subplot_gap, colorbar_gap, *margins.values()) < 0:
        raise ValueError("Margins and colorbar must leave a positive plotting area.")
    height_mm = matrix_size + margins["bottom"] + margins["top"]
    fig = plt.figure(figsize=(width_mm * MM_TO_INCH, height_mm * MM_TO_INCH))
    for index, (correlation, label) in enumerate(correlations):
        ax = fig.add_axes([
            (margins["left"] + index * (matrix_size + subplot_gap)) / width_mm,
            margins["bottom"] / height_mm,
            matrix_size / width_mm,
            matrix_size / height_mm,
        ])
        im = _plot_corr_matrix(ax, correlation, labels, figure_args)
        ax.set_title(label, pad=6, fontproperties=_font(figure_args, "english", "title_font_size"))
    colorbar_ax = fig.add_axes([
        (width_mm - margins["right"] - colorbar_width) / width_mm,
        margins["bottom"] / height_mm,
        colorbar_width / width_mm,
        matrix_size / height_mm,
    ])
    cbar = fig.colorbar(im, cax=colorbar_ax, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(length=3, pad=3)
    cbar.outline.set_linewidth(section.get("spine_width", 0.8))
    if section.get("show_suptitle", False):
        fig.suptitle(
            f"{vehicle_name} Axle Weight Pearson Correlations",
            y=section.get("suptitle_y", 0.98),
            fontproperties=_font(figure_args, "english", "title_font_size"),
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    if data_path is not None:
        arrays = {"generated_corr": generated_corr}
        if has_real:
            arrays["real_corr"] = real_corr
        save_array_xlsx(data_path, arrays)


def plot_vehicle_total_weight_summary(work_path, vehicle_types, figure_args=None, exp_root="exp", output_variant=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    section = _plot_section(figure_args, "vehicle_total_weight_summary")
    rows, cols = _grid_shape(len(vehicle_types), max_cols=section.get("max_cols", 3))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(figure_args, section, rows=rows, cols=cols), squeeze=False)
    paths = ExperimentPaths(work_path, exp_root=exp_root).create()
    create_bridge_figure_scopes(work_path, model_names=("V_W",), exp_root=exp_root)
    scope_name = "All" if output_variant is None else str(output_variant)
    output_parts = ["V_W"] if output_variant is None else ["V_W", scope_name]
    data_xlsx = os.path.join(paths.data_dir, *output_parts, "fig6_total_weight_comparison.xlsx")
    data_frames = {}

    for idx, vehicle_type in enumerate(vehicle_types):
        variant = str(vehicle_type)
        vehicle_dir = paths.model_data_dir("V_W", variant)
        unit_scale = section.get("unit_scale", 0.001)
        real_path = os.path.join(vehicle_dir, "flow_total_weight.npy")
        real = np.load(real_path) * unit_scale if os.path.isfile(real_path) else None
        generated = np.load(os.path.join(vehicle_dir, "samples_total_weight.npy")) * unit_scale
        ax = axes.reshape(-1)[idx]
        real_values, generated_values, bin_edges = _plot_real_hist_generated_pdf(
            ax,
            real,
            generated,
            xlabel=section.get("xlabel", "Total Weight (T)"),
            xlim=_section_xlim(section, vehicle_type=variant),
            figure_args=figure_args,
        )
        data_frames[variant] = histogram_frame(real_values, generated_values, bin_edges)
        vehicle_label = variant.split('_')[-1] if variant.startswith('Bridge_') else variant
        ax.set_title(
            section.get("title_template", "Vehicle {vehicle_type}").format(vehicle_type=vehicle_label),
            pad=6,
            fontproperties=_font(figure_args, "english", "title_font_size"),
        )
        ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                  prop=_font(figure_args, "english", "legend_font_size"))

    _hide_unused_axes(axes, len(vehicle_types))
    if section.get("show_suptitle", False):
        fig.suptitle(
            "Observed and Generated Vehicle Total Weight Distributions",
            y=section.get("suptitle_y", 0.98),
            fontproperties=_font(figure_args, "english", "title_font_size"),
        )
    _apply_subplot_layout(fig, section)
    save_path = os.path.join(
        paths.figure_dir, "V_W", scope_name, "total_weight_distribution.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    save_frames_xlsx(data_xlsx, data_frames)


def plot_line_distribution_summary(work_path, lines, model_name, xlabel, title, xlim=None, figure_args=None, exp_root="exp", output_variant=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    section = _plot_section(figure_args, "line_distribution_summary")
    section = deep_merge(section, section.get("model_overrides", {}).get(model_name, {}))
    xlabel = section.get("xlabels", {}).get(model_name, xlabel)
    xlim = _section_xlim(section, xlim=xlim, model_name=model_name)
    rows, cols = _grid_shape(len(lines), max_cols=section.get("max_cols", 3))
    fig, axes = plt.subplots(rows, cols, figsize=_figure_size(figure_args, section, rows=rows, cols=cols), squeeze=False)
    paths = ExperimentPaths(work_path, exp_root=exp_root).create()
    save_name = "fig6_spacing_distribution.png" if model_name == "V_S" else "fig6_speed_distribution.png"
    output_parts = [model_name] if output_variant is None else [model_name, str(output_variant)]
    data_xlsx = os.path.join(paths.data_dir, *output_parts, save_name.replace(".png", ".xlsx"))
    data_frames = {}

    for idx, line in enumerate(lines):
        scope = str(line)
        if scope == "All":
            variant = None
            axis_title = "Overall"
        elif scope.startswith("Bridge_") and "_Line_" not in scope:
            variant = scope
            axis_title = scope.replace("_", " ")
        else:
            variant = scope if scope.startswith("Bridge_") or scope.startswith("Line_") else f"Line_{scope}"
            line_label = variant.split('_')[-1] if variant.startswith('Bridge_') else variant.replace('Line_', '')
            lane_name = _lane_display_name(line_label)
            axis_title = section.get("title_template", "{line} Lane").format(line=lane_name)
        line_dir = paths.model_data_dir(model_name, variant)
        real_path = os.path.join(line_dir, "flow.npy")
        real = np.load(real_path) if os.path.isfile(real_path) else None
        generated = np.load(os.path.join(line_dir, "samples.npy"))
        ax = axes.reshape(-1)[idx]
        real_values, generated_values, bin_edges = _plot_real_hist_generated_pdf(ax, real, generated, xlabel=xlabel, xlim=xlim, figure_args=figure_args)
        data_frames[scope] = histogram_frame(real_values, generated_values, bin_edges)
        ax.set_title(
            axis_title,
            pad=6,
            fontproperties=_font(figure_args, "english", "title_font_size"),
        )
        legend_font = _font(figure_args, "english", "legend_font_size")
        legend_font.set_size(section.get("legend_font_size", legend_font.get_size()))
        legend = ax.legend(loc="upper right", frameon=True, framealpha=0.9, prop=legend_font)
        if model_name == "V_V":
            legend.set_zorder(0)

    _hide_unused_axes(axes, len(lines))
    if section.get("show_suptitle", False):
        fig.suptitle(title, y=section.get("suptitle_y", 0.98), fontproperties=_font(figure_args, "english", "title_font_size"))
    _apply_subplot_layout(fig, section)
    if model_name == "V_S":
        create_bridge_figure_scopes(work_path, model_names=("V_S",), exp_root=exp_root)
        if output_variant is None or output_variant == "All":
            scope_name = "All"
            save_name = "overall_spacing_distribution.png"
        elif str(output_variant).endswith("_Overall"):
            scope_name = str(output_variant)[:-len("_Overall")]
            save_name = "overall_spacing_distribution.png"
        else:
            scope_name = str(output_variant)
            save_name = "lane_spacing_distributions.png"
        save_path = os.path.join(paths.figure_dir, model_name, scope_name, save_name)
    else:
        save_path = os.path.join(paths.figure_dir, *output_parts, save_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    save_frames_xlsx(data_xlsx, data_frames)


def plot_vehicle_speed_timeseries(samples_by_condition, flow_by_condition, save_path, figure_args=None,
                                  data_path=None, lane_labels=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    general = _general(figure_args)
    section = _plot_section(figure_args, "vehicle_speed_timeseries")

    if hasattr(samples_by_condition, "detach"):
        samples_by_condition = samples_by_condition.detach().cpu().numpy()
    if hasattr(flow_by_condition, "detach"):
        flow_by_condition = flow_by_condition.detach().cpu().numpy()
    samples_by_condition = np.asarray(samples_by_condition)
    has_flow = flow_by_condition is not None
    if has_flow:
        flow_by_condition = np.asarray(flow_by_condition)
    if samples_by_condition.ndim == 6:
        samples_by_condition = samples_by_condition.mean(axis=3)
    if has_flow and flow_by_condition.ndim == 5:
        flow_by_condition = flow_by_condition.mean(axis=2)

    sample_index = int(section.get("sample_index", 0))
    condition_index = int(section.get("condition_index", 0))
    time_steps = section.get("time_steps", 2016)
    if time_steps is None:
        time_steps = samples_by_condition.shape[2]
    time_steps = min(int(time_steps), samples_by_condition.shape[2])
    if has_flow:
        time_steps = min(time_steps, flow_by_condition.shape[1])
    time_axis = np.arange(time_steps)
    time_ticks, time_tick_labels = _clock_time_ticks(
        time_steps,
        section.get("time_step_minutes", 5),
        section.get("tick_interval_hours", 6),
    )
    num_lines = samples_by_condition.shape[3]
    if has_flow:
        num_lines = min(num_lines, flow_by_condition.shape[2])
    lane_labels = list(range(1, num_lines + 1)) if lane_labels is None else list(lane_labels)
    colors = section.get("colors", ["#2f6fbb", "#d14b3a", "#2f8f5b", "#8a5ab8", "#d9902f", "#4f9da6", "#7a7a7a", "#bf6f30"])
    prediction_label = section.get("prediction_label", "Generated")
    real_label = section.get("real_label", "Real")
    prediction_linestyle = section.get("prediction_linestyle", "--")
    real_linestyle = section.get("real_linestyle", "-")
    prediction_line_width = section.get("prediction_line_width", general.get("line_width", 2.0))
    real_line_width = section.get("real_line_width", general.get("line_width", 2.0))

    fig, ax = plt.subplots(figsize=_figure_size(figure_args, section))
    has_real = False
    for line_index in range(num_lines):
        lane_label = lane_labels[line_index]
        color = colors[line_index % len(colors)]
        generated = samples_by_condition[condition_index, sample_index, :time_steps, line_index, 0]
        ax.plot(
            time_axis,
            generated,
            color=color,
            linestyle=prediction_linestyle,
            linewidth=prediction_line_width,
            alpha=section.get("prediction_alpha", 0.90),
            label=f"Lane {lane_label} {prediction_label}",
        )
        if has_flow:
            real = flow_by_condition[condition_index, :time_steps, line_index, 0]
        if has_flow and np.isfinite(real).any():
            has_real = True
            ax.plot(
                time_axis,
                real,
                color=color,
                linestyle=real_linestyle,
                linewidth=real_line_width,
                alpha=section.get("real_alpha", 0.95),
                label=f"Lane {lane_label} {real_label}",
            )
    ax.set_xlabel(section.get("xlabel", "Time"), fontproperties=_font(figure_args, "english", "label_font_size"))
    ax.set_ylabel(section.get("ylabel", "Speed (m/s)"), fontproperties=_font(figure_args, "english", "label_font_size"))
    if time_axis.size > 1:
        ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_tick_labels)
    if section.get("show_title", False):
        ax.set_title(section.get("title", "Vehicle speed time series comparison"), fontproperties=_font(figure_args, "english", "title_font_size"))
    ax.tick_params(direction="in", top=False, right=False, bottom=True, left=True)
    ax.grid(True, color=general["grid_color"], linestyle="--", linewidth=general.get("grid_width", 0.8))
    legend_handles = [
        Line2D([0], [0], color=colors[index % len(colors)], linewidth=1.5, label=f"Lane {lane_label}")
        for index, lane_label in enumerate(lane_labels)
    ]
    if has_real:
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=1.5, linestyle=real_linestyle, label=real_label)
        )
    legend_handles.append(
        Line2D([0], [0], color="black", linewidth=1.5, linestyle=prediction_linestyle, label=prediction_label)
    )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(legend_handles),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        prop=_font(figure_args, "english", "legend_font_size"),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(general.get("spine_width", 0.8))
    _apply_subplot_layout(fig, section)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    if data_path is not None:
        arrays = {"time": time_axis}
        for line_index in range(num_lines):
            lane_label = lane_labels[line_index]
            arrays[f"generated_lane_{lane_label}"] = samples_by_condition[condition_index, sample_index, :time_steps, line_index, 0]
            if has_flow:
                arrays[f"real_lane_{lane_label}"] = flow_by_condition[condition_index, :time_steps, line_index, 0]
        save_array_xlsx(data_path, arrays)


def plot_vehicle_flow_timeseries(samples_by_condition, observations_by_condition, save_path,
                                 figure_args=None, data_path=None, lane_labels=None):
    figure_args = _resolve_figure_args(figure_args)
    figure_args["plots"]["vehicle_speed_timeseries"] = deepcopy(
        _plot_section(figure_args, "vehicle_flow_timeseries")
    )
    return plot_vehicle_speed_timeseries(
        samples_by_condition,
        observations_by_condition,
        save_path,
        figure_args=figure_args,
        data_path=data_path,
        lane_labels=lane_labels,
    )


def plot_v_v_marginal_distributions(samples_by_condition, observations_by_condition, valid_mask,
                                    lane_labels, figure_dir, data_dir, figure_args=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    section = _plot_section(figure_args, "v_v_marginal_distributions")

    for name, values in (
        ("samples_by_condition", samples_by_condition),
        ("observations_by_condition", observations_by_condition),
        ("valid_mask", valid_mask),
    ):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if name == "samples_by_condition":
            samples_by_condition = np.asarray(values)
        elif name == "observations_by_condition":
            observations_by_condition = np.asarray(values)
        else:
            valid_mask = np.asarray(values, dtype=bool)

    if samples_by_condition.ndim != 6 or samples_by_condition.shape[-1] != 2:
        raise ValueError("samples_by_condition must have shape [condition, repeat, time, bridge, lane, 2].")
    if observations_by_condition.ndim != 5 or observations_by_condition.shape[-1] != 2:
        raise ValueError("observations_by_condition must have shape [condition, time, bridge, lane, 2].")
    if valid_mask.ndim != 5 or valid_mask.shape[-1] != 1:
        raise ValueError("valid_mask must have shape [condition, time, bridge, lane, 1].")
    if samples_by_condition.shape[0] != observations_by_condition.shape[0] \
            or samples_by_condition.shape[2:] != observations_by_condition.shape[1:]:
        raise ValueError("Sample and observation condition/time/bridge/lane/channel dimensions must match.")
    if valid_mask.shape[:-1] != observations_by_condition.shape[:-1]:
        raise ValueError("valid_mask condition/time/bridge/lane dimensions must match observations.")

    lane_labels = list(lane_labels)
    lane_count = observations_by_condition.shape[-2]
    if not lane_labels or len(lane_labels) != lane_count:
        raise ValueError(f"lane_labels must contain exactly {lane_count} labels.")

    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    variable_specs = (
        (0, "speed", "Speed (m/s)", "Speed Marginal Distributions"),
        (1, "minute_flow", "Minute flow (vehicles/min)", "Minute Flow Marginal Distributions"),
    )
    rows, cols = _grid_shape(lane_count, max_cols=section.get("max_cols", 3))

    for channel_index, variable_name, xlabel, suptitle in variable_specs:
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=_figure_size(figure_args, section, rows=rows, cols=cols),
            squeeze=False,
        )
        data_frames = {}
        xlim = _normalize_xlim(section.get("xlim_by_variable", {}).get(variable_name))
        for lane_index, lane_label in enumerate(lane_labels):
            lane_mask = valid_mask[..., lane_index, 0]
            real = observations_by_condition[..., lane_index, channel_index][lane_mask]
            generated_matrix = np.moveaxis(
                samples_by_condition[..., lane_index, channel_index], 1, -1
            )
            generated = generated_matrix[lane_mask].reshape(-1)
            ax = axes.reshape(-1)[lane_index]
            real, generated, bin_edges = _plot_real_hist_generated_pdf(
                ax,
                real,
                generated,
                xlabel=xlabel,
                xlim=xlim,
                figure_args=figure_args,
            )
            lane_name = _lane_display_name(lane_label)
            ax.set_title(
                section.get("title_template", "{line} Lane").format(line=lane_name),
                pad=6,
                fontproperties=_font(figure_args, "english", "title_font_size"),
            )
            legend_font = _font(figure_args, "english", "legend_font_size")
            legend_font.set_size(section.get("legend_font_size", legend_font.get_size()))
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, prop=legend_font)
            data_frames[f"lane_{lane_label}"] = histogram_frame(real, generated, bin_edges)

        _hide_unused_axes(axes, lane_count)
        if section.get("show_suptitle", False):
            fig.suptitle(
                suptitle,
                y=section.get("suptitle_y", 0.98),
                fontproperties=_font(figure_args, "english", "title_font_size"),
            )
        _apply_subplot_layout(fig, section)
        file_stem = f"fig_{variable_name}_marginal_distribution"
        fig.savefig(os.path.join(figure_dir, f"{file_stem}.png"))
        plt.close(fig)
        save_frames_xlsx(os.path.join(data_dir, f"{file_stem}.xlsx"), data_frames)

def _vehicle_class_lane_ratios(sample, flow_ratio, flow_total, condition_index=0, sample_index=0,
                               line_index=0, time_steps=None):
    if time_steps is None:
        time_steps = sample.shape[2]
        if flow_ratio is not None:
            time_steps = min(time_steps, flow_ratio.shape[1])
    if sample.ndim == 6:
        probabilities = sample[condition_index, sample_index, :time_steps, :, line_index]
        generated_fallback = probabilities.mean(axis=1)
    else:
        generated_fallback = sample[condition_index, sample_index, :time_steps, line_index]
    if flow_ratio is None or flow_total is None:
        return generated_fallback, np.full_like(generated_fallback, np.nan)

    total = flow_total[condition_index, :time_steps, :, line_index]
    if sample.ndim == 6:
        generated_count = (probabilities * total[..., None]).sum(axis=1)
    else:
        generated_count = generated_fallback * total.sum(axis=1, keepdims=True)
    real_probabilities = flow_ratio[condition_index, :time_steps, :, line_index]
    real_count = (real_probabilities * total[..., None]).sum(axis=1)
    denominator = total.sum(axis=1, keepdims=True)
    generated = np.divide(
        generated_count, denominator, out=np.full_like(generated_count, np.nan), where=denominator > 0
    )
    missing = denominator[:, 0] <= 0
    generated[missing] = generated_fallback[missing]
    real = np.divide(real_count, denominator, out=np.full_like(real_count, np.nan), where=denominator > 0)
    return generated, real


def _vehicle_class_all_lane_ratios(sample, flow_ratio, flow_total, condition_index=0, sample_index=0, time_steps=None):
    if time_steps is None:
        time_steps = sample.shape[2]
        if flow_ratio is not None:
            time_steps = min(time_steps, flow_ratio.shape[1])
    if sample.ndim == 6:
        probabilities = sample[condition_index, sample_index, :time_steps]
        generated_fallback = probabilities.mean(axis=(1, 2))
    else:
        probabilities = sample[condition_index, sample_index, :time_steps]
        generated_fallback = probabilities.mean(axis=1)
    if flow_ratio is None or flow_total is None:
        return generated_fallback, np.full_like(generated_fallback, np.nan)

    total = flow_total[condition_index, :time_steps]
    if sample.ndim == 6:
        generated_count = (probabilities * total[..., None]).sum(axis=(1, 2))
    else:
        total_by_line = total.sum(axis=1)
        generated_count = (probabilities * total_by_line[..., None]).sum(axis=1)
    real_count = (flow_ratio[condition_index, :time_steps] * total[..., None]).sum(axis=(1, 2))
    total = total.sum(axis=(1, 2))
    denominator = total[:, None]
    generated = np.divide(
        generated_count, denominator, out=np.full_like(generated_count, np.nan), where=denominator > 0
    )
    missing = denominator[:, 0] <= 0
    generated[missing] = generated_fallback[missing]
    real = np.divide(real_count, denominator, out=np.full_like(real_count, np.nan), where=denominator > 0)
    return generated, real


def plot_vehicle_class_ratio_timeseries(sample, flow, save_path, figure_args=None, data_path=None,
                                        lane_labels=None):
    figure_args = _resolve_figure_args(figure_args)
    setup_plot_style(figure_args)
    general = _general(figure_args)
    section = _plot_section(figure_args, "vehicle_class_ratio_timeseries")

    if hasattr(sample, "detach"):
        sample = sample.detach().cpu().numpy()
    if hasattr(flow, "detach"):
        flow = flow.detach().cpu().numpy()
    sample = np.asarray(sample)
    has_flow = flow is not None
    if has_flow:
        flow = np.asarray(flow)
        flow_ratio = np.divide(
            flow[..., :6], flow[..., 6:], out=np.zeros_like(flow[..., :6]), where=flow[..., 6:] > 0
        )
        flow_total = flow[..., 6]
    else:
        flow_ratio = None
        flow_total = None

    condition_index = int(section.get("condition_index", 0))
    sample_index = int(section.get("sample_index", 0))
    time_steps = section.get("time_steps", 2016)
    colors = section.get("colors", ["#2f6fbb", "#d14b3a", "#2f8f5b", "#8a5ab8", "#d9902f", "#4f9da6"])
    vc_list = section.get("vehicle_types", ["2C", "2F", "3", "4", "5", "6"])
    if time_steps is None:
        time_steps = sample.shape[2]
    time_steps = min(int(time_steps), sample.shape[2])
    if has_flow:
        time_steps = min(time_steps, flow_ratio.shape[1])
    time_axis = np.arange(time_steps)
    time_ticks, time_tick_labels = _clock_time_ticks(
        time_steps,
        section.get("time_step_minutes", 5),
        section.get("tick_interval_hours", 6),
    )

    prediction_label = section.get("prediction_label", "Generated")
    real_label = section.get("real_label", "Real")
    prediction_linestyle = section.get("prediction_linestyle", "--")
    real_linestyle = section.get("real_linestyle", "-")
    prediction_line_width = section.get("prediction_line_width", general.get("line_width", 2.0))
    real_line_width = section.get("real_line_width", general.get("line_width", 2.0))
    prediction_alpha = section.get("prediction_alpha", 0.90)
    real_alpha = section.get("real_alpha", 0.95)

    def style_axis(ax, title, has_real):
        ax.set_xlabel(section.get("xlabel", "Time"), fontproperties=_font(figure_args, "english", "label_font_size"))
        ax.set_ylabel(section.get("ylabel", "Vehicle class ratio"), fontproperties=_font(figure_args, "english", "label_font_size"))
        if time_axis.size > 1:
            ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_tick_labels)
        if section.get("show_title", False):
            ax.set_title(title, fontproperties=_font(figure_args, "english", "title_font_size"))
        ax.tick_params(direction="in", top=False, right=False, bottom=True, left=True)
        ax.grid(True, color=general["grid_color"], linestyle="--", linewidth=general.get("grid_width", 0.8))
        for spine in ax.spines.values():
            spine.set_linewidth(general.get("spine_width", 0.8))
        legend_handles = [
            Line2D([0], [0], color=colors[index % len(colors)], linewidth=1.5, label=vehicle_type)
            for index, vehicle_type in enumerate(vc_list)
        ]
        if has_real:
            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=1.5, linestyle=real_linestyle, label=real_label)
            )
        legend_handles.append(
            Line2D([0], [0], color="black", linewidth=1.5,
                   linestyle=prediction_linestyle, label=prediction_label)
        )
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
            prop=_font(figure_args, "english", "legend_font_size"),
        )

    def plot_lane_series(ax, line_index, show_labels=True):
        generated_lane, real_lane = _vehicle_class_lane_ratios(
            sample, flow_ratio, flow_total, condition_index, sample_index, line_index, time_steps
        )
        for class_index, vehicle_type in enumerate(vc_list):
            color = colors[class_index % len(colors)]
            ax.plot(
                time_axis,
                generated_lane[:, class_index],
                color=color,
                linestyle=prediction_linestyle,
                linewidth=prediction_line_width,
                alpha=prediction_alpha,
                label=f"{vehicle_type} {prediction_label}" if show_labels else "_nolegend_",
            )
            real_values = real_lane[:, class_index]
            if np.isfinite(real_values).any():
                ax.plot(
                    time_axis,
                    real_values,
                    color=color,
                    linestyle=real_linestyle,
                    linewidth=real_line_width,
                    alpha=real_alpha,
                    label=f"{vehicle_type} {real_label}" if show_labels else "_nolegend_",
                )
        return np.isfinite(real_lane).any()

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    num_lines = sample.shape[-2]
    if has_flow:
        num_lines = min(num_lines, flow_ratio.shape[3])
    lane_labels = list(range(1, num_lines + 1)) if lane_labels is None else list(lane_labels)

    for line_index in range(num_lines):
        lane_label = lane_labels[line_index]
        fig, ax = plt.subplots(figsize=_figure_size(figure_args, section))
        has_real = plot_lane_series(ax, line_index, show_labels=True)
        style_axis(ax, f"{_lane_display_name(lane_label)} Lane vehicle class ratio", has_real)
        _apply_subplot_layout(fig, section)
        lane_save_path = save_path_obj.with_name(f"{save_path_obj.stem}_lane_{lane_label}{save_path_obj.suffix}")
        fig.savefig(lane_save_path)
        plt.close(fig)

    all_section = deep_merge(section, {
        "aspect_ratio": section.get("all_lanes_aspect_ratio", section.get("aspect_ratio", 0.38)),
        "margins": section.get("all_lanes_margins", section.get("margins", {})),
    })
    generated_all, real_all = _vehicle_class_all_lane_ratios(sample, flow_ratio, flow_total, condition_index, sample_index, time_steps)
    fig, ax = plt.subplots(figsize=_figure_size(figure_args, all_section))
    for class_index, vehicle_type in enumerate(vc_list):
        color = colors[class_index % len(colors)]
        ax.plot(time_axis, generated_all[:, class_index], color=color, linestyle=prediction_linestyle,
                linewidth=prediction_line_width, alpha=prediction_alpha, label=f"{vehicle_type} {prediction_label}")
        real_values = real_all[:, class_index]
        if np.isfinite(real_values).any():
            ax.plot(time_axis, real_values, color=color, linestyle=real_linestyle,
                    linewidth=real_line_width, alpha=real_alpha, label=f"{vehicle_type} {real_label}")
    style_axis(ax, "All lanes vehicle class ratio", np.isfinite(real_all).any())
    _apply_subplot_layout(fig, all_section)
    all_lanes_save_path = save_path_obj.with_name(f"{save_path_obj.stem}_all_lanes{save_path_obj.suffix}")
    fig.savefig(all_lanes_save_path)
    plt.close(fig)

    if data_path is not None:
        arrays = {"time": time_axis}
        for line_index in range(num_lines):
            lane_label = lane_labels[line_index]
            generated_lane, real_lane = _vehicle_class_lane_ratios(
                sample, flow_ratio, flow_total, condition_index, sample_index, line_index, time_steps
            )
            for class_index, vehicle_type in enumerate(vc_list):
                arrays[f"lane_{lane_label}_{vehicle_type}_generated"] = generated_lane[:, class_index]
                if has_flow:
                    arrays[f"lane_{lane_label}_{vehicle_type}_real"] = real_lane[:, class_index]
        for class_index, vehicle_type in enumerate(vc_list):
            arrays[f"all_lanes_{vehicle_type}_generated"] = generated_all[:, class_index]
            if has_flow:
                arrays[f"all_lanes_{vehicle_type}_real"] = real_all[:, class_index]
        save_array_xlsx(data_path, arrays)

