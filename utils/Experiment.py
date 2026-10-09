import json
import logging
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


EXCEL_MAX_ROWS = 1048576
EXCEL_MAX_COLS = 16384


def save_resolved_config(experiment_root, args):
    path = Path(experiment_root) / "resolved_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(args, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return str(path)


def load_resolved_config(experiment_root):
    path = Path(experiment_root) / "resolved_config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class ExperimentPaths:
    def __init__(self, work_path, exp_root="exp"):
        self.work_path = work_path
        self.exp_root = exp_root
        self.root = os.path.join(exp_root, work_path)
        self.log_dir = os.path.join(self.root, "log")
        self.checkpoint_dir = os.path.join(self.root, "checkpoint")
        self.plot_dir = os.path.join(self.root, "plot")
        self.figure_dir = os.path.join(self.plot_dir, "figure")
        self.data_dir = os.path.join(self.plot_dir, "data")

    def create(self):
        for path in [self.root, self.log_dir, self.checkpoint_dir, self.figure_dir, self.data_dir]:
            os.makedirs(path, exist_ok=True)
        return self

    def model_log_file(self, model_name, variant=None):
        suffix = _variant_suffix(variant)
        return os.path.join(self.log_dir, f"{model_name}{suffix}.log")

    def tensorboard_dir(self, model_name, variant=None):
        suffix = _variant_suffix(variant)
        return os.path.join(self.log_dir, "tensorboard", f"{model_name}{suffix}")

    def model_figure_dir(self, model_name, variant=None):
        parts = [self.figure_dir, model_name]
        if variant is not None:
            parts.append(str(variant))
        return os.path.join(*parts)

    def model_data_dir(self, model_name, variant=None):
        parts = [self.data_dir, model_name]
        if variant is not None:
            parts.append(str(variant))
        return os.path.join(*parts)

    def checkpoint_path(self, model_name, variant=None):
        return os.path.join(self.checkpoint_dir, checkpoint_filename(model_name, variant))


def _variant_suffix(variant):
    return "" if variant is None else f"_{sanitize_name(variant)}"


def sanitize_name(value):
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def checkpoint_filename(model_name, variant=None):
    return f"{sanitize_name(model_name)}{_variant_suffix(variant)}_best.pth"


def manifest_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, "checkpoints.json")


def load_checkpoint_manifest(checkpoint_dir):
    path = manifest_path(checkpoint_dir)
    if not os.path.exists(path):
        return {"version": 1, "updated_at": None, "checkpoints": []}
    with open(path, encoding="utf-8") as file:
        manifest = json.load(file)
    manifest.setdefault("version", 1)
    manifest.setdefault("checkpoints", [])
    return manifest


def save_checkpoint_manifest_entry(paths, model_name, variant, filename, epoch, metric_name, metric_value, metadata=None):
    os.makedirs(paths.checkpoint_dir, exist_ok=True)
    manifest = load_checkpoint_manifest(paths.checkpoint_dir)
    entry = {
        "model_name": model_name,
        "variant": None if variant is None else str(variant),
        "filename": filename,
        "epoch": int(epoch),
        "metric_name": metric_name,
        "metric_value": _json_safe_number(metric_value),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        entry.update(metadata)

    manifest["checkpoints"] = [
        item for item in manifest["checkpoints"]
        if not (item.get("model_name") == model_name and item.get("variant") == entry["variant"])
    ]
    manifest["checkpoints"].append(entry)
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(manifest_path(paths.checkpoint_dir), "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return entry


def find_checkpoint(checkpoint_dir, model_name, variant=None):
    variant_text = None if variant is None else str(variant)
    manifest = load_checkpoint_manifest(checkpoint_dir)
    for entry in manifest.get("checkpoints", []):
        if entry.get("model_name") == model_name and entry.get("variant") == variant_text:
            path = os.path.join(checkpoint_dir, entry["filename"])
            if os.path.exists(path):
                return path

    fallback = os.path.join(checkpoint_dir, checkpoint_filename(model_name, variant))
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(f"Cannot find checkpoint for {model_name} {variant_text or ''} in {checkpoint_dir}")


def get_experiment_logger(log_file, name=None):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger_name = name or f"experiment.{log_file}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_experiment_writer(tensorboard_dir):
    os.makedirs(tensorboard_dir, exist_ok=True)
    return SummaryWriter(tensorboard_dir)


def log_training_start(logger, model_name, args, paths, variant=None, device="cuda:0", extra=None):
    logger.info("========== Training Start ==========")
    logger.info(f"model: {model_name}")
    logger.info(f"variant: {variant if variant is not None else 'none'}")
    logger.info(f"work_path: {paths.work_path}")
    logger.info(f"experiment_root: {paths.root}")
    logger.info(f"log_dir: {paths.log_dir}")
    logger.info(f"tensorboard_dir: {paths.tensorboard_dir(model_name, variant)}")
    logger.info(f"checkpoint_dir: {paths.checkpoint_dir}")
    logger.info(f"figure_dir: {paths.model_figure_dir(model_name, variant)}")
    logger.info(f"data_dir: {paths.model_data_dir(model_name, variant)}")
    logger.info(f"cwd: {os.getcwd()}")
    logger.info(f"python: {platform.python_version()}")
    logger.info(f"torch: {torch.__version__}")
    logger.info(f"cuda_available: {torch.cuda.is_available()}")
    logger.info(f"device: {device}")
    logger.info(f"debug: {args.get('debug', False)}")
    logger.info(f"epoch: {args.get('epoch')}")
    logger.info(f"batch_size: {args.get('batch_size')}")
    logger.info(f"lr: {args.get('lr')}")
    logger.info(f"val_freq: {args.get('val_freq')}")
    dataset_args = args.get("dataset", {})
    logger.info(f"dataset: {json.dumps(dataset_args, ensure_ascii=False, default=str)}")
    if extra:
        for key, value in extra.items():
            logger.info(f"{key}: {value}")


def save_training_curves(data_dir, ave_losses, log_p_values):
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "ave_losses.npy"), np.array(ave_losses))
    np.save(os.path.join(data_dir, "log_p_values.npy"), np.array(log_p_values))
    frame = pd.DataFrame({"epoch": np.arange(1, len(ave_losses) + 1), "ave_loss": ave_losses})
    if log_p_values:
        val_epochs = np.arange(1, len(log_p_values) + 1)
        val_frame = pd.DataFrame({"validation_index": val_epochs, "val_log_p": log_p_values})
        with pd.ExcelWriter(os.path.join(data_dir, "training_curves.xlsx")) as writer:
            frame.to_excel(writer, sheet_name="loss", index=False)
            val_frame.to_excel(writer, sheet_name="val_log_p", index=False)
    else:
        frame.to_excel(os.path.join(data_dir, "training_curves.xlsx"), sheet_name="loss", index=False)


def save_array_xlsx(path, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = {}
    matrices = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        if array.ndim <= 1:
            columns[str(name)] = array.reshape(-1)
        else:
            matrices[str(name)] = array.reshape(array.shape[0], -1)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if columns:
            max_length = max(len(values) for values in columns.values())
            frame_data = {}
            for name, values in columns.items():
                padded = np.full(max_length, np.nan)
                padded[:len(values)] = values
                frame_data[name] = padded
            _write_dataframe_chunks(writer, pd.DataFrame(frame_data), "data")
        for name, matrix in matrices.items():
            _write_matrix_chunks(writer, matrix, name[:31])


def save_histogram_xlsx(path, sheet_name, real, generated, bin_edges):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    real_density = _hist_density(real, bin_edges)
    generated_density = _hist_density(generated, bin_edges)
    frame = pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "bin_center": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "real_density": real_density,
        "generated_density": generated_density,
    })
    _write_or_replace_sheet(path, sheet_name, frame)
    return frame


def histogram_frame(real, generated, bin_edges):
    real_density = _hist_density(real, bin_edges)
    generated_density = _hist_density(generated, bin_edges)
    return pd.DataFrame({
        "bin_left": bin_edges[:-1],
        "bin_right": bin_edges[1:],
        "bin_center": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "real_density": real_density,
        "generated_density": generated_density,
    })


def _hist_density(values, bin_edges):
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    counts, _ = np.histogram(values, bins=bin_edges)
    widths = np.diff(bin_edges)
    total = values.size
    if total == 0 or np.any(widths <= 0):
        return np.zeros(len(bin_edges) - 1, dtype=float)
    return np.nan_to_num(counts / (total * widths))


def save_frames_xlsx(path, frames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            _write_dataframe_chunks(writer, frame, str(sheet_name)[:31])


def _write_dataframe_chunks(writer, frame, sheet_name):
    max_rows = int(globals().get("EXCEL_MAX_ROWS", 1048576)) - 1
    if len(frame) <= max_rows:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
        return

    chunk_count = (len(frame) + max_rows - 1) // max_rows
    for chunk_index, start in enumerate(range(0, len(frame), max_rows), start=1):
        chunk = frame.iloc[start:start + max_rows]
        if chunk_count == 1:
            chunk_sheet_name = sheet_name
        else:
            suffix = f"_{chunk_index:03d}"
            available = max(1, 31 - len(suffix))
            chunk_sheet_name = f"{sheet_name[:available]}{suffix}"
        chunk.to_excel(writer, sheet_name=chunk_sheet_name, index=False)


def _write_matrix_chunks(writer, matrix, sheet_name):
    max_rows = int(globals().get("EXCEL_MAX_ROWS", 1048576)) - 1
    max_cols = int(globals().get("EXCEL_MAX_COLS", 16384))
    matrix = np.asarray(matrix)
    row_count, col_count = matrix.shape
    row_chunk_count = (row_count + max_rows - 1) // max_rows
    col_chunk_count = (col_count + max_cols - 1) // max_cols

    for row_chunk_index, row_start in enumerate(range(0, row_count, max_rows), start=1):
        row_stop = min(row_start + max_rows, row_count)
        for col_chunk_index, col_start in enumerate(range(0, col_count, max_cols), start=1):
            col_stop = min(col_start + max_cols, col_count)
            chunk = pd.DataFrame(matrix[row_start:row_stop, col_start:col_stop])
            if row_chunk_count == 1 and col_chunk_count == 1:
                chunk_sheet_name = sheet_name
            else:
                suffix = f"_r{row_chunk_index:03d}_c{col_chunk_index:03d}"
                available = max(1, 31 - len(suffix))
                chunk_sheet_name = f"{sheet_name[:available]}{suffix}"
            chunk.to_excel(writer, sheet_name=chunk_sheet_name, index=False)


def _write_or_replace_sheet(path, sheet_name, frame):
    sheet_name = str(sheet_name)[:31]
    if os.path.exists(path):
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _json_safe_number(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
