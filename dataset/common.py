import os.path as osp
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigh


class FeatureStandardizer:
    def __init__(self, indices, mean, std):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, values, embedding_schema, fit_rows=None):
        values = np.asarray(values, dtype=np.float32)
        indices = np.flatnonzero(np.asarray(embedding_schema) == 1)
        selected = values if fit_rows is None else values[np.asarray(fit_rows, dtype=bool)]
        selected = selected.reshape(-1, values.shape[-1])[:, indices]
        mean = selected.mean(axis=0)
        std = selected.std(axis=0)
        std[std == 0] = 1.0
        return cls(indices, mean, std)

    def transform(self, values):
        result = np.asarray(values, dtype=np.float32).copy()
        result[..., self.indices] = (result[..., self.indices] - self.mean) / self.std
        return result

    def state_dict(self):
        return {
            "method": "zscore",
            "indices": self.indices.copy(),
            "mean": self.mean.copy(),
            "std": self.std.copy(),
        }

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["indices"], state["mean"], state["std"])


def normalize_ste(values, normalization):
    return FeatureStandardizer.from_state_dict(normalization).transform(values)


def split_range(split, length, ratios, order=None):
    split_names = ("Train", "Val", "Test")
    order = tuple(order or split_names)
    if len(ratios) != len(split_names):
        raise ValueError("Split ratios must contain Train, Val, and Test values.")
    if len(order) != len(split_names) or set(order) != set(split_names):
        raise ValueError("Split order must contain Train, Val, and Test exactly once.")

    decimal_ratios = [Decimal(str(value)) for value in ratios]
    total = sum(decimal_ratios)
    if total <= 0:
        raise ValueError("Split ratios must have a positive sum.")

    ratio_by_split = dict(zip(split_names, decimal_ratios))
    ranges = {}
    start = 0
    cumulative = Decimal(0)
    for index, name in enumerate(order):
        cumulative += ratio_by_split[name]
        end = length if index == len(order) - 1 else int(cumulative / total * length)
        ranges[name] = (start, end)
        start = end
    return ranges[split]


def groupwise_split_mask(group_index, order_index, split, ratios):
    group_index = np.asarray(group_index)
    order_index = np.asarray(order_index)
    if group_index.ndim != 1 or order_index.ndim != 1 or group_index.shape != order_index.shape:
        raise ValueError(
            "Group and order indices must be one-dimensional arrays with matching shapes."
        )

    selected = np.zeros(len(group_index), dtype=bool)
    for group in np.unique(group_index):
        rows = np.flatnonzero(group_index == group)
        order = order_index[rows]
        if np.any(order[1:] < order[:-1]):
            rows = rows[np.argsort(order, kind="mergesort")]
        start, end = split_range(split, len(rows), ratios)
        selected[rows[start:end]] = True
    return selected


def timeline_split_mask(timeline_index, split, ratios, num_time_steps, order=None):
    timeline_index = np.asarray(timeline_index, dtype=np.int64)
    if timeline_index.ndim != 1:
        raise ValueError("Timeline indices must be one-dimensional.")
    start, end = split_range(split, int(num_time_steps), ratios, order=order)
    return (timeline_index >= start) & (timeline_index < end)


def dataset_sample_indices(selected, cfgs, split, family, vehicle_class=None, debug=False):
    indices = np.flatnonzero(selected)
    limit_key = {
        "Train": "max_train_samples",
        "Val": "max_val_samples",
        "Test": "max_test_samples",
    }.get(split)
    if limit_key is None:
        return indices
    limit = cfgs.get(limit_key)
    if isinstance(limit, dict):
        limit = limit.get(str(vehicle_class))
    if limit is None:
        return indices
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise ValueError(f"{limit_key} must be a positive integer or null.")
    if len(indices) <= limit:
        return indices
    seed = cfgs.get("sampling_seed", 2026)
    indices = np.sort(np.random.default_rng(seed).choice(indices, size=limit, replace=False))
    directory = cfgs["debug_npy_dir"] if debug else cfgs["npy_dir"]
    output_dir = Path(cfgs["dataset_root"]) / directory / "sampling_indices"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = family if vehicle_class is None else f"{family}_{vehicle_class}"
    np.save(output_dir / f"{name}_{split.lower()}_seed{seed}_limit{limit}.npy", indices)
    return indices


def build_windows(*arrays, length):
    return tuple(
        torch.from_numpy(array).unfold(0, length, 1).movedim(-1, 1)
        for array in arrays
    )


def apply_training_bridge_mask(observed, train_bridge_nodes, split):
    result = np.asarray(observed, dtype=bool).copy()
    if split == "Train":
        bridge_mask = np.zeros(result.shape[-3], dtype=bool)
        bridge_mask[np.asarray(train_bridge_nodes, dtype=np.int64)] = True
        result &= bridge_mask.reshape((1,) * (result.ndim - 3) + (-1, 1, 1))
    return result


def group_balance_weights(bridge_index, standard_lane, num_lanes=6):
    bridge_index = np.asarray(bridge_index, dtype=np.int64).reshape(-1)
    standard_lane = np.asarray(standard_lane, dtype=np.int64).reshape(-1)
    if bridge_index.shape != standard_lane.shape:
        raise ValueError("Bridge and lane indices must have the same shape.")
    if bridge_index.size == 0:
        raise ValueError("Cannot balance an empty training dataset.")
    if np.any(bridge_index < 0) or np.any((standard_lane < 0) | (standard_lane >= num_lanes)):
        raise ValueError("Bridge or lane index is out of range.")

    counts = np.zeros((int(bridge_index.max()) + 1, num_lanes), dtype=np.int64)
    np.add.at(counts, (bridge_index, standard_lane), 1)
    observed = counts > 0
    weights = np.zeros_like(counts, dtype=np.float32)
    weights[observed] = bridge_index.size / (observed.sum() * counts[observed])
    return weights


def rbf_similarity(distances):
    distances = np.asarray(distances, dtype=np.float32)
    positive = distances[distances > 0]
    scale = positive.std()
    result = np.zeros_like(distances)
    valid = distances > 0
    result[valid] = np.exp(-(distances[valid] ** 2) / (scale ** 2))
    np.fill_diagonal(result, 1.0)
    return result


def top_k_and_normalize(matrix, k):
    matrix = np.asarray(matrix, dtype=np.float32)
    result = np.zeros_like(matrix)
    for row in range(matrix.shape[0]):
        indices = np.flatnonzero(matrix[row] > 0)
        indices = indices[np.argsort(matrix[row, indices])[-min(k, len(indices)):]]
        result[row, indices] = matrix[row, indices] / matrix[row, indices].sum()
    return result


def compute_eigenmaps(adjacency, k):
    adjacency = np.asarray(adjacency, dtype=np.float32).copy()
    row, column = adjacency.nonzero()
    adjacency[row, column] = adjacency[column, row] = 1
    degree = adjacency.sum(axis=1) ** -0.5
    laplacian = np.eye(len(adjacency)) - (adjacency * degree).T * degree
    _, vectors = eigh(laplacian)
    return vectors[:, 1:k + 1].astype(np.float32)


def load_external_graphs(root, num_nodes, top_k, eigenmaps_k):
    distance = np.load(osp.join(root, "G_distance.npy"))[:num_nodes, :num_nodes]
    duration = np.load(osp.join(root, "G_duration.npy"))[:num_nodes, :num_nodes]
    adjacency = np.eye(num_nodes, dtype=np.float32)
    adjacency[distance > 0] = 1
    transition_matrices = np.stack([
        top_k_and_normalize(rbf_similarity(distance), top_k),
        top_k_and_normalize(rbf_similarity(duration), top_k),
    ])
    return compute_eigenmaps(adjacency, eigenmaps_k), transition_matrices


def spatial_factors(root):
    frame = pd.read_excel(osp.join(root, "bridge_information.xlsx"))
    longitude_latitude = frame["location"].str.split(",", expand=True).astype(float)
    longitude = np.deg2rad(longitude_latitude[0].to_numpy())
    latitude = np.deg2rad(longitude_latitude[1].to_numpy())
    longitude0 = longitude.mean()
    latitude0 = latitude.mean()
    radius = 6_371_000.0
    x = radius * np.cos(latitude0) * (longitude - longitude0)
    y = radius * (latitude - latitude0)
    return np.column_stack([
        frame["总车道数"].to_numpy(),
        frame["建成年份编码"].to_numpy(),
        frame["道路等级编码"].to_numpy(),
        x,
        y,
    ]).astype(np.float32)


def tensor_tree(value):
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value
