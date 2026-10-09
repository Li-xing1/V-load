import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.common import (
    FeatureStandardizer,
    apply_training_bridge_mask,
    build_windows,
    load_external_graphs,
    spatial_factors,
    split_range,
    tensor_tree,
)


def fit_vehicle_class_normalizer(data):
    class_mass = np.asarray(data, dtype=np.float64)[..., :6].sum(axis=(0, 1, 2))
    return (class_mass / class_mass.sum()).astype(np.float32)


def normalize_vehicle_classes(data, frequencies):
    result = np.asarray(data, dtype=np.float32).copy()
    counts = result[..., :6]
    total = result[..., 6:7]
    proportions = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    balanced = proportions / np.asarray(frequencies, dtype=np.float32)
    balanced_sum = balanced.sum(axis=-1, keepdims=True)
    result[..., :6] = np.divide(
        balanced, balanced_sum, out=np.zeros_like(balanced), where=balanced_sum > 0
    ) * total
    return result


class V_c_handle:
    def __init__(self, cfgs, debug=False):
        self.root = cfgs["dataset_root"]
        directory = cfgs["debug_npy_dir"] if debug else cfgs["npy_dir"]
        family = osp.join(self.root, directory, "V_C")
        self.data = np.load(osp.join(family, "V_C_data.npy")).astype(np.float32, copy=False)
        self.TF = np.load(osp.join(family, "V_C_tf.npy")).astype(np.float32, copy=False)
        self.V = np.load(osp.join(family, "V_C_v.npy")).astype(np.float32, copy=False)
        self.mask = np.load(osp.join(family, "V_C_mask.npy")).astype(bool, copy=False)
        self.N = self.data.shape[1]
        self.top_k = cfgs["top_k"]
        self.eigenmaps_k = cfgs["eigenmaps_k"]

    def get_predefine_graphs(self):
        return load_external_graphs(self.root, self.N, self.top_k, self.eigenmaps_k)

    def get_spatial_factors(self):
        return spatial_factors(self.root)[:self.N]

    def return_data(self):
        return self.data, self.TF, self.V, self.mask


class V_C_dataset(Dataset):
    def __init__(self, cfgs, debug=False, split="Train", normalization=None):
        handle = V_c_handle(cfgs, debug)
        raw_data, raw_tf, raw_v, observed_mask = handle.return_data()
        raw_sf = handle.get_spatial_factors()
        train_nodes = np.asarray(cfgs["train_bridge_nodes"], dtype=np.int64)
        split_order = cfgs.get("split_order")
        train_start, train_end = split_range(
            "Train", len(raw_data), cfgs["split"], order=split_order
        )

        if normalization is None:
            temporal = FeatureStandardizer.fit(
                raw_tf[train_start:train_end], cfgs["temporal_embedding_list"]
            )
            spatial = FeatureStandardizer.fit(
                raw_sf, cfgs["spatial_embedding_list"],
                fit_rows=np.isin(np.arange(len(raw_sf)), train_nodes),
            )
            speed = FeatureStandardizer.fit(
                raw_v[train_start:train_end, train_nodes], [1]
            )
            frequencies = fit_vehicle_class_normalizer(
                raw_data[train_start:train_end, train_nodes]
            )
            normalization = {
                "temporal": temporal.state_dict(),
                "spatial": spatial.state_dict(),
                "V_C_v": speed.state_dict(),
                "vehicle_class_frequencies": frequencies,
                "balance_classes": bool(cfgs["balance_classes"]),
            }
        else:
            temporal = FeatureStandardizer.from_state_dict(normalization["temporal"])
            spatial = FeatureStandardizer.from_state_dict(normalization["spatial"])
            speed = FeatureStandardizer.from_state_dict(normalization["V_C_v"])
            frequencies = np.asarray(normalization["vehicle_class_frequencies"], dtype=np.float32)

        model_data = (
            normalize_vehicle_classes(raw_data, frequencies)
            if normalization["balance_classes"] else raw_data
        )
        normalized_tf = temporal.transform(raw_tf)
        normalized_v = speed.transform(raw_v)
        normalized_sf = spatial.transform(raw_sf)
        start, end = split_range(
            split, len(raw_data), cfgs["split"], order=split_order
        )
        split_mask = apply_training_bridge_mask(observed_mask[start:end], train_nodes, split)
        self.data, self.TF, self.V, self.mask = build_windows(
            model_data[start:end], normalized_tf[start:end], normalized_v[start:end], split_mask,
            length=cfgs["len"]
        )
        self.raw_data = torch.from_numpy(raw_data[start:end])
        self.raw_tf = torch.from_numpy(normalized_tf[start:end])
        self.raw_v = torch.from_numpy(normalized_v[start:end])
        self.raw_mask = torch.from_numpy(observed_mask[start:end])
        self.normalization = normalization
        self.class_frequencies = frequencies

        if split == "Train":
            eigenmaps, transition_matrices = handle.get_predefine_graphs()
            self.statics = tensor_tree({
                "eigenmaps": eigenmaps,
                "transition_matrices": transition_matrices,
                "SF": normalized_sf,
                "normalization": normalization,
                "train_bridge_nodes": train_nodes,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.TF[index], self.V[index], self.mask[index]
