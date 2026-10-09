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


class MaskedLogStandardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (2,) or self.std.shape != (2,):
            raise ValueError("V_V normalization mean and std must each contain two channels.")

    @classmethod
    def fit(cls, values, mask):
        values = np.asarray(values, dtype=np.float32)
        pair_mask = cls._pair_mask(values, mask)
        if not pair_mask.any():
            raise ValueError("Cannot fit V_V normalization without valid speed-flow pairs.")
        selected = values[pair_mask]
        if not np.isfinite(selected).all() or (selected < 0).any():
            raise ValueError("V_V speed-flow targets must be finite and non-negative.")
        transformed = np.log1p(selected)
        std = transformed.std(axis=0)
        std = np.where(std > np.finfo(np.float32).eps, std, 1.0)
        return cls(transformed.mean(axis=0), std)

    def transform(self, values, mask):
        values = np.asarray(values, dtype=np.float32)
        pair_mask = self._pair_mask(values, mask)
        result = np.zeros_like(values, dtype=np.float32)
        result[pair_mask] = (np.log1p(values[pair_mask]) - self.mean) / self.std
        return result

    def inverse_transform(self, values):
        return np.expm1(np.asarray(values) * self.std + self.mean).astype(np.float32)

    def state_dict(self):
        return {"method": "log1p_zscore", "mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["mean"], state["std"])

    @staticmethod
    def _pair_mask(values, mask):
        if values.ndim < 1 or values.shape[-1] != 2:
            raise ValueError(f"V_V targets must end with two channels, got shape {values.shape}.")
        mask = np.asarray(mask, dtype=bool)
        expected_shape = values.shape[:-1] + (1,)
        if mask.shape != expected_shape:
            raise ValueError(f"V_V mask must have shape {expected_shape}, got {mask.shape}.")
        return mask[..., 0]


class V_v_handle:
    def __init__(self, cfgs, debug=False):
        self.root = cfgs["dataset_root"]
        directory = cfgs["debug_npy_dir"] if debug else cfgs["npy_dir"]
        family = osp.join(self.root, directory, "V_V")
        self.data = np.load(osp.join(family, "V_V_data.npy")).astype(np.float32, copy=False)
        self.TF = np.load(osp.join(family, "V_V_tf.npy")).astype(np.float32, copy=False)
        self.mask = np.load(osp.join(family, "V_V_mask.npy")).astype(bool, copy=False)
        expected_mask_shape = self.data.shape[:-1] + (1,)
        if self.data.ndim != 4 or self.data.shape[-1] != 2:
            raise ValueError(f"V_V_data.npy must have shape [T, N, L, 2], got {self.data.shape}.")
        if self.mask.shape != expected_mask_shape:
            raise ValueError(f"V_V_mask.npy must have shape {expected_mask_shape}, got {self.mask.shape}.")
        if self.TF.shape[0] != self.data.shape[0]:
            raise ValueError("V_V temporal features and targets must have the same time dimension.")
        self.N = self.data.shape[1]
        self.top_k = cfgs["top_k"]
        self.eigenmaps_k = cfgs["eigenmaps_k"]

    def get_predefine_graphs(self):
        return load_external_graphs(self.root, self.N, self.top_k, self.eigenmaps_k)

    def get_spatial_factors(self):
        return spatial_factors(self.root)[:self.N]

    def return_data(self):
        return self.data, self.TF, self.mask


class V_V_dataset(Dataset):
    def __init__(self, cfgs, debug=False, split="Train", normalization=None):
        handle = V_v_handle(cfgs, debug)
        raw_data, raw_tf, observed_mask = handle.return_data()
        raw_sf = handle.get_spatial_factors()
        train_nodes = np.asarray(cfgs["train_bridge_nodes"], dtype=np.int64)
        split_order = cfgs.get("split_order")
        train_start, train_end = split_range(
            "Train", len(raw_data), cfgs["split"], order=split_order
        )

        if normalization is None:
            fit_mask = np.zeros_like(observed_mask[train_start:train_end])
            fit_mask[:, train_nodes] = observed_mask[train_start:train_end, train_nodes]
            target = MaskedLogStandardizer.fit(raw_data[train_start:train_end], fit_mask)
            temporal = FeatureStandardizer.fit(
                raw_tf[train_start:train_end], cfgs["temporal_embedding_list"]
            )
            spatial = FeatureStandardizer.fit(
                raw_sf, cfgs["spatial_embedding_list"],
                fit_rows=np.isin(np.arange(len(raw_sf)), train_nodes),
            )
            normalization = {
                "target": target.state_dict(),
                "temporal": temporal.state_dict(),
                "spatial": spatial.state_dict(),
                "target_channels": ["speed_mps", "minute_flow"],
                "max_speed_mps": float(cfgs["max_speed_mps"]),
                "min_minute_flow": 0.0,
            }
        else:
            target = MaskedLogStandardizer.from_state_dict(normalization["target"])
            temporal = FeatureStandardizer.from_state_dict(normalization["temporal"])
            spatial = FeatureStandardizer.from_state_dict(normalization["spatial"])

        transformed = target.transform(raw_data, observed_mask)
        normalized_tf = temporal.transform(raw_tf)
        normalized_sf = spatial.transform(raw_sf)
        start, end = split_range(
            split, len(raw_data), cfgs["split"], order=split_order
        )
        split_mask = apply_training_bridge_mask(observed_mask[start:end], train_nodes, split)
        self.data, self.TF, self.mask = build_windows(
            transformed[start:end], normalized_tf[start:end], split_mask, length=cfgs["len"]
        )
        self.raw_data = torch.from_numpy(transformed[start:end])
        self.raw_tf = torch.from_numpy(normalized_tf[start:end])
        self.raw_mask = torch.from_numpy(observed_mask[start:end])
        self.normalizer = target
        self.normalization = normalization

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
        return self.data[index], self.TF[index], self.mask[index]
