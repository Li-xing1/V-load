import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.V_W_dataset import AxisWeightStandardizer
from dataset.common import (
    FeatureStandardizer, dataset_sample_indices, group_balance_weights, timeline_split_mask,
)


class V_S_dataset(Dataset):
    def __init__(
        self,
        cfgs,
        debug=False,
        split="Train",
        target_normalizer=None,
        context_normalizer=None,
    ):
        directory = cfgs["debug_npy_dir"] if debug else cfgs["npy_dir"]
        archive = np.load(osp.join(cfgs["dataset_root"], directory, "V_S", "V_S.npz"))
        target = archive["target"].astype(np.float32, copy=False)
        context = archive["context"].astype(np.float32, copy=False)
        bridge_index = archive["bridge_index"].astype(np.int64, copy=False)
        timeline_index = archive["timeline_index"].astype(np.int64, copy=False)
        standard_lane = archive["standard_lane"].astype(np.int64, copy=False)
        num_time_steps = int(archive["num_time_steps"])
        archive.close()

        selected = timeline_split_mask(
            timeline_index,
            split,
            cfgs["split"],
            num_time_steps,
            order=cfgs.get("split_order"),
        )
        if split in {"Train", "Val"}:
            selected &= np.isin(bridge_index, np.asarray(cfgs["train_bridge_nodes"]))
        self.original_size = int(selected.sum())
        self.sample_indices = dataset_sample_indices(
            selected, cfgs, split, "V_S", debug=debug,
        )

        if target_normalizer is None:
            target_normalizer = AxisWeightStandardizer.fit(target[selected])
        if context_normalizer is None:
            context_normalizer = FeatureStandardizer.fit(
                context[selected], cfgs["temporal_embedding_list"]
            )
        target = target[self.sample_indices]
        context = context[self.sample_indices]
        bridge_index = bridge_index[self.sample_indices]
        timeline_index = timeline_index[self.sample_indices]
        standard_lane = standard_lane[self.sample_indices]
        self.target_normalizer = target_normalizer
        self.context_normalizer = context_normalizer
        self.data = torch.from_numpy(target_normalizer.transform(target))
        self.TF = torch.from_numpy(context_normalizer.transform(context))
        self.bridge_index = torch.from_numpy(bridge_index)
        self.timeline_index = torch.from_numpy(timeline_index)
        self.standard_lane = torch.from_numpy(standard_lane)
        self.group_weight_lookup = (
            torch.from_numpy(group_balance_weights(bridge_index, standard_lane))
            if split == "Train" else None
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            self.data[index],
            self.TF[index],
            self.bridge_index[index],
            self.standard_lane[index],
            self.timeline_index[index],
        )
