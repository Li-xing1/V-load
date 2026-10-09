import random
from collections import OrderedDict

import numpy as np
import torch
from torch.nn.parallel import DataParallel
from torch.nn.parallel.scatter_gather import scatter_kwargs
from utils.TrainingConfig import move_tensors_to_device


def set_random_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def group_balanced_mean(values, bridge_index, standard_lane, weight_lookup):
    sample_weights = weight_lookup[bridge_index.long(), standard_lane.long()]
    while sample_weights.ndim < values.ndim:
        sample_weights = sample_weights.unsqueeze(-1)
    return (values * sample_weights).mean()


# Visualization of forecast results
def tensorboard(write, target, output, out_len=0):
    B, T, N, C = target.shape
    B = min(B, 1000)
    index = random.sample(range(0, N), 10)
    for b in range(B):
        for c in range(C):
            for node in index:
                write.add_scalars(main_tag=f'result-feature{c}/node{node}',
                                  tag_scalar_dict={'target': target[b, 0, node, c]},
                                  global_step=b)
                for lenl in range(out_len):
                    write.add_scalars(main_tag=f'result-feature{c}/node{node}',
                                      tag_scalar_dict={f'output-{lenl + 1}': output[b, lenl, node, c]},
                                      global_step=b + lenl)



# ancillary
# Error Aggregate Averager
class Average(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def add(self, value, count):
        self._sum += value * count
        self._count += count

    def average(self):
        if self._count == 0:
            raise ValueError('Cannot average zero samples.')
        return self._sum / self._count


def distribution_statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise FloatingPointError('No finite values are available for statistics.')
    return {
        'mean': float(np.mean(finite_values)),
        'std': float(np.std(finite_values)),
        'median': float(np.median(finite_values)),
        'p95': float(np.percentile(finite_values, 95)),
        'p99': float(np.percentile(finite_values, 99)),
        'max': float(np.max(finite_values)),
        'nonfinite_count': int(values.size - finite_values.size),
    }


def _device_to_index(device):
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        text = device.strip()
        if text == 'cpu':
            return None
        if text.startswith('cuda:'):
            return int(text.split(':', 1)[1])
        return int(text)
    raise TypeError(f'Unsupported device id type: {type(device)!r}')


def parse_device_ids(device):
    if isinstance(device, (list, tuple)):
        return [idx for idx in (_device_to_index(item) for item in device) if idx is not None]
    if device is None:
        return []
    if isinstance(device, str) and ',' in device:
        return [idx for idx in (_device_to_index(item) for item in device.split(',') if item.strip()) if idx is not None]
    device_id = _device_to_index(device)
    return [] if device_id is None else [device_id]


def primary_device(device):
    if isinstance(device, (list, tuple)):
        if not device:
            raise ValueError('device list cannot be empty')
        return device[0]
    if isinstance(device, str) and ',' in device:
        return device.split(',', 1)[0].strip()
    return device


def should_use_data_parallel(device):
    return len(parse_device_ids(device)) > 1


def parallel_model_module(model):
    return model.module if isinstance(model, DataParallel) else model


def data_parallel_state_dict(model):
    return parallel_model_module(model).state_dict()


def build_data_parallel_model(model, device, static_arg_index=2):
    device_ids = parse_device_ids(device)
    if len(device_ids) <= 1:
        return model
    return StaticDataParallel(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
        static_arg_index=static_arg_index,
    )


class StaticDataParallel(DataParallel):
    """DataParallel that broadcasts one positional argument as static data."""

    def __init__(self, module, static_arg_index=2, **kwargs):
        super().__init__(module, **kwargs)
        self.static_arg_index = static_arg_index

    def scatter(self, inputs, kwargs, device_ids):
        inputs = tuple(inputs)
        kwargs = dict(kwargs)
        static_index = self.static_arg_index
        has_positional_static = len(inputs) > static_index
        static = inputs[static_index] if has_positional_static else kwargs.pop('statics', None)
        batch_inputs = (
            inputs[:static_index] + inputs[static_index + 1:]
            if has_positional_static else inputs
        )
        scattered_inputs, scattered_kwargs = scatter_kwargs(
            batch_inputs, kwargs, device_ids, dim=self.dim,
        )

        static_copies = [
            move_tensors_to_device(static, f'cuda:{device_id}')
            for device_id in device_ids
        ]
        if has_positional_static:
            scattered_inputs = tuple(
                tuple(chunk[:static_index]) + (static_copy,) + tuple(chunk[static_index:])
                for chunk, static_copy in zip(scattered_inputs, static_copies)
            )
        else:
            scattered_kwargs = tuple(
                {**chunk_kwargs, 'statics': static_copy}
                for chunk_kwargs, static_copy in zip(scattered_kwargs, static_copies)
            )
        return tuple(scattered_inputs), tuple(scattered_kwargs)


# Multi-GPU parameter conversion
def state_dict_2_1(state_dict):
    return normalize_state_dict(state_dict)


def normalize_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict
