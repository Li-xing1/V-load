import copy
import os
from datetime import datetime

import torch
import yaml


MODEL_NAMES = ('V_C', 'V_W', 'V_S', 'V_V')


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def read_yaml(path):
    with open(path, encoding='utf-8') as file:
        return yaml.safe_load(file) or {}


def normalize_split(split):
    total = sum(split)
    if total > 1.0:
        return [item / total for item in split]
    return split


def load_merged_args(args_dir='args', dataset_args_path=None, figure_args=None, debug=None):
    dataset_args_path = dataset_args_path or os.path.join(args_dir, 'dataset_args.yaml')
    args = read_yaml(os.path.join(args_dir, 'args.yaml'))
    debug = args.get('debug', False) if debug is None else bool(debug)
    args['debug'] = debug
    if debug:
        args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset_args = read_yaml(dataset_args_path)
    dataset_root = dataset_args['dataset_root']
    known_lanes = dataset_args.get('known_lanes', {})

    for model_name in MODEL_NAMES:
        model_args = read_yaml(os.path.join(args_dir, f'{model_name}.yaml'))
        model_dataset_args = copy.deepcopy(dataset_args.get(model_name, {}))
        for key in (
            'start_time', 'end_time', 'time_interval_minutes', 'train_bridge_nodes',
            'unknown_bridge_nodes', 'core_known_bridge', 'core_unknown_bridge', 'canonical_lanes',
        ):
            if key in dataset_args:
                model_dataset_args[key] = copy.deepcopy(dataset_args[key])
        model_dataset_args['dataset_root'] = model_dataset_args.get('dataset_root', dataset_root)
        model_dataset_args['data_root'] = model_dataset_args['dataset_root']
        model_dataset_args['npy_dir'] = model_dataset_args.get('npy_dir', 'npy')
        model_dataset_args['debug_npy_dir'] = model_dataset_args.get('debug_npy_dir', 'debug_npy')
        if 'split' in model_dataset_args:
            model_dataset_args['split'] = normalize_split(model_dataset_args['split'])

        model_args['dataset'] = deep_merge(model_args.get('dataset', {}), model_dataset_args)
        model_args['debug'] = debug
        model_args['seed'] = args.get('seed', 2026)
        if model_name in ('V_C', 'V_V'):
            model_args['known_lanes'] = known_lanes
        if figure_args is not None:
            model_args['figure'] = figure_args
        if model_name in ('V_C', 'V_V'):
            model_args['model'] = deep_merge(model_args.get('model', {}), model_dataset_args)
        elif 'temporal_embedding_list' in model_dataset_args:
            model_args['model']['temporal_num_embeddings'] = model_dataset_args['temporal_embedding_list']
            model_args['model']['location_feature_indices'] = model_dataset_args['location_feature_indices']
            model_args['model']['location_fourier_levels'] = model_dataset_args['location_fourier_levels']

        if debug:
            model_args['epoch'] = 2
            model_args['batch_size'] = 32
            if model_name in ('V_C', 'V_V'):
                model_args['model']['d_model'] = 2

        args[model_name] = model_args

    args['dataset_args'] = dataset_args
    args['known_lanes'] = known_lanes
    args['bridge_nodes'] = sorted(args['known_lanes'], key=lambda item: int(item))
    return args


def resolve_work_path(args, fallback):
    save_name = args.get('save_name', fallback)
    if save_name == 'time':
        return datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return save_name


def discover_v_w_vehicle_types(args):
    dataset_args = args['dataset']
    npy_dir = dataset_args.get('debug_npy_dir' if args.get('debug', False) else 'npy_dir')
    family_dir = os.path.join(dataset_args['dataset_root'], npy_dir, 'V_W')
    files = sorted(name for name in os.listdir(family_dir) if name.startswith('V_W_') and name.endswith('.npz'))
    return [name[4:-4] for name in files]


def move_tensors_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_device(item, device) for item in value)
    return value


def move_tensors_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_tensors_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(item) for item in value)
    return value
