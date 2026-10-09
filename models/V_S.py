# -*- coding: utf-8 -*-
# @Time : 2026/3/10 15:13
# @Author : Xing Li
# @Email : 2434147605@qq.com
# @File : V_S.py
# @Project : V_load

import torch
import yaml
import torch.nn.functional as F
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation

from models.common import (
    ContextualAffineCouplingTransform1D,
    FlowTemporalEmbedding,
    LearnableMixtureOfGaussians,
)


class V_S(Flow):
    def __init__(self, args):
        num_features = args.get('num_features', 1)
        hidden_dim = args['hidden_dim']
        context_dim = args['context_dim']
        num_components = args['num_components']
        temporal_num_embeddings = args['temporal_num_embeddings']
        num_layer = args['num_layer']
        autoregressive_args = args.get('masked_piecewise_rational_quadratic_autoregressive', {})
        coupling_args = args.get('contextual_affine_coupling', {})

        def create_autoregressive_transform():
            return MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=num_features,
                hidden_features=autoregressive_args.get('hidden_features', hidden_dim),
                context_features=autoregressive_args.get('context_features', context_dim),
                num_bins=autoregressive_args.get('num_bins', 8),
                tails=autoregressive_args.get('tails', 'linear'),
                tail_bound=autoregressive_args.get('tail_bound', 3.0),
                num_blocks=autoregressive_args.get('num_blocks', 2),
                use_residual_blocks=autoregressive_args.get('use_residual_blocks', True),
                random_mask=autoregressive_args.get('random_mask', False),
                activation=F.relu,
                dropout_probability=autoregressive_args.get('dropout_probability', 0.25),
                use_batch_norm=autoregressive_args.get('use_batch_norm', False),
                min_bin_width=autoregressive_args.get('min_bin_width', 1e-3),
                min_bin_height=autoregressive_args.get('min_bin_height', 1e-3),
                min_derivative=autoregressive_args.get('min_derivative', 1e-3),
            )

        transforms = []
        for _ in range(num_layer):
            transforms.extend([
                RandomPermutation(features=num_features),
                LULinear(features=num_features, identity_init=True),
                create_autoregressive_transform(),
                ContextualAffineCouplingTransform1D(
                    hidden_dim,
                    context_dim,
                    num_features,
                    positive_output_eps=coupling_args.get('positive_output_eps', True),
                    log_scale_bound=coupling_args.get('log_scale_bound', 5.0),
                    eps=coupling_args.get('eps', 1e-30),
                ),
            ])
        transforms.append(LULinear(features=num_features, identity_init=True))

        base_distribution = LearnableMixtureOfGaussians(
            num_components,
            num_features,
            min_log_std=args.get('base_min_log_std', -3.0),
            max_log_std=args.get('base_max_log_std', 1.0),
        )
        transform = CompositeTransform(transforms)
        embedding_net = FlowTemporalEmbedding(
            temporal_num_embeddings, context_dim, log_features=False,
            fourier_feature_indices=args['location_feature_indices'],
            fourier_levels=args['location_fourier_levels'],
        )
        super().__init__(transform=transform, distribution=base_distribution, embedding_net=embedding_net)


if __name__ == '__main__':
    args = yaml.safe_load(open('../args/V_S.yaml'))['model']
    args['temporal_num_embeddings'] = [288, 2]
    flow_model = V_S(args)
    batch_size = 12
    inputs = torch.rand(size=(batch_size, 1))

    c1 = torch.randint(low=0, high=288, size=[batch_size, 1], dtype=torch.float32)
    c2 = torch.randint(low=0, high=2, size=[batch_size, 1], dtype=torch.float32)
    TF = torch.cat([c1, c2], dim=1)

    log_probs = flow_model.log_prob(inputs, context=TF)
    print("Log probabilities:", log_probs)

    flow_model.eval()
    num_samples = 10
    c1 = torch.randint(low=0, high=288, size=[1, 1], dtype=torch.float32)
    c2 = torch.randint(low=0, high=2, size=[1, 1], dtype=torch.float32)
    TF = torch.cat([c1, c2], dim=1)
    samples = flow_model.sample(num_samples=num_samples, context=TF)
    print("Samples:", samples.shape)
