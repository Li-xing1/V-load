# -*- coding: utf-8 -*-
# @Time : 2026/3/10 15:13
# @Author : Xing Li
# @Email : 2434147605@qq.com
# @File : V_W.py
# @Project : V_load

import torch
import yaml
import torch.nn.functional as F
from pathlib import Path
import sys
from nflows.flows.base import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import ReversePermutation, RandomPermutation
from nflows.utils import create_alternating_binary_mask

if __package__ is None or __package__ == '':
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.common import (
    ContextualAffineCouplingTransform1D,
    FlowTemporalEmbedding,
    LearnableMixtureOfGaussians,
)


class ContextIgnoringNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, context=None):
        return self.net(inputs)


class V_W(Flow):
    def __init__(self, args):
        num_features = args.get('num_features', 1)
        hidden_dim = args['hidden_dim']
        context_dim = args['context_dim']
        num_components = args['num_components']
        temporal_num_embeddings = args['temporal_num_embeddings']
        num_layer = args['num_layer']
        coupling_args = args.get('contextual_affine_coupling', {})
        rq_args = args.get('rational_quadratic_coupling', {})

        def create_rq_coupling(layer_index):
            even_mask = True if num_features == 1 else (layer_index % 2 == 0)
            return PiecewiseRationalQuadraticCouplingTransform(
                mask=create_alternating_binary_mask(num_features, even=even_mask),
                transform_net_create_fn=lambda in_features, out_features:
                    ResidualNet(
                        in_features=in_features,
                        out_features=out_features,
                        hidden_features=rq_args.get('hidden_features', hidden_dim),
                        context_features=context_dim,
                        num_blocks=rq_args.get('num_transform_blocks', 2),
                        activation=F.relu,
                        dropout_probability=rq_args.get('dropout_probability', 0.25),
                        use_batch_norm=rq_args.get('use_batch_norm', False),
                ),
                num_bins=rq_args.get('num_bins', 8),
                tails='linear',
                tail_bound=rq_args.get('tail_bound', 3.0),
                apply_unconditional_transform=rq_args.get('apply_unconditional_transform', False),
            )

        def create_layer(layer_index):
            return CompositeTransform([
                RandomPermutation(features=num_features),
                LULinear(features=num_features, identity_init=True),
                create_rq_coupling(layer_index),
                ContextualAffineCouplingTransform1D(
                    hidden_dim,
                    context_dim,
                    num_features,
                    positive_output_eps=coupling_args.get('positive_output_eps', True),
                    log_scale_bound=coupling_args.get('log_scale_bound', 5.0),
                    eps=coupling_args.get('eps', 1e-30),
                )
            ])

        base_distribution = LearnableMixtureOfGaussians(
            num_components,
            num_features,
            min_log_std=args.get('base_min_log_std', -3.0),
            max_log_std=args.get('base_max_log_std', 1.0),
        )
        transform = CompositeTransform([create_layer(i) for i in range(num_layer)]+[LULinear(features=num_features, identity_init=True)])
        embedding_net = FlowTemporalEmbedding(
            temporal_num_embeddings, context_dim, log_features=False,
            fourier_feature_indices=args['location_feature_indices'],
            fourier_levels=args['location_fourier_levels'],
        )
        super().__init__(transform=transform, distribution=base_distribution, embedding_net=embedding_net)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    num_features = 3
    args = yaml.safe_load(open(project_root / 'args.yaml' / 'args.yaml.yaml'))['V_W']['model']
    args['num_features'] = num_features
    args['temporal_num_embeddings'] = [288, 2]
    flow_model = V_W(args)
    batch_size = 8
    inputs = torch.rand(size=(batch_size, num_features))

    c1 = torch.randint(low=0, high=288, size=[batch_size, 1], dtype=torch.float32)
    c2 = torch.randint(low=0, high=2, size=[batch_size, 1], dtype=torch.float32)
    TF = torch.cat([c1, c2], dim=1)

    log_probs = flow_model.log_prob(inputs, context=TF)
    print("Log probabilities:", log_probs)

    # flow_model.eval()
    # num_samples = 10
    # c1 = torch.randint(low=0, high=288, size=[1, 1], dtype=torch.float32)
    # c2 = torch.randint(low=0, high=2, size=[1, 1], dtype=torch.float32)
    # TF = torch.cat([c1, c2], dim=1)
    # samples = flow_model.sample(num_samples=num_samples, context=TF)
    # print("Samples:", samples.shape)
