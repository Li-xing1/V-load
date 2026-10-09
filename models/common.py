# -*- coding: utf-8 -*-

import itertools
import math
from functools import lru_cache

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.nn.functional as F

try:
    from nflows.distributions.base import Distribution
except ImportError:  # Allows V_C to import this module without nflows installed.
    Distribution = nn.Module


def nozero_softmax(data, dim):
    mask = data.ne(0)
    max = torch.max(data, dim=-1, keepdim=True)[0]
    data = data - max
    exps = torch.exp(data)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + 0.00001
    alpha = masked_exps / masked_sums
    return alpha


def multihead_linear_transform(W, inputs):
    B, P, N, H, d_k, d_model = W.shape
    inputs = inputs.reshape((B, P, N, 1, d_model, 1))
    out = torch.matmul(W, inputs).squeeze(-1)  # (B, P, N, H, d_k)
    return out


def gauss_quadrature_node_gen(d, device=None, dtype=torch.float32):
    node_values = [-0.8611363115940526, -0.3399810435848563,
                   0.3399810435848563, 0.8611363115940526]
    weight_values = [0.3478548451374538, 0.6521451548625461,
                     0.6521451548625461, 0.3478548451374538]
    nodes = torch.tensor(list(itertools.product(node_values, repeat=d)), dtype=dtype, device=device)
    weights = torch.tensor(list(itertools.product(weight_values, repeat=d)), dtype=dtype, device=device)
    w = weights.prod(dim=1)
    return nodes, w


@lru_cache(maxsize=16)
def predefine_graph_data(transition_matrices, selection):
    matrices = transition_matrices[list(selection)]
    return matrices, tuple(matrix.nonzero() for matrix in matrices)


def multihead_spatial_attention_predefine_graph(Q, K, V, transition_matrix, index):
    '''
    math: [softmax(QK/SQRT(d_k))*transition_matrix]V
    :param Q:shape：[B,P,N,H,d_K]
    :param K:shape：[B,P,N,H,d_K]
    :param V:shape：[B,P,N,H,d_K]
    :param transition_matrix: 类似与邻接矩阵 shape:[N,N]
    :return: shapr[B, P, N, H * d_k]
    '''
    B, P, N, H, d_k = Q.shape

    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)

    Q = Q[:, :, :, index[:, 0], :].unsqueeze(dim=-2)
    K = K[:, :, :, index[:, 1], :].unsqueeze(dim=-1)
    scaled = torch.matmul(Q, K).squeeze(dim=-1).squeeze(dim=-1) / math.sqrt(d_k)
    scaled_dot_product = torch.full(size=[B, P, H, N, N], fill_value=0., dtype=torch.float32, device=Q.device)
    scaled_dot_product[:, :, :, index[:, 0], index[:, 1]] = scaled
    alpha = nozero_softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    graph_attention = alpha * transition_matrix
    graph_attention = graph_attention / graph_attention.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(graph_attention.dtype).tiny
    )
    out = torch.matmul(graph_attention, V)  # (B, P, H, N, d_k)

    out = out.permute((2, 0, 1, 3, 4))
    return out


def multihead_spatial_attention_adaptive_graph(Q, K, V, transition_matrix):
    '''
    math: [softmax(QK/SQRT(d_k))*transition_matrix]V
    :param Q:shape：[B,P,N,H,d_K]
    :param K:shape：[B,P,N,H,d_K]
    :param V:shape：[B,P,N,H,d_K]
    :param transition_matrix: 类似与邻接矩阵 shape:[N,N]
    :return: shapr[B, P, N, H * d_k]
    '''
    B, P, N, H, d_k = Q.shape
    Q = Q.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    K = K.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    V = V.permute((0, 1, 3, 2, 4))  # (B, P, H, N, d_k)
    index = transition_matrix.nonzero()
    Q = Q[index[:, 0], index[:, 1], :, index[:, 2], :].unsqueeze(dim=-2)
    K = K[index[:, 0], index[:, 1], :, index[:, 3], :].unsqueeze(dim=-1)
    scaled = torch.matmul(Q, K).squeeze(dim=-1).squeeze(dim=-1) / math.sqrt(d_k)
    scaled_dot_product = torch.full(size=[B, P, H, N, N], fill_value=0., dtype=torch.float32, device=Q.device)
    scaled_dot_product[index[:, 0], index[:, 1], :, index[:, 2], index[:, 3]] = scaled
    alpha = nozero_softmax(scaled_dot_product, dim=-1)  # (B, P, H, N, N)
    graph_attention = alpha * transition_matrix.unsqueeze(2)
    graph_attention = graph_attention / graph_attention.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(graph_attention.dtype).tiny
    )
    out = torch.matmul(graph_attention, V)  # (B, P, H, N, d_k)
    out = out.permute((2, 0, 1, 3, 4))
    return out


def gauss_legendre_node_gen_1d(num_points, device=None, dtype=torch.float32, low=-1.0, high=1.0):
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    nodes = torch.tensor(nodes, dtype=dtype, device=device)
    weights = torch.tensor(weights, dtype=dtype, device=device)
    if low != -1.0 or high != 1.0:
        scale = 0.5 * (high - low)
        shift = 0.5 * (high + low)
        nodes = nodes * scale + shift
        weights = weights * scale
    return nodes, weights


@lru_cache(maxsize=32)
def latent_gauss_quadrature(num_points, device, dtype):
    nodes, weights = gauss_legendre_node_gen_1d(
        num_points, device=device, dtype=dtype, low=0.0, high=1.0,
    )
    nodes = math.sqrt(2.0) * torch.erfinv(nodes.mul(2.0).sub(1.0))
    return nodes, torch.log(weights)


def nonan_softmax(data, dim):
    max_value, _ = torch.max(data, dim=dim, keepdim=True)
    data = data - max_value
    return torch.softmax(data, dim=dim)


class NodewiseLinear(nn.Module):
    def __init__(self, num_nodes, in_features, out_features):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_nodes, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_nodes, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        bound = math.sqrt(6.0 / (self.in_features + self.out_features)) if (self.in_features + self.out_features) > 0 else 0.0
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        out = torch.einsum('sbni,nio->sbno', x, self.weight)
        return out + self.bias.unsqueeze(0).unsqueeze(0)


class ContextualLinear(nn.Module):
    def __init__(self, in_features, out_features, context_features, hidden_features, rank):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.meta = nn.Sequential(
            nn.Linear(context_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, rank * (in_features + out_features) + out_features),
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, context):
        parameters = self.meta(context)
        left_end = self.in_features * self.rank
        right_end = left_end + self.rank * self.out_features
        left = parameters[..., :left_end].reshape(*context.shape[:-1], self.in_features, self.rank)
        right = parameters[..., left_end:right_end].reshape(*context.shape[:-1], self.rank, self.out_features)
        generated_bias = parameters[..., right_end:]
        base = torch.matmul(inputs, self.weight) + self.bias
        update = torch.einsum('sbni,bnir,bnro->sbno', inputs, left, right) / math.sqrt(self.rank)
        return base + update + generated_bias.unsqueeze(0)


class My_Multinormal:
    def __init__(self, mean, covariance_matrix, cov_factor=None, cov_diag=None, low=-1.0, high=1.0,
                 latent_quad_points=16):
        self.mean = mean
        self.low = float(low)
        self.high = float(high)
        self.latent_quad_points = latent_quad_points
        self.covariance_matrix = covariance_matrix
        self.cov_factor = cov_factor
        self.cov_diag = cov_diag

        if self.cov_factor is None and self.cov_diag is not None and covariance_matrix is not None:
            if covariance_matrix.dim() == mean.dim() + 1 and covariance_matrix.shape[-2] == mean.shape[-1] and \
                    covariance_matrix.shape[-1] == 1:
                self.cov_factor = covariance_matrix
                self.covariance_matrix = None

        if self.cov_factor is not None:
            if self.cov_factor.shape[-1] != 1:
                raise NotImplementedError('My_Multinormal currently supports rank-1 low-rank covariances only.')
            if self.cov_diag is None:
                self.cov_diag = torch.full_like(self.mean, torch.finfo(self.mean.dtype).eps)
            self.multivariate_normal_dist = torch.distributions.LowRankMultivariateNormal(
                loc=self.mean,
                cov_factor=self.cov_factor,
                cov_diag=self.cov_diag,
            )
        else:
            self.multivariate_normal_dist = torch.distributions.MultivariateNormal(
                loc=mean,
                covariance_matrix=self.covariance_matrix,
            )
        self.factor = self._cube_normalizer()

    @staticmethod
    def _standard_normal_cdf(x):
        return torch.special.ndtr(x)

    @staticmethod
    def _stable_log_ndtr(x):
        log_cdf = torch.special.log_ndtr(x)
        if not x.requires_grad:
            return log_cdf

        log_density = -0.5 * x.square() - 0.5 * math.log(2.0 * math.pi)
        regular_gradient = torch.exp(log_density - log_cdf)
        tail = (-x).clamp_min(1.0)
        inverse_tail = tail.reciprocal()
        tail_gradient = tail + inverse_tail - 2.0 * inverse_tail.pow(3) + 10.0 * inverse_tail.pow(5)
        gradient = torch.where(x < -10.0, tail_gradient, regular_gradient)
        return log_cdf.detach() + gradient.detach() * (x - x.detach())

    @staticmethod
    def _log_normal_interval_probability(lower, upper):
        log_cdf_lower = My_Multinormal._stable_log_ndtr(lower)
        log_cdf_upper = My_Multinormal._stable_log_ndtr(upper)
        log_left = log_cdf_upper + torch.log(
            (-torch.expm1(log_cdf_lower - log_cdf_upper)).clamp_min(torch.finfo(lower.dtype).tiny)
        )

        log_sf_lower = My_Multinormal._stable_log_ndtr(-lower)
        log_sf_upper = My_Multinormal._stable_log_ndtr(-upper)
        log_right = log_sf_lower + torch.log(
            (-torch.expm1(log_sf_upper - log_sf_lower)).clamp_min(torch.finfo(lower.dtype).tiny)
        )
        return torch.where(lower + upper <= 0, log_left, log_right)

    def _full_cube_normalizer(self):
        a_len = max(self.mean.dim() - 2, 0)
        nodes, weights = gauss_quadrature_node_gen(
            self.covariance_matrix.shape[-1],
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        for _ in range(a_len):
            nodes = nodes.unsqueeze(1)
            weights = weights.unsqueeze(1)
        log_p = self.multivariate_normal_dist.log_prob(nodes.unsqueeze(1))
        return torch.logsumexp(torch.log(weights).unsqueeze(1) + log_p, dim=0, keepdim=False)

    def _low_rank_cube_normalizer(self):
        latent_nodes, log_latent_weights = latent_gauss_quadrature(
            self.latent_quad_points,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        batch_ndim = max(self.mean.dim() - 1, 0)
        latent_nodes = latent_nodes.view(-1, *([1] * batch_ndim))

        mean = self.mean.unsqueeze(0)
        factor = self.cov_factor.squeeze(-1).unsqueeze(0)
        std = self.cov_diag.sqrt().unsqueeze(0)
        low = torch.as_tensor(self.low, dtype=self.mean.dtype, device=self.mean.device)
        high = torch.as_tensor(self.high, dtype=self.mean.dtype, device=self.mean.device)

        latent_nodes = latent_nodes.unsqueeze(-1)
        upper = (high - mean - factor * latent_nodes) / std
        lower = (low - mean - factor * latent_nodes) / std
        log_cube_prob = self._log_normal_interval_probability(lower, upper).sum(dim=-1)
        log_latent_weights = log_latent_weights.view(-1, *([1] * batch_ndim))
        return torch.logsumexp(log_latent_weights + log_cube_prob, dim=0)

    def _cube_normalizer(self):
        if self.cov_factor is not None:
            return self._low_rank_cube_normalizer()
        return self._full_cube_normalizer()

    def log_prob(self, x):
        return self.multivariate_normal_dist.log_prob(x) - self.factor

    def prob(self, x):
        return self.log_prob(x).exp()

    def _sample_low_rank(self, num_samples):
        latent_nodes, log_latent_weights = latent_gauss_quadrature(
            self.latent_quad_points,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

        batch_ndim = max(self.mean.dim() - 1, 0)
        mean = self.mean.unsqueeze(0)
        factor = self.cov_factor.squeeze(-1).unsqueeze(0)
        std = self.cov_diag.sqrt().unsqueeze(0)
        low = torch.as_tensor(self.low, dtype=self.mean.dtype, device=self.mean.device)
        high = torch.as_tensor(self.high, dtype=self.mean.dtype, device=self.mean.device)

        latent_nodes_for_batch = latent_nodes.view(-1, *([1] * batch_ndim)).unsqueeze(-1)
        upper = (high - mean - factor * latent_nodes_for_batch) / std
        lower = (low - mean - factor * latent_nodes_for_batch) / std
        log_cube_prob = self._log_normal_interval_probability(lower, upper).sum(dim=-1)
        log_latent_weights = log_latent_weights.view(-1, *([1] * batch_ndim))
        latent_logits = (log_latent_weights + log_cube_prob).movedim(0, -1)

        latent_indices = torch.distributions.Categorical(logits=latent_logits).sample((num_samples,))
        selected_latent = latent_nodes[latent_indices].unsqueeze(-1)

        conditional_mean = self.mean.unsqueeze(0) + self.cov_factor.squeeze(-1).unsqueeze(0) * selected_latent
        conditional_std = self.cov_diag.sqrt().unsqueeze(0)
        conditional_lower = (low - conditional_mean) / conditional_std
        conditional_upper = (high - conditional_mean) / conditional_std
        lower_cdf = self._standard_normal_cdf(conditional_lower)
        upper_cdf = self._standard_normal_cdf(conditional_upper)
        uniform = torch.rand_like(lower_cdf)
        cdf_sample = (lower_cdf + uniform * (upper_cdf - lower_cdf)).clamp(
            torch.finfo(self.mean.dtype).eps,
            1.0 - torch.finfo(self.mean.dtype).eps,
        )
        standard_sample = math.sqrt(2.0) * torch.erfinv(cdf_sample.mul(2.0).sub(1.0))
        return (conditional_mean + conditional_std * standard_sample).clamp(self.low, self.high)

    def sample(self, num_samples=1, max_attempts=1024):
        if num_samples < 1:
            raise ValueError('num_samples must be at least 1')
        if self.cov_factor is not None:
            return self._sample_low_rank(num_samples)

        samples = []
        attempts = 0
        while len(samples) < num_samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    'Unable to draw a sample inside the truncation range; '
                    'use a low-rank covariance or increase max_attempts.'
                )
            sample = self.multivariate_normal_dist.sample()
            if self.is_within_truncation_range(sample):
                samples.append(sample)
        return torch.stack(samples)

    def is_within_truncation_range(self, sample):
        return torch.all((sample >= -1) & (sample <= 1))

    def find_first_valid_matrix(self, tensor):
        valid_mask = torch.all((tensor >= -1) & (tensor <= 1), dim=(1, 2))
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        if valid_indices.numel() == 0:
            return None
        first_valid_index = valid_indices[0].item()
        return tensor[first_valid_index]


class My_Multinomial:
    def __init__(self, probs):
        self.probs = probs
        self.len_shape = len(probs.shape)
        self.shape = self.probs.shape

    def sample(self, num_samples=20):
        props = self.probs.reshape(-1, self.shape[-1])
        samples = torch.multinomial(props, num_samples, replacement=True)
        return samples.reshape(self.shape[:-1] + (num_samples,))

    def factorial_tensor(self, n):
        return torch.lgamma(n + 1)

    def combination(self, summ, numm):
        a_sum = torch.zeros_like(summ, dtype=torch.float32, device=summ.device)
        a_sum[summ == 0] = -10000000000.
        return self.factorial_tensor(summ) - torch.sum(self.factorial_tensor(numm), dim=-1, keepdim=True)

    def log_prob(self, x):
        summ = x[..., -1:]
        numm = x[..., :-1]
        y1 = self.combination(summ, numm)
        y2 = torch.log(self.probs.clamp_min(torch.finfo(self.probs.dtype).tiny))

        y3 = y2 * numm
        out = y1 + torch.sum(y3, dim=-1, keepdim=True)
        out = torch.sum(out.squeeze(dim=-1), dim=-1, keepdim=False)
        # print(out.shape)
        return out

    def prob(self, x):
        return self.log_prob(x).exp()


class My_TruncatedNormal:
    def __init__(self, loc, scale, low=-1.0, high=1.0):
        self.loc = loc
        self.scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
        self.low = low
        self.high = high
        self.normal = torch.distributions.Normal(self.loc, self.scale)

    def log_prob(self, value):
        low = torch.as_tensor(self.low, dtype=value.dtype, device=value.device)
        high = torch.as_tensor(self.high, dtype=value.dtype, device=value.device)
        log_prob = self.normal.log_prob(value)
        normalizer = (self.normal.cdf(high) - self.normal.cdf(low)).clamp_min(torch.finfo(value.dtype).tiny)
        log_prob = log_prob - torch.log(normalizer)
        in_range = (value >= low) & (value <= high)
        log_prob = torch.where(in_range, log_prob, torch.full_like(log_prob, -torch.inf))
        return log_prob.sum(dim=(-1, -2))

    @property
    def mean(self):
        calculation_dtype = torch.float64 if self.loc.dtype in (
            torch.float16, torch.bfloat16, torch.float32
        ) else self.loc.dtype
        loc = self.loc.to(calculation_dtype)
        scale = self.scale.to(calculation_dtype)
        low = torch.as_tensor(self.low, dtype=calculation_dtype, device=loc.device)
        high = torch.as_tensor(self.high, dtype=calculation_dtype, device=loc.device)
        lower = (low - loc) / scale
        upper = (high - loc) / scale

        log_normalizer = My_Multinormal._log_normal_interval_probability(lower, upper)
        log_constant = 0.5 * math.log(2.0 * math.pi)
        log_pdf_lower = -0.5 * lower.square() - log_constant
        log_pdf_upper = -0.5 * upper.square() - log_constant
        larger = torch.maximum(log_pdf_lower, log_pdf_upper)
        smaller = torch.minimum(log_pdf_lower, log_pdf_upper)
        log_abs_difference = larger + torch.log(
            (-torch.expm1(smaller - larger)).clamp_min(torch.finfo(calculation_dtype).tiny)
        )
        correction = torch.sign(log_pdf_lower - log_pdf_upper) * torch.exp(
            log_abs_difference - log_normalizer
        )
        both_pdf_zero = torch.isneginf(log_pdf_lower) & torch.isneginf(log_pdf_upper)
        correction = torch.where(both_pdf_zero, torch.zeros_like(correction), correction)
        return (loc + scale * correction).clamp(min=low, max=high).to(self.loc.dtype)

    def sample(self):
        low = torch.as_tensor(self.low, dtype=self.loc.dtype, device=self.loc.device)
        high = torch.as_tensor(self.high, dtype=self.loc.dtype, device=self.loc.device)
        cdf_low = self.normal.cdf(low)
        cdf_high = self.normal.cdf(high)
        eps = torch.finfo(self.loc.dtype).eps
        probability = cdf_low + torch.rand_like(self.loc) * (cdf_high - cdf_low)
        probability = probability.clamp(eps, 1.0 - eps)
        return self.normal.icdf(probability).clamp(min=low, max=high)


class Gated_Dynamic_Connection(nn.Module):
    def __init__(self, num_gate, d_k, d_model, noGC=False):
        super().__init__()
        self.noGC = noGC
        if not noGC:
            self.Weight1 = nn.Parameter(torch.Tensor(num_gate, d_k, d_model))
            self.Weight2 = nn.Parameter(torch.Tensor(num_gate, d_k, d_model))
            nn.init.xavier_normal_(self.Weight1)
            nn.init.xavier_normal_(self.Weight2)
        else:
            self.Linear = nn.Linear(d_k * num_gate, d_model)

    def forward(self, data):
        if not self.noGC:
            if len(data.shape) == 4:
                data = data.permute(1, 2, 0, 3).unsqueeze(-2)
            elif len(data.shape) == 5:
                data = data.permute(1, 2, 3, 0, 4).unsqueeze(-2)
            elif len(data.shape) == 6:
                data = data.permute(1, 2, 3, 4, 0, 5).unsqueeze(-2)
            else:
                raise ValueError(f"Unsupported gated connection input rank: {len(data.shape)}")

            data_out = torch.matmul(data, self.Weight1).transpose(-1, -3)
            data_softmax = nonan_softmax(
                F.relu(torch.matmul(data, self.Weight2)).transpose(-1, -3),
                dim=-1,
            ).transpose(-1, -2)
            return torch.matmul(data_out, data_softmax).squeeze(dim=-1).squeeze(dim=-1)

        if len(data.shape) == 5:
            _, batch_size, seq_len, num_nodes, _ = data.shape
            data = data.reshape(batch_size, seq_len, num_nodes, -1)
            return self.Linear(data)

        raise ValueError(f"Unsupported linear gated connection input rank: {len(data.shape)}")


def split_temporal_embeddings(embedding_list):
    num_embeddings = []
    index_embeddings = []
    index_feature = []
    for i, item in enumerate(embedding_list):
        if item == 1:
            index_feature.append(i)
        if item > 1:
            num_embeddings.append(item)
            index_embeddings.append(i)
    return num_embeddings, index_embeddings, index_feature


def fourier_encode(features, levels):
    frequencies = torch.pow(
        2.0,
        torch.arange(levels, dtype=features.dtype, device=features.device),
    ) * math.pi
    angles = features.unsqueeze(-1) * frequencies
    return torch.cat([features, torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)], dim=-1)


class PositionalTemporalEmbedding(nn.Module):
    def __init__(self, temporal_embedding_list, d_model, noGC=False, max_len=36):
        super().__init__()
        self.num_embeddings, self.index_embeddings, self.index_feature = split_temporal_embeddings(
            temporal_embedding_list,
        )
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
        self.register_buffer('pe', pe)

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in self.num_embeddings])
        if len(self.index_feature) > 0:
            self.feature_tran = nn.Linear(len(self.index_feature), d_model)
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 2, d_model, d_model, noGC)
        else:
            self.gat = Gated_Dynamic_Connection(len(self.index_embeddings) + 1, d_model, d_model, noGC)

    def embeddings(self, embedding_list):
        return split_temporal_embeddings(embedding_list)

    def forward(self, temporal_characterisrics):
        extras_onehot = temporal_characterisrics[self.index_embeddings].to(torch.int32)
        batch_size, seq_len = extras_onehot[0].shape

        pe = self.pe[:seq_len, :].expand(batch_size, seq_len, self.d_model)

        if len(self.index_feature) > 0:
            extras_feature = temporal_characterisrics[self.index_feature].permute(1, 2, 0)
            extras_feature = self.feature_tran(extras_feature)
            extras_embedding = torch.stack(
                [self.embedding_modules[i](extras_onehot[i]) for i in range(len(self.num_embeddings))]
                + [extras_feature, pe],
                dim=0,
            ).unsqueeze(dim=2)
        else:
            extras_embedding = torch.stack(
                [self.embedding_modules[i](extras_onehot[i]) for i in range(len(self.num_embeddings))] + [pe],
                dim=0,
            ).unsqueeze(dim=2)

        return self.gat(extras_embedding).squeeze(dim=1)


class FlowTemporalEmbedding(nn.Module):
    def __init__(self, temporal_embedding_list, d_model, noGC=False, log_features=True,
                 fourier_feature_indices=None, fourier_levels=0):
        super(FlowTemporalEmbedding, self).__init__()
        self.num_embeddings, self.index_embeddings, self.index_feature = split_temporal_embeddings(
            temporal_embedding_list,
        )
        self.fourier_feature_indices = list(fourier_feature_indices or [])
        self.regular_feature_indices = [
            index for index in self.index_feature if index not in self.fourier_feature_indices
        ]
        self.d_model = d_model
        self.log_features = log_features

        self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in self.num_embeddings])
        if self.regular_feature_indices:
            self.feature_tran = nn.Linear(len(self.regular_feature_indices), d_model)
        if self.fourier_feature_indices:
            self.fourier_levels = fourier_levels
            fourier_dim = len(self.fourier_feature_indices) * (1 + 2 * fourier_levels)
            self.fourier_tran = nn.Linear(fourier_dim, d_model)
        num_inputs = len(self.index_embeddings) + bool(self.regular_feature_indices) + bool(self.fourier_feature_indices)
        self.gat = Gated_Dynamic_Connection(num_inputs, d_model, d_model, noGC)

    def forward(self, temporal_characterisrics):
        extras_onehot = temporal_characterisrics[..., self.index_embeddings].to(torch.int32)
        if temporal_characterisrics.dim() == 2:
            extras_onehot = extras_onehot.transpose(0, 1)
        else:
            extras_onehot = extras_onehot.permute(2, 0, 1)
        categorical_embeddings = [
            self.embedding_modules[i](extras_onehot[i]) for i in range(len(self.num_embeddings))
        ]

        extras = list(categorical_embeddings)
        if self.regular_feature_indices:
            extras_feature = temporal_characterisrics[..., self.regular_feature_indices]
            if self.log_features:
                extras_feature = torch.sign(extras_feature) * torch.log1p(torch.abs(extras_feature))
            extras.append(self.feature_tran(extras_feature))
        if self.fourier_feature_indices:
            location = temporal_characterisrics[..., self.fourier_feature_indices]
            extras.append(self.fourier_tran(fourier_encode(location, self.fourier_levels)))
        extras_embedding = torch.stack(extras, dim=0).unsqueeze(dim=2)
        output = self.gat(extras_embedding).squeeze(dim=1)

        return output


class FlowSpatialEmbedding(nn.Module):
    def __init__(self, spatial_embedding_list, d_model, eigenmaps_k, noGC=False, log_features=True,
                 location_feature_indices=None, location_fourier_levels=0):
        super(FlowSpatialEmbedding, self).__init__()
        self.num_embeddings, self.index_embeddings, self.index_feature = split_temporal_embeddings(
            spatial_embedding_list,
        )
        self.d_model = d_model
        self.eigenmaps_k = eigenmaps_k
        self.log_features = log_features
        self.location_feature_indices = list(location_feature_indices or [])
        self.regular_feature_indices = [
            index for index in self.index_feature if index not in self.location_feature_indices
        ]

        self.eigenmaps_k_trans = nn.Linear(eigenmaps_k, d_model)
        num_gat_feature = 1
        if len(self.num_embeddings) > 0:
            self.embedding_modules = nn.ModuleList([nn.Embedding(item, d_model) for item in self.num_embeddings])
            num_gat_feature += len(self.num_embeddings)
        if self.regular_feature_indices:
            self.feature_tran = nn.Linear(len(self.regular_feature_indices), d_model)
            num_gat_feature += 1
        if self.location_feature_indices:
            self.location_fourier_levels = location_fourier_levels
            location_dim = len(self.location_feature_indices) * (1 + 2 * location_fourier_levels)
            self.location_tran = nn.Linear(location_dim, d_model)
            num_gat_feature += 1
        self.gat = Gated_Dynamic_Connection(num_gat_feature, d_model, d_model, noGC)

    def forward(self, spatial_characterisrics, eigenmaps):
        '''

        :param eigenmaps: shape[N,eigenmaps_k]
        :return: u:shape[N,d_model]
        '''

        eigenmap_embedding = self.eigenmaps_k_trans(eigenmaps)
        if spatial_characterisrics is not None:
            extras_list = [eigenmap_embedding]
            if self.regular_feature_indices:
                extras_feature = spatial_characterisrics[:, self.regular_feature_indices]
                extras_feature = self.feature_tran(extras_feature)
                extras_list.append(extras_feature)
            if self.location_feature_indices:
                location = spatial_characterisrics[:, self.location_feature_indices]
                extras_list.append(
                    self.location_tran(fourier_encode(location, self.location_fourier_levels))
                )
            if len(self.index_embeddings) > 0:
                extras_onehot = spatial_characterisrics[:, self.index_embeddings].to(torch.int32)
                extras_list += [self.embedding_modules[i](extras_onehot[:, i]) for i in range(len(self.num_embeddings))]
            extras_embedding = torch.stack(extras_list, dim=0).unsqueeze(dim=1).unsqueeze(dim=1)
            spatial_embedding = self.gat(extras_embedding).squeeze(dim=0).squeeze(dim=0)

        else:
            spatial_embedding = eigenmap_embedding

        return spatial_embedding


class FlowSpatialTemporalEmbedding(nn.Module):
    def __init__(self, temporal_embedding_list, spatial_embedding_list, d_model, num_nodes, eigenmaps_k, noGC=False,
                 log_features=True, noTE=False, noSE=False, location_feature_indices=None,
                 location_fourier_levels=0, node_temporal_dim=0):
        super(FlowSpatialTemporalEmbedding, self).__init__()
        self.noTE = noTE
        self.noSE = noSE
        self.d_model = d_model
        self.N = num_nodes
        self.node_temporal_dim = node_temporal_dim
        if not noTE:
            self.TF = FlowTemporalEmbedding(temporal_embedding_list, d_model, noGC, log_features)
        if not noSE:
            self.SF = FlowSpatialEmbedding(
                spatial_embedding_list, d_model, eigenmaps_k, noGC, log_features,
                location_feature_indices, location_fourier_levels,
            )
        if node_temporal_dim:
            self.node_temporal_tran = nn.Linear(node_temporal_dim, d_model)
        num_contexts = int(not noTE) + int(not noSE) + int(bool(node_temporal_dim))
        if num_contexts > 1:
            self.gat = Gated_Dynamic_Connection(num_contexts, d_model, d_model, noGC=noGC)

    def forward(self, temporal_characterisrics, spatial_characterisrics, eigenmaps, node_temporal_characteristics=None):
        B, T, _ = temporal_characterisrics.shape
        N = self.N
        contexts = []
        if not self.noTE:
            TE = self.TF(temporal_characterisrics)
            TE = TE.unsqueeze(dim=2).expand(B, T, N, self.d_model)
            contexts.append(TE)
        if not self.noSE:
            SE = self.SF(spatial_characterisrics, eigenmaps)
            SE = SE.expand(B, T, N, self.d_model)
            contexts.append(SE)
        if self.node_temporal_dim:
            contexts.append(self.node_temporal_tran(node_temporal_characteristics))
        STE = self.gat(torch.stack(contexts, dim=0)) if len(contexts) > 1 else contexts[0]

        return STE


class MetaLearner(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param num_weight_matrices: 生产转换矩阵数量，自注意力为3
        '''
        super(MetaLearner, self).__init__()
        self.num_weight_matrices = num_weight_matrices
        self.num_heads = num_heads
        self.d_k = d_k

        self.linear1 = nn.Linear(d_model, d_hidden_mt)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_hidden_mt, num_weight_matrices * num_heads * d_k * d_model)

    def forward(self, c_inputs):
        '''

        :param c_inputs: shape[B,P,N,d_model]
        :return: W shape[3,B,P,N,H,d_k,d_model]
        '''
        B, P, N, d_model = c_inputs.shape
        out = self.relu(self.linear1(c_inputs))
        out = self.linear2(out)
        out = out.reshape((B, P, N, self.num_weight_matrices, self.num_heads, self.d_k, d_model))
        out = out.permute((3, 0, 1, 2, 4, 5, 6))  # (num_weight_matrices, B, P, N, num_heads, d_k, d_model

        return out


class Adaptive_Graph(nn.Module):
    def __init__(self, d_model, d_hidden_gm, top_k):
        super(Adaptive_Graph, self).__init__()
        self.top_k = top_k
        self.weight = nn.Parameter(torch.Tensor(2, d_model, d_hidden_gm))
        nn.init.xavier_normal_(self.weight)

    def forward(self, c_input):
        A1 = torch.matmul(c_input, self.weight[0])
        # print(A1.shape)
        A2 = torch.matmul(c_input, self.weight[1]).transpose(2, 3)
        gen_graph = torch.matmul(A1, A2)
        gen_graph = self.top_kf(F.relu(gen_graph), self.top_k)
        gen_graph = nozero_softmax(gen_graph, dim=-1)
        return gen_graph

    def top_kf(self, matrix, top_k):
        top_k_per_row = matrix.topk(top_k, dim=3)[0]
        min = top_k_per_row[:, :, :, top_k - 1:top_k]
        matrix2 = matrix.clone()
        matrix2[matrix < min] = 0.
        return matrix2


class Swish(nn.Module):
    def __init__(self, d_model):
        super(Swish, self).__init__()
        self.linear_wg = nn.Linear(d_model, d_model)
        self.linear_wo = nn.Linear(d_model, d_model)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, inputs, out):
        out = self.linear_wo(self.swish(self.linear_wg(inputs) * out))

        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_hidden_mt, num_heads, which_transition_matrices, dropout, noML, noAG,
                 noGC):
        '''

        :param d_model:
        :param d_k:
        :param d_hidden_mt:
        :param num_heads:
        :param which_transition_matrices: list,len=特征图数量，[是否采用的意思]
        :param dropout:
        :param noML:
        '''
        super(SpatialSelfAttention, self).__init__()
        if "get_para":
            self.which_transition_matrices = which_transition_matrices
            self.num_transition_matrices = sum(which_transition_matrices)
            assert self.num_transition_matrices > 0
            self.noML = noML
            self.noAG = noAG
            if self.noAG:
                hid = self.num_transition_matrices
            else:
                hid = self.num_transition_matrices + 1
        if self.noML:
            self.num_heads = num_heads
            self.d_k = d_k
            self.linear_q = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
            self.linear_k = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
            self.linear_v = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                           for _ in range(hid)])
        else:
            self.meta_learners = nn.ModuleList([MetaLearner(
                d_model, d_k, d_hidden_mt, num_heads, num_weight_matrices=3)
                for _ in range(hid)])
        self.gat1 = nn.ModuleList(
            [Gated_Dynamic_Connection(num_gate=num_heads, d_k=d_k, d_model=d_model, noGC=noGC) for _ in
             range(hid)])
        self.gat2 = Gated_Dynamic_Connection(num_gate=hid, d_k=d_model, d_model=d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.swish = Swish(d_model)

    def forward(self, inputs, c_inputs, transition_matrices, adaptive_graph):
        '''

        :param inputs: shape[B,P,N,d_model]
        :param c_inputs: shape[B,P,N,d_model]
        :param transition_matrices: 转换矩阵，与图数量相同[num_maps,N,N]
        :return: shape[B,P,N,d_model]
        '''

        assert transition_matrices.shape[0] == len(self.which_transition_matrices)
        transition_matrices, graph_indices = predefine_graph_data(
            transition_matrices, tuple(self.which_transition_matrices),
        )

        out = []
        fu = []
        for i in range(self.num_transition_matrices):
            fu.append(jit.fork(self.fork, i, inputs, c_inputs, transition_matrices, graph_indices[i]))
        if not self.noAG:
            fu.append(jit.fork(self.forkgg, inputs, c_inputs, adaptive_graph))
        for i in fu:
            i = jit.wait(i)
            out.append(i)

        out = torch.stack(out, dim=0)
        out = self.gat2(out)
        out = self.dropout(out)
        out = self.swish(inputs, out)
        out = self.layer_norm(out + inputs)  # (B, P, N, d_model)

        return out

    def fork(self, i, inputs, c_inputs, transition_matrices, graph_index):
        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v[i](inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learners[i](c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        mid_result = multihead_spatial_attention_predefine_graph(Q, K, V, transition_matrices[i], graph_index)
        return self.gat1[i](mid_result)

    def forkgg(self, inputs, c_inputs, adaptive_graph):

        if self.noML:
            B, P, N, _ = inputs.shape
            Q = self.linear_q[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            K = self.linear_k[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
            V = self.linear_v[-1](inputs).reshape((B, P, N, self.num_heads, self.d_k))
        else:
            W_q, W_k, W_v = self.meta_learners[-1](c_inputs)  # (B, P, N, H, d_k, d_model)
            Q = multihead_linear_transform(W_q, inputs)  # (B, P, N, H, d_k)
            K = multihead_linear_transform(W_k, inputs)  # (B, P, N, H, d_k)
            V = multihead_linear_transform(W_v, inputs)  # (B, P, N, H, d_k)
        mid_result = multihead_spatial_attention_adaptive_graph(Q, K, V, adaptive_graph)
        return self.gat1[-1](mid_result)


class LearnableMixtureOfGaussians(Distribution):
    def __init__(self, num_components, dim, min_log_std=-3.0, max_log_std=1.0):
        super(LearnableMixtureOfGaussians, self).__init__()
        self.num_components = num_components
        self.dim = dim
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.logit_weights = nn.Parameter(torch.zeros(num_components))
        self.means = nn.Parameter(torch.randn(num_components, dim))
        self.log_stds = nn.Parameter(torch.zeros(num_components, dim))

    def _stds(self):
        return torch.exp(torch.clamp(self.log_stds, self.min_log_std, self.max_log_std))

    def _log_prob(self, inputs, context=None):
        stds = self._stds()
        inputs = inputs.unsqueeze(1)
        means = self.means.unsqueeze(0)
        stds = stds.unsqueeze(0)
        centered = (inputs - means) / stds
        log_component_gaussians = -0.5 * (
                torch.sum(centered ** 2, dim=-1) + torch.sum(torch.log(2 * torch.pi * stds ** 2), dim=-1)
        )
        log_weights = F.log_softmax(self.logit_weights, dim=0)
        return torch.logsumexp(log_weights + log_component_gaussians, dim=1)

    def _sample(self, num_samples, context=None):
        with torch.no_grad():
            context_size = 1 if context is None else context.shape[0]
            weights = F.softmax(self.logit_weights, dim=0)
            component_indices = torch.multinomial(weights, num_samples, replacement=True)
            means = self.means[component_indices]
            stds = self._stds()[component_indices]
            samples = torch.randn(num_samples * context_size, self.dim, device=self.means.device)
            samples = samples.reshape(context_size, num_samples, self.dim)
            samples = samples * stds + means
        return samples


class ContextualAffineCouplingTransform1D(nn.Module):
    def __init__(
            self,
            hidden_dim,
            context_dim,
            num_features=1,
            positive_output_eps=False,
            log_scale_bound=5.0,
            eps=1e-30,
    ):
        super(ContextualAffineCouplingTransform1D, self).__init__()
        self.num_features = num_features
        self.positive_output_eps = positive_output_eps
        self.log_scale_bound = float(log_scale_bound)
        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * num_features),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _shift_and_log_scale(self, context):
        shift_and_scale = self.net(context)
        shift, log_scale = shift_and_scale.split(self.num_features, dim=-1)
        log_scale = self.log_scale_bound * torch.tanh(log_scale / self.log_scale_bound)
        return shift, log_scale

    def _affine_transform(self, inputs, shift, log_scale):
        outputs = inputs * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=-1)
        return outputs, log_det

    def _inverse_affine_transform(self, inputs, shift, log_scale):
        outputs = (inputs - shift) / torch.exp(log_scale)
        log_det = -log_scale.sum(dim=-1)
        return outputs, log_det

    def forward(self, inputs, context=None):
        if context is None:
            raise ValueError("Context cannot be None.")
        shift, log_scale = self._shift_and_log_scale(context)
        return self._affine_transform(inputs, shift, log_scale)

    def inverse(self, inputs, context=None):
        if context is None:
            raise ValueError("Context cannot be None.")
        shift, log_scale = self._shift_and_log_scale(context)
        return self._inverse_affine_transform(inputs, shift, log_scale)
