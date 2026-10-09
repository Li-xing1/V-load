# -*- coding: utf-8 -*-
# @Time : 2026/3/6 13:37
# @Author : Xing Li
# @Email : 2434147605@qq.com
# @File : V_V.py
# @Project : V_load

import yaml

from models.common import *


class NET_A(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.d_model = cfgs['d_model']
        self.num_hidden = cfgs['num_hidden_A']
        self.d_k = cfgs['d_k'] if 'd_k' in cfgs else cfgs['d_K']
        self.num_heads = cfgs['num_heads']
        self.dropout = cfgs['dropout']
        self.num_nodes = cfgs['num_nodes']
        self.num_hidden_layer = cfgs['num_hidden_layer_A']
        which_transition_matrices = cfgs['which_transition_matrices']
        d_hidden_mt = cfgs['d_hidden_mt']
        dropout = cfgs['dropout']
        noML = cfgs.get('noML', False)
        noAG = cfgs.get('noAG', False)
        noGC = cfgs.get('noGC', False)
        self.SSA1 = SpatialSelfAttention(d_model=self.d_model, d_k=self.d_k, d_hidden_mt=d_hidden_mt,
                                         num_heads=self.num_heads,
                                         which_transition_matrices=which_transition_matrices, dropout=dropout,
                                         noML=noML,
                                         noAG=noAG,
                                         noGC=noGC)
        self.SSA2 = SpatialSelfAttention(d_model=self.d_model, d_k=self.d_k, d_hidden_mt=d_hidden_mt,
                                         num_heads=self.num_heads,
                                         which_transition_matrices=which_transition_matrices, dropout=dropout,
                                         noML=noML,
                                         noAG=noAG,
                                         noGC=noGC)

        self.fun = nn.Tanh()

    def forward(self, x, c, transition_matrices, AG):
        num_samples = x.shape[0]
        B, N, d_model = c.shape
        x = x.expand(num_samples, B, N, d_model)
        c = c.unsqueeze(0).expand(num_samples, B, N, d_model)

        out1 = self.SSA1(x, c, transition_matrices, AG)
        out2 = self.SSA2(x, c, transition_matrices, AG)
        out = self.fun(torch.stack([out1, out2], dim=0))

        return out


class NET_B(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.d_model = cfgs['d_model']
        self.L = cfgs['num_lines']
        self.num_hidden = cfgs['num_hidden_B']
        self.num_hidden_layer_B = cfgs['num_hidden_layer_B']
        self.min_scale = float(cfgs.get('min_obs_scale', 1e-3))
        self.correlation_margin = float(cfgs.get('correlation_margin', 1e-4))
        meta_hidden = cfgs['d_hidden_mt']
        meta_rank = cfgs.get('emission_meta_rank', 4)
        self.linear_list = nn.ModuleList(
            [ContextualLinear(self.d_model, self.num_hidden, self.d_model, meta_hidden, meta_rank)]
            + [ContextualLinear(self.num_hidden, self.num_hidden, self.d_model, meta_hidden, meta_rank)
               for _ in range(self.num_hidden_layer_B)]
            + [ContextualLinear(self.num_hidden, self.L * 5, self.d_model, meta_hidden, meta_rank)]
        )
        self.activation = nn.SiLU()

    def forward(self, Z, context):

        num_samples, _, num_nodes, _ = Z.shape
        batch_size = context.shape[0]
        out = Z.expand(num_samples, batch_size, num_nodes, -1)
        for index, lay in enumerate(self.linear_list):
            out = lay(out, context)
            if index < len(self.linear_list) - 1:
                out = self.activation(out)
        out = out.reshape(num_samples, batch_size, num_nodes, self.L, 5)
        loc = out[..., 0:2]
        scale = F.softplus(out[..., 2:4]) + self.min_scale
        correlation = torch.tanh(out[..., 4:5]) * (1.0 - self.correlation_margin)

        return loc, scale, correlation


def pair_observation_mask(observations, mask=None):
    expected_shape = observations.shape[:-1] + (1,)
    if mask is None:
        return torch.ones(expected_shape, dtype=torch.bool, device=observations.device)
    if mask.shape != expected_shape:
        raise ValueError(
            f"V_V mask must contain one shared speed-flow pair channel with shape "
            f"{expected_shape}, got {tuple(mask.shape)}."
        )
    return mask.to(dtype=torch.bool)


@lru_cache(maxsize=32)
def observation_quadrature(num_points, device, dtype):
    nodes, weights = gauss_legendre_node_gen_1d(
        num_points, device=device, dtype=dtype, low=0.0, high=1.0
    )
    return nodes, torch.log(weights)


def truncated_bivariate_log_prob(value, loc, scale, correlation, low, high,
                                   quadrature_points=16):
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    correlation = correlation.squeeze(-1)
    one_minus_rho2 = (1.0 - correlation.square()).clamp_min(torch.finfo(scale.dtype).eps)
    low = torch.as_tensor(low, dtype=loc.dtype, device=loc.device)
    high = torch.as_tensor(high, dtype=loc.dtype, device=loc.device)
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError("Bivariate observation bounds must each contain two values.")
    if not torch.isfinite(low).all() or not torch.isfinite(high[0]):
        raise ValueError("Observation lower bounds and the speed upper bound must be finite.")

    standardized = (value.unsqueeze(0) - loc) / scale
    z_speed, z_flow = standardized.unbind(dim=-1)
    quadratic = (
        z_speed.square() - 2.0 * correlation * z_speed * z_flow + z_flow.square()
    ) / one_minus_rho2
    log_density = (
        -math.log(2.0 * math.pi)
        - torch.log(scale).sum(dim=-1)
        - 0.5 * torch.log(one_minus_rho2)
        - 0.5 * quadratic
    )

    speed_low = (low[0] - loc[..., 0]) / scale[..., 0]
    speed_high = (high[0] - loc[..., 0]) / scale[..., 0]
    flow_low = (low[1] - loc[..., 1]) / scale[..., 1]
    flow_has_no_upper_bound = bool(torch.isposinf(high[1]).item())
    if flow_has_no_upper_bound:
        flow_high = torch.full_like(flow_low, torch.inf)
    else:
        flow_high = (high[1] - loc[..., 1]) / scale[..., 1]
    width = (speed_high - speed_low).clamp_min(torch.finfo(loc.dtype).eps)

    nodes, log_weights = observation_quadrature(
        quadrature_points, loc.device, loc.dtype
    )
    view_shape = (quadrature_points,) + (1,) * speed_low.ndim
    nodes = nodes.view(view_shape)
    log_weights = log_weights.view(view_shape)
    speed_node = speed_low.unsqueeze(0) + width.unsqueeze(0) * nodes
    conditional_std = one_minus_rho2.sqrt().unsqueeze(0)
    conditional_mean = correlation.unsqueeze(0) * speed_node
    conditional_low = (flow_low.unsqueeze(0) - conditional_mean) / conditional_std
    if flow_has_no_upper_bound:
        conditional_high = torch.full_like(conditional_low, torch.inf)
    else:
        conditional_high = (flow_high.unsqueeze(0) - conditional_mean) / conditional_std
    log_flow_probability = My_Multinormal._log_normal_interval_probability(
        conditional_low, conditional_high
    )
    log_standard_speed_density = -0.5 * (
        speed_node.square() + math.log(2.0 * math.pi)
    )
    log_normalizer = torch.log(width) + torch.logsumexp(
        log_weights + log_standard_speed_density + log_flow_probability, dim=0
    )

    in_range = ((value >= low) & (value <= high)).all(dim=-1).unsqueeze(0)
    log_prob = log_density - log_normalizer
    return torch.where(in_range, log_prob, torch.full_like(log_prob, -torch.inf))


def _sample_truncated_standard_normal(lower, upper, max_attempts):
    lower, upper = torch.broadcast_tensors(lower, upper)
    result = torch.empty_like(lower)
    pending = torch.ones_like(lower, dtype=torch.bool)
    uniform_threshold = math.sqrt(2.0 * math.pi)

    for _ in range(max_attempts):
        right_tail = pending & (lower >= 0.0)
        left_tail = pending & (upper <= 0.0)
        tail = right_tail | left_tail
        if tail.any():
            tail_lower = torch.where(right_tail, lower, -upper)
            tail_upper = torch.where(right_tail, upper, -lower)
            rate = 0.5 * (tail_lower + torch.sqrt(tail_lower.square() + 4.0))
            interval_mass = -torch.expm1(-rate * (tail_upper - tail_lower))
            proposal = tail_lower - torch.log1p(-torch.rand_like(lower) * interval_mass) / rate
            accepted = tail & (
                torch.log(torch.rand_like(lower)) <= -0.5 * (proposal - rate).square()
            )
            result[accepted] = torch.where(right_tail, proposal, -proposal)[accepted]
            pending = pending & ~accepted

        central = pending & (lower < 0.0) & (upper > 0.0)
        narrow = central & ((upper - lower) < uniform_threshold)
        if narrow.any():
            proposal = lower + torch.rand_like(lower) * (upper - lower)
            accepted = narrow & (
                torch.log(torch.rand_like(lower)) <= -0.5 * proposal.square()
            )
            result[accepted] = proposal[accepted]
            pending = pending & ~accepted

        wide = pending & (lower < 0.0) & (upper > 0.0)
        if wide.any():
            proposal = torch.randn_like(lower)
            accepted = wide & (proposal >= lower) & (proposal <= upper)
            result[accepted] = proposal[accepted]
            pending = pending & ~accepted

        if not pending.any():
            return result

    raise RuntimeError(
        f"Unable to sample {int(pending.sum())} truncated normal observations "
        f"after {max_attempts} attempts."
    )


def sample_truncated_bivariate(loc, scale, correlation, low, high, num_gibbs_steps=32):
    output_dtype = loc.dtype
    calculation_dtype = torch.float64 if loc.dtype in (
        torch.float16, torch.bfloat16, torch.float32
    ) else loc.dtype
    loc = loc.to(calculation_dtype)
    scale = scale.to(calculation_dtype).clamp_min(torch.finfo(calculation_dtype).eps)
    correlation = correlation.squeeze(-1).to(calculation_dtype)
    low = torch.as_tensor(low, dtype=calculation_dtype, device=loc.device)
    high = torch.as_tensor(high, dtype=calculation_dtype, device=loc.device)
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError("Bivariate observation bounds must each contain two values.")

    conditional_scale = (1.0 - correlation.square()).clamp_min(
        torch.finfo(scale.dtype).eps
    ).sqrt()
    speed_lower = (low[0] - loc[..., 0]) / scale[..., 0]
    speed_upper = (high[0] - loc[..., 0]) / scale[..., 0]
    flow_lower = (low[1] - loc[..., 1]) / scale[..., 1]
    flow_upper = (high[1] - loc[..., 1]) / scale[..., 1]
    speed_noise = torch.maximum(
        torch.minimum(torch.zeros_like(speed_lower), speed_upper), speed_lower
    )
    flow_noise = torch.maximum(
        torch.minimum(correlation * speed_noise, flow_upper), flow_lower
    )

    for _ in range(num_gibbs_steps):
        speed_conditional_mean = correlation * flow_noise
        speed_standard = _sample_truncated_standard_normal(
            (speed_lower - speed_conditional_mean) / conditional_scale,
            (speed_upper - speed_conditional_mean) / conditional_scale,
            max_attempts=1024,
        )
        speed_noise = speed_conditional_mean + conditional_scale * speed_standard

        flow_conditional_mean = correlation * speed_noise
        flow_standard = _sample_truncated_standard_normal(
            (flow_lower - flow_conditional_mean) / conditional_scale,
            (flow_upper - flow_conditional_mean) / conditional_scale,
            max_attempts=1024,
        )
        flow_noise = flow_conditional_mean + conditional_scale * flow_standard

    result = torch.stack([
        loc[..., 0] + scale[..., 0] * speed_noise,
        loc[..., 1] + scale[..., 1] * flow_noise,
    ], dim=-1)
    return torch.maximum(torch.minimum(result, high), low).to(output_dtype)


def truncated_bivariate_mean(loc, scale, correlation, low, high, quadrature_points=16):
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    correlation = correlation.squeeze(-1)
    low = torch.as_tensor(low, dtype=loc.dtype, device=loc.device)
    high = torch.as_tensor(high, dtype=loc.dtype, device=loc.device)
    one_minus_rho2 = (1.0 - correlation.square()).clamp_min(torch.finfo(scale.dtype).eps)
    speed_low = (low[0] - loc[..., 0]) / scale[..., 0]
    speed_high = (high[0] - loc[..., 0]) / scale[..., 0]
    flow_low = (low[1] - loc[..., 1]) / scale[..., 1]
    flow_high = torch.full_like(flow_low, torch.inf) if torch.isposinf(high[1]) else (
        high[1] - loc[..., 1]
    ) / scale[..., 1]

    nodes, log_weights = observation_quadrature(
        quadrature_points, loc.device, loc.dtype
    )
    view_shape = (quadrature_points,) + (1,) * speed_low.ndim
    nodes = nodes.view(view_shape)
    log_weights = log_weights.view(view_shape)
    speed_node = speed_low.unsqueeze(0) + (speed_high - speed_low).unsqueeze(0) * nodes
    conditional_std = one_minus_rho2.sqrt().unsqueeze(0)
    conditional_loc = correlation.unsqueeze(0) * speed_node
    conditional_low = flow_low.unsqueeze(0)
    conditional_high = flow_high.unsqueeze(0)
    log_flow_probability = My_Multinormal._log_normal_interval_probability(
        (conditional_low - conditional_loc) / conditional_std,
        torch.full_like(conditional_loc, torch.inf) if torch.isposinf(high[1]) else (
            conditional_high - conditional_loc
        ) / conditional_std,
    )
    log_speed_density = -0.5 * (speed_node.square() + math.log(2.0 * math.pi))
    quadrature_weight = torch.softmax(
        log_weights + log_speed_density + log_flow_probability, dim=0
    )
    mean_speed_z = (quadrature_weight * speed_node).sum(dim=0)
    mean_flow_given_speed = My_TruncatedNormal(
        conditional_loc, conditional_std, low=conditional_low, high=conditional_high
    ).mean
    mean_flow_z = (quadrature_weight * mean_flow_given_speed).sum(dim=0)
    mean = torch.stack([
        loc[..., 0] + scale[..., 0] * mean_speed_z,
        loc[..., 1] + scale[..., 1] * mean_flow_z,
    ], dim=-1)
    return torch.maximum(torch.minimum(mean, high), low)


class V_V(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        if "get_para":
            self.d_model = cfgs['d_model']
            self.N = cfgs['num_nodes']
            self.quad_points = int(cfgs.get('quad_points', self.d_model))
            self.observation_quad_points = int(cfgs.get('observation_quad_points', 16))
            temporal_num_embeddings = cfgs['temporal_embedding_list']
            spatial_embedding_list = cfgs['spatial_embedding_list']
            num_nodes = cfgs['num_nodes']
            eigenmaps_k = cfgs['eigenmaps_k']
            self.noTE = cfgs.get('noTE', False)
            self.noSE = cfgs.get('noSE', False)
            self.noGC = cfgs.get('noGC', False)
            self.noLog = cfgs.get('noLog', False)
            self.len = cfgs['len']
            self.epsilon = float(cfgs['epsilon'])
            self.spatial_temporal_embedding = FlowSpatialTemporalEmbedding(temporal_num_embeddings,
                                                                           spatial_embedding_list, self.d_model,
                                                                           num_nodes,
                                                                           eigenmaps_k, self.noGC, self.noLog,
                                                                           self.noTE, self.noSE,
                                                                           cfgs.get('location_feature_indices', []),
                                                                           cfgs.get('location_fourier_levels', 0))
            d_hidden_gm = cfgs['d_hidden_gm'] if 'd_hidden_gm' in cfgs else cfgs['d_hidden_mt']
            self.AG = Adaptive_Graph(d_model=self.d_model, d_hidden_gm=d_hidden_gm, top_k=cfgs['top_k'])

        quad_nodes, quad_weight = gauss_legendre_node_gen_1d(self.quad_points, low=0.0, high=1.0)
        self.register_buffer('quad_nodes', quad_nodes)
        self.register_buffer('quad_weight', quad_weight)
        self.register_buffer('quad_log_weight', torch.log(quad_weight.clamp_min(torch.finfo(quad_weight.dtype).tiny)))

        self.A = NET_A(cfgs)
        self.B = NET_B(cfgs)

        self.u = nn.Parameter(torch.Tensor(1, self.N, self.d_model))
        self.E = nn.Parameter(torch.Tensor(1, self.N, self.d_model))
        nn.init.uniform_(self.u, a=-0.1, b=0.1)
        nn.init.xavier_normal_(self.E, 1)

    def state_nodes(self):
        return self.u + self.quad_nodes.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (self.E - self.u)

    def initial_state_log_probs(self, nodes=None):
        nodes = self.state_nodes() if nodes is None else nodes
        logits = self.normal_dist(self.u, self.E).log_prob(nodes).sum(dim=-1)
        logits = logits + self.quad_log_weight.view(-1, 1)
        return torch.log_softmax(logits, dim=0)

    def transition_log_probs(self, nodes, condition, transition_matrices, AG):
        parameters = self.A(nodes, condition, transition_matrices, AG)
        transition = self.normal_dist(parameters[0], parameters[1])
        logits = transition.log_prob(nodes.unsqueeze(dim=1)).sum(dim=-1)
        logits = logits + self.quad_log_weight.view(-1, 1, 1)
        return torch.log_softmax(logits, dim=0)

    @staticmethod
    def observation_bounds(statics, observation_low=None, observation_high=None):
        normalization = statics['normalization']['target']
        mean = torch.as_tensor(normalization['mean'])
        std = torch.as_tensor(normalization['std'])
        if mean.shape != (2,) or std.shape != (2,):
            raise ValueError("V_V checkpoints must contain two-channel target normalization.")
        if observation_low is None:
            observation_low = -mean / std
        if observation_high is None:
            max_speed = float(statics['normalization']['max_speed_mps'])
            observation_high = torch.stack([
                (math.log1p(max_speed) - mean[0]) / std[0],
                torch.full_like(mean[1], torch.inf),
            ])
        return observation_low, observation_high

    def forward(self, flow, TF, statics, mask=None, observation_low=None, observation_high=None):

        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'])
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c)

        mask = pair_observation_mask(flow, mask)
        observation_low, observation_high = self.observation_bounds(
            statics, observation_low, observation_high
        )
        log_p1 = self.forward_p(
            flow, c, transition_matrices, AG, mask, observation_low, observation_high
        )
        log_p2 = self.backward_p(
            flow, c, transition_matrices, AG, mask, observation_low, observation_high
        )
        log_sum = torch.logaddexp(log_p1, log_p2)
        log_mean = log_sum - math.log(2.0)
        return log_mean

    def observation_log_prob(self, hidden, flow, mask=None, low=-10.0, high=float('inf')):
        loc, scale, correlation = hidden
        log_prob = truncated_bivariate_log_prob(
            flow, loc, scale, correlation, low, high,
            quadrature_points=int(getattr(self, 'observation_quad_points', 16)),
        )
        mask = pair_observation_mask(flow, mask)
        log_prob = torch.where(mask[..., 0].unsqueeze(0), log_prob, torch.zeros_like(log_prob))
        return torch.sum(log_prob, dim=(-1, -2)).unsqueeze(dim=1)

    def forward_p(self, flow, c, transition_matrices, AG, mask, observation_low, observation_high):
        B, time_steps, N, L, _ = flow.shape
        flow = flow.transpose(0, 1)
        mask = mask.transpose(0, 1)
        c = c.transpose(0, 1)
        AG = AG.transpose(0, 1)

        nodes = self.state_nodes()
        log_a = self.initial_state_log_probs(nodes)

        for t in range(time_steps):
            log_transition = self.transition_log_probs(
                nodes, c[t], transition_matrices, AG[t:t + 1]
            )
            hidden = self.B(nodes, c[t])
            log_emission = self.observation_log_prob(
                hidden, flow[t], mask[t], observation_low, observation_high
            ).squeeze(1)
            log_a = torch.logsumexp(log_a.unsqueeze(dim=0) + log_transition, dim=1)
            log_a = log_a + log_emission

        return torch.logsumexp(log_a, dim=0)

    def backward_p(self, flow, c, transition_matrices, AG, mask, observation_low, observation_high):
        batch_size, time_steps, N, L, _ = flow.shape
        flow = flow.transpose(0, 1)
        mask = mask.transpose(0, 1)
        c = c.transpose(0, 1)
        AG = AG.transpose(0, 1)

        nodes = self.state_nodes()
        log_initial = self.initial_state_log_probs(nodes)
        log_b = torch.zeros(nodes.shape[0], batch_size, dtype=flow.dtype, device=flow.device)
        for t in range(time_steps - 1, -1, -1):
            log_transition = self.transition_log_probs(
                nodes, c[t], transition_matrices, AG[t:t + 1]
            )
            hidden = self.B(nodes, c[t])
            log_emission = self.observation_log_prob(
                hidden, flow[t], mask[t], observation_low, observation_high
            )
            log_b = torch.logsumexp(
                log_b.unsqueeze(dim=1) + log_transition + log_emission,
                dim=0,
            )

        return torch.logsumexp(log_initial + log_b, dim=0)

    def normal_dist(self, U, E):
        u = U
        factor = (E - u).unsqueeze(dim=-1)
        cov_diag = torch.full_like(u, self.epsilon)
        return My_Multinormal(u, factor, cov_diag=cov_diag, low=-1.0, high=1.0, latent_quad_points=self.quad_points)

    def sample(self, TF, statics, observation_low=None, observation_high=None):
        emission_parameters = []

        TF = TF.unsqueeze(0)
        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'])
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c).transpose(0, 1)
        c = c.transpose(0, 1)

        observation_low, observation_high = self.observation_bounds(
            statics, observation_low, observation_high
        )
        nodes = self.state_nodes()
        batch_size = TF.shape[0]
        batch_indices = torch.arange(batch_size, device=TF.device)
        initial = self.initial_state_log_probs(nodes).transpose(0, 1).expand(batch_size, -1)
        state = torch.distributions.Categorical(logits=initial).sample()
        for t in range(c.shape[0]):
            transition = self.transition_log_probs(
                nodes, c[t], transition_matrices, AG[t:t + 1]
            )
            state_logits = transition[:, state, batch_indices].transpose(0, 1)
            state = torch.distributions.Categorical(logits=state_logits).sample()
            locations, scales, correlations = self.B(nodes, c[t])
            loc = locations[state, 0]
            scale = scales[state, 0]
            correlation = correlations[state, 0]
            emission_parameters.append((loc, scale, correlation))

        locations, scales, correlations = zip(*emission_parameters)
        return sample_truncated_bivariate(
            torch.stack(locations, dim=1),
            torch.stack(scales, dim=1),
            torch.stack(correlations, dim=1),
            observation_low,
            observation_high,
        )

    def Predict(self, flow, TF, statics, mask=None, observation_low=None, observation_high=None):

        flow = flow.unsqueeze(0)
        TF = TF.unsqueeze(0)
        mask = pair_observation_mask(flow, None if mask is None else mask.unsqueeze(0))
        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'])
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c).transpose(0, 1)
        c = c.transpose(0, 1)

        history_steps = flow.shape[1]
        total_steps = TF.shape[1]
        flow = flow.transpose(0, 1)
        mask = mask.transpose(0, 1)

        nodes = self.state_nodes()
        log_a = self.initial_state_log_probs(nodes).expand(-1, flow.shape[1])

        observation_low, observation_high = self.observation_bounds(
            statics, observation_low, observation_high
        )

        for t in range(history_steps):
            log_transition = self.transition_log_probs(
                nodes, c[t], transition_matrices, AG[t:t + 1]
            )
            hidden = self.B(nodes, c[t])
            log_emission = self.observation_log_prob(
                hidden, flow[t], mask[t], observation_low, observation_high
            ).squeeze(1)
            log_a = torch.logsumexp(log_a.unsqueeze(0) + log_transition, dim=1)
            log_a = torch.log_softmax(log_a + log_emission, dim=0)

        result = []

        for i in range(history_steps, total_steps):
            log_transition = self.transition_log_probs(
                nodes, c[i], transition_matrices, AG[i:i + 1]
            )
            log_a = torch.logsumexp(log_a.unsqueeze(0) + log_transition, dim=1)
            weights = torch.softmax(log_a, dim=0)
            hidden = self.B(nodes, c[i])
            hidden_mean = truncated_bivariate_mean(
                hidden[0], hidden[1], hidden[2], observation_low, observation_high,
                quadrature_points=self.observation_quad_points,
            )
            result.append(
                (weights[..., None, None, None] * hidden_mean).sum(dim=0)
            )
        p = torch.stack(result, dim=1).squeeze(dim=0)
        return p


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    device = 'cuda:0'

    from torch.utils.data import DataLoader
    from dataset.V_V_dataset import V_V_dataset
    from utils.Auxiliary import move2device

    debug = True
    generator = torch.Generator()
    cfgs = yaml.safe_load(open('../args/dataset_args.yaml'))['V_V']
    cfgs['dataset_root'] = '../' + cfgs['dataset_root']

    model_cfgs = yaml.safe_load(open('../args/V_V.yaml'))['model']
    for key in cfgs.keys():
        model_cfgs[key] = cfgs[key]

    V_V = V_V(model_cfgs).to(device)

    dataset = V_V_dataset(model_cfgs, debug, split="Train")
    stastic = dataset.statics
    stastic = move2device(stastic, device)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
    for data, tf in dataloader:
        data, tf = move2device([data, tf], device)
        y = V_V(data, tf, stastic)
        print(y)
        break
