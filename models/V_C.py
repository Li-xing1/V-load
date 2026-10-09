# -*- coding: utf-8 -*-
# @Time : 2026/3/6 13:37
# @Author : Xing Li
# @Email : 2434147605@qq.com
# @File : V_C.py
# @Project : V_load

import yaml

from models.common import *



class NET_A(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.d_model = cfgs['d_model']
        self.num_hidden = cfgs['num_hidden_A']
        self.d_k = cfgs['d_k']
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
        meta_hidden = cfgs['d_hidden_mt']
        meta_rank = cfgs.get('emission_meta_rank', 4)
        self.linear_list = nn.ModuleList(
            [ContextualLinear(self.d_model, self.num_hidden, self.d_model, meta_hidden, meta_rank)]
            + [ContextualLinear(self.num_hidden, self.num_hidden, self.d_model, meta_hidden, meta_rank)
               for _ in range(self.num_hidden_layer_B)]
            + [ContextualLinear(self.num_hidden, self.L * 6, self.d_model, meta_hidden, meta_rank)]
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
        out = out.reshape(num_samples, batch_size, num_nodes, self.L, 6)
        out = torch.softmax(out, dim=-1)
        return out


class V_C(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        if "get_para":
            self.d_model = cfgs['d_model']
            self.N = cfgs['num_nodes']
            self.quad_points = int(cfgs.get('quad_points', self.d_model))
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
                                                                           cfgs.get('location_fourier_levels', 0),
                                                                           node_temporal_dim=1)
            self.AG = Adaptive_Graph(d_model=self.d_model, d_hidden_gm=cfgs['d_hidden_gm'], top_k=cfgs['top_k'])

        quad_nodes, quad_weight = gauss_legendre_node_gen_1d(self.quad_points, low=0.0, high=1.0)
        # print('node and weight:', quad_nodes.shape, quad_weight.shape)
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

    def forward(self, flow, TF, V_C_v, statics, mask):

        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'], V_C_v)
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c)

        log_p1 = self.forward_p(flow, c, transition_matrices, AG, mask)
        # #print('fp:',log_p1.shape)
        log_p2 = self.backward_p(flow, c, transition_matrices, AG, mask)
        log_sum = torch.logaddexp(log_p1, log_p2)
        log_mean = log_sum - math.log(2.0)
        # log_mean = log_p2
        return log_mean

    @staticmethod
    def emission_log_prob(probabilities, flow, mask):
        counts = flow[..., :6].unsqueeze(0)
        total = flow[..., 6:7].unsqueeze(0)
        log_coefficient = torch.lgamma(total + 1) - torch.lgamma(counts + 1).sum(dim=-1, keepdim=True)
        log_probability = log_coefficient + (
            counts * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny))
        ).sum(dim=-1, keepdim=True)
        log_probability = torch.where(mask.unsqueeze(0), log_probability, torch.zeros_like(log_probability))
        return log_probability.sum(dim=(-1, -2, -3))

    def forward_p(self, flow, c, transition_matrices, AG, mask):
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
            probabilities = self.B(nodes, c[t])
            log_emission = self.emission_log_prob(probabilities, flow[t], mask[t])
            log_a = torch.logsumexp(log_a.unsqueeze(0) + log_transition, dim=1)
            log_a = log_a + log_emission

        return torch.logsumexp(log_a, dim=0)

    def backward_p(self, flow, c, transition_matrices, AG, mask):
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
            probabilities = self.B(nodes, c[t])
            log_emission = self.emission_log_prob(probabilities, flow[t], mask[t]).unsqueeze(1)
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

    @staticmethod
    def restore_vehicle_class_probabilities(probabilities, statics):
        normalization = statics.get('normalization') if isinstance(statics, dict) else None
        if not normalization or not bool(normalization.get('balance_classes', True)):
            return probabilities
        frequencies = normalization['vehicle_class_frequencies'].to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        restored = probabilities * frequencies
        return restored / restored.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(restored.dtype).tiny)

    def sample(self, TF, V_C_v, statics):
        """

        :param TF: [T,F]
        :param statics:
        :return:
        """

        result = []

        TF = TF.unsqueeze(0)
        V_C_v = V_C_v.unsqueeze(0)
        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'], V_C_v)
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c).transpose(0, 1)
        c = c.transpose(0, 1)

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
            emissions = self.B(nodes, c[t])
            probabilities = emissions[state, batch_indices]
            result.append(self.restore_vehicle_class_probabilities(probabilities, statics))

        return torch.stack(result, dim=1)

    def Predict(self, flow, TF, V_C_v, statics, mask):

        flow = flow.unsqueeze(0)
        mask = mask.unsqueeze(0)
        TF = TF.unsqueeze(0)
        V_C_v = V_C_v.unsqueeze(0)
        c = self.spatial_temporal_embedding(TF, statics['SF'], statics['eigenmaps'], V_C_v)
        transition_matrices = statics['transition_matrices']
        AG = self.AG(c).transpose(0, 1)
        c = c.transpose(0, 1)

        # print(c.shape)
        # print(AG.shape)


        history_steps = flow.shape[1]
        total_steps = TF.shape[1]
        flow = flow.transpose(0, 1)
        mask = mask.transpose(0, 1)

        nodes = self.state_nodes()
        log_a = self.initial_state_log_probs(nodes).expand(-1, flow.shape[1])
        for t in range(history_steps):
            log_transition = self.transition_log_probs(
                nodes, c[t], transition_matrices, AG[t:t + 1]
            )
            emissions = self.B(nodes, c[t])
            log_emission = self.emission_log_prob(emissions, flow[t], mask[t])
            log_a = torch.logsumexp(log_a.unsqueeze(0) + log_transition, dim=1)
            log_a = torch.log_softmax(log_a + log_emission, dim=0)

        result = []
        # print(log_a.shape)

        for i in range(history_steps, total_steps):
            log_transition = self.transition_log_probs(
                nodes, c[i], transition_matrices, AG[i:i + 1]
            )
            log_a = torch.logsumexp(log_a.unsqueeze(0) + log_transition, dim=1)
            log_a = torch.log_softmax(log_a, dim=0)
            emissions = self.B(nodes, c[i])
            log_p = torch.logsumexp(
                log_a.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                + torch.log(emissions.clamp_min(torch.finfo(emissions.dtype).tiny)),
                dim=0,
            )

            result.append(self.restore_vehicle_class_probabilities(log_p.exp(), statics))
        p = torch.stack(result, dim=1).squeeze(dim=0)
        return p


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    # torch.set_default_device(torch.device('cuda:0'))

    device = 'cuda:0'

    from torch.utils.data import DataLoader
    from dataset.V_C_dataset import V_C_dataset
    from utils.Auxiliary import move2device

    debug = True
    generator = torch.Generator()
    cfgs = yaml.safe_load(open('../args/dataset_args.yaml'))['V_C']
    cfgs['dataset_root'] = '../' + cfgs['dataset_root']

    model_cfgs = yaml.safe_load(open('../args/V_C.yaml'))['model']
    for key in cfgs.keys():
        model_cfgs[key] = cfgs[key]

    V_C = V_C(model_cfgs).to(device)

    dataset = V_C_dataset(model_cfgs, debug, split="Train")
    stastic = dataset.statics
    stastic = move2device(stastic, device)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
    for data, tf, v_c_v, mask in dataloader:
        Flow = data.to(device)
        tf = tf.to(device)
        v_c_v = v_c_v.to(device)
        mask = mask.to(device)
        # a = V_C(Flow, tf, v_c_v, stastic, mask)
        break
    # print(a)

    tf = tf[0]
    v_c_v = v_c_v[0]

    sample = V_C.sample(tf, v_c_v, statics=stastic)
    print(sample.shape)

    # flow = Flow[0, :12]
    # pre = V_C.Predict(flow, tf, v_c_v, statics=stastic, mask=mask[0])
    # print(pre.shape)
