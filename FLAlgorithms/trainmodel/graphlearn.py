import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import VERY_SMALL_NUMBER, INF
from FLAlgorithms.trainmodel.layers import *
from FLAlgorithms.trainmodel.graph_generator import *
from utils.model_utils import *


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm

class GCN_DAE(nn.Module):
    #def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k, knn_metric, i_,
                 #non_linearity, normalization, mlp_h, mlp_epochs, gen_mode, sparse, mlp_act):     
    def __init__(self, nlayers, in_dim, hidden_dim, nclasses, dropout, dropout_adj, features, k,
                 knn_metric, i, non_linearity, normalization, mlp_h, mlp_epochs, mlp_act, 
                 feature_hidden_size, graph_learn_hidden_size, graph_learn_epsilon, graph_learn_num_pers, graph_metric_type, 
                 gen_mode, sparse, device, batch_adj=False):
        super(GCN_DAE, self).__init__()

        self.layers = nn.ModuleList()

        #self.layers.append(GCNLayer(in_dim, hidden_dim, bias=True))
        #for i in range(nlayers - 2):
            #self.layers.append(GCNLayer(hidden_dim, hidden_dim, bias=True))
        #self.layers.append(GCNLayer(hidden_dim, nclasses, bias=True))

        if sparse:
            self.layers.append(GCNConv_dgl(in_dim, hidden_dim))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dgl(hidden_dim, nclasses))

        else:
            self.layers.append(GCNConv_dense(in_dim, hidden_dim))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv_dense(hidden_dim, hidden_dim))
            self.layers.append(GCNConv_dense(hidden_dim, nclasses))

        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)
        self.dropout_adj_p = dropout_adj
        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.non_linearity = non_linearity
        self.normalization = normalization
        self.nnodes = features.shape[0]
        self.mlp_h = mlp_h
        self.mlp_epochs = mlp_epochs
        self.sparse = sparse
        self.graph_metric_type = graph_metric_type
        self.gen_mode = gen_mode
        self.device=device
        self.batch_adj=batch_adj

        if gen_mode == 0:
            self.graph_gen = FullParam(features, non_linearity, k, knn_metric, self.i, sparse, device, batch_adj).to(self.device)
        elif gen_mode == 1:
            self.graph_gen = MLP(2, features, features.shape[-1], math.floor(math.sqrt(features.shape[-1] * self.mlp_h)), # sqrt(input_dim * output_dim)
                                 self.mlp_h, mlp_epochs, k, knn_metric, self.non_linearity, self.i, self.sparse,
                                 mlp_act, device, batch_adj).to(self.device)
        elif gen_mode == 2:
            self.graph_gen = MLP_Diag(2, features.shape[-1], k, knn_metric, self.non_linearity, self.i, sparse,
                                      mlp_act, device, batch_adj).to(self.device)
        elif gen_mode == 3:
            self.graph_gen = GraphLearner(feature_hidden_size, #args.user_feature_hidden_size feature_embedding_dim
                                          graph_learn_hidden_size, 
                                          k, 
                                          graph_learn_epsilon,
                                          graph_learn_num_pers,
                                          graph_metric_type,
                                          non_linearity,
                                          i).to(self.device)
    
    def get_adj(self, h):
        adj = self.graph_gen(h)
       
        if self.gen_mode == 3: 
            if self.graph_metric_type in ('kernel', 'weighted_cosine'):
                assert adj.min().item() >= 0
                adj = adj / torch.clamp(torch.sum(adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

            elif self.graph_metric_type == 'cosine':
                adj = (adj > 0).float()
                adj = normalize_adj(adj)

            else:
                adj = torch.softmax(adj, dim=-1) #使adj的行在0到1之间，且和为1
        else:
            if not self.sparse:
                if self.batch_adj:
                    adj_res = []
                    for i in range(32):
                        adj_i = symmetrize(adj[i])
                        adj_i = normalize(adj_i, self.normalization, self.sparse)
                        adj_res.append(adj_i)
                    adj = torch.stack(adj_res)
                else:
                    adj = symmetrize(adj)
                    adj = normalize(adj, self.normalization, self.sparse)
    
        return adj


    def forward(self, features, x):  # x corresponds to masked_fearures
        Adj_ = self.get_adj(features)
        if self.sparse:
            Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, Adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, Adj)
        return x, Adj_
    

class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='attention', non_linearity=None, i = 0, device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type
        self.non_linearity = non_linearity
        self.i = i
        if metric_type == 'attention':
            self.linear_sims = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=False) for _ in range(num_pers)])
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))


        elif metric_type == 'gat_attention':
            self.linear_sims1 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])
            self.linear_sims2 = nn.ModuleList([nn.Linear(input_size, 1, bias=False) for _ in range(num_pers)])

            self.leakyrelu = nn.LeakyReLU(0.2)

            print('[ GAT_Attention GraphLearner]')

        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))
        elif metric_type == 'transformer':
            self.linear_sim1 = nn.Linear(input_size, hidden_size, bias=False)
            self.linear_sim2 = nn.Linear(input_size, hidden_size, bias=False)


        elif metric_type == 'cosine':
            pass

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'attention':
            attention = 0
            for _ in range(len(self.linear_sims)):
                context_fc = torch.relu(self.linear_sims[_](context))
                #print(context_fc)
                #input()
                attention += torch.matmul(context_fc, context_fc.transpose(-1, -2))
                #print(attention)
                #input()

            attention /= len(self.linear_sims)
            markoff_value = -INF

        elif self.metric_type == 'weighted_cosine':
            expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            if len(context.shape) == 3:
                expand_weight_tensor = expand_weight_tensor.unsqueeze(1)

            context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0


        elif self.metric_type == 'transformer':
            Q = self.linear_sim1(context)
            attention = torch.matmul(Q, Q.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
            markoff_value = -INF


        elif self.metric_type == 'gat_attention':
            attention = []
            for _ in range(len(self.linear_sims1)):
                a_input1 = self.linear_sims1[_](context)
                a_input2 = self.linear_sims2[_](context)
                attention.append(self.leakyrelu(a_input1 + a_input2.transpose(-1, -2)))

            attention = torch.mean(torch.stack(attention, 0), 0)
            markoff_value = -INF


        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

            markoff_value = 0

        elif self.metric_type == 'cosine':
            context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
            attention = torch.mm(context_norm, context_norm.transpose(-1, -2)).detach()
            markoff_value = 0


        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        #if self.epsilon is not None:
            #attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)

        #if self.non_linearity is not None:
            #attention = self.apply_non_linearity(attention, self.non_linearity, self.i)
        return attention

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = (markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val).to(self.device) 
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists
    
    def apply_non_linearity(self, tensor, non_linearity, i):
        if non_linearity == 'elu':
            return F.elu(tensor * i - i) + 1
        elif non_linearity == 'relu':
            return F.relu(tensor)
        elif non_linearity == 'none':
            return tensor
        else:
            raise NameError('We dont support the non-linearity yet')


def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1)
    adj = torch.zeros_like(attention).scatter_(-1, knn_ind, 1).to(device)
    return adj
