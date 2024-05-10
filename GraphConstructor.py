import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConstructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes # nodes_num
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim) # Embedding(100, 40)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx) # torch.Size([100, 40])
            nodevec2 = self.emb2(idx) # torch.Size([100, 40])
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1)) # tanh一种激活函数  torch.Size([100, 40])
        # print("nodevec1", nodevec1)
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2)) # torch.Size([100, 40])
        # print("nodevec2", nodevec2)
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0)) # torch.Size([100, 100])
        # print("a", a)
        adj = F.relu(torch.tanh(self.alpha*a)) # torch.Size([100, 100])
        # print("adj", adj)

        return adj

    def eval(self, idx, full=False):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
            #print(nodevec1)
            #print(nodevec2)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))-torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))

        # 前k条边保留，其余为0
        if not full:
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device) #[100,100]
            mask.fill_(float('0'))
            s1, t1 = adj.topk(self.k, 1) # dim=1
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj*mask

        return adj
