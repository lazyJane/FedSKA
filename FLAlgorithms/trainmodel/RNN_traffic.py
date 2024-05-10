import dgl
import dgl.function as fn
import dgl.nn as dglnn
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
from dgl.base import DGLError
from dgl.nn.functional import edge_softmax


class GraphGRUCell(nn.Module):
    """Graph GRU unit which can use any message passing
    net to replace the linear layer in the original GRU
    Parameter
    ==========
    in_feats : int
        number of input features

    out_feats : int
        number of output features

    net : torch.nn.Module
        message passing network
    """

    def __init__(self, in_feats, out_feats, net):
        super(GraphGRUCell, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dir = dir
        # net can be any GNN model
        self.r_net = net(in_feats + out_feats, out_feats)
        self.u_net = net(in_feats + out_feats, out_feats)
        self.c_net = net(in_feats + out_feats, out_feats)
        # Manually add bias Bias
        self.r_bias = nn.Parameter(torch.rand(out_feats))
        self.u_bias = nn.Parameter(torch.rand(out_feats))
        self.c_bias = nn.Parameter(torch.rand(out_feats))

    def forward(self, g, x, h):
        r = torch.sigmoid(self.r_net(g, torch.cat([x, h], dim=1)) + self.r_bias)
        u = torch.sigmoid(self.u_net(g, torch.cat([x, h], dim=1)) + self.u_bias)
        h_ = r * h
        c = torch.sigmoid(
            self.c_net(g, torch.cat([x, h_], dim=1)) + self.c_bias
        )
        new_h = u * h + (1 - u) * c
        return new_h


class StackedEncoder(nn.Module):
    """One step encoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    net : torch.nn.Module
        message passing network for graph computation
    """

    def __init__(self, in_feats, out_feats, num_layers, net):
        super(StackedEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0! ")
        self.layers.append(
            GraphGRUCell(self.in_feats, self.out_feats, self.net)
        )
        for _ in range(self.num_layers - 1):
            self.layers.append(
                GraphGRUCell(self.out_feats, self.out_feats, self.net)
            )

    # hidden_states should be a list which for different layer
    def forward(self, g, x, hidden_states):
        hiddens = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x, hidden_states[i])
            hiddens.append(x)
        return x, hiddens


class StackedDecoder(nn.Module):
    """One step decoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    hid_feats : int
        number of feature before the linear output layer

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    net : torch.nn.Module
        message passing network for graph computation
    """

    def __init__(self, in_feats, hid_feats, out_feats, num_layers, net):
        super(StackedDecoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.out_layer = nn.Linear(self.hid_feats, self.out_feats)
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0!")
        self.layers.append(GraphGRUCell(self.in_feats, self.hid_feats, net))
        for _ in range(self.num_layers - 1):
            self.layers.append(
                GraphGRUCell(self.hid_feats, self.hid_feats, net)
            )

    def forward(self, g, x, hidden_states):
        hiddens = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x, hidden_states[i])
            hiddens.append(x)
        x = self.out_layer(x)
        return x, hiddens
'''
class Encoder(nn.Module):
    """One step encoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    net : torch.nn.Module
        message passing network for graph computation
    """

    def __init__(self, in_feats, out_feats, num_layers):
        super(StackedEncoder, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0! ")
        self.layers.append(
            
        )
        for _ in range(self.num_layers - 1):
            self.layers.append(
                
            )

    # hidden_states should be a list which for different layer
    def forward(self, x, hidden_states):
        hiddens = []
        for i, layer in enumerate(self.layers):
            x = 
            hiddens.append(x)
        return x, hiddens

class Decoder(nn.Module):
    """One step decoder unit for hidden representation generation
    it can stack multiple vertical layers to increase the depth.

    Parameter
    ==========
    in_feats : int
        number if input features

    hid_feats : int
        number of feature before the linear output layer

    out_feats : int
        number of output features

    num_layers : int
        vertical depth of one step encoding unit

    """

    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super(StackedDecoder, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.out_layer = nn.Linear(self.hid_feats, self.out_feats)
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0!")
        self.layers.append(GraphGRUCell(self.in_feats, self.hid_feats, net))
        for _ in range(self.num_layers - 1):
            self.layers.append(
                GraphGRUCell(self.hid_feats, self.hid_feats, net)
            )

    def forward(self, g, x, hidden_states):
        hiddens = []
        for i, layer in enumerate(self.layers):
            x = layer(g, x, hidden_states[i])
            hiddens.append(x)
        x = self.out_layer(x)
        return x, hiddens
'''    
class GraphRNN(nn.Module):
    """Graph Sequence to sequence prediction framework
    Support multiple backbone GNN. Mainly used for traffic prediction.

    Parameter
    ==========
    in_feats : int
        number of input features

    out_feats : int
        number of prediction output features

    seq_len : int
        input and predicted sequence length

    num_layers : int
        vertical number of layers in encoder and decoder unit

    net : torch.nn.Module
        Message passing GNN as backbone

    decay_steps : int
        number of steps for the teacher forcing probability to decay
    """

    def __init__(
        self, in_feats, out_feats, seq_len, num_layers, net, decay_steps
    ):
        super(GraphRNN, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.net = net
        self.decay_steps = decay_steps

        self.encoder = StackedEncoder(
            self.in_feats, self.out_feats, self.num_layers, self.net
        )

        self.decoder = StackedDecoder(
            self.in_feats,
            self.out_feats,
            self.in_feats,
            self.num_layers,
            self.net,
        )

    # Threshold For Teacher Forcing

    def compute_thresh(self, batch_cnt):
        return self.decay_steps / (
            self.decay_steps + np.exp(batch_cnt / self.decay_steps)
        )

    def encode(self, g, inputs, device):
        hidden_states = [
            torch.zeros(g.num_nodes(), self.out_feats).to(device)
            for _ in range(self.num_layers)
        ]
        for i in range(self.seq_len):
            _, hidden_states = self.encoder(g, inputs[i], hidden_states)

        return hidden_states

    def decode(self, g, teacher_states, hidden_states, batch_cnt, device):
        outputs = []
        inputs = torch.zeros(g.num_nodes(), self.in_feats).to(device)
        for i in range(self.seq_len):
            if (
                np.random.random() < self.compute_thresh(batch_cnt)
                and self.training
            ):
                inputs, hidden_states = self.decoder(
                    g, teacher_states[i], hidden_states
                )
            else:
                inputs, hidden_states = self.decoder(g, inputs, hidden_states)
            outputs.append(inputs)
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, g, inputs, teacher_states, batch_cnt, device):
        hidden = self.encode(g, inputs, device)
        outputs = self.decode(g, teacher_states, hidden, batch_cnt, device)
        return outputs

class GRUSeq2Seq(nn.Module):
    """
    """

    def __init__(
        self, input_size, hidden_size, output_size, dropout, gru_num_layers
    ):
        super(GRUSeq2Seq, self).__init__()
      
        self.encoder = nn.GRU(
            input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
        )

        self.decoder = nn.GRU(
                input_size, hidden_size, num_layers=gru_num_layers, dropout=dropout
            )
        self.fc1 = nn.Linear(hidden_size, output_size)

    # Threshold For Teacher Forcing
    def forward_encoder(self, x_input):
        _, h_encode = self.encoder(x_input) # T x (B x N) x F  #torch.Size([2, 64, 64])
        return h_encode

    def forward_decoder(self, y_input,  h_encode,  return_encoding=False, server_graph_encoding=None):
        out_hidden, _ = self.decoder(y_input, h_encode)
        out = self.fc1(out_hidden)
        return out

    def forward(self, x_input, y_input, return_encoding=False, server_graph_encoding=None):
        h_encode = self.forward_encoder(x_input)
        out_hidden, _ = self.decoder(y_input, h_encode)
        out = self.fc1(out_hidden)
        return out

class GRUSeq2SeqWithGraphNet(GRUSeq2Seq):
    """
    """

    def __init__(
        self, input_size, hidden_size, output_size, dropout, gru_num_layers
    ):
        super(GRUSeq2SeqWithGraphNet, self).__init__(input_size, hidden_size, output_size, dropout, gru_num_layers)
      
       
        self.decoder = nn.GRU(
                input_size, hidden_size * 2, num_layers=gru_num_layers, dropout=dropout
            )
        self.fc1 = nn.Linear(hidden_size * 2, output_size)

    # Threshold For Teacher Forcing
    def forward_encoder(self, x_input):
        _, h_encode = self.encoder(x_input) # T x (B x N) x F  #torch.Size([2, 64, 64])
        return h_encode
    
    def forward_decoder(self, y_input,  h_encode,  return_encoding=False, server_graph_encoding=None):
        encoder_h = h_encode

        graph_encoding = server_graph_encoding
        
        #print("h_encode.shape", h_encode.shape) #torch.Size([2, 64, 64])
        #print("graph_encoding.shape", graph_encoding.shape) #B x (T x N) x F
        h_encode = torch.cat([h_encode, graph_encoding], dim=-1)
        out_hidden, _ = self.decoder(y_input, h_encode)
        out = self.fc1(out_hidden)
        if return_encoding:
            return out, encoder_h
        else:
            return out

    def forward(self, x_input, y_input, return_encoding=False, server_graph_encoding=None):
        h_encode = self.forward_encoder(x_input)
        return  self.forward_decoder(y_input, h_encode, return_encoding=return_encoding, server_graph_encoding=server_graph_encoding)
       
       