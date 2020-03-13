import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter_

from .gcn_base_models import NodeModelBase
from .common import activation, softmax


class NodeModelAttention(NodeModelBase):
    """
    Multi-head soft attention over a node's neighborhood.
    Note:
        - Inheritance to :class:`NodeModelBase` is only for organization purpose, which is actually not necessary
          So deg_norm=None, edge_gate=None, aggr='add' (defaults), and they are not currently used.
        - When `att_combine` is 'cat', out_channels for 1 head is out_channels / nheads;
          otherwise, it is out_channels for every head.
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None,
                 nheads=1, att_act='none', att_dropout=0, att_combine='cat', att_dir='in', bias=False, **kwargs):
        assert att_act in ['none', 'lrelu', 'relu']
        assert att_combine in ['cat', 'add', 'mean']
        assert att_dir in ['in', 'out']

        super(NodeModelAttention, self).__init__(in_channels, out_channels, in_edgedim)

        self.nheads = nheads
        if att_combine == 'cat':
            self.out_channels_1head = out_channels // nheads
            assert self.out_channels_1head * nheads == out_channels, 'out_channels should be divisible by nheads'
        else:
            self.out_channels_1head = out_channels

        self.att_combine = att_combine
        self.att_dir = att_dir

        if att_combine == 'cat':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else:    # 'add' or 'mean':
            self.weight = Parameter(torch.Tensor(in_channels, out_channels * nheads))
        self.att_weight = Parameter(torch.Tensor(1, nheads, 2 * self.out_channels_1head))
        self.att_act = activation(att_act)
        self.att_dropout = nn.Dropout(p=att_dropout)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, attn_store=None, **kwargs):
        """
        'deg' and 'edge_weight' are not used. Just to be consistent for API.
        """
        x = torch.mm(x, self.weight).view(-1, self.nheads, self.out_channels_1head)  # size (N, n_heads, C_out_1head)

        # lift the features to source and target nodes, size (E, nheads, C_out_1head) for each
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])

        # calculate attention coefficients, size (E, nheads)
        alpha = self.att_act((torch.cat([x_j, x_i], dim=-1) * self.att_weight).sum(dim=-1))

        # softmax over each node's neighborhood, size (E, nheads)
        if self.att_dir == 'out':
            # random walk
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))
        else:
            # attend over nodes that all points to the current one
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))

        # dropout on attention coefficients (which means that during training, the neighbors are stochastically sampled)
        alpha = self.att_dropout(alpha)

        ''' 
        # check attention entropy
        if self.att_dir == 'out':
            entropy = scatter_('add', -alpha * torch.log(alpha + 1e-16), edge_index[0], dim_size=x.size(0))
        else:    # size (N, nheads)
            entropy = scatter_('add', -alpha * torch.log(alpha + 1e-16), edge_index[1], dim_size=x.size(0))
        # breakpoint()
        entropy = entropy[deg > 100, :].mean()
        entropy_max = (torch.log(deg[deg > 100] + 1e-16)).mean()
        print(f'average attention entropy {entropy.item()} (average max entropy {entropy_max.item()})')
        '''

        # normalize messages on each edges with attention coefficients
        x_j = x_j * alpha.view(-1, self.nheads, 1)

        # aggregate features to nodes, resulting in size (N, n_heads, C_out_1head)
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # combine multi-heads, resulting in size (N, C_out)
        if self.att_combine == 'cat':
            x = x.view(-1, self.out_channels)
        elif self.att_combine == 'add':
            x = x.sum(dim=1)
        else:
            x = x.mean(dim=1)

        # add bias
        if self.bias is not None:
            x = x + self.bias
  
        if attn_store is not None:    # attn_store is a callback list in case we want to get the attention scores out
            attn_store.append(alpha)

        return x

    def __repr__(self):
        return ('{} (in_channels: {}, out_channels: {}, in_edgedim: {}, nheads: {}, att_activation: {},'
                'att_dropout: {}, att_combine: {}, att_dir: {} | number of parameters: {}').format(
                self.__class__.__name__, self.in_channels, self.out_channels, self.in_edgedim,
                self.nheads, self.att_act, self.att_dropout.p, self.att_combine, self.att_dir, self.num_parameters())
