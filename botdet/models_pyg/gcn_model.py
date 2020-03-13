import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .gcn_base_models import NodeModelAdditive, NodeModelMLP
from .graph_attention import NodeModelAttention
from .common import activation


class GCNModel(nn.Module):
    """
    Graph convolutional model, composed of several GCN layers, residual connections, and final output layers.

    Args:
        in_channels (int): input channels
        enc_sizes (List[int]): output channels of each layer, e.g. [32, 64, 64, 32]
        num_classes (int): number of classes for final prediction
        non_linear (str): non-linear activation function
        non_linear_layer_wise (str): non-linear activation function inside each layer before residual addition.
            Default: 'none' (should not be changed in most of the cases)
        residual_hop (int): hops between layers to add residual connections. If the dimensions are the same, output from
            previous layer is directly added; otherwise, a linear transformation (no bias) is applied before adding.
        dropout (float): dropout applied to node hidden representations (not on initial input feature).
        final_layer_config (dict): configuration arguments for the final layer, if it is different from previous layers.
            This is useful when the last layer is the direct output layer, and you want to change some setup, such as
            the attention heads, etc.
        final_type (str): final layer type for the predicted scores. Default: 'none'.
        pred_on (str): whether the prediction task is on nodes or on the whole graph. Default: 'node'.
        **kwargs: could include other configuration arguments for each layer, such as for graph attention layers.

    Input:
        - x (torch.Tensor): node features of size (B * N, C_in)
        - edge_index (torch.LongTensor): COO format edge index of size (2, E)
        - edge_attr (torch.Tensor, optional): edge attributes/features of size (E, D_in)
        - deg (torch.Tensor, optional): node degrees of size (B * N,); this could save computation and memory for
            computing the node degrees every forward pass when message normalization is dependent on degrees.
        - edge_weight (torch.Tensor, optional): currently not used in most cases.

    Output:
        - x (torch.Tensor): updated node features of size (B * N, num_classes) for node prediction, or (B, num_classes)
            for graph level prediction

    where
        B: number of graphs in a batch (batch size)
        N: number of nodes
        E: number of edges
        C_in: dimension of input node features
        num_classes: number of classes to predict
        D_in: dimension of input edge features
    """

    def __init__(self, in_channels, enc_sizes, num_classes, non_linear='relu', non_linear_layer_wise='none',
                 residual_hop=None, dropout=0.0, final_layer_config=None, final_type='none', pred_on='node', **kwargs):
        assert final_type in ['none', 'proj']
        assert pred_on in ['node', 'graph']
        super().__init__()

        self.in_channels = in_channels
        self.enc_sizes = [in_channels, *enc_sizes]
        self.num_layers = len(self.enc_sizes) - 1
        self.num_classes = num_classes
        self.residual_hop = residual_hop
        self.non_linear_layer_wise = non_linear_layer_wise
        self.final_type = final_type
        self.pred_on = pred_on

        # allow different layers to have different attention heads
        # particularly for the last attention layer to be directly the output layer
        if 'nheads' in kwargs:
            if isinstance(kwargs['nheads'], int):
                self.nheads = [kwargs['nheads']] * self.num_layers
            elif isinstance(kwargs['nheads'], list):
                self.nheads = kwargs['nheads']
                assert len(self.nheads) == self.num_layers
            else:
                raise ValueError
            del kwargs['nheads']
        else:
            # otherwise just a placeholder for 'nheads'
            self.nheads = [1] * self.num_layers

        if final_layer_config is None:
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes, self.enc_sizes[1:], self.nheads)])
        else:
            assert isinstance(final_layer_config, dict)
            self.gcn_net = nn.ModuleList([GCNLayer(in_c, out_c, nheads=nh, non_linear=non_linear_layer_wise, **kwargs)
                                          for in_c, out_c, nh in zip(self.enc_sizes[:-2],
                                                                     self.enc_sizes[1:-1],
                                                                     self.nheads[:-1])])
            kwargs.update(final_layer_config)    # this will update with the new values in final_layer_config
            self.gcn_net.append(GCNLayer(self.enc_sizes[-2], self.enc_sizes[-1], nheads=self.nheads[-1],
                                         non_linear=non_linear_layer_wise, **kwargs))

        self.dropout = nn.Dropout(dropout)

        if residual_hop is not None and residual_hop > 0:
            self.residuals = nn.ModuleList([nn.Linear(self.enc_sizes[i], self.enc_sizes[j], bias=False)
                                            if self.enc_sizes[i] != self.enc_sizes[j]
                                            else
                                            nn.Identity()
                                            for i, j in zip(range(0, len(self.enc_sizes), residual_hop),
                                                            range(residual_hop, len(self.enc_sizes), residual_hop))])
            self.num_residuals = len(self.residuals)

        self.non_linear = activation(non_linear)

        if self.final_type == 'none':
            self.final = nn.Identity()
        elif self.final_type == 'proj':
            self.final = nn.Linear(self.enc_sizes[-1], num_classes)
        else:
            raise ValueError

    def reset_parameters(self):
        for net in self.gcn_net:
            net.reset_parameters()
        if self.residual_hop is not None:
            for net in self.residuals:
                net.reset_parameters()
        if self.final_type != 'none':
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        xr = None
        add_xr_at = -1

        for n, net in enumerate(self.gcn_net):
            # pass to a GCN layer with non-linear activation
            xo = net(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
            xo = self.dropout(xo)
            # deal with residual connections
            if self.residual_hop is not None and self.residual_hop > 0:
                if n % self.residual_hop == 0 and (n // self.residual_hop) < self.num_residuals:
                    xr = self.residuals[n // self.residual_hop](x)
                    add_xr_at = n + self.residual_hop - 1
                if n == add_xr_at:
                    if n < self.num_layers - 1:  # before the last layer
                        # non_linear is applied both after each layer (by default: 'none') and after residual sum
                        xo = self.non_linear(xo + xr)
                    else:  # the last layer (potentially the output layer)
                        if self.final_type == 'none':
                            # no non_linear is important for binary classification since this is to be passed to sigmoid
                            # function to calculate loss, and ReLU will directly kill all the negative parts
                            xo = xo + xr
                        else:
                            xo = self.non_linear(xo + xr)
            else:
                if n < self.num_layers - 1:  # before the last layer
                    xo = self.non_linear(xo)
                else:
                    if self.final_type == 'none':
                        pass
                    else:
                        xo = self.non_linear(xo)

            x = xo
        # size of x: (B * N, self.enc_sizes[-1]) -> (B * N, num_classes)
        x = self.final(x)

        # graph level pooling for graph classification
        # use mean pooling here
        if self.pred_on == 'graph':
            assert 'batch_slices_x' in kwargs
            batch_slices_x = kwargs['batch_slices_x']
            if len(batch_slices_x) == 2:
                # only one graph in the batch
                x = x.mean(dim=0, keepdim=True)  # size (1, num_classes)
            else:
                # more than one graphs in the batch
                x_batch, lengths = zip(*[(x[i:j], j - i) for (i, j) in zip(batch_slices_x, batch_slices_x[1:])])
                x_batch = pad_sequence(x_batch, batch_first=True,
                                       padding_value=0)  # size (batch_size, max_num_nodes, num_classes)
                x = x_batch.sum(dim=1) / x_batch.new_tensor(lengths)  # size (batch_size, num_classes)

        return x


class GCNLayer(nn.Module):
    """
    Graph convolutional layer. A wrapper of the node update models such as basic additive, MLP, or attention, etc.
    Can also be extended to include edge update models and extra read out operations.

    Args:
        in_channels (int): input channels
        out_channels (int): output channels
        in_edgedim (int, optional): input edge feature dimension
        deg_norm (str, optional): method of (out-)degree normalization. Choose from ['none', 'sm', 'rw']. Default: 'sm'.
            'sm': symmetric, better for undirected graphs. 'rw': random walk, better for directed graphs.
            Note that 'sm' for directed graphs might have some problems, when a target node does not have any out-degree.
        edge_gate (str, optional): method of apply edge gating mechanism. Choose from ['none', 'proj', 'free'].
            Note that when set to 'free', should also provide `num_edges` as an argument (but then it can only work with
            fixed edge graph).
        aggr (str, optional): method of aggregating the neighborhood features. Choose from ['add', 'mean', 'max'].
            Default: 'add'.
        bias (bool, optional): whether to include bias vector in the model. Default: True.
        nodemodel (str, optional): node model name
        non_linear (str, optional): non-linear activation function
        **kwargs: could include `num_edges`, etc.
    """
    nodemodel_dict = {'additive': NodeModelAdditive,
                      'mlp': NodeModelMLP,
                      'attention': NodeModelAttention}

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='add',
                 bias=True, nodemodel='additive', non_linear='relu', **kwargs):
        assert nodemodel in ['additive', 'mlp', 'attention']
        super().__init__()
        self.gcn = self.nodemodel_dict[nodemodel](in_channels,
                                                  out_channels,
                                                  in_edgedim,
                                                  deg_norm=deg_norm,
                                                  edge_gate=edge_gate,
                                                  aggr=aggr,
                                                  bias=bias,
                                                  **kwargs)

        self.non_linear = activation(non_linear)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        xo = self.gcn(x, edge_index, edge_attr, deg, edge_weight, **kwargs)
        xo = self.non_linear(xo)
        return xo
