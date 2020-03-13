import torch.nn as nn
import torch_scatter


# activations = nn.ModuleDict([
#     ['lrelu', nn.LeakyReLU()],
#     ['relu', nn.ReLU()],
#     ])


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.
    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def activation(act, negative_slope=0.2):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU(negative_slope)],
        ['relu', nn.ReLU()],
        ['elu', nn.ELU()],
        ['none', Identity()],
    ])
    return activations[act]


def scatter_(name, src, index, dim_size=None, out=None):
    r"""Aggregates all values from the :attr:`botdet` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        botdet (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name == 'max' else 0

    out = op(src, index, 0, out, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out


def softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`botdet`, this function first groups those values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        botdet (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    if num_nodes is None:
        num_nodes = index.max().item() + 1

    out = src - torch_scatter.scatter_max(src, index, dim=0, dim_size=num_nodes,
                                          fill_value=-1e16)[0][index]
    # fill_value here above is crucial for correct operation!!
    out = out.exp()
    out = out / (
            torch_scatter.scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
