from itertools import chain

import torch


def h5group_to_dict(h5group):
    group_dict = {k: v[()] for k, v in chain(h5group.items(), h5group.attrs.items())}
    return group_dict


def sub_dict(full_dict, *keys, to_tensor):
    return {k: torch.tensor(full_dict[k]) if to_tensor else full_dict[k] for k in keys if k in full_dict}


def sub_dict_ft(full_dict, *keys, to_tensor):
    """For DGL, make all to float tensor."""
    return {k: torch.tensor(full_dict[k]).float() if to_tensor else full_dict[k] for k in keys if k in full_dict}


def build_graph_from_dict_pyg(graph_dict, to_tensor=True):
    from torch_geometric.data import Data

    g = Data(**sub_dict(graph_dict, 'edge_index', 'x', 'y', 'edge_attr', 'edge_y', to_tensor=to_tensor))
    return g


def build_graph_from_dict_dgl(graph_dict, to_tensor=True):
    import dgl

    g = dgl.DGLGraph()
    g.add_nodes(graph_dict['num_nodes'], data=sub_dict_ft(graph_dict, 'x', 'y', to_tensor=to_tensor))
    g.add_edges(*graph_dict['edge_index'], data=sub_dict_ft(graph_dict, 'edge_attr', 'edge_y', to_tensor=to_tensor))
    return g


def build_graph_from_dict_nx(graph_dict):
    import networkx as nx

    # currently only undirected graph, with no feature stored
    # TODO: option for directed graph, and see features
    g = nx.Graph()
    g.add_nodes_from(range(graph_dict['num_nodes']))
    g.add_edges_from(zip(graph_dict['edge_index'][0], graph_dict['edge_index'][1]))
    return g
