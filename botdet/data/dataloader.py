from torch.utils.data import DataLoader


class GraphDataLoader(DataLoader):
    """
    Graph data loader, for a series of static graphs.

    Args:
        dataset (BotnetDataset): botnet graph dataset object
        batch_size (int, optional): batch size
        num_workers (int, optional): number of workers for multiple subprocesses
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):

        def collate_graph(graph_obj_list):
            """
            Collating function to form graph mini-batch.
            It takes in a list of graph Data objects and returns a graph Batch.
            """
            graph_format = dataset.graph_format
            if graph_format == 'pyg':
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(graph_obj_list)
            elif graph_format == 'dgl':
                import dgl
                batch = dgl.batch(graph_obj_list)
            elif graph_format == 'nx' or graph_format == 'dict':
                batch = graph_obj_list
            else:
                raise ValueError
            return batch

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_graph,
            num_workers=num_workers)
