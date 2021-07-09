## Graph Data

For each graph, the associated data are:
- `x`: node signals/features, of size (N, C)
- `edge_index`: edge indexes, of size (2, E)
- `y` (for node classification): node labels, of size (N,), each element an integer in \[0, num_class)
- `edge_attr` (optional): edge attributes/features, of size (E, D)
- `edge_y` (for edge classification): edge labels, of size (E,), each element an integer in \[0, num_class)
- any others

where
* N: number of nodes
* E: number of edges
* C: dimension of node features
* D: dimension of edge features
* num_class: number of labels (this is 2 for chord data based on whether a node is "evil")

## Storage Format

The graph data is stored in HDF5 files. 

The Python library to handle HDF5 data io is `h5py` (basic) and `deepdish` (higher level).

The HDF5 file works like dictionaries of NumPy arrays. Here are how the graph data is structured in storage.

### Collection of graphs

```
\ (file root)
├── '0' (graph id)
│   ├── 'x'
│   ├── 'edge_index'
│   ├── 'y'
│   ├── 'edge_attr'
│   ├── 'edge_y'
│   ├── ... (any other data)
│   └── attrs (metadata attributes for the current graph)
|       ├── 'num_nodes'
|       ├── 'num_edges'
|       ├── 'num_evils'
|       ├── 'num_evil_edges'
│       └── ... (any other attributes, such as 'is_directed', 'contains_self_loops', etc.)
├── ...
├── '800'
│   ├── 'x'
│   ├── 'edge_index'
│   ├── 'y'
│   ├── 'edge_attr'
│   ├── 'edge_y'
│   ├── ... (any other data)
│   └── attrs
|       ├── 'num_nodes'
|       ├── 'num_edges'
|       ├── 'num_evils'
|       ├── 'num_evil_edges'
│       └── ... (any other attributes, such as 'is_directed', 'contains_self_loops', etc.)
├── ...
└── attrs (metadata attributes for the whole graph data set)
    ├── 'num_graphs'
    ├── 'num_nodes_avg'
    ├── 'num_edges_avg'
    ├── 'num_evils_avg'
    ├── 'is_directed'
    ├── 'contains_self_loops'
    └── ... (any other attributes, such as 'contains_multi_edges', etc.)
```

**Note:**
- The root level is an `h5py.File` object, and then the `h5py.Group`s under it are different graphs.
- The top level keys are graph ids (keys need to be `str`), and each graph contains different `h5py.Dataset`s such as `x` and `edge_index`.
- Each graph can have some *attributes* aside to describe the graph, such as "num_nodes" and "num_edges" in the graph.
- The root level file could also have some *attributes* to describe some statistics of the graph dataset, such as "num_graphs" in the dataset.

### Write and Read

For writing graph data, check code in this directory.

For reading graph data, one can do:
```
import h5py
with h5py.File('filename', 'r') as f:
    e = f['0']['edge_index'][()]             # take out the edge indexes from the first graph with id '0'
    num_nodes = f['0'].attrs['num_nodes']    # access the statistics stored in attributes of the first graph with id '0'
    num_graphs = f.attrs['num_graphs']       # access the statistics stored in attributes of the dataset file
```

Or using `deepdish`:
```
import deepdish as dd
graphs = dd.io.load('filename')        # this will return a dictionary, with each graph as a sub-dictionary, and
                                       # attributes are key-item pairs in the dictionaries as well
```
`deepdish` also allows partial loading of the data based on data path and slice indexes.
