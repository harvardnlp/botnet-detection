# botnet-detection

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](http://img.shields.io/badge/paper-arxiv.2003.06344-B31B1B.svg)](https://arxiv.org/abs/2003.06344)

Topological botnet detection datasets and automatic detection with graph neural networks.

<!--The graphs are of relatively large scale and featureless. Each dataset contains a specific botnet topology, with 960 graphs in total, randomly split to train/val/test sets. There are labels on both nodes and edges indicating whether they were in the botnet (evil) community. Learning tasks could target at predicting on nodes to detect whether they are botnet nodes, or recovering the whole botnet community by also predicting on edges as whether they belong to the original botnet.-->

<p align="left">
  <img width="30%" src=./pictures/p2p.png />
</p>

A collection of different botnet topologyies overlaid onto normal background network traffic, containing featureless graphs of relatively large scale for inductive learning.

## Installation

From source 
```
git clone https://github.com/harvardnlp/botnet-detection
cd botnet-detection
python setup.py install
```

## To Load the Botnet Data

We provide standard and easy-to-use dataset and data loaders, which automatically handle the dataset dnowloading as well as standard data splitting, and can be compatible with most of the graph learning libraries by specifying the `graph_format` argument:

```
from botdet.data.dataset_botnet import BotnetDataset
from botdet.data.dataloader import GraphDataLoader

botnet_dataset_train = BotnetDataset(name='chord', split='train', graph_format='pyg')
botnet_dataset_val = BotnetDataset(name='chord', split='val', graph_format='pyg')
botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')

train_loader = GraphDataLoader(botnet_dataset_train, batch_size=2, shuffle=False, num_workers=0)
val_loader = GraphDataLoader(botnet_dataset_val, batch_size=1, shuffle=False, num_workers=0)
test_loader = GraphDataLoader(botnet_dataset_test, batch_size=1, shuffle=False, num_workers=0)
```

The choices for dataset `name` are (indicating different botnet topologies):
- `'chord'` (synthetic, 10k botnet nodes)
- `'debru'` (synthetic, 10k botnet nodes)
- `'kadem'` (synthetic, 10k botnet nodes)
- `'leet'` (synthetic, 10k botnet nodes)
- `'c2'` (real, ~3k botnet nodes)
- `'p2p'` (real, ~3k botnet nodes)

The choices for dataset `graph_format` are (for different graph data format according to different graph libraries):
- `'pyg'` for [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- `'dgl'` for [DGL](https://github.com/dmlc/dgl) 
- `'nx'` for [NetworkX](https://github.com/networkx/networkx)
- `'dict'` for [plain python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)

Based on different choices of the above argument, when indexing the botnet dataset object, it will return a corresponding graph data object defined by the specified graph library.

The data loader handles automatic batching and is agnostic to the specific graph learning library.


## To Evaluate a Model Predictor

We prepare a standardized evaluator for easy evaluation and comparison of different models.
First load the dataset class with `BotnetDataset` and the evaluation function `eval_predictor`.
Then define a simple wrapper of your model as a predictor function (see [examples](botdet/eval/evaluation.py#L99)), which takes in a graph from the dataset and returns the prediction probabilities for the positive class (as well as the loss from the forward pass, optionally).

We mainly use the average F1 score to compare across models. For example, to get evaluations on the `chord` test set:

```
from botdet.data.dataset_botnet import BotnetDataset
from botdet.eval.evaluation import eval_predictor
from botdet.eval.evaluation import PygModelPredictor

botnet_dataset_test = BotnetDataset(name='chord', split='test', graph_format='pyg')
predictor = PygModelPredictor(model)    # 'model' is some graph learning model
result_dict_avg, loss_avg = eval_predictor(botnet_dataset_test, predictor)

print(f'Testing --- loss: {loss_avg:.5f}')
print(' ' * 10 + ', '.join(['{}: {:.5f}'.format(k, v) for k, v in result_dict_avg.items()]))

test_f1 = result_dict_avg['f1']
```

## To Train a Graph Neural Network for Topological Botnet Detection

We provide a set of graph convolutional neural network (GNN) models [here](./botdet/models_pyg) with PyTorch Geometric, along with the corresponding [training script](./train_botnet.py).
Various basic GNN models can be constructed and tested by specifing configuration arguments:
- number of layers, hidden size
- node updating model each layer (e.g. direct message passing, MLP, gated edges, or graph attention)
- message normalization
- residual hops
- final layer type
- etc. (check the [model API](./botdet/models_pyg/gcn_model.py#L9) and the [training script](./train_botnet.py#L71))

<!--One can use our main [model API](./botdet/models_pyg/gcn_model.py#L9) to construct various basic GNN models, by specifing different number of layers, how in each layer node representations are updated (e.g. with direct message passing, MLP, or with graph attention), different choices of non-linear activation functions, whether to use residual connections and how many hops to connect, whether to add a final projection layer or not, etc. For a complete list of model configuration arguments, check our [example training script](./train_botnet.py#L71).-->

As an example, to train a GNN model on the topological botnet datasets, simply run:
```
bash run_botnet.sh
```

With the above configuration, we run graph neural network models (with 12 layers, 32 hidden dimension, random walk normalization, and residual connections) on each of the topologies, and results are as below:

<!--| Topology | Chord | de Bruijn | Kademlia | LEET-Chord | C2 | P2P |-->
<!--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|-->
<!--| Test F1 | | | | | | |-->
<!--| Average Over Topologies <td colspan=6> 0 </td>|-->

<table align="center">
  <tr>
    <td> Topology </td>
    <td> Chord </td>
    <td> de Bruijn </td>
    <td> Kademlia </td>
    <td> LEET-Chord </td>
    <td> C2 </td>
    <td> P2P </td>
  </tr>
    
  <tr>
    <td> Test F1 (%) </td>
    <td>  99.061 </td>
    <td>  99.926 </td>
    <td>  98.935 </td>
    <td>  99.231 </td>
    <td>  98.992 </td>
    <td>  98.692 </td>
  </tr>
  <tr>
    <td style="text-align:center"> Average </td>
    <td colspan="6"> 99.140 </td>
  </tr>
</table>

#### Note

We also provide labels on the edges under the name `edge_y`, which can be used for the complete botnet community recovery task, or for interpretation matters.

## Citing

```
@article{zhou2020auto,
  title={Automating Botnet Detection with Graph Neural Networks},
  author={Jiawei Zhou*, Zhiying Xu*, Alexander M. Rush, and Minlan Yu},
  journal={AutoML for Networking and Systems Workshop of MLSys 2020 Conference},
  year={2020}
}
```
