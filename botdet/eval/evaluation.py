import torch
from sklearn.metrics import roc_auc_score

from .metrics import *


def eval_metrics(target, pred_prob, threshold=0.5):
    """
    Calculate a group of evaluation metrics for a model's prediction for target binary labels.

    :param target: must be numpy.ndarray or torch.Tensor.
    :param pred_prob: should be the probabilities, instead of binary classification results.

    :return: dict
    """
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.cpu().numpy()

    pred = (pred_prob >= threshold).astype(int)

    acc = accuracy(pred, target)
    fpr = false_positive_rate(pred, target)
    fnr = false_negative_rate(pred, target)
    rec = recall(pred, target)
    prc = precision(pred, target)
    f1 = f1_score(pred, target)
    auroc = roc_auc_score(target, pred_prob)
    result_dict = {'acc': acc, 'fpr': fpr, 'fnr': fnr, 'rec': rec, 'prc': prc, 'f1': f1, 'auroc': auroc}

    return result_dict


def dict_value_add(dict1, dict2):
    """Add values with same keys from two dictionaries."""
    result = {key: dict1.get(key, 0) + dict2.get(key, 0)
              for key in set(dict1) | set(dict2)}
    '''
    # This has an issue of only keeping track of >0 values!
    from collections import Counter
    result = dict(Counter(dict1) + Counter(dict2))
    '''
    return result


def dict_value_div(dict, n):
    """Divide all the values in the dictionary by a number."""
    result = {key: value / n for key, value in dict.items()}
    return result


def eval_predictor(dataset, predictor):
    """Evaluate a predictor on the botnet dataset.

    `dataset` is a `BotnetDataset` object containing train/val/test graphs.
        A data loader can also be used, but to be consistent for the average evaluation results, the batch size of
        the data loader should set to 1.
    `predictor` is a callable function that takes in a graph and returns prediction probabilities for node labels,
        as well as the loss if it's from a model.
        This is a simple wrapper of models for one prediction forward pass, and some examples are shown below.
    """
    result_dict_avg = {}
    loss_avg = 0

    for data in dataset:
        # prediction
        try:
            pred_prob, loss = predictor(data)
            loss_avg += loss
        except ValueError:  # if "too many values to unpack"
            pred_prob = predictor(data)

        # get the ground truth target
        if dataset.graph_format == 'pyg':
            target = data.y
        elif dataset.graph_format == 'dgl':
            target = data.ndata['y']
        elif dataset.graph_format == 'nx':
            raise NotImplementedError
        elif dataset.graph_format == 'dict':
            target = data['y']
        else:
            raise ValueError

        # compute the evaluation metrics
        result_dict = eval_metrics(target, pred_prob)

        result_dict_avg = dict_value_add(result_dict_avg, result_dict)

    # average the metrics across all graphs in the dataset as final results
    result_dict_avg = dict_value_div(result_dict_avg, len(dataset))
    loss_avg = loss_avg / len(dataset)

    return result_dict_avg, loss_avg


# =================================================================================================================
# some examples of the 'predictor' model wrapper to be fed into the above evaluation function (for PyG Data format)
# =================================================================================================================
class PygRandomPredictor:
    def __init__(self):
        # torch.manual_seed(0)
        pass

    def __call__(self, data):
        pred_prob = torch.rand(len(data.y))
        return pred_prob


class PygModelPredictor:
    def __init__(self, model, loss_fcn=torch.nn.CrossEntropyLoss()):
        self.model = model
        self.loss_fcn = loss_fcn
        self.device = next(model.parameters()).device

    def __call__(self, data):
        self.model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            # custom the below line to adjust to your model's input format for forward pass
            out = self.model(data.x, data.edge_index)
            loss = self.loss_fcn(out, data.y.long())
            pred_prob = torch.softmax(out, dim=1)[:, 1]
        return pred_prob, loss.float()
