"""
Evaluation metrics for binary classification.
`pred` and `target` are both `torch.LongTensor`s of the same length.
Also works when `pred` and `target` are both `numpy.ndarray`s of the same length.
"""


def accuracy(pred, target):
    return (pred == target).sum().item() / len(target)


def true_positive(pred, target):
    return (target[pred == 1] == 1).sum().item()


def false_positive(pred, target):
    return (target[pred == 1] == 0).sum().item()


def true_negative(pred, target):
    return (target[pred == 0] == 0).sum().item()


def false_negative(pred, target):
    return (target[pred == 0] == 1).sum().item()


def recall(pred, target):
    """
    Or true positive rate.
    """
    try:
        return true_positive(pred, target) / (target == 1).sum().item()
    except:  # divide by zero
        return -1


def precision(pred, target):
    try:
        prec = true_positive(pred, target) / (pred == 1).sum().item()
        return prec
    except:  # divide by zero
        return -1


def f1_score(pred, target):
    prec = precision(pred, target)
    rec = recall(pred, target)
    try:
        return 2 * (prec * rec) / (prec + rec)
    except:
        return 0


def false_positive_rate(pred, target):
    try:
        return false_positive(pred, target) / (target == 0).sum().item()
    except:  # divide by zero
        return -1


def false_negative_rate(pred, target):
    """
    Or 1 - recall/true_positive_rate
    """
    try:
        return false_negative(pred, target) / (target == 1).sum().item()
    except:  # divide by zero
        return -1
