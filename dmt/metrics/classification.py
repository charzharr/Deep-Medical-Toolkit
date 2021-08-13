""" Module metrics.py (By: Charley Zhang, Nov 2020)

Classification metrics:
  Note: these are computed for each class. You must average the results.
    - Confusion matrix (TP, FP, FN, TN)
    - AUCs
    - Specificities
    - Recalls
    - Precisions
    - F1s
    - Accuracies
    - Balanced Accuracies
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pdb

# Turn off sklearn warnings
import warnings
warnings.filterwarnings(action='ignore')


### ======================================================================== ###
### * ### * ### * ### *      Metrics Bundle Coverage     * ### * ### * ### * ###
### ======================================================================== ###


def multiclass_metrics(pred, targ):
    """ IPMI: classification for CIFAR & ISIC.
    Metrics: AUC, F1, acc, sens, spec.
    Parameters:
        pred (NxC tensor) - per class probabilities
        targ (NxC tensor) - one-hot ground truth
    """
    def to_np(tens):
        return tens.float().cpu().detach().numpy()
    
    targ = to_np(targ)
    pred_1hot = to_np(pred.scatter_(1, pred.argmax(1).view(-1,1), 1)).astype('uint8')
    pred_prob = to_np(pred)

    AUCs = compute_aucs(pred_prob, targ)
    ACCs = compute_accuracies(pred_1hot, targ)
    F1s  = compute_f1s(pred_1hot, targ)
    SENs = compute_recalls(pred_1hot, targ)
    bal_ACC = compute_balanced_accuracies(pred_1hot.argmax(1), targ.argmax(1))
    SPEs, Ss = compute_specificities(pred_1hot, targ)
    
    # Ss = (tns, fps, fns, tps)
    TNs, FPs, FNs, TPs = Ss
    overall_f1 = ((2 * TPs) / (2 * TPs + FPs + FNs)).mean()

    return {
        'AUCs': AUCs,
        'ACCs': ACCs,
        'F1s': F1s,
        'SENs': SENs,
        'SPEs': SPEs,
        'TNs': TNs,
        'FPs': FPs,
        'FNs': FNs,
        'TPs': TPs,
        'oF1': overall_f1,
        'F1': F1s.mean(),
        'AUC': AUCs.mean(),
        'ACC': ACCs.mean(),
        'SEN': SENs.mean(),
        'SPE': SPEs.mean(),
        'bACC': bal_ACC
    }


def multilabel_metrics(pred, targ, thresh=0.5):
    pred_1hot = pred >= thresh
    pass


### ======================================================================== ###
### * ### * ### * ### * Per Class Classification Metrics * ### * ### * ### * ###
### ======================================================================== ###


def compute_confusion_matrix(pred, targ, C):
    """ Compute per class classification precision scores.
    Parameters:
        pred (Nx1 tensor or np) - prediction one-hot
        targ (Nx1 tensor or np) - target one-hot
        C - num_classes
    Returns:
        tuple(TN, FP, FN, TP)
    """
    assert targ.shape == pred.shape, f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    if pred.ndim == 2 and pred.shape[1] > 1:
        pred = pred.argmax(1)
        targ = targ.argmax(1) 
    tn, fp, fn, tp = confusion_matrix(targ, pred, labels=range(C)).ravel()
    return tn, fp, fn, tp


def compute_aucs(pred, targ):
    """
    Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        targ: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    AUROCs = np.zeros(targ.shape[-1])
    for i in range(targ.shape[-1]):
        try:
            AUROCs[i] = roc_auc_score(targ[:, i], pred[:, i])
        except ValueError:
            AUROCs[i] = 0
    return AUROCs


def compute_specificities(pred, targ):
    """ Compute per class specificity scores: TN/(TN+FP)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    dims = targ.shape[-1]
    specs = np.zeros(dims)
    tns, fps, fns, tps = np.zeros(dims), np.zeros(dims), np.zeros(dims), np.zeros(dims)
    for i in range(targ.shape[-1]):
        try:
            tn, fp, fn, tp = compute_confusion_matrix(targ[:, i], (pred[:, i]), 2)
            specs[i] = tn / (tn + fp + 1e-7)
            tns[i] = tn
            fps[i] = fp
            fns[i] = fn
            tps[i] = tp
        except ValueError as error:
            print('Error in computing specificity for {}.\n Error msg:{}'.format(i, error))
            specs[i] = 0
    return specs, (tns, fps, fns, tps)


def compute_recalls(pred, targ):
    """ Compute per class classification recall/sensitivity scores: TP/(TP+FN)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    recalls = np.zeros(targ.shape[-1])
    for i in range(targ.shape[-1]):
        try:
            recalls[i] = recall_score(targ[:, i], (pred[:, i]))
        except ValueError as error:
            print('Error in computing recall for {}.\n Error msg:{}'.format(i, error))
            recalls[i] = 0
    return recalls


def compute_precisions(pred, targ):
    """ Compute per class classification precision scores: TP/(TP+FP)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    precisions = np.zeros(targ.shape[-1])
    for i in range(targ.shape[-1]):
        try:
            precisions[i] = recall_score(targ[:, i], (pred[:, i]))
        except ValueError as error:
            print('Error in computing precision for {}.\n Error msg:{}'.format(i, error))
            precisions[i] = 0
    return precisions


def compute_f1s(pred, targ):
    """ Compute per class classification F1 scores: 2*TP/(2*TP+FP+FN)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    f1s = np.zeros(targ.shape[-1])
    for i in range(targ.shape[-1]):
        try:
            f1s[i] = f1_score(targ[:, i], (pred[:, i]))
        except ValueError as error:
            print('Error in computing f1 for {}.\n Error msg:{}'.format(i, error))
            f1s[i] = 0
    return f1s


def compute_accuracies(pred, targ):
    """ Compute per class classification accuracies: (TP+TN)/(TP+TN+FP+FN)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape[1] == pred.shape[1], f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    accuracies = np.zeros(targ.shape[-1])
    for i in range(targ.shape[-1]):
        try:
            accuracies[i] = accuracy_score(targ[:, i], (pred[:, i]))
        except ValueError as error:
            print('Error in computing accuracy for {}.\n Error msg:{}'.format(i, error))
            accuracies[i] = 0
    return accuracies


def compute_balanced_accuracies(pred, targ):
    """ Compute per class classification accuracies: (TP+TN)/(TP+TN+FP+FN)
    Parameters:
        pred (NxC tensor or np) - prediction one-hot
        targ (NxC tensor or np) - target one-hot
    """
    assert targ.shape == pred.shape, f"pred:{pred.shape} targ:{targ.shape}"

    if isinstance(pred, torch.Tensor):
        pred = pred.float().cpu().detach().numpy()
    if isinstance(targ, torch.Tensor):
        targ = targ.float().cpu().detach().numpy()
    assert isinstance(pred, np.ndarray), 'pred must be tensor or np array'
    assert isinstance(targ, np.ndarray), 'targ must be tensor or np array'

    balanced_acc = balanced_accuracy_score(targ, pred)
    return balanced_acc




### Sanity Checks

if __name__ == '__main__':
    # Multi-Class (4x3 tensors)
    pred = torch.tensor([[0.1485, 0.4298, 0.4216],
                         [0.4826, 0.4813, 0.0361],
                         [0.6463, 0.0945, 0.2593],
                         [0.1980, 0.5257, 0.2763]])
    pred_oh = torch.tensor([[0., 1., 0.],
                            [1., 0., 0.],
                            [1., 0., 0.],
                            [0., 1., 0.]])
    targ = torch.tensor([[0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.],
                         [0., 1., 0.]])

    aucs = compute_aucs(pred, targ)
    accs = compute_accuracies(pred_oh, targ)
    f1s = compute_f1s(pred_oh, targ)
    sens = compute_recalls(pred_oh, targ)
    specs = compute_specificities(pred_oh, targ)
    import IPython; IPython.embed(); 

