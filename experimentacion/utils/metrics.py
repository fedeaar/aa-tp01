from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_curve
import numpy as np


#
# metrics
#

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred)


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def aucroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, recall, _ = roc_curve(y_true, y_prob)
    return auc(fpr, recall)
