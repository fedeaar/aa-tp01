from typing import Iterable, Callable
import numpy as np
from sklearn.model_selection import BaseCrossValidator

from utils.cross_validation import cross_validate
from utils.estimators import EstimatorWrapper


def complexity_curve(
    estimator: EstimatorWrapper,
    complexity_param: str, 
    param_range: list | Iterable,
    X: np.ndarray, 
    y: np.ndarray, 
    metric: Callable[[np.ndarray, np.ndarray], float],
    use_decision: bool,
    cv: BaseCrossValidator,
    ax
):
    k = cv.get_n_splits()
    cols_val = [f"split_val_{i}" for i in range(k)]
    cols_train = [f"split_train_{i}" for i in range(k)]
    mean_val_scores = []
    std_val_scores = []
    mean_train_scores = []
    std_train_scores = []
    kwargs = {}
    
    for value in param_range:
        kwargs[complexity_param] = value
        model = estimator.clone(**kwargs)
        auc_scores, _ = cross_validate(model, X, y, metric=metric, use_decision=use_decision, cv=cv)
        mean_val_scores.append(auc_scores.mean_val.iloc[0])
        std_val_scores.append(auc_scores[cols_val].std(axis=1).iloc[0])
        mean_train_scores.append(auc_scores.mean_train.iloc[0])
        std_train_scores.append(auc_scores[cols_train].std(axis=1).iloc[0])

    ax.errorbar(param_range, mean_val_scores, std_val_scores, fmt='-o', label="validation")
    ax.errorbar(param_range, mean_train_scores, std_train_scores, fmt='-o', label="train")

    return mean_val_scores, std_val_scores, mean_train_scores, std_train_scores


def learning_curve(
    estimator: EstimatorWrapper,
    train_range: list | Iterable,
    X: np.ndarray, 
    y: np.ndarray, 
    metric: Callable[[np.ndarray, np.ndarray], float],
    use_decision: bool,
    cv: BaseCrossValidator,
    ax
):
    k = cv.get_n_splits()
    cols_val = [f"split_val_{i}" for i in range(k)]
    cols_train = [f"split_train_{i}" for i in range(k)]
    mean_val_scores = []
    std_val_scores = []
    mean_train_scores = []
    std_train_scores = []
    
    for idx in train_range:
        model = estimator.clone()
        auc_scores, _ = cross_validate(model, X[:idx,], y[:idx,], metric=metric, use_decision=use_decision, cv=cv)
        mean_val_scores.append(auc_scores.mean_val.iloc[0])
        std_val_scores.append(auc_scores[cols_val].std(axis=1).iloc[0])
        mean_train_scores.append(auc_scores.mean_train.iloc[0])
        std_train_scores.append(auc_scores[cols_train].std(axis=1).iloc[0])

    ax.errorbar(train_range, mean_val_scores, std_val_scores, fmt='-o', label="validation")
    ax.errorbar(train_range, mean_train_scores, std_train_scores, fmt='-o', label="train")

    return mean_val_scores, std_val_scores, mean_train_scores, std_train_scores
