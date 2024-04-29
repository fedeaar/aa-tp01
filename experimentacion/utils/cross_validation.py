from utils.estimators import EstimatorWrapper
from typing import Tuple, Callable
import numpy as np
import pandas as pd 
from sklearn.model_selection import BaseCrossValidator


#
# cross validation
#

def cross_validate(
    estimator: EstimatorWrapper,
    X: np.ndarray, 
    y: np.ndarray, 
    metric: Callable[[np.ndarray, np.ndarray], float],
    use_decision: bool,
    cv: BaseCrossValidator,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Params
    ------
    use_decision : bool
        should metric be called with estimator's .predict or .decision output
    
    Returns
    -------
    results : pd.DataFrame
        pandas DataFrame with columns:
            mean_val : float
                the mean value of split_val

            mean_train : float
                the mean value of split_train

            tot_val : float
                the evaluation of the metric on y and y_pred
    
            split_val_[0:k) : float
                each cross-validation model's evaluated metric on its validation set
    
            split_train_[0:k): float
                each cross-validation model's evaluated metric on its train set

    y_pred : np.ndarray
        the whole prediction / decision results on validation. ie. each 
        instance's prediction / decision value from the cross-validation model 
        that evaluated it on validation.
    """
    y_pred = np.zeros(y.shape, dtype=float)
    k = cv.get_n_splits()
    split_metrics_train = np.zeros(k)
    split_metrics_val = np.zeros(k)

    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[test_idx], y[test_idx]
        model = estimator.clone()
        model.fit(X_train, y_train)
        if use_decision:
            pred_train = model.decision(X_train)
            pred_val = model.decision(X_val)
        else:
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
        split_metrics_train[idx] = metric(y_train, pred_train)
        split_metrics_val[idx] = metric(y_val, pred_val)
        y_pred[test_idx] = pred_val

    data = np.hstack((np.array([
        split_metrics_val.mean(),
        split_metrics_train.mean(),
        metric(y, y_pred)
    ]), split_metrics_val, split_metrics_train)).reshape((1, 3+2*k))
    result = pd.DataFrame(
        data,
        index=[metric.__name__],
        columns=["mean_val", "mean_train", "tot_val"] + \
            [f"split_val_{i}" for i in range(k)] + \
            [f"split_train_{i}" for i in range(k)]
    )

    return result, y_pred
