from utils.estimators import EstimatorWrapper
from typing import Tuple, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator


#
# cross validation
#

def cross_validate(
    estimator: EstimatorWrapper,
    X: np.ndarray, 
    y: np.ndarray, 
    metric: Callable[[np.ndarray, np.ndarray], float],
    use_decision: bool,
    cv: BaseCrossValidator = StratifiedKFold(n_splits=5, shuffle=True)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Params
    -------
    use_decision : bool
        should metric be called with estimator's .predict or .decision output
    
    Returns
    -------
    split_metrics_train: np.ndarray
        each cross-validation model's evaluated metric on its train set

    split_metrics_val : np.ndarray
        each cross-validation model's evaluated metric on its validation set
    
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

    return \
        split_metrics_train,\
        split_metrics_val,\
        y_pred
