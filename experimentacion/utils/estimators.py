from abc import ABC, abstractmethod
from typing_extensions import Self
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np


#
# model wrappers
#

class EstimatorWrapper(ABC):
    estimator: BaseEstimator
    
    @abstractmethod
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    @abstractmethod
    def decision(self, X: np.ndarray) -> np.ndarray:
        """
        wrapper for predict_proba or decision_function
        """
        pass

    def clone(self, **params) -> Self:
        kwargs = self.estimator.get_params()
        kwargs.update(params)
        return type(self)(**kwargs)


class DecisionTreeWrapper(EstimatorWrapper):
    estimator: DecisionTreeClassifier

    def __init__(self, **kwargs):
        self.estimator = DecisionTreeClassifier(**kwargs)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class ForestTreeWrapper(EstimatorWrapper):
    estimator: RandomForestClassifier

    def __init__(self, **kwargs):
        self.estimator = RandomForestClassifier(**kwargs)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


class SVCWrapper(EstimatorWrapper):
    estimator: SVC

    def __init__(self, **kwargs):
        self.estimator = SVC(**kwargs)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.decision_function(X)


class LDAWrapper(EstimatorWrapper):
    estimator: LinearDiscriminantAnalysis

    def __init__(self, **kwargs):
        self.estimator = LinearDiscriminantAnalysis(**kwargs)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.decision_function(X)
