from abc import ABC, abstractmethod
from typing_extensions import Self
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
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

    def clone(self) -> Self:
        return type(self)(**self.estimator.get_params())


class DecisionTree(EstimatorWrapper):
    estimator: DecisionTreeClassifier

    def __init__(self, **kwargs):
        self.estimator = DecisionTreeClassifier(**kwargs)

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]
