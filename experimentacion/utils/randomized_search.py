import numpy as np 

import pandas as pd
import sklearn as sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, ParameterGrid

from sklearn.model_selection import RandomizedSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.fixes import loguniform

from utils.base_set import X_train, y_train, seed
from utils.cross_validation import cross_validate
import utils.metrics as metrics
from utils.estimators import DecisionTree

class logUnifD:
    distr = loguniform
    def __init__(self,*args,**kwds):
        self.distr = loguniform(*args,**kwds)
    
    def rvs(self,*args,**kwds):
        samples = self.distr.rvs(*args,**kwds)
        discretized_samples = np.round(samples).astype(int)
        return discretized_samples
"""
def randomized_search(
    estimators: np.array,
    params: np.array, 
    scoring: str,
    cv: int
): 
    for enumerate
"""
