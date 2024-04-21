from typing import Tuple
import numpy as np


#
# data wrangling
#

def train_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_size: float = 0.9, 
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    params
    ------
        y : np.ndarray 0-1 target labels corresponding to X
    """
    generator = np.random.default_rng(seed)
    
    pMask = y == 1
    pInstances = X[pMask]
    nInstances = X[~pMask]

    generator.shuffle(pInstances)
    generator.shuffle(nInstances)

    X_pTrain, X_pTest = np.split(pInstances, [int(len(pInstances)*train_size)])
    X_nTrain, X_nTest = np.split(nInstances, [int(len(nInstances)*train_size)])

    X_train = np.concatenate((X_pTrain, X_nTrain))
    y_train = np.array([1] * len(X_pTrain) + [0] * len(X_nTrain))
    trainShuffle = generator.permutation(len(y_train))

    X_test = np.concatenate((X_pTest, X_nTest))
    y_test = np.array([1] * len(X_pTest) + [0] * len(X_nTest))
    testShuffle = generator.permutation(len(y_test))

    return X_train[trainShuffle], X_test[testShuffle], y_train[trainShuffle], y_test[testShuffle]
