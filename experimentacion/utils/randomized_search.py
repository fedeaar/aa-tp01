import numpy as np 
import pandas as pd
import scipy.stats as stats 

class logUnifD:
    distr: stats.loguniform

    def __init__(self, *args, **kwds):
        self.distr = stats.loguniform(*args, **kwds)
    
    def rvs(self, *args, **kwds):
        samples = self.distr.rvs(*args,**kwds)
        discretized_samples = np.round(samples).astype(int)
        return discretized_samples


class pNpUniform:
    distr: stats.uniform

    def __init__(self ,*args, **kwds):
        self.distr = stats.uniform(*args,**kwds)
    
    def rvs(self, *args, **kwds):
        sample = self.distr.rvs(*args, **kwds)
        return sample, 1 - sample


class pNpNormal:
    distr: stats.norm

    def __init__(self ,*args, **kwds):
        self.distr = stats.norm(*args,**kwds)
    
    def rvs(self, *args, **kwds):
        sample = self.distr.rvs(*args, **kwds)
        if sample < 0:
            sample = 0
        if sample > 1:
            sample = 1
        return sample, 1 - sample


def rs_results(cv_results: dict, params: dict) -> pd.DataFrame:
    columns_to_keep = [f"param_{param}" for param in params.keys()] + ['mean_test_score', 'rank_test_score']
    return pd.DataFrame(cv_results).sort_values("rank_test_score")[columns_to_keep]
