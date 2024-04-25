import numpy as np 
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
