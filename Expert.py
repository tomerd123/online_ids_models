import dA as ae
import math
import netStat as ns
import numpy as np
import AfterImage as ai
class Expert:
    def __init__(self,hiddenRatio,featureIndxs,gracePeriod=10000): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        self.gracePeriod = gracePeriod
        self.featureIndxs = featureIndxs
        self.n = 0 #number of instances seen by this expert so far
        numFeats = len(featureIndxs)
        rng = np.random.RandomState(123)
        self.AE = ae.dA(n_visible=numFeats, n_hidden=math.ceil(numFeats*hiddenRatio), rng=rng)
        self.train_stats = ai.incStat(0,np.Inf) #used to track rolling std

    def train(self,x):
        self.n = self.n + 1
        error = self.AE.train(corruption_level=0.0, input=x[self.featureIndxs])
        if self.n > self.gracePeriod:
            self.train_stats.insert(error)

    def score(self,x):
        return self.AE.score(x[self.featureIndxs])

    def vote(self,x):
        std = self.train_stats.std()
        if std == 0:
            return np.nan
        else:
            return (self.score(x[self.featureIndxs])-self.train_stats.mean())/std #zscore normalized error
