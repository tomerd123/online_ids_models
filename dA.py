# -*- coding: utf-8 -*-

import sys
import numpy
import AfterImage as af
#from utils import *
#from sklearn.preprocessing import scale
#from scipy import stats
class dA(object):
    def __init__(self, indexesMap=None,input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, rng=None):
        self.indexesMap=indexesMap
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer


        if rng is None:
            rng = numpy.random.RandomState(1234)

        if W is None:
            a = 1. / n_visible
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_visible)  # initialize v bias 0

        self.rng = rng
        self.x = input
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias


    def sigmoid(self, z):

        eps=10**-10
        z = 1.0 / (1.0 + numpy.exp(-z))
        return z

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        rngBinRes=self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level)

        return  rngBinRes* input

    # Encode
    def get_hidden_values(self, input):
        return self.sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return self.sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)


    def train(self, lr=0.1, corruption_level=0.3, input=None):
        if input is not None:

            if max(input)>1:
                m1=min(input)
                m2=max(input)
                input=numpy.array([(input[i]-m1)/(m2-m1+1) for i in range(len(input))])
            self.x = input
        #if has indexesMap
        if self.indexesMap!=None:
            input = numpy.array(input)
            self.x=numpy.array(input[self.indexesMap])
            self.x=self.x.astype(float)


        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += lr * L_W
        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)
        eps = 10 ** -10
        #minVal = min(z);
        #maxVal = max(z);
        #z = (z - minVal) / (maxVal - minVal + eps)
        A = self.x * numpy.log(z + eps)
        B = (1 - self.x) * numpy.log(1 - z + eps)

        #cross_entropy  = - numpy.mean(self.x * numpy.log(z+eps) + (1 - self.x) * numpy.log(1 - z+eps))

        #return cross_entropy
        mse = numpy.array(((numpy.array(x) - numpy.array(z)) ** 2)).mean()  # MSE
        return mse

    def execute(self, x, t=-1):  # returns MSE of the reconstruction of x

            z = self.reconstruct(x)
            mse = ((x - z) ** 2).mean()  # MSE

            # rollmean?
            if self.params.rollmean_L != numpy.Inf:
                self.rollmean.insert(mse, t)
                mse = self.rollmean.mean()
            return mse
    def feedForward(self,  lr=0.1, corruption_level=0.3, input=None,t=-1):
        if input is not None:
            self.x = input

        # if has indexesMap
        if self.indexesMap != None:
            input = numpy.array(input)
            self.x = numpy.array(input[self.indexesMap])
            self.x = self.x.astype(float)

        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        """
        self.W += lr * L_W
        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)
        """
        eps = 10 ** -10
        #minVal = min(z);
        #maxVal = max(z);
        #z = (z - minVal) / (maxVal - minVal + eps)
        A = self.x * numpy.log(z + eps)
        B = (1 - self.x) * numpy.log(1 - z + eps)

        cross_entropy = - numpy.mean(self.x * numpy.log(z + eps) + (1 - self.x) * numpy.log(1 - z + eps))
        mse=numpy.array(((numpy.array(x) - numpy.array(z)) ** 2)).mean()  # MSE


        return mse
        #return cross_entropy
    def negative_log_likelihood(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        cross_entropy = - numpy.mean(self.x * numpy.log(z) +
                      (1 - self.x) * numpy.log(1 - z))

        return cross_entropy

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def score(self,x):
        eps=10**-10
        z = self.reconstruct(x)
        return - numpy.mean(self.x * numpy.log(z+eps) + (1 - self.x) * numpy.log(1 - z+eps))