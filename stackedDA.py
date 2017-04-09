# -*- coding: utf-8 -*-

import sys
import numpy
#from utils import *
#from sklearn.preprocessing import scale
#from scipy import stats
class stackedDA(object):
    def __init__(self, indexesMap=None,input=None, n_visible=2, n_hidden=3, n_hidden2=3,  \
                 W=None,W2=None,W3=None, hbias=None, vbias=None, vbias2=None, rng=None):
        self.indexesMap=indexesMap
        self.n_visible = n_visible  # num of units in visible (input) layer
        self.n_hidden = n_hidden  # num of units in hidden layer
        self.n_hidden2=n_hidden2
        if rng is None:
            rng = numpy.random.RandomState(1234)

        if W is None:

            a = 1. / n_visible
            W = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_visible, n_hidden)))

        if W2 is None:
            a = 1. / n_hidden
            W2 = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_hidden, n_hidden2)))


        if W3 is None:

            a = 1. / n_hidden2
            W3 = numpy.array(rng.uniform(  # initialize W uniformly
                low=-a,
                high=a,
                size=(n_hidden2, n_visible)))


        if hbias is None:
            hbias = numpy.zeros(n_hidden)  # initialize h bias 0

        if vbias is None:
            vbias = numpy.zeros(n_hidden2)  # initialize v bias 0


        if vbias2 is None:
            vbias2 = numpy.zeros(n_visible)  # initialize v bias 0


        self.rng = rng
        self.x = input
        self.W = W
        self.W2=W2
        self.W3=W3
        #self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias
        self.vbias2=vbias2

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

    def get_hidden_values2(self, input):
        return self.sigmoid(numpy.dot(input, self.W2) + self.vbias)


    # Decode
    def get_reconstructed_input(self, hidden):
        return self.sigmoid(numpy.dot(hidden, self.W3) + self.vbias2)


    def train(self, lr=0.1, corruption_level=0.3, input=None):
        if input is not None:


            self.x = input
        #if has indexesMap
        if self.indexesMap!=None:
            input = numpy.array(input)
            self.x=numpy.array(input[self.indexesMap])
            self.x=self.x.astype(float)


        x = self.x
        tilde_x = self.get_corrupted_input(x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        y2=self.get_hidden_values2(y)
        z = self.get_reconstructed_input(y2)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)
        L_h0=numpy.dot(L_h1,self.W2)*y2*(1-y2)
        L_h00=numpy.dot(L_h0,self.W3)*z*(1-z)

        L_vbias = L_h2
        L_hbias = L_h1
        L_vbias2=L_h0

        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)
        L_W2=numpy.outer(y.T,L_h0)+numpy.outer(L_h1.T,y2)
        L_W3=numpy.outer(y2.T,L_h00)+numpy.outer(L_h0.T,z)

        self.W += lr * L_W
        self.W2+=lr*L_W2
        self.W3+=lr*L_W3

        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)
        self.vbias2+=lr*numpy.mean(L_vbias2,axis=0)
        eps = 10 ** -10
        minVal = min(z);
        maxVal = max(z);
        z = (z - minVal) / (maxVal - minVal + eps)
        A = self.x * numpy.log(z + eps)
        B = (1 - self.x) * numpy.log(1 - z + eps)

        cross_entropy  = - numpy.mean(self.x * numpy.log(z+eps) + (1 - self.x) * numpy.log(1 - z+eps))

        return cross_entropy

    def feedForward(self,  lr=0.1, corruption_level=0.3, input=None):
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
        y2 = self.get_hidden_values2(y)
        z = self.get_reconstructed_input(y2)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)
        L_h0 = numpy.dot(L_h1, self.W2) * y2 * (1 - y2)
        L_h00 = numpy.dot(L_h0, self.W3) * z * (1 - z)

        L_vbias = L_h2
        L_hbias = L_h1
        L_vbias2 = L_h0

        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)
        L_W2 = numpy.outer(y.T, L_h0) + numpy.outer(L_h1.T, y2)
        L_W3 = numpy.outer(y2.T, L_h00) + numpy.outer(L_h0.T, z)

        """
        self.W += lr * L_W
        self.hbias += lr * numpy.mean(L_hbias, axis=0)
        self.vbias += lr * numpy.mean(L_vbias, axis=0)
        """
        eps = 10 ** -10
        minVal = min(z);
        maxVal = max(z);
        z = (z - minVal) / (maxVal - minVal + eps)
        A = self.x * numpy.log(z + eps)
        B = (1 - self.x) * numpy.log(1 - z + eps)

        cross_entropy = - numpy.mean(self.x * numpy.log(z + eps) + (1 - self.x) * numpy.log(1 - z + eps))

        return cross_entropy
    def negative_log_likelihood(self, corruption_level=0.3):
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        y2 = self.get_hidden_values2(y)
        z = self.get_reconstructed_input(y2)

        cross_entropy = - numpy.mean(self.x * numpy.log(z) +
                      (1 - self.x) * numpy.log(1 - z))

        return cross_entropy

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        y2 = self.get_hidden_values2(y)
        z = self.get_reconstructed_input(y)
        return z

    def score(self,x):
        eps=10**-10
        z = self.reconstruct(x)
        return - numpy.mean(self.x * numpy.log(z+eps) + (1 - self.x) * numpy.log(1 - z+eps))



"""
sda=stackedDA()
for x in range(10000):
    print(sda.train(0.1,0.3,numpy.array([0.1,0.2])))
for x in range(3):
    print(sda.feedForward(0.1,0.3,numpy.array([0.11,0.21])))
"""

