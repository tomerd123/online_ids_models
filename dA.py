# -*- coding: utf-8 -*-

import json
import sys
from utils import *
import numpy



class dA(object):
    def __init__(self, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, rng=None):

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
        self.W = W
        self.W_prime = self.W.T
        self.hbias = hbias
        self.vbias = vbias

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input

    # Encode
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x, lr=0.1, corruption_level=0.3):
        if corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, corruption_level)
        else:
            tilde_x = x
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
        #A = self.x * numpy.log(z)
        #B = (1 - self.x) * numpy.log(1 - z)
        #return - numpy.mean(x * numpy.log(z) + (1 - x) * numpy.log(1 - z))
        return ((x - z) ** 2).mean()  # MSE
        #
        # cross_entropy = - numpy.mean(x * numpy.log(z) +
        #               (1 - x) * numpy.log(1 - z))
        #
        # return cross_entropy

    def reconstruct(self, x):
        y = self.get_hidden_values(x)
        z = self.get_reconstructed_input(y)
        return z

    def score(self,x):
        z = self.reconstruct(x)
        #return - numpy.mean(x * numpy.log(z) + (1 - x) * numpy.log(1 - z))
        return ((x - z) ** 2).mean() #MSE

    def toJSON(self):
        AE = {}
        AE['n_visible'] = self.n_visible
        AE['n_hidden'] = self.n_hidden
        AE['W'] = self.W.tolist()
        AE['W_prime'] = self.W_prime.tolist()
        AE['hbias'] = self.hbias.tolist()
        AE['vbias'] = self.vbias.tolist()
        return json.dumps(AE)

    def loadFromJSON(self,JSONstring):
        J = json.loads(JSONstring)
        self.n_visible = J['n_visible']
        self.n_hidden = J['n_hidden']
        self.W = numpy.array(J['W'])
        self.W_prime = numpy.array(J['W_prime'])
        self.hbias = numpy.array(J['hbias'])
        self.vbias = numpy.array(J['vbias'])