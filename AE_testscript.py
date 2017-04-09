# -*- coding: utf-8 -*-

import sys
import numpy
import csv
import dA
from utils import *
#import AutoEncoder as ae
import Expert as exp

def test_dA(learning_rate=0.1, corruption_level=0.0, training_epochs=50):
    data = numpy.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    rng = numpy.random.RandomState(123)

    # construct dA
    da = dA(input=data, n_visible=20, n_hidden=7, rng=rng)


    # train
    for epoch in range(training_epochs):
        da.train(lr=learning_rate, corruption_level=corruption_level)
        # cost = da.negative_log_likelihood(corruption_level=corruption_level)
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        # learning_rate *= 0.95

    # test
    x = numpy.array([[1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]])


def test_ndA(learning_rate=0.1):
    rng = numpy.random.RandomState(123)

    maxs = numpy.ones((111,)) * -numpy.Inf
    mins = numpy.ones((111,)) * numpy.Inf

    with open('datasets\\physMIMCsv.csv', 'rt') as csvin:
        csvin = csv.reader(csvin, delimiter=',')
        data = list()
        for i, row in enumerate(csvin):
            if i % 10000 == 0:
                print(i)
            if i > 0: #not header
                x = numpy.asarray(row[64:175]).astype(float)
                data.append([x])
                maxs[x>maxs]=x[x>maxs]
                mins[x < mins] = x[x < mins]

    # construct dA
    da = dA(n_visible=112, n_hidden=30, rng=rng)

    pred =list()
    for i, x in enumerate(data):
            x[0] = (x[0]+mins)/(maxs-mins)
            if i % 10000 == 0:
                print(i)
            if i < 1750648:
                da.train(input=numpy.array(x), lr=learning_rate, corruption_level=0.1)
            else:
                p = da.reconstruct(x)
                mse = ((p[0] - x[0]) ** 2).mean(axis=0)
                pred.append(mse)

def test_bdA(learning_rate=0.1):
    rng = numpy.random.RandomState(123)

    maxs = numpy.ones((111,)) * -numpy.Inf
    mins = numpy.ones((111,)) * numpy.Inf

    with open('Datasets//physMIMCsv.csv', 'rt') as csvin:
        csvin = csv.reader(csvin, delimiter=',')
        data = list()
        L = list()
        for i, row in enumerate(csvin):

            if i % 10000 == 0:
                print(i)
            if i > 0:  # not header
                x = numpy.asarray(row[64:175]).astype(float)
                #x = x[numpy.array([1,2,4,5,7,8,10,11,13,14,16,17])-1]
                data.append([x])
                for idx in range(0,len(maxs)):
                    if x[idx]>maxs[idx]:
                        maxs[idx]=x[idx]
                    if x[idx]<mins[idx]:
                        mins[idx]=x[idx]
                #maxs[x > maxs] = x[x > maxs]
                #mins[x < mins] = x[x < mins]
                L.append(row[175])
            #if i > 100000:
            #    break

    # construct dA

    #da = dA(n_visible=50, n_hidden=30, rng=rng)
    da = exp.Expert(float(1.0/5.0),range(0,111),30000)# ae.AutoEncoder(50,10)

    pred = list()
    labels = list()
    #predstd = list()
    for i, x in enumerate(data):
        x[0] = (x[0] - mins) / (maxs - mins+1)
        if i % 30000 == 0:
            print(i)
        if i < 15000: #1206268: #
            da.train(numpy.array(x[0]))
        else:
            da.AE.feedForward(numpy.array(x[0]))
        pred.append(da.score(x[0]))
        labels.append(L[i])

        #predstd.append(da.vote(x[0]))
            #p = da.feedForward(x)
            #mse = da.MSE(p,x)
            #pred.append(mse)

    with open('Datasets//out5.csv', 'wt') as csvout:
        csvout = csv.writer(csvout)
        for i,v in enumerate(pred):
            csvout.writerow([pred[i],labels[i]])
    # cost = da.negative_log_likelihood(corruption_level=corruption_level)
    # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
    # learning_rate *= 0.95



if __name__ == "__main__":
    with open('Datasets//out5.csv', 'w') as csvout1:
        x=1
    test_bdA()

    # test_dA()

#5179941,5551800