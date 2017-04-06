import dA
import sys
import numpy
from utils import *
from sklearn.preprocessing import scale
from scipy import stats
import csv


class Executor (object):

    def trainAndexecute (self,dbPath,scoresPath,visible,trainToFeedForwardThresh):
        da=dA.dA(None,None,visible,int(visible/3))

        with open(dbPath, 'rt') as csvin:
            with open(scoresPath, 'wt') as csvout:
                csvin = csv.reader(csvin, delimiter=',')

                maxs,mins=self.findMaxsAndMins(dbPath)

                for i, row in enumerate(csvin):
                    if i % 10000 == 0:
                        print(i)
                    if i > 0:  # not header
                        for j in range(len(row)-1):
                            row[j]=float(row[j])
                    else:
                        continue
                    if len(row)-1!=len(maxs):
                        print ("rejected "+str(i))
                        continue
                    else:
                        for number in range(len(row)-1):
                            row[number]=(row[number]-mins[number])/(maxs[number]-mins[number])

                    #row = (row - mins) / (maxs - mins)

                    if i<trainToFeedForwardThresh:
                        score=da.train(0.1,0.3,numpy.array(row[:len(row)-1]))
                    else:

                        score=da.feedForward(0.1,0.3,numpy.array(row[:len(row)-1]))

                    csvout.write(str(score)+","+str(row[len(row)-1])+"\n")

    def findMaxsAndMins(self,dbPath):


            maxs = numpy.ones((111,)) * -numpy.Inf
            mins = numpy.ones((111,)) * numpy.Inf

            with open(dbPath, 'rt') as csvin:
                csvin = csv.reader(csvin, delimiter=',')

                for i, row in enumerate(csvin):

                    if i % 10000 == 0:
                        print(i)
                    if i > 0:  # not header
                        for j in range(len(row) - 1):
                            row[j] = float(row[j])
                        # x = x[numpy.array([1,2,4,5,7,8,10,11,13,14,16,17])-1]

                        if len (row)<len(maxs):
                            continue
                        for idx in range(0, len(maxs)):
                            if row[idx] > maxs[idx]:
                                maxs[idx] = row[idx]
                            if row[idx] < mins[idx]:
                                mins[idx] = row[idx]
            print (maxs)
            print (mins)
            return maxs,mins



ex=Executor()
#ex.trainAndexecute('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/videoJak_full_onlyNetstat.csv','/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/videoJak_full_onlyNetstat_scores.csv',111,1750648)

#ex.trainAndexecute('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/SYN_full_onlyNetstat.csv','/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/SYN_full_onlyNetstat_scores.csv',111,1536268)

ex.trainAndexecute('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/piddle_FULL_onlyNetstat.csv','/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/piddle_FULL_onlyNetstat_scores.csv',111,5179941)
