import csv
import numpy as np
from operator import itemgetter
def defineClusterForEachAttr (matPath,numOfClusters=16):

    with open(matPath,'rb') as clFile:
        mat=csv.reader(clFile)
        clList=np.array(list(mat))
        clusters={}
        for i in range(numOfClusters):
            clusters[i]=[]
        maxClusterSize=int(111 / numOfClusters)
        for j in range(len(clList[0])): #foreach column=attribute in the 2d array
            col=np.array(clList[...,j]).astype(float)

            #minIdx=sorted(e for i,e in enumerate(col))
            indices, L_sorted = zip(*sorted(enumerate(col), key=itemgetter(1)))
            for k in range(len(indices)):

                if len(clusters[indices[k]])<maxClusterSize:
                    clusters[indices[k]].append(j)
                    break
        clusterDistribution=[]
        for i in range(111):
            for j in range(len(clusters)):
                if i in clusters[j]:
                    clusterDistribution.append(j+1)
                    break
        print ("finished clustering")
        return clusterDistribution

#defineClusterForEachAttr(matPath='D:/datasets/featureClustering/fcm/fcm_syn.csv')