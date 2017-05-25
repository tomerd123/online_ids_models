import numpy
#from utils import *
#from sklearn.preprocessing import scale
#from scipy import stats
import dA
import csv
import stackedDA
import ClusteringHandler as ch

class dAEnsemble(object):
    def __init__(self, AEsNumber,indexesMap,input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, rng=None):

        self.AEsList=[]


        counter=1
        for aeIndex in range(AEsNumber):

            ae=dA.dA(indexesMap[counter],n_visible=len(indexesMap[counter]),n_hidden=numpy.amax([ int(len(indexesMap[counter])/3),1]))
            self.AEsList.append(ae)
            self.aggDA=dA.dA(n_visible=AEsNumber,n_hidden=numpy.amax([ int(AEsNumber/3),1])) #added agg

            counter+=1
        self.AEsNumber=AEsNumber

    def trainAndExecute(self,dsPath,ensembleCMeans,maxs,mins,threshold,numOfClusters):
        encounteredAnomaly=0

        with open(dsPath, 'rt') as csvin:
            csvin = csv.reader(csvin, delimiter=',')
            with open(ensembleCMeans,'w') as ensembleCmeansFile:
                for i, row in enumerate(csvin):

                    try:
                        if i!=0 and float(row[111])!=0:
                            encounteredAnomaly=1

                        if i==0:
                            continue
                        if i%10000==0:
                            print(i)
                        totalScore=0
                        input = (numpy.array(row[:111]).astype(float) - numpy.array(mins)) / (
                        numpy.array((numpy.array(maxs) - numpy.array(mins)+1)))
                        for m in range(len(input)):
                            if numpy.isnan(input[m])==True:
                                input[m]=0

                        scoresList=[] #added agg
                        totalScore=0

                        aeCount=1
                        for ae in self.AEsList:
                            #if encounteredAnomaly==0:
                            """
                            if i==threshold-1: #added mse
                                lastTrainPacket=input #added mse
                            """
                            if i<threshold:

                                score=ae.train(input=input)
                            else:
                                score=ae.feedForward(input=input)

                            aeCount+=1
                            if aeCount == 7:
                                score *= 40
                            if (score<0) :
                                print("not match")
                                continue

                            scoresList.append(score) #added agg
                            totalScore+=score*(ae.n_visible)

                        # added aggr
                        if len(scoresList)==numOfClusters:

                            if encounteredAnomaly==0:
                                score=self.aggDA.train(input=numpy.array(scoresList))
                            else:
                                score=self.aggDA.feedForward(input=numpy.array(scoresList))


                        else:
                            continue
                        # end added aggr


                        totalScore/=len(self.AEsList*len(input))

                        """
                        #added mse
                        if i>=threshold:
                            mse=0
                            for me in range(len(input)):
                                mse+=numpy.abs(float(input[me])-float(lastTrainPacket[me]))
                            #mse/=len(input)
                        #added mse
                        """


                        totalScore=score # added aggr

                        """
                        if i>=threshold: #added mse
                            totalScore+=(mse*mse) #added mse
                        """

                        #for inp in range(len(input)):
                         #   ensembleCmeansFile.write(str(input[inp]) + ",")
                        ensembleCmeansFile.write(str(totalScore) + "," + str(row[111]) + "\n")
                    except Exception as ex:
                        print(ex.message)
                        print("observation rejected")
                        continue
                    #except:
                     #   print("packet rejected")
                      #  continue
    def findMaxsAndMins(self,dsPath):


            maxs = numpy.ones((111,)) * -numpy.Inf
            mins = numpy.ones((111,)) * numpy.Inf



            with open(dsPath, 'rt') as csvin:
                csvin = csv.reader(csvin, delimiter=',')

                for i, row in enumerate(csvin):
                    try:
                        if i % 10000 == 0:
                            print(i)

                        if i > 0:  # not header
                            x = numpy.asarray(row[0:111]).astype(float)
                            # x = x[numpy.array([1,2,4,5,7,8,10,11,13,14,16,17])-1]

                            for idx in range(0, len(maxs)):
                                if x[idx] > maxs[idx]:
                                    maxs[idx] = x[idx]
                                if x[idx] < mins[idx]:
                                    mins[idx] = x[idx]
                    except:
                        print("packet rejected")
                        continue

            print(maxs)
            print(mins)
            return maxs,mins

    def createNormalizedDataset (self,oldDS,newDS,maxs,mins):
            with open(newDS, 'w') as csvinNew:
                with open(oldDS, 'rt') as csvin:
                    csvin = csv.reader(csvin, delimiter=',')
                    for i, row in enumerate(csvin):
                        if i==0:
                            continue
                        if i%10000==0:
                            print(i)
                        norm=((numpy.array(row[64:175]).astype(float) - numpy.array(mins)) / (
                        numpy.array(maxs) - numpy.array(mins) + 1))
                        for number in norm:
                            csvinNew.write(str(number)+",")
                        csvinNew.write(str(row[175])+"\n")

    def getLabels (self,dsPath,dsLabels):
        with open(dsPath,'rt') as csvin:
            with open(dsLabels,'w') as labelsFp:
             for i, row in enumerate(csvin):
                if i%10000==0:
                     print(i)
                labelsFp.write(str(row[175])+",")


clustersDistribution = [10,8,1,10,8,9,8,8,11,13,8,14,3,8,14,12,8,14,10,8,10,8,2,10,10,8,8,10,8,2,10,10,8,8,10
    ,8,2,10,10,13,8,4,10,8,7,10,8,7,10,8,4,8,8,4,13,8,4,13,8,4,10,8,10,8,5,10,10,8,8,10,8,6,10,10,13,8,10,8,6,10
    ,10,10,10,10,8,10,10,13,10,10,10,8,10,8,5,10,10,8,8,10,8,6,10,10,13,8,10,8,6,10,10]


clustersDistribution = [1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5,5
    ,6,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,11,11,11,11,11,11,11,12,12,12
    ,12,12,12,12,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14]

clustersDistribution=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4,4,4,4,4,
                      4, 4, 4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,
                      6, 6, 6, 6, 6, 6, 6,6,6,6,6,6,6,6]

clustersDistribution=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4,4,4,4,4,
                      4, 4, 4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,
                      6, 6, 6, 6, 6, 6, 6,6,6,6,6,6,6,6]


clustersDistribution=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,4,4,4,4,4,4,
                      4, 4, 4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,7,
                      7, 7, 7, 7, 7, 7, 8,8,8,8,8,8,8,8]



clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))




indexesMap={}
for key in range(len(clusterMap)):
    if indexesMap.keys().__contains__(clusterMap[key][1])==False:
        indexesMap[clusterMap[key][1]]=[]
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    else:
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])





#aes=dAEnsemble(14,indexesMap)
aes=dAEnsemble(8,indexesMap)
#aes.getLabels('Datasets//physMIMCsv.csv','Datasets//physMIMCsvLabels.csv')
#aes.createNormalizedDataset('Datasets//physMIMCsv.csv','/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5//datasets//physMIMCsvNormalized.csv',maxs,mins)
#aes.findMaxsAndMins()

#maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')
#aes.trainAndExecute('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','D:/thesis_data/datasets/videoJak_full_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)


#maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
#aes.trainAndExecute('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv','D:/thesis_data/datasets/SYN_full_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)

#maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
#aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)
"""
maxs,mins=aes.findMaxsAndMins('D:/datasets/ctu_818_52_Full_changedTotalBytes.csv')
aes.trainAndExecute('D:/datasets/ctu_818_52_Full_changedTotalBytes.csv','D:/datasets/ctu_818_52_Full_changedTotalBytes_scores_pi3.csv',maxs,mins, 53000,8)

#maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
#aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_microMindindCluster_scores.csv',maxs,mins, 5179941)

"""
clustersDistribution=[1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,12,12,12,13,13,13,
                      14, 14, 14,15,15,15,16,16,16,17,17,17,18,18,18,18,18,18,18,19,19,19,19,19,19,19,20,20,20,20,20,20,20,21,21,21,22,22,22,23,23,23,24,24,24,24,24,24,24,
                      25, 25, 25, 25, 25, 25, 25,26,26,26,26,26,26,26]
def getIndexesMap(path,numOfClusters=16):
    clustersDistribution=ch.defineClusterForEachAttr(matPath=path,numOfClusters=numOfClusters)
    clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))
    indexesMap={}
    for key in range(len(clusterMap)):
        if indexesMap.keys().__contains__(clusterMap[key][1])==False:
            indexesMap[clusterMap[key][1]]=[]
            indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
        else:
            indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    return indexesMap

def getIndexesMapFromClusterDistribution(clustersDistribution):
    clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))
    indexesMap = {}
    for key in range(len(clusterMap)):
        if indexesMap.keys().__contains__(clusterMap[key][1]) == False:
            indexesMap[clusterMap[key][1]] = []
            indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
        else:
            indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    return indexesMap

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_rtsp.csv')
aes=dAEnsemble(16,indexesMap)


"""
maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')

aes.trainAndExecute('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','D:/thesis_data/datasets/videoJak_full_onlyNetstat_microMindCluster_scores.csv',maxs,mins, 1750648,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_syn.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv','D:/thesis_data/datasets/SYN_full_onlyNetstat_microMindCluster_scores.csv',maxs,mins, 1536268,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_piddle.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_microMindCluster_scores.csv',maxs,mins, 5179941,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_ctu.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv')
aes.trainAndExecute('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv','D:/thesis_data/datasets/ctu_818_52_NetstatOnly_microMindCluster_scores.csv',maxs,mins, 53000,16)

"""

#knn ordered
clustersDistribution=ch.getClusterDistributionFromFile('D:/thesis_data/datasets/featureClustering/knn/KNN_Clustering_rtsp.txt')

indexesMap=getIndexesMapFromClusterDistribution(clustersDistribution)

aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')

aes.trainAndExecute('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','D:/thesis_data/datasets/videoJak_full_onlyNetstat_KNNCluster_scores.csv',maxs,mins, 1750648,16)


clustersDistribution=ch.getClusterDistributionFromFile('D:/thesis_data/datasets/featureClustering/knn/KNN_Clustering_syn.txt')

indexesMap=getIndexesMapFromClusterDistribution(clustersDistribution)
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv','D:/thesis_data/datasets/SYN_full_onlyNetstat_KNNCluster_scores.csv',maxs,mins, 1536168,16)

clustersDistribution=ch.getClusterDistributionFromFile('D:/thesis_data/datasets/featureClustering/knn/KNN_Clustering_piddle.txt')

indexesMap=getIndexesMapFromClusterDistribution(clustersDistribution)
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_KNNCluster_scores.csv',maxs,mins, 5179941,16)

clustersDistribution = ch.getClusterDistributionFromFile('D:/thesis_data/datasets/featureClustering/knn/KNN_Clustering_ctu.txt')

indexesMap = getIndexesMapFromClusterDistribution(clustersDistribution)
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv')
aes.trainAndExecute('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv','D:/thesis_data/datasets/ctu_818_52_NetstatOnly_KNNCluster_scores.csv',maxs,mins, 53000,16)






#ordered by topic
clustersDistribution=[1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,12,12,12,13,13,13,
                      14, 14, 14,15,15,15,16,16,16,17,17,17,18,18,18,18,18,18,18,19,19,19,19,19,19,19,20,20,20,20,20,20,20,21,21,21,22,22,22,23,23,23,24,24,24,24,24,24,24,
                      25, 25, 25, 25, 25, 25, 25,26,26,26,26,26,26,26]

clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))
indexesMap = {}
for key in range(len(clusterMap)):
    if indexesMap.keys().__contains__(clusterMap[key][1]) == False:
        indexesMap[clusterMap[key][1]] = []
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    else:
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])

aes=dAEnsemble(26,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')

aes.trainAndExecute('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','D:/thesis_data/datasets/videoJak_full_onlyNetstat_OrderedCluster_scores.csv',maxs,mins, 1750648,26)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_syn.csv')
aes=dAEnsemble(26,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv','D:/thesis_data/datasets/SYN_full_onlyNetstat_OrderedCluster_scores.csv',maxs,mins, 1536268,26)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_piddle.csv')
aes=dAEnsemble(26,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_OrderedCluster_scores.csv',maxs,mins, 5179941,26)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_ctu.csv')
aes=dAEnsemble(26,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv')
aes.trainAndExecute('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv','D:/thesis_data/datasets/ctu_818_52_NetstatOnly_OrderedCluster_scores.csv',maxs,mins, 53000,26)





#randomally ordered

clustersDistribution=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
                      1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,
                      2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15]

clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))
indexesMap = {}
for key in range(len(clusterMap)):
    if indexesMap.keys().__contains__(clusterMap[key][1]) == False:
        indexesMap[clusterMap[key][1]] = []
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    else:
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])

aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')

aes.trainAndExecute('D:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','D:/thesis_data/datasets/videoJak_full_onlyNetstat_RandomCluster_scores.csv',maxs,mins, 1750648,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_syn.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/SYN_full_onlyNetstat.csv','D:/thesis_data/datasets/SYN_full_onlyNetstat_RandomCluster_scores.csv',maxs,mins, 1536168,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_piddle.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
aes.trainAndExecute('D:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','D:/thesis_data/datasets/piddle_FULL_onlyNetstat_RandomCluster_scores.csv',maxs,mins, 5179941,16)

indexesMap=getIndexesMap('D:/thesis_data/datasets/featureClustering/fcm/fcm_ctu.csv')
aes=dAEnsemble(16,indexesMap)

maxs,mins=aes.findMaxsAndMins('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv')
aes.trainAndExecute('D:/thesis_data/datasets/ctu_818_52_NetstatOnly.csv','D:/thesis_data/datasets/ctu_818_52_NetstatOnly_RandomCluster_scores.csv',maxs,mins, 53000,16)

