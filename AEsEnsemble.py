import numpy
from utils import *
from sklearn.preprocessing import scale
from scipy import stats
import dA
import csv
class dAEnsemble(object):
    def __init__(self, AEsNumber,indexesMap,input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, rng=None):

        self.AEsList=[]


        counter=1
        for aeIndex in range(AEsNumber):

            ae=dA.dA(indexesMap[counter],n_visible=len(indexesMap[counter]),n_hidden=n_hidden)
            self.AEsList.append(ae)

            counter+=1
            self.AEsNumber=AEsNumber

    def train(self,dsPath,ensembleCMeans,maxs,mins):
        with open(dsPath, 'rt') as csvin:
            csvin = csv.reader(csvin, delimiter=',')
            with open(ensembleCMeans,'w') as ensembleCmeansFile:
                for i, row in enumerate(csvin):
                    if i==0:
                        continue
                    if i%10000==0:
                        print(i)
                    totalScore=0

                    for ae in self.AEsList:
                        score=ae.train(input=row[:111])
                        if (score<0 or score>1) :
                            print("not match")
                            continue
                        totalScore+=score
                    totalScore/=len(self.AEsList)
                    ensembleCmeansFile.write(str(totalScore) + "," + str(row[111]) + "\n")
    def findMaxsAndMins(self):


            maxs = numpy.ones((111,)) * -numpy.Inf
            mins = numpy.ones((111,)) * numpy.Inf

            with open('Datasets//physMIMCsv.csv', 'rt') as csvin:
                csvin = csv.reader(csvin, delimiter=',')

                for i, row in enumerate(csvin):

                    if i % 10000 == 0:
                        print(i)
                    if i > 0:  # not header
                        x = numpy.asarray(row[64:175]).astype(float)
                        # x = x[numpy.array([1,2,4,5,7,8,10,11,13,14,16,17])-1]

                        for idx in range(0, len(maxs)):
                            if x[idx] > maxs[idx]:
                                maxs[idx] = x[idx]
                            if x[idx] < mins[idx]:
                                mins[idx] = x[idx]
            print(maxs)
            print(mins)

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

maxs=[  5.63452864e+02 ,  1.29400000e+03   ,3.71285200e+05,   8.40584668e+02,
   1.29400000e+03,   3.71947018e+05,   2.09836446e+03,   1.29400000e+03,
   3.71491559e+05 ,  1.49708295e+04 ,  1.29400000e+03 ,  3.46730465e+05,
   1.30162963e+05  , 1.29400000e+03  , 3.44401023e+05  , 1.12414728e+06,
   1.29400000e+03   ,3.44400902e+05   ,1.41953974e+03   ,1.29400000e+03,
   6.08349308e+02,   1.29539029e+03,   3.70091650e+05,   1.61786102e+04,
   4.64198341e+00,   3.84626038e+03 ,  1.29400000e+03 ,  6.04771535e+02,
   1.29539029e+03  , 3.65750293e+05  , 1.16812117e-01  , 1.20216425e-03,
   1.01343562e+04,   1.29400000e+03,   2.94003771e+02  , 1.29539029e+03,
   8.64382176e+04 ,  3.52575493e-02 ,  6.32278455e-04 ,  4.17288597e+04,
   1.29400000e+03  , 3.01001510e+05  , 1.95264700e+02  , 1.29400000e+03,
   3.43048100e+05   ,2.80684153e+02,   1.29400000e+03   ,3.53131369e+05,
   6.77671935e+02,   1.29400000e+03 ,  3.58829120e+05   ,4.80932086e+03,
   1.29400000e+03 ,  3.35034802e+05  , 4.17288597e+04   ,1.29400000e+03,
   3.01001510e+05 ,  3.60593097e+05,   1.29400000e+03,   2.78463257e+05,
   6.39672610e+02  , 1.29400000e+03 ,  5.99023472e+02 ,  1.29539029e+03,
   3.58831792e+05   ,2.75459654e+04  , 2.62123938e+00  , 3.38675428e+03,
   1.29400000e+03,   5.93627752e+02,   1.29539029e+03,   3.52400483e+05,
   3.65815515e+03 ,  6.44847467e-01 ,  3.00516188e+04 ,  1.29400000e+03,
   5.93972009e+02   ,1.29539029e+03  , 3.52809386e+05  , 3.69582843e+03,
   6.53824206e-01  , 6.39672610e+02,   4.11896912e+03,   2.49526112e+05,
   3.38675428e+03,   4.11896912e+03 ,  2.68992706e+05 ,  3.00516188e+04,
   4.11896912e+03 ,  2.69144362e+05   ,6.39672610e+02  , 1.29400000e+03,
   5.99023469e+02  , 1.29539029e+03  , 3.58831870e+05,   4.21481446e+04,
   1.09459651e+00,   3.38675428e+03 ,  1.29400000e+03,   6.02646538e+02,
   1.29539029e+03,   3.63196318e+05  , 3.97007955e+04,   1.08678189e+00,
   3.00516188e+04 ,  1.29400000e+03 ,  6.02888369e+02 ,  1.29539029e+03,
   3.63494091e+05  , 3.94171307e+04,   1.08925946e+00]
mins=[  1.00000000e+00 ,  5.97483751e+01 ,  0.00000000e+00,   1.00000000e+00,
   6.90682961e+01   ,0.00000000e+00,   1.00000000e+00,   7.74994847e+01,
   0.00000000e+00   ,1.00000000e+00 ,  8.17327182e+02 ,  0.00000000e+00,
   1.00000000e+00   ,8.17332718e+02  , 0.00000000e+00  , 1.00000000e+00,
   8.17333272e+02   ,0.00000000e+00,   1.00000000e+00  , 4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01,   0.00000000e+00  ,-6.59306889e+03,
  -1.96030060e+01   ,1.00000000e+00 ,  4.20000000e+01  , 0.00000000e+00,
   4.20000000e+01   ,0.00000000e+00  ,-2.34038235e+04  ,-1.29628886e+00,
   1.00000000e+00  , 4.20000000e+01,   0.00000000e+00  , 4.20000000e+01,
   0.00000000e+00  ,-9.26117995e+02 , -7.27972848e-01  , 1.00000000e+00,
   6.00000000e+01   ,0.00000000e+00  , 1.00000000e+00  , 4.24999988e+01,
   0.00000000e+00   ,1.00000000e+00,   4.70855267e+01  , 0.00000000e+00,
   1.00000000e+00   ,6.00000000e+01 ,  0.00000000e+00  , 1.00000000e+00,
   6.00000000e+01   ,0.00000000e+00  , 1.00000000e+00  , 6.00000000e+01,
   0.00000000e+00   ,1.00000000e+00   ,6.00000000e+01   ,0.00000000e+00,
   1.00000000e+00   ,4.20000000e+01,   0.00000000e+00   ,4.20000000e+01,
   0.00000000e+00  ,-1.79235009e+04 , -2.01909081e+01   ,1.00000000e+00,
   4.20000000e+01  , 0.00000000e+00  , 4.20000000e+01   ,0.00000000e+00,
  -1.83754771e+04  ,-9.09003175e-01,   1.00000000e+00   ,4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01 ,  0.00000000e+00  ,-1.84191413e+04,
  -7.73821256e-01   ,1.00000000e+00  , 0.00000000e+00  , 0.00000000e+00,
   1.00000000e+00   ,0.00000000e+00   ,0.00000000e+00 ,  1.00000000e+00,
   0.00000000e+00   ,0.00000000e+00,   1.00000000e+00  , 4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01 ,  0.00000000e+00 , -4.55026572e+03,
  -5.67606590e-01  , 1.00000000e+00  , 4.20000000e+01  , 0.00000000e+00,
   4.20000000e+01   ,0.00000000e+00  ,-4.77090681e+03 , -5.75883435e-01,
   1.00000000e+00  , 4.20000000e+01  , 0.00000000e+00,   4.20000000e+01,
   0.00000000e+00 , -4.63804302e+03  ,-5.76183881e-01]

clustersDistribution = [10,8,1,10,8,9,8,8,11,13,8,14,3,8,14,12,8,14,10,8,10,8,2,10,10,8,8,10,8,2,10,10,8,8,10
    ,8,2,10,10,13,8,4,10,8,7,10,8,7,10,8,4,8,8,4,13,8,4,13,8,4,10,8,10,8,5,10,10,8,8,10,8,6,10,10,13,8,10,8,6,10
    ,10,10,10,10,8,10,10,13,10,10,10,8,10,8,5,10,10,8,8,10,8,6,10,10,13,8,10,8,6,10,10]
clusterMap = map(lambda x: (x, clustersDistribution[x]), range(len(clustersDistribution)))


indexesMap={}
for key in range(len(clusterMap)):
    if indexesMap.keys().__contains__(clusterMap[key][1])==False:
        indexesMap[clusterMap[key][1]]=[]
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])
    else:
        indexesMap[clusterMap[key][1]].append(clusterMap[key][0])



aes=dAEnsemble(14,indexesMap)
#aes.getLabels('Datasets//physMIMCsv.csv','Datasets//physMIMCsvLabels.csv')
#aes.createNormalizedDataset('Datasets//physMIMCsv.csv','/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5//datasets//physMIMCsvNormalized.csv',maxs,mins)
#aes.findMaxsAndMins()
aes.train('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5//datasets//physMIMCsvNormalized.csv','Datasets//ensembleCMeans.csv',maxs,mins)
