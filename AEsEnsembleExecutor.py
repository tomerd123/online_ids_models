import numpy
#from utils import *
#from sklearn.preprocessing import scale
#from scipy import stats
import dA
import csv
import stackedDA

class dAEnsemble(object):
    def __init__(self, AEsNumber,indexesMap,input=None, n_visible=2, n_hidden=3, \
                 W=None, hbias=None, vbias=None, rng=None):

        self.AEsList=[]


        counter=1
        for aeIndex in range(AEsNumber):

            ae=dA.dA(indexesMap[counter],n_visible=len(indexesMap[counter]),n_hidden=numpy.amax([ int(len(indexesMap[counter])/3),1]))
            self.AEsList.append(ae)

            counter+=1
        self.AEsNumber=AEsNumber

    def trainAndExecute(self,dsPath,ensembleCMeans,maxs,mins,threshold):
        encounteredAnomaly=0

        with open(dsPath, 'rt') as csvin:
            csvin = csv.reader(csvin, delimiter=',')
            with open(ensembleCMeans,'w') as ensembleCmeansFile:
                for i, row in enumerate(csvin):

                    if i!=0 and float(row[111])!=0:
                        encounteredAnomaly=1

                    if i==0:
                        continue
                    if i%10000==0:
                        print(i)
                    totalScore=0
                    input = (numpy.array(row[:111]).astype(float) - numpy.array(mins)) / (
                    numpy.array((numpy.array(maxs) - numpy.array(mins))))

                    for ae in self.AEsList:
                        if encounteredAnomaly==0:
                            score=ae.train(input=input)
                        else:
                            score=ae.feedForward(input=input)
                        if (score<0) :
                            print("not match")
                            continue
                        totalScore+=score*(ae.n_visible)
                    totalScore/=len(self.AEsList*len(input))
                    ensembleCmeansFile.write(str(totalScore) + "," + str(row[111]) + "\n")
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

maxs=[  3.16286392e+02,   7.69358582e+02,   2.92139829e+05,   4.59172242e+02,
   7.69255215e+02,   2.92140098e+05,   1.17984437e+03,   7.69151760e+02,
   2.92140233e+05,   1.09198147e+04 ,  7.69105177e+02 ,  2.92140250e+05,
   1.03275268e+05 ,  7.69100518e+02  , 2.92140250e+05  , 7.18786252e+05,
   2.86553068e+05 ,  2.92140250e+05   ,1.17665317e+03,   2.79758214e+05,
   1.11615709e+03  , 7.69151760e+02,   2.92140233e+05 ,  1.07971946e+04,
   7.28190993e+02   ,2.68439395e+05 ,  1.01866981e+05  , 7.28634684e+02,
   2.66861009e+05,   7.13099466e+05  , 7.28683015e+02,   2.66298511e+05,
   1.08789278e+04 ,  7.69105177e+02   ,5.40500000e+02 ,  7.69105177e+02,
   2.92140250e+05  , 1.58886879e+02,   1.26346341e-02  , 3.78266391e+04,
   7.71550450e+02   ,2.96182383e+05 ,  1.46517540e+02   ,8.24489619e+02,
   3.11230661e+05,   2.04405556e+02  , 7.86579923e+02,   2.96240641e+05,
   4.88653464e+02 ,  7.71595019e+02   ,2.96201711e+05 ,  4.17799055e+03,
   7.71554504e+02  , 2.96184142e+05,   3.78266391e+04  , 7.71550450e+02,
   2.96182383e+05   ,2.46528626e+05 ,  7.71550045e+02   ,2.96182207e+05,
   7.17273084e+02,   9.32480139e+02  , 5.65870344e+02,   9.41899443e+02,
   3.20209246e+05 ,  3.47838345e+03   ,7.00000000e+01 ,  6.11860859e+03,
   9.32451464e+02  , 5.65862051e+02,   9.32451464e+02  , 3.20199861e+05,
   5.68447060e+02   ,2.17900188e-02 ,  5.52725385e+04   ,9.32448595e+02,
   5.65861220e+02,   9.32448595e+02  , 3.20198921e+05,   1.92384455e+02,
   1.50020037e-02 ,  4.88653464e+02   ,2.05806331e+03 ,  1.05890610e+06,
   4.17799055e+03  , 2.05806331e+03,   1.05890610e+06  , 3.78234507e+04,
   2.05803712e+03   ,1.05890604e+06 ,  7.15323171e+02   ,9.44406010e+02,
   5.76056911e+02,   9.44406010e+02  , 3.31841564e+05,   2.60617025e+03,
   6.01661103e+02 ,  9.04990195e+04   ,9.44370603e+02 ,  6.01661103e+02,
   9.04990195e+04  , 3.31809703e+05,   6.01481556e+02  , 9.04883863e+04,
   5.52714947e+04   ,9.44367060e+02 ,  5.76026488e+02   ,9.44367060e+02,
   3.31806514e+05,   2.15238646e+02  , 1.84827413e-02]
mins=[ -1.00000000e+00,  -1.00000000e+00,  -1.00000000e+00,  -1.00000000e+00,
  -1.00000000e+00,  -1.00000000e+00,  -1.00000000e+00,  -1.00000000e+00,
  -1.00000000e+00 , -1.00000000e+00 , -1.00000000e+00 , -1.00000000e+00,
  -1.00000000e+00  ,-1.00000000e+00  , 0.00000000e+00  , 1.00000000e+00,
   2.09000000e+02,   0.00000000e+00   ,1.00000000e+00   ,4.20000000e+01,
   0.00000000e+00 ,  4.20000000e+01,   0.00000000e+00,  -1.06787491e+03,
  -3.97096608e-02  , 1.00000000e+00 ,  4.20000000e+01 ,  0.00000000e+00,
   4.20000000e+01,   0.00000000e+00  ,-5.96712844e+02  ,-2.22781844e-02,
   1.00000000e+00 ,  4.20000000e+01   ,0.00000000e+00   ,4.20000000e+01,
   0.00000000e+00  ,-2.84343702e+02,  -1.45431164e-02,   1.00000000e+00,
   4.20000000e+01,   0.00000000e+00 ,  1.00000000e+00 ,  0.00000000e+00,
   0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00  , 0.00000000e+00,
   0.00000000e+00  , 4.20000000e+01   ,0.00000000e+00   ,0.00000000e+00,
   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
   0.00000000e+00 ,  1.00000000e+00 ,  0.00000000e+00 ,  0.00000000e+00,
   1.00000000e+00  , 0.00000000e+00  , 0.00000000e+00  , 4.20000000e+01,
   0.00000000e+00,  -2.42019962e+03 , -1.17920504e-01   ,0.00000000e+00,
   1.00000000e+00 ,  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
  -6.02329024e+02  ,-3.26490219e-02 ,  1.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00,   4.20000000e+01  , 0.00000000e+00  ,-3.80411686e+02,
  -3.08169886e-02 ,  1.00000000e+00   ,0.00000000e+00   ,0.00000000e+00,
   1.00000000e+00  , 0.00000000e+00 ,  0.00000000e+00,   0.00000000e+00,
   0.00000000e+00,   0.00000000e+00  , 0.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00  , 0.00000000e+00   ,0.00000000e+00  ,-2.58662466e+03,
  -1.16249734e-01 ,  0.00000000e+00,   1.00000000e+00   ,0.00000000e+00,
   0.00000000e+00  , 0.00000000e+00 , -6.13524533e+02,  -2.65058008e-02,
   1.00000000e+00   ,4.20000000e+01  , 0.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00 , -2.92943249e+02  ,-2.28298812e-02]

maxs=[  6.77531882e+02,   8.30104222e+02,   3.07888542e+05,   9.32301289e+02,
   8.05501520e+02  , 2.94194916e+05,   1.77866352e+03,   7.75570710e+02,
   2.92140233e+05,   1.09198147e+04 ,  7.69105177e+02,   2.92140250e+05,
   1.03275268e+05 ,  7.69100518e+02  , 2.92140250e+05 ,  7.30223216e+05,
   7.69100052e+02  , 2.92140250e+05   ,1.77290307e+03  , 1.14127012e+03,
   5.40499984e+02   ,1.14127012e+03,   2.92140233e+05   ,1.25291977e+03,
   3.93958999e-02,   3.87152859e+03 ,  8.08835671e+02,   5.40499999e+02,
   8.08835671e+02 ,  2.92140248e+05  , 6.08542720e+02 ,  2.28627810e-02,
   1.08789278e+04  , 7.69105177e+02   ,5.40500000e+02  , 7.69105177e+02,
   2.92140250e+05   ,2.86646923e+02,   1.26346341e-02   ,3.78266391e+04,
   7.71550450e+02,   2.96182383e+05 ,  5.80070371e+02,   1.14884392e+03,
   4.84217213e+05 ,  7.65590759e+02  , 1.14346881e+03 ,  4.80628947e+05,
   1.27761343e+03  , 9.58925293e+02   ,4.63494780e+05  , 4.77560658e+03,
   8.17414519e+02   ,3.51578691e+05,   3.78266391e+04   ,7.71550450e+02,
   2.96182383e+05,   2.50281899e+05 ,  7.71550045e+02,   2.96182207e+05,
   1.89874526e+03 ,  1.14127012e+03  , 6.80834066e+02 ,  1.14127012e+03,
   4.63569948e+05  , 5.02755256e+04   ,9.87378828e-01  , 6.97806798e+03,
   9.55446245e+02   ,6.22008429e+02,   9.58007542e+02   ,3.86961888e+05,
   4.42195192e+04,   1.39383243e+00 ,  5.52725385e+04,   9.32448595e+02,
   5.65861220e+02 ,  9.32448595e+02  , 3.20198921e+05 ,  1.92384455e+02,
   1.50020037e-02  , 1.27761343e+03   ,2.05806331e+03  , 1.05890610e+06,
   4.77560323e+03   ,2.05806331e+03,   1.05890610e+06   ,3.78234507e+04,
   2.05803712e+03,   1.05890604e+06 ,  1.89697151e+03,   1.14900000e+03,
   6.99166149e+02 ,  1.14900000e+03  , 4.88833303e+05 ,  5.17383437e+04,
   1.02611268e+00  , 6.97625852e+03   ,1.14900000e+03  , 6.99170182e+02,
   1.14900000e+03   ,4.88838944e+05,   4.47429091e+04,   1.40937026e+00,
   5.52714947e+04,   1.14900000e+03 ,  6.99170576e+02 ,  1.14900000e+03,
   4.88839495e+05 ,  1.45690407e+04  , 1.03704500e+00]
mins=[  1.00000000e+00  , 2.09000000e+02   ,0.00000000e+00 ,  1.00000000e+00,
   2.09000000e+02   ,0.00000000e+00   ,1.00000000e+00,   2.09000000e+02,
   0.00000000e+00,   1.00000000e+00   ,2.09000000e+02 ,  0.00000000e+00,
   1.00000000e+00 ,  2.09000000e+02   ,0.00000000e+00  , 1.00000000e+00,
   2.09000000e+02  , 0.00000000e+00   ,1.00000000e+00   ,4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01   ,0.00000000e+00,  -3.28047285e+03,
  -6.76522188e-02,   1.00000000e+00   ,4.20000000e+01 ,  0.00000000e+00,
   4.20000000e+01 ,  0.00000000e+00  ,-1.34967536e+03  ,-3.09067604e-02,
   1.00000000e+00  , 4.20000000e+01   ,0.00000000e+00,   4.20000000e+01,
   0.00000000e+00  ,-5.00605978e+02  ,-2.05123943e-02 ,  1.00000000e+00,
   4.20000000e+01   ,0.00000000e+00   ,1.00000000e+00  , 4.20000000e+01,
   0.00000000e+00   ,1.00000000e+00   ,4.20000000e+01   ,0.00000000e+00,
   1.00000000e+00   ,4.20000000e+01   ,0.00000000e+00,   1.00000000e+00,
   4.20000000e+01   ,0.00000000e+00   ,1.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00   ,1.00000000e+00   ,4.20000000e+01  , 0.00000000e+00,
   1.00000000e+00   ,4.20000000e+01   ,0.00000000e+00   ,4.20000000e+01,
   0.00000000e+00  ,-6.12071421e+03  ,-1.01691842e+00,   1.00000000e+00,
   4.20000000e+01   ,0.00000000e+00   ,4.20000000e+01 ,  0.00000000e+00,
  -1.46257362e+04  ,-9.15624958e-01   ,1.00000000e+00  , 4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01   ,0.00000000e+00 , -2.88828280e+03,
  -6.75651344e-02   ,1.00000000e+00   ,0.00000000e+00,   0.00000000e+00,
   1.00000000e+00   ,0.00000000e+00   ,0.00000000e+00 ,  1.00000000e+00,
   0.00000000e+00   ,0.00000000e+00   ,1.00000000e+00  , 4.20000000e+01,
   0.00000000e+00   ,4.20000000e+01   ,0.00000000e+00  ,-2.58662466e+03,
  -9.16814795e-01   ,1.00000000e+00   ,4.20000000e+01,   0.00000000e+00,
   4.20000000e+01,   0.00000000e+00  ,-2.13461462e+03 , -9.18883684e-01,
   1.00000000e+00 ,  4.20000000e+01   ,0.00000000e+00  , 4.20000000e+01,
   0.00000000e+00  ,-2.42701016e+03  ,-9.19090509e-01]

maxs=[  4.12168445e+02,   8.30104222e+02,   2.96459749e+05,   6.13069582e+02,
   8.05501520e+02 ,  2.88478995e+05 ,  1.43436971e+03 ,  7.75570710e+02,
   2.82139439e+05,   9.20294587e+03 ,  7.53502813e+02 ,  2.72467755e+05,
   8.26808610e+04 ,  7.24165765e+02  , 2.67703314e+05  , 7.30223216e+05,
   7.11291336e+02  , 2.66126828e+05   ,1.43037902e+03   ,7.78023037e+02,
   5.24936314e+02   ,7.80969142e+02,   2.75558211e+05,   1.06612419e+03,
   3.53340153e-02,   3.76224466e+03 ,  7.64241549e+02 ,  5.20403083e+02,
   7.67726877e+02 ,  2.70819439e+05  , 5.75802734e+02  , 1.70296774e-02,
   9.12461696e+03   ,7.57503557e+02   ,5.18518543e+02   ,7.61286933e+02,
   2.68861900e+05  , 2.86646923e+02,   1.08931645e-02,   2.84269826e+04,
   7.46463060e+02,   2.71479651e+05 ,  3.13844749e+02 ,  9.41055918e+02,
   3.14018119e+05 ,  4.26125579e+02  , 9.08183444e+02  , 3.06752699e+05,
   9.27717766e+02  , 8.52300243e+02   ,2.87151653e+05   ,4.25050307e+03,
   7.95566157e+02   ,2.81081442e+05,   2.84269826e+04,   7.46463060e+02,
   2.71479651e+05,   2.49620440e+05 ,  7.33112370e+02 ,  2.71479826e+05,
   1.37432653e+03 ,  9.76111786e+02  , 5.46628434e+02  , 9.78093222e+02,
   2.98852439e+05  , 3.65827717e+04   ,9.08491211e-01   ,6.24360027e+03,
   9.45451736e+02   ,5.66998047e+02,   9.47695501e+02,   3.21486788e+05,
   4.42195192e+04,   1.39383243e+00 ,  4.11666957e+04 ,  9.14268972e+02,
   5.10705056e+02 ,  9.16980973e+02  , 2.60822230e+05  , 7.54286815e+01,
   4.51565499e-03  , 9.27717765e+02   ,1.20005675e+03   ,5.05845699e+04,
   4.25013628e+03   ,1.20005675e+03,   5.05872461e+04,   2.84256030e+04,
   1.20005668e+03,   2.82375292e+05 ,  1.37336270e+03 ,  9.76942375e+02,
   5.47777452e+02 ,  9.78922129e+02  , 3.00109723e+05  , 3.75095870e+04,
   1.00268791e+00  , 6.24269503e+03   ,9.45473096e+02   ,5.41547467e+02,
   9.47716810e+02   ,2.93294132e+05,   4.47429091e+04,   1.40937026e+00,
   4.11647062e+04,   9.14282437e+02 ,  5.10699678e+02 ,  9.16994397e+02,
   2.60816737e+05 ,  1.45690407e+04  , 1.03704500e+00]
mins=[  1.54020858e+00,   3.86159057e+02,   1.51891494e+05,   9.58761998e+00,
   6.02834741e+02,   2.49286662e+05,   2.35906808e+02,   6.32938441e+02,
   2.57556403e+05 ,  4.24862588e+03 ,  6.99237867e+02 ,  2.63021548e+05,
   6.01746231e+04  , 7.05150232e+02  , 2.64850690e+05  , 7.07531007e+05,
   7.09180090e+02   ,2.65821812e+05   ,1.00000000e+00   ,4.20000000e+01,
   0.00000000e+00,   4.20000000e+01,   0.00000000e+00,  -2.48586495e+03,
  -6.76522188e-02 ,  1.00000000e+00 ,  4.20000000e+01 ,  0.00000000e+00,
   4.20000000e+01  , 0.00000000e+00  ,-1.02579283e+03  ,-3.07803431e-02,
   1.00000000e+00   ,4.20000000e+01   ,0.00000000e+00   ,4.20000000e+01,
   0.00000000e+00,  -4.38671019e+02,  -2.05123943e-02,   1.00000000e+00,
   4.20000000e+01 ,  0.00000000e+00 ,  1.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00  , 1.00000000e+00  , 4.20000000e+01  , 0.00000000e+00,
   1.00000000e+00   ,4.20000000e+01   ,0.00000000e+00   ,1.00000000e+00,
   4.20000000e+01,   0.00000000e+00,   1.00000000e+00,   4.20000000e+01,
   0.00000000e+00 ,  1.00000000e+00 ,  4.20000000e+01 ,  0.00000000e+00,
   1.00000000e+00  , 4.20000000e+01  , 0.00000000e+00  , 4.20000000e+01,
   0.00000000e+00  ,-6.12071421e+03  ,-1.01691842e+00   ,1.00000000e+00,
   4.20000000e+01,   0.00000000e+00,   4.20000000e+01,   0.00000000e+00,
  -1.46257362e+04 , -9.15624958e-01 ,  1.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00  , 4.20000000e+01  , 0.00000000e+00  ,-3.06719611e+02,
  -2.00620691e-02   ,1.00000000e+00   ,0.00000000e+00   ,0.00000000e+00,
   1.00000000e+00,   0.00000000e+00,   0.00000000e+00,   1.00000000e+00,
   0.00000000e+00 ,  0.00000000e+00 ,  1.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00  , 4.20000000e+01  , 0.00000000e+00  ,-2.01724956e+03,
  -5.63470382e-01   ,1.00000000e+00   ,4.20000000e+01,   0.00000000e+00,
   4.20000000e+01   ,0.00000000e+00,  -2.13461462e+03,  -5.64962656e-01,
   1.00000000e+00,   4.20000000e+01 ,  0.00000000e+00 ,  4.20000000e+01,
   0.00000000e+00 , -2.42701016e+03  ,-7.12221832e-01]

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

#maxs,mins=aes.findMaxsAndMins('E:/thesis_data/datasets/videoJak_full_onlyNetstat.csv')
#aes.trainAndExecute('E:/thesis_data/datasets/videoJak_full_onlyNetstat.csv','E:/thesis_data/datasets/videoJak_full_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)


#maxs,mins=aes.findMaxsAndMins('E:/thesis_data/datasets/SYN_full_onlyNetstat.csv')
#aes.trainAndExecute('E:/thesis_data/datasets/SYN_full_onlyNetstat.csv','E:/thesis_data/datasets/SYN_full_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)

#maxs,mins=aes.findMaxsAndMins('E:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv')
#aes.trainAndExecute('E:/thesis_data/datasets/piddle_FULL_onlyNetstat.csv','E:/thesis_data/datasets/piddle_FULL_onlyNetstat_scoresAEEnsemble.csv',maxs,mins, 1750648)

#maxs,mins=aes.findMaxsAndMins('E:/thesis_data/datasets/videoJak_full_onlyNetstat_testSamples.csv')
aes.trainAndExecute('E:/thesis_data/datasets/videoJak_full_onlyNetstat_testSamples.csv','E:/thesis_data/datasets/videoJakNetStat_testSamples_scoresAEEnsemble.csv',maxs,mins, 1750648)
