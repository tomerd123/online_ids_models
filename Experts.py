import json
import math
import utils
import AfterImage as af
import dA as ae
import numpy as np



class Expert: #abstract class
    def init(self,numFeats,name,Auth,hiddenRatio,gracePeriod, corruption, threshold, rm_winsize=0, learningRate = 0.1):
        self.authority = Auth
        self.threshold = threshold
        self.name = name
        self.gracePeriod = gracePeriod
        self.numFeats = numFeats
        self.corruption = corruption
        self.n = 0  # number of instances seen by this expert so far
        self.hiddenRatio = hiddenRatio
        self.score_stats = af.incStat(0.00001)  # used to track rolling std
        # for 0-1 normlaization
        self.norm_max = np.ones((self.numFeats,)) * -np.Inf
        self.norm_min = np.ones((self.numFeats,)) * np.Inf
        self.rm_winsize = rm_winsize
        self.learningRate = learningRate
        if rm_winsize == 0:
            self.rollmean = None
        else:
            self.rollmean = utils.rollmean(rm_winsize)

    def toJSON(self):
        jsonobj = {}
        jsonobj['expert_type'] = 'Expert'  # will be overriden by implimenting class
        jsonobj['name'] = self.name
        jsonobj['authority'] = self.authority
        jsonobj['threshold'] = self.threshold
        jsonobj['gracePeriod'] = self.gracePeriod
        jsonobj['featureIndxs'] = self.featureIndxs
        jsonobj['hiddenRatio'] = self.hiddenRatio
        jsonobj['corruption'] = self.corruption
        jsonobj['n'] = self.n
        if self.rollmean == None:
            jsonobj['rollmean'] = 0
        else:
            jsonobj['rollmean'] = self.rollmean.winsize
        jsonobj['score_stats'] = self.score_stats.toJSON()
        jsonobj['norm_max'] = self.norm_max.tolist()
        jsonobj['norm_min'] = self.norm_min.tolist()
        jsonobj = self.toJSONmore(jsonobj)
        return json.dumps(jsonobj)

    def toJSONmore(self, jsonObj):
        return jsonObj

    # x_ref is a mutable object [] to the numrical vector 'x'
    def vote(self, x_ref):
        if self.n < self.gracePeriod * 2:
            return 0.0
        # std = self.train_stats.std()
        return self.score(x_ref)

    def description(self):
        return (self.name, self.authority)

    #overridden if a condidtion need to pass for an Expert to be trained/excuted on an instance
    def isApplicable(self,x_ref):
        return True

    #### pure Virtual functions which must be implimented in cpp ####
    def score(self,x_ref):
        return 0.0
    # 0 = train(...)

class Expert_Basic(Expert):
    def __init__(self,numFeats,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0, rm_winsize=0, learningRate = 0.1): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        self.init(numFeats,name,Auth,hiddenRatio,gracePeriod, corruption, threshold, rm_winsize,learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))

    #this helper function is over ridden by diffrent kinds of experts to add additonal fields
    def toJSONmore(self,jsonobj):
        jsonobj['AE'] = self.AE.toJSON()
        jsonobj['expert_type'] = 'Expert_Basic'
        return jsonobj

    #pure virtual
    def prepInstance(self,x_ref,readOnly=False):
        return np.zeros(self.numFeats)*np.nan

    #x_ref is a mutable object [] to the feature vector 'x'
    def train(self,x_ref):
        if self.isApplicable(x_ref):
            x = self.prepInstance(x_ref)
            self.n = self.n + 1
            error = self.AE.train(x,corruption_level=self.corruption,lr=self.learningRate)
            if self.n > self.gracePeriod:
                if error > 0:
                    self.score_stats.insert(np.log(error),x_ref[0][len(x_ref[0])-1]) #the latter argument is the timestamp of the isntance

    def score(self,x_ref):
        if (self.n < self.gracePeriod*2) or (not self.isApplicable(x_ref)):
            return 0.0
        x = self.prepInstance(x_ref,True)#we don't update max-min normalization bounds when executing (only train)
        mse = np.log(self.AE.score(x))
        logprob = utils.invLogCDF(mse,self.score_stats.mean(),self.score_stats.std())
        score = logprob/self.threshold
        if self.rollmean is None:
            return score  #return the normalized score (>=1 indicates anomaly, <1 is normal)
        else:
            return self.rollmean.apply(score) #update rollign average and return the current windowed average

class Expert_Complex(Expert):
    def __init__(self, numFeats, name="unknown_expert",maxSubExperts=100, Auth=0, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000, corruption=0.0,
                 rm_winsize=0,learningRate=0.1):  # gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        self.init(numFeats=numFeats, name=name, Auth=Auth, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod, corruption=corruption, threshold=threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.subExperts = {} #unordered_map
        self.maxSubExperts = maxSubExperts
        self.subExperts[''] = self.makeSubExpert(self.name + " (default subExpert)")

    # this helper function is over ridden by diffrent kinds of experts to add additonal fields
    def toJSONmore(self, jsonobj):
        jsonobj['expert_type'] = 'Expert_Complex'
        subExperts = {}
        for key, value in self.subExperts.items():
            subExperts[key]= value[0].toJSON()
        jsonobj['subExperts'] = json.dumps(subExperts)
        return jsonobj

    # ****make pure virtual!!
    def getSubExpertKey(self,x_ref): #****make pure virtual!!
        return ''

    # ****make pure virtual!!
    def makeSubExpert(self,name):
        return [Expert_Basic(name=name, rm_winsize=self.rm_winsize, Auth=self.authority, threshold=self.threshold, hiddenRatio=self.hiddenRatio, gracePeriod=self.gracePeriod, corruption=self.corruption, learningRate=self.learningRate)]

    def vote(self,x_ref):
        if not self.isApplicable(x_ref):
            return 0.0
        key = self.getSubExpertKey(x_ref)
        Exp = self.subExperts.get(key)
        if Exp is None: #ask the default subExpert
            return self.subExperts.get('')[0].vote(x_ref)
        else: #as the respective subExpert
            return Exp[0].vote(x_ref)

    def train(self,x_ref):
        if self.isApplicable(x_ref):
            key = self.getSubExpertKey(x_ref)
            Exp = self.subExperts.get(key)
            if Exp is None:
                #can we add a new subExpert?
                if len(self.subExperts) < self.maxSubExperts:  # add new subExpert for this direction
                    Exp = self.makeSubExpert(self.name + ": "+ key)
                    Exp[0].train(x_ref)
                    self.subExperts[key] = Exp
                    self.n = self.n + 1
                else: #train the default sub-expert
                    self.subExperts.get('')[0].train(x_ref)
                    self.n = self.n + 1
            else:
                Exp[0].train(x_ref)
                self.n = self.n + 1

class Expert_ICMP_Ch(Expert_Basic):
    def __init__(self,term,name="ICMP Channel",Auth=2,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0, rm_winsize=0, learningRate=0.1):
        if term == 0:#term enum:
            numFeats = 6
        elif term == 1:
            numFeats = 6
        elif term == 2:
            numFeats = 5
        else:
            raise Exception("Term Type unacceptable")
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.term = term
        if self.term == 0:
            self.name = self.name + ": short-term"
        elif self.term == 1:
            self.name = self.name + ": long-term"
        else:
            self.name = self.name + ": multi-term"

    def prepInstance(self,x_ref,readOnly=False):
        indxs = featureIndex()
        if self.term == 0:
            findx = indxs.HpHp_L1_mean, indxs.HpHp_L1_std, indxs.HpHp_L1_magnitude, indxs.HpHp_L1_radius, indxs.HpHp_L1_covariance, indxs.HpHp_L1_pcc
        elif self.term == 1:
            findx = indxs.HpHp_L0_1_mean, indxs.HpHp_L0_1_std, indxs.HpHp_L0_1_magnitude, indxs.HpHp_L0_1_radius, indxs.HpHp_L0_1_covariance, indxs.HpHp_L0_1_pcc
        else: #multiTerm
            findx = indxs.HpHp_L1_mean, indxs.HpHp_L1_pcc, indxs.HpHp_L0_1_mean, indxs.HpHp_L0_1_pcc, indxs.HpHp_L0_1_pcc

        x = x_ref[0][findx,].astype(float)
        #update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        #0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

    def isApplicable(self,x_ref):
        #** cpp: check instance for the enum!   if (this->otherProtocol == x_ref->otherProtocol) return true, else false
        if x_ref[0][featureIndex().icmp_code] != '':
            return True
        else:
            return False

class Expert_IP_Ch(Expert_Basic):
    def __init__(self, term,IPtype, name="IP Channel", Auth=2, threshold=-50, hiddenRatio=0.7, gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        if term == 0: #term enum:
            numFeats = 6
        elif term == 1:
            numFeats = 6
        elif term == 2:
            numFeats = 5
        else:
            raise Exception("Term Type unacceptable")
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.term = term
        self.IPtype = IPtype
        if IPtype == 0:
            ip = "IPv4"
        else:
            ip = "IPv6"
        if self.term == 0:
            self.name = self.name + ": "+ ip + " short-term"
        elif self.term == 1:
            self.name = self.name + ": "+ ip + " long-term"
        else:
            self.name = self.name + ": "+ ip + " multi-term"

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        if self.term == 0:#term enum:
            findx = indxs.HH_L1_mean, indxs.HH_L1_std, indxs.HH_L1_magnitude, indxs.HH_L1_radius, indxs.HH_L1_covariance, indxs.HH_L1_pcc
        elif self.term == 1:
            findx = indxs.HH_L0_1_mean, indxs.HH_L0_1_std, indxs.HH_L0_1_magnitude, indxs.HH_L0_1_radius, indxs.HH_L0_1_covariance, indxs.HH_L0_1_pcc
        else:  # multiTerm
            findx = indxs.HH_L1_mean, indxs.HH_L1_pcc, indxs.HH_L0_1_mean, indxs.HH_L0_1_pcc, indxs.HH_L0_1_pcc
        x = x_ref[0][findx,].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

    def isApplicable(self, x_ref):
        #** cpp: check instance for the enum!   if (this->IPtype == x_ref->IPtype) return true, else false
        if self.IPtype == 0: #IPv4
            if x_ref[0][featureIndex().ip_src] != '':
                return True
        else: #IPv6
            if x_ref[0][featureIndex().ipv6_src] != '':
                return True
        return False

class Expert_TCPUDP_Ch(Expert_Basic):
    def __init__(self, term,TCPUDPtype, name="TCP/UDP Channel", Auth=2, threshold=-50, hiddenRatio=0.7, gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        if term == 0:  # term enum:
            numFeats = 6
        elif term == 1:
            numFeats = 6
        elif term == 2:
            numFeats = 5
        else:
            raise Exception("Term Type unacceptable")
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold,rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.term = term
        self.TCPUDPtype = TCPUDPtype
        if TCPUDPtype == 0:
            prot = "TCP Well-known"
        elif TCPUDPtype == 1:
            prot = "TCP Registered"
        elif TCPUDPtype == 2:
            prot = "TCP Dynamic"
        elif TCPUDPtype == 3:
            prot = "UDP Well-known"
        elif TCPUDPtype == 4:
            prot = "UDP Registered"
        else:
            prot = "UDP Dynamic"
        if self.term == 0:
            self.name = prot + " Channel: short-term"
        elif self.term == 1:
            self.name = prot + " Channel: long-term"
        else:
            self.name = prot + " Channel: multi-term"

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        if self.term == 0:
            findx = indxs.HpHp_L1_mean, indxs.HpHp_L1_std, indxs.HpHp_L1_magnitude, indxs.HpHp_L1_radius, indxs.HpHp_L1_covariance, indxs.HpHp_L1_pcc
        elif self.term == 1:
            findx = indxs.HpHp_L0_1_mean, indxs.HpHp_L0_1_std, indxs.HpHp_L0_1_magnitude, indxs.HpHp_L0_1_radius, indxs.HpHp_L0_1_covariance, indxs.HpHp_L0_1_pcc
        else:  # multiTerm
            findx = indxs.HpHp_L1_mean, indxs.HpHp_L1_pcc, indxs.HpHp_L0_1_mean, indxs.HpHp_L0_1_pcc, indxs.HpHp_L0_1_pcc

        x = x_ref[0][findx,].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

    def isApplicable(self, x_ref):
        #** cpp: check instance for the enum!   if (this->TCPUDPtype == x_ref->TCPUDPtype) return true, else false
        indxs = featureIndex()
        if x_ref[0][indxs.tcp_srcport] != '':
            if self.TCPUDPtype == 0: #TCP_wkn
                port = int(x_ref[0][indxs.tcp_srcport])
                if (port >= 0) and (port < 1024):
                    return True
            elif self.TCPUDPtype == 1: #TCP_reg
                port = int(x_ref[0][indxs.tcp_srcport])
                if (port >= 1024) and (port < 49152):
                    return True
            elif self.TCPUDPtype == 2: #TCP_dyn
                port = int(x_ref[0][indxs.tcp_srcport])
                if (port >= 49152) and (port < 66535):
                    return True
        elif x_ref[0][indxs.udp_srcport] != '':
            if self.TCPUDPtype == 3: #UDP_wkn
                port = int(x_ref[0][indxs.udp_srcport])
                if (port >= 0) and (port < 1024):
                    return True
            elif self.TCPUDPtype == 4: #UDP_reg
                port = int(x_ref[0][indxs.udp_srcport])
                if (port >= 1024) and (port < 49152):
                    return True
            elif self.TCPUDPtype == 5: #UDP_dyn
                port = int(x_ref[0][indxs.udp_srcport])
                if (port >= 49152) and (port < 66535):
                    return True
        return False

class Expert_ICMP_Pr(Expert_Basic):
    def __init__(self, version, name="ICMP Protocol", Auth=2, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        if version == 0:
            numFeats = 5
        elif version == 1:
            numFeats = 6
        elif version == 2:
            numFeats = 5
        else:  # =3
            numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.version = version
        if self.version == 0:
            self.name = self.name + ": General"
        elif self.version == 1:
            self.name = self.name + ": Error Msg"
        elif self.version == 2:
            self.name = self.name + ": Router Msg"
        else:
            self.name = self.name + ": Queury"

    def prepInstance(self,x_ref,readOnly=False):
        x = np.zeros(self.numFeats)
        indxs = featureIndex()
        if self.version == 0:
            ipsrc = str.split(x_ref[0][indxs.ip_src],'.')
            ipdst = str.split(x_ref[0][indxs.ip_dst],'.')
            x[0] = float(ipsrc[3]) #take last byte of IP (e.g. 192.168.1.X)
            x[1] = float(ipdst[3])
            x[2:] = x_ref[0][(indxs.icmp_type, indxs.icmp_code, indxs.HpHp_L0_1_mean),].astype(float)
        elif self.version == 1:
            ipsrc = str.split(x_ref[0][indxs.ip_src],'.')
            x[0] = float(ipsrc[2]) + float(ipsrc[3]) #take last two bytes of IP (e.g. 192.168.X.X)
            x[1] = float(x_ref[0][indxs.icmp_code,]) if float(x_ref[0][indxs.icmp_type,]) == 3 else -1.0 #the Destination Unreachable code if it exists
            x[2] = float(x_ref[0][indxs.icmp_code,]) if float(x_ref[0][indxs.icmp_type,]) == 11 else -1.0 #the TTL Exceeded Code if it exists
            x[3] = float(x_ref[0][indxs.icmp_code,]) if float(x_ref[0][indxs.icmp_type,]) == 5 else -1.0 #the Redirect Code if it exists
            x[4] = float(x_ref[0][indxs.icmp_code,]) if float(x_ref[0][indxs.icmp_type,]) == 12 else -1.0 #the param problem Code if it exists
            x[5] = float(x_ref[0][indxs.HpHp_L0_1_mean,])
        elif self.version == 2:
            ipsrc = str.split(x_ref[0][indxs.ip_src],'.')
            ipdst = str.split(x_ref[0][indxs.ip_src],'.')
            x[0] = float(ipsrc[2]) + float(ipsrc[3]) #take last two bytes of IP (e.g. 192.168.X.X)
            x[1] = float(ipdst[3]) # last  byte of IP (e.g. 192.168.1.X)
            x[2] = 0.0 if int(x_ref[0][indxs.icmp_type,]) == 5 else 1.0 #router Advert, or Redirect message?
            x[3:] = x_ref[0][(indxs.HpHp_L0_1_mean, indxs.H_L0_1_mean),].astype(float)
        else: #==3
            ipsrc = str.split(x_ref[0][indxs.ip_src], '.')
            ipdst = str.split(x_ref[0][indxs.ip_src], '.')
            x[0] = float(ipsrc[3])  # take last byte of IP (e.g. 192.168.1.X)
            x[1] = float(ipdst[3])
            icmp_type = int(x_ref[0][indxs.icmp_type,])
            x[2] = icmp_type if (icmp_type == 0 or icmp_type == 8) else -1.0 #is this icmp echo request or reply?
            x[3] = 1.0 if icmp_type == 10 else 0.0  # is this icmp router solicitation?
            x[4:] = x_ref[0][(indxs.HpHp_L0_1_mean, indxs.H_L0_1_mean),].astype(float)

        #update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        #0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

    def isApplicable(self,x_ref):
        #** cpp: check instance for the enum!   if (x_ref->otherProtocol == ICMP) return true, else false
        if x_ref[0][featureIndex().icmp_code] != '':
            return True
        return False

class Expert_ARP_Pr(Expert_Basic):
    def __init__(self, term, name="ARP Protocol", Auth=2, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        if term == 0:  # term enum:
            numFeats = 6
        elif term == 1:
            numFeats = 6
        else:
            raise Exception("Term Type unacceptable")
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold,rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.term = term
        if self.term == 0:
            self.name = self.name + ": short-term"
        elif self.term == 1:
            self.name = self.name + ": long-term"

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        if self.term == 0:
            findx = indxs.MI_dir_L0_01_mean, indxs.MI_dir_L0_01_variance, indxs.HH_L1_mean, indxs.HH_L1_std, indxs.HH_L1_radius, indxs.HH_L1_pcc
        else: #=1
            findx = indxs.MI_dir_L0_01_mean, indxs.MI_dir_L0_01_variance, indxs.HH_L0_01_mean, indxs.HH_L0_01_std, indxs.HH_L0_01_radius, indxs.HH_L0_01_pcc
        x = x_ref[0][findx,].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

    def isApplicable(self, x_ref):
        #** cpp: check instance for the enum!   if (x_ref->otherProtocol == ARP) return true, else false
        if x_ref[0][featureIndex().arp_opcode] != '':
            return True
        return False

class Expert_IP_Pr(Expert_Basic):
    def __init__(self, IPtype, version, name="IP Protocol", Auth=1, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        numFeats = 0
        if IPtype == 0: #IPv4
            if version == 0 or version == 2:
                numFeats = 7
            elif version == 1:
                numFeats = 6
        else:# IPtype == 1 #IPv6
            if version == 0 or version == 1:
                numFeats = 7
            elif version == 2:
                numFeats = 6
        if numFeats == 0: #bad enum detected
            raise Exception("Enum Type unacceptable")
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.IPtype = IPtype
        self.version = version
        if IPtype == 0:
            ip = "IPv4"
        else:
            ip = "IPv6"
        self.name = ip + " Protocol: " + str(version)

    def isApplicable(self,x_ref):
        #** cpp: check instance for the enum!   if (this->IPtype == x_ref->IPtype) return true, else false
        indxs = featureIndex()
        if self.IPtype == 0 and x_ref[0][indxs.ip_src] != '': #IPv4
            if x_ref[0][featureIndex().ip_src] != 0:
                return True
        if self.IPtype == 1 and x_ref[0][indxs.ipv6_src] != '': #IPv6
            if x_ref[0][featureIndex().ipv6_src] != 0:
                return True
        return False

    def prepInstance(self,x_ref,readOnly=False):
        x = np.zeros(self.numFeats)
        indxs = featureIndex()
        if self.IPtype == 0: #IPv4
            if self.version == 0:
                x[0] = float(str.split(x_ref[0][indxs.ip_src], '.')[3])  # take last byte of IP (e.g. 192.168.1.X)
                x[1] = float(str.split(x_ref[0][indxs.ip_dst], '.')[3])
                x[2:] = x_ref[0][(indxs.ip_hdr_len, indxs.ip_len, indxs.frame_len, indxs.ip_ttl, indxs.HH_L0_1_mean),].astype(float)
            elif self.version == 1:
                x[0] = float(str.split(x_ref[0][indxs.ip_src], '.')[3])  # take last byte of IP (e.g. 192.168.1.X)
                x[1:] = x_ref[0][(indxs.ip_flags_rb, indxs.ip_flags_df, indxs.ip_flags_mf, indxs.ip_ttl, indxs.ip_hdr_len),].astype(float)
            else:  # ==2
                x = x_ref[0][(indxs.ip_hdr_len, indxs.ip_len, indxs.ip_flags_rb, indxs.ip_flags_df, indxs.ip_flags_mf, indxs.ip_ttl, indxs.HH_L0_1_mean),].astype(float)
        else: # IPv6
            if self.version == 0:
                srcAddr = str.split(x_ref[0][indxs.ipv6_src], ':')
                dstAddr = str.split(x_ref[0][indxs.ipv6_dst], ':')
                src_o_1 = int(srcAddr[len(srcAddr) - 2], 16) if srcAddr[len(srcAddr) - 2] != '' else 0
                src_o_2 = int(srcAddr[len(srcAddr) - 1], 16) if srcAddr[len(srcAddr) - 1] != '' else 0
                dst_o_1 = int(dstAddr[len(dstAddr) - 2], 16) if dstAddr[len(dstAddr) - 2] != '' else 0
                dst_o_2 = int(dstAddr[len(dstAddr) - 1], 16) if dstAddr[len(dstAddr) - 1] != '' else 0
                x[0] = src_o_1 + src_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
                x[1] = dst_o_1 + dst_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
                x[2] = float(x_ref[0][indxs.ipv6_plen,])  # payload length
                x[3] = float(x_ref[0][indxs.frame_len,])  # actual payload length
                dstopts = x_ref[0][indxs.ipv6_dstopts_nxt,]  # ipv6.dstopts.nxt
                if dstopts == '':
                    dstopts = '0'
                x[4] = float(dstopts)
                x[5] = int(x_ref[0][indxs.ipv6_flow,], 16)  # ipv6.flow: convert from hex string '0x0000AF'
                x[6] = float(x_ref[0][indxs.HH_L0_1_mean,])  # HH L0.01 Mean
            elif self.version == 1:
                srcAddr = str.split(x_ref[0][indxs.ipv6_src], ':')
                dstAddr = str.split(x_ref[0][indxs.ipv6_dst], ':')
                src_o_1 = int(srcAddr[len(srcAddr) - 2], 16) if srcAddr[len(srcAddr) - 2] != '' else 0
                src_o_2 = int(srcAddr[len(srcAddr) - 1], 16) if srcAddr[len(srcAddr) - 1] != '' else 0
                dst_o_1 = int(dstAddr[len(dstAddr) - 2], 16) if dstAddr[len(dstAddr) - 2] != '' else 0
                dst_o_2 = int(dstAddr[len(dstAddr) - 1], 16) if dstAddr[len(dstAddr) - 1] != '' else 0
                x[0] = src_o_1 + src_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
                x[1] = dst_o_1 + dst_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
                x[2] = float(x_ref[0][indxs.ipv6_tclass_dscp,])  # ipv6.tclass.dscp
                x[3] = float(x_ref[0][indxs.ipv6_tclass_ecn,])  # ipv6.tclass.ecn
                x[4] = float(x_ref[0][indxs.ipv6_hlim,])  # ipv6.hlim
                x[5] = int(x_ref[0][indxs.ipv6_tclass,], 16)  # ipv6.tclass: convert from hex string '0x0000AF'
                x[6] = float(x_ref[0][indxs.ipv6_plen])  # payload len
            else:  # ==2
                x[0] = int(x_ref[0][indxs.ipv6_flow,], 16)  # ipv6.flow: convert from hex string '0x0000AF'
                optnext = x_ref[0][indxs.ipv6_dstopts_nxt,]
                x[1] = int(optnext, 16) if optnext != '' else 0  # ipv6.dstopts.nxt
                x[2:] = x_ref[0][(indxs.ipv6_hlim, indxs.ipv6_plen, indxs.frame_len, indxs.ipv6_version),].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_TCPUDP_Pr(Expert_Basic):
    def __init__(self, TCPUDPtype, version, name="TCP/UDP Protocol", Auth=2, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        numFeats = 7
        if TCPUDPtype == 0 or TCPUDPtype == 1 or TCPUDPtype == 2: # is TCP
            if version == 2:
                numFeats = 6
        else: #UDP
            numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold,rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.version = version
        self.TCPUDPtype = TCPUDPtype
        if TCPUDPtype == 0 or TCPUDPtype == 1 or TCPUDPtype == 2:
            if version == 0:
                self.name = "TCP Protocol: Scanning"
            elif version == 1:
                self.name = "TCP Fuzzing"
            else:
                self.name = "TCP Payload Inconsistency"
                self.Auth = 1
        else:
            self.name = "UDP Protocol"

    def isApplicable(self,x_ref):
        #** cpp: check instance for the enum!   if (has TCP AND i'm TCP AND ((i'm ver1 AND has IPv4) OR (i'm not ver1))) return true, else false
        # can do with single null pointer check  elseif (has UDP AND i'm UDP AND has IPv4)...
        indxs = featureIndex()
        #is TCP?
        if self.TCPUDPtype == 0 or self.TCPUDPtype == 1 or self.TCPUDPtype == 2:
            if self.version == 0:
                if (x_ref[0][indxs.ip_src] != '') and (x_ref[0][indxs.tcp_srcport] != ''):
                    return True
            else: #version is 1 or 2
                if x_ref[0][indxs.tcp_srcport] != '':
                    return True
        #is UDP
        if self.TCPUDPtype == 3 or self.TCPUDPtype == 4 or self.TCPUDPtype == 5:
            if (x_ref[0][indxs.ip_src] != '') and (x_ref[0][indxs.udp_srcport] != ''):
                return True
        return False

    def prepInstance(self,x_ref, readOnly=False):
        indxs = featureIndex()
        #TCP
        if self.TCPUDPtype == 0 or self.TCPUDPtype == 1 or self.TCPUDPtype == 2:
            if self.version == 0:
                x = np.zeros(self.numFeats)
                x[0] = float(str.split(x_ref[0][indxs.ip_src],'.')[3]) #take last byte of IP (e.g. 192.168.1.X)
                x[1:] = x_ref[0][(indxs.tcp_ack, indxs.tcp_flags_ack, indxs.tcp_flags_syn, indxs.HH_L1_mean, indxs.HH_L1_radius, indxs.HH_L1_pcc),].astype(float)
            elif self.version == 1:
                x = x_ref[0][(indxs.tcp_window_size_value, indxs.tcp_urgent_pointer, indxs.tcp_flags_ack, indxs.tcp_flags_cwr, indxs.tcp_flags_ecn, indxs.tcp_flags_syn, indxs.tcp_flags_urg),].astype(float)
            else:
                x = x_ref[0][(indxs.tcp_seq, indxs.tcp_ack, indxs.frame_len, indxs.tcp_window_size_value, indxs.HH_L0_1_mean, indxs.HpHp_L0_1_mean),].astype(float)
        else: #UDP - version0
            x = x_ref[0][(indxs.udp_length, indxs.frame_len, indxs.HH_L1_mean, indxs.HH_L1_std, indxs.HH_L1_radius, indxs.HH_L1_pcc),].astype(float)

        #update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        #0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_hosthostBW(Expert_Basic):
    def __init__(self, term, name="Inter-Host Channel", Auth=1, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=0,learningRate=0.1):
        if term == 0 or term ==1:
            numFeats = 6
        else: #=2
            numFeats = 5
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.term = term
        if self.term == 0:
            self.name = self.name + ": short-term"
        elif self.term == 1:
            self.name = self.name + ": long-term"
        else:
            self.name = self.name + ": multi-term"

    def prepInstance(self,x_ref,readOnly=False):
        indxs = featureIndex()
        if self.term == 0:
            findx = indxs.HH_L1_mean, indxs.HH_L1_std, indxs.HH_L1_magnitude, indxs.HH_L1_radius, indxs.HH_L1_covariance, indxs.HH_L1_pcc
        elif self.term == 1:
            findx = indxs.HH_L0_1_mean, indxs.HH_L0_1_std, indxs.HH_L0_1_magnitude, indxs.HH_L0_1_radius, indxs.HH_L0_1_covariance, indxs.HH_L0_1_pcc
        else:  # multiTerm
            findx = indxs.HH_L1_mean, indxs.HH_L1_pcc, indxs.HH_L0_1_mean, indxs.HH_L0_1_pcc, indxs.HH_L0_1_pcc

        x = x_ref[0][findx,].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_BW(Expert_Basic):
    def __init__(self, name="Total Bandwidth", Auth=1, threshold=-15, hiddenRatio=0.7,
                 gracePeriod=1000,
                 corruption=0.0, rm_winsize=5,learningRate=0.1):
        numFeats = 7
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        x = x_ref[0][(indxs.BW_L5_weight, indxs.BW_L5_mean, indxs.BW_L5_variance, indxs.BW_L3_mean, indxs.BW_L3_variance, indxs.BW_L1_mean, indxs.BW_L1_variance),].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_dirBW(Expert_Complex):
    def __init__(self, name="Directional Bandwidth", maxDirections=100, Auth=1, threshold=-15, hiddenRatio=0.7,
                 gracePeriod=1000, corruption=0.0,
                 rm_winsize=20,learningRate=0.1):  # gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize,learningRate=learningRate)
        self.subExperts = {}  # unordered_map
        self.maxSubExperts = maxDirections
        self.subExperts[''] = self.makeSubExpert(self.name + " (default subExpert)")

    def isApplicable(self,x_ref):
        # ** cpp: check instance for the enum!   if (x_ref->direction != NULL) return true, else false
        return True

    def findDirection(self,x_ref): #cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        indxs = featureIndex()
        if x_ref[0][indxs.ip_src] != '': #is IPv4
            lstP = x_ref[0][indxs.ip_src].rfind('.')
            src_subnet = x_ref[0][indxs.ip_src][0:lstP:]
            lstP = x_ref[0][indxs.ip_dst].rfind('.')
            dst_subnet = x_ref[0][indxs.ip_dst][0:lstP:]
        elif x_ref[0][indxs.ipv6_src] != '': #is IPv6
            src_subnet = x_ref[0][indxs.ipv6_src][0:round(len(x_ref[0][indxs.ipv6_src])/2):]
            dst_subnet = x_ref[0][indxs.ipv6_dst][0:round(len(x_ref[0][indxs.ipv6_dst])/2):]
        else: #no Network layer, use MACs
            src_subnet = x_ref[0][indxs.eth_src]
            dst_subnet = x_ref[0][indxs.eth_dst]

        dir = src_subnet + ">" + dst_subnet
        return dir

    def getSubExpertKey(self,x_ref):
        # cpp: this is all given to you in the direction string of the instance (just return the string)
        return self.findDirection(x_ref)

    def makeSubExpert(self,name):
        return [subExpert_BWdir(name=name, rm_winsize=self.rm_winsize, Auth=self.authority, threshold=self.threshold, hiddenRatio=self.hiddenRatio, gracePeriod=self.gracePeriod, corruption=self.corruption, learningRate=self.learningRate)]

class subExpert_BWdir(Expert_Basic):
    def __init__(self, name="subDirectional Bandwidth", Auth=1, threshold=-15, hiddenRatio=0.7, gracePeriod=1000,
                 corruption=0.0, rm_winsize=20, learningRate=0.1):
        numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        x = x_ref[0][(indxs.BW_dir_L0_3_mean, indxs.BW_dir_L0_3_std, indxs.BW_dir_L0_3_magnitude, indxs.BW_dir_L0_3_radius, indxs.BW_dir_L0_3_covariance, indxs.BW_dir_L0_3_pcc),].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_hostBW(Expert_Complex):
    def __init__(self, name="Host Bandwidth", maxHosts=100, Auth=1, threshold=-15, hiddenRatio=0.7,
                 gracePeriod=1000, corruption=0.0,
                 rm_winsize=20,learningRate=0.1):  # gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize, learningRate=learningRate)
        self.subExperts = {}  # unordered_map
        self.maxSubExperts = maxHosts
        self.subExperts[''] = self.makeSubExpert(self.name + " (default subExpert)")

    def isApplicable(self, x_ref):
        # ** cpp: check instance for the enum!   if (x_ref->IPtype != NA) return true, else false
        indxs = featureIndex()
        if x_ref[0][indxs.ip_src,] != '' or x_ref[0][indxs.ipv6_src,] != '':
            return True
        return False

    def getSubExpertKey(self,x_ref):
        srcHost = x_ref[0][featureIndex().ip_src]
        if srcHost == '':
            srcHost = x_ref[0][featureIndex().ipv6_src]
        return srcHost

    def makeSubExpert(self,name):
        return [subExpert_hostBW(name=name, rm_winsize=self.rm_winsize, Auth=self.authority,
                                threshold=self.threshold, hiddenRatio=self.hiddenRatio,
                                gracePeriod=self.gracePeriod, corruption=self.corruption, learningRate=self.learningRate)]

class subExpert_hostBW(Expert_Basic):
    def __init__(self, name="subHost Bandwidth", Auth=1, threshold=-15, hiddenRatio=0.7, gracePeriod=1000,
                 corruption=0.0, rm_winsize=20,learningRate=0.1):
        numFeats = 6
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        x = x_ref[0][(indxs.H_L5_mean, indxs.H_L5_variance, indxs.H_L3_mean, indxs.H_L3_variance, indxs.H_L1_mean, indxs.H_L1_variance),].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x

class Expert_RTS(Expert_Complex):
    def __init__(self, version, serverPorts = None, name="Real-time Stream", maxSubExperts=100, Auth=2, threshold=-50, hiddenRatio=0.7,
                 gracePeriod=1000, corruption=0.0,
                 rm_winsize=0,learningRate=0.1):  # gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
        numFeats = 7
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize,learningRate=learningRate)
        self.subExperts = {}  # unordered_map
        self.maxSubExperts = maxSubExperts
        self.version = version
        if serverPorts is None:
            self.serverPorts = (80, 554, 5000)
        else:
            self.serverPorts = serverPorts
        self.subExperts[''] = self.makeSubExpert(self.name + " (default subExpert)")

    def getServerSocket(self,x_ref):
        indxs = featureIndex()
        #cpp: do check with enums!!
        if x_ref[0][indxs.ip_src,] != '': #is IPv4
            ip = x_ref[0][indxs.ip_src,]
        else:
            ip = x_ref[0][indxs.ipv6_src,]
        if x_ref[0][indxs.tcp_srcport,]!='':
            return ip + x_ref[0][indxs.tcp_srcport,] # IP:port
        elif x_ref[0][indxs.udp_srcport,]!='':
            return ip + x_ref[0][indxs.udp_srcport,]  # IP:port

    def isApplicable(self,x_ref): #cpp: if (x_ref has TCP and TCPsrcport is in this->serverPorts... same for UDP)
        indxs = featureIndex()
        if x_ref[0][indxs.tcp_srcport,] != '':
            if int(x_ref[0][indxs.tcp_srcport,]) in self.serverPorts:
                return True
        elif x_ref[0][indxs.udp_srcport,] != '':
            if int(x_ref[0][indxs.udp_srcport,]) in self.serverPorts:
                return True
        return False

    def getSubExpertKey(self, x_ref):
        return self.getServerSocket(x_ref)

    def makeSubExpert(self,name):
        return [subExpert_RTS(version = self.version, name=name, rm_winsize=self.rm_winsize,
                                 Auth=self.authority,
                                 threshold=self.threshold, hiddenRatio=self.hiddenRatio,
                                 gracePeriod=self.gracePeriod, corruption=self.corruption, learningRate=self.learningRate)]

class subExpert_RTS(Expert_Basic):
    def __init__(self, version, name="subReal-time Stream", Auth=2, threshold=-50, hiddenRatio=0.7, gracePeriod=1000,
                 corruption=0.0, rm_winsize=20,learningRate=0.1):
        numFeats = 7
        self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption, threshold, rm_winsize=rm_winsize,learningRate=learningRate)
        self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio),
                        rng=np.random.RandomState(123))
        self.version = version
        self.name = self.name + ": (version " + str(version) + ")"

    def prepInstance(self, x_ref, readOnly=False):
        indxs = featureIndex()
        if self.version == 0: #in cpp, this is the enum 'Version'
            x = x_ref[0][(indxs.HpHp_L1_mean, indxs.HpHp_L1_std, indxs.HpHp_L1_pcc, indxs.HH_jit_L1_mean, indxs.HH_jit_L1_variance, indxs.HH_jit_L0_1_mean, indxs.HH_jit_L0_1_variance),].astype(float)
        else: #version is 1
            x = x_ref[0][(indxs.HpHp_L0_1_mean, indxs.HH_jit_L1_mean, indxs.HH_jit_L1_variance, indxs.HH_jit_L0_3_mean, indxs.HH_jit_L0_3_variance, indxs.HH_jit_L0_1_mean, indxs.HH_jit_L0_1_variance),].astype(float)
        # update norms?
        if not readOnly:
            self.norm_max[x > self.norm_max] = x[x > self.norm_max]
            self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
        return x



# out of date: -some oew fields and feature indx management
def expertFromJSON(JSONstring):
    return ''

# def expertFromJSON(JSONstring):
#     Js = json.loads(JSONstring)
#     # load general traits
#     name = Js['name']
#     authority = Js['authority']
#     threshold = Js['threshold']
#     hiddenRatio = Js['hiddenRatio']
#     gracePeriod = Js['gracePeriod']
#     featureIndxs = np.array(Js['featureIndxs'])
#     corruption = Js['corruption']
#     n = Js['n']
#     rm_winsize = Js['rollmean']
#
#     J_AE = Js.get('AE')  # is None is null
#
#     # create experts by type
#     expert_type = Js['expert_type']
#
#     if expert_type == 'Expert_cond':
#         expert = Expert_cond(featureIndxs=featureIndxs, conditional_Indx=Js['conditional_Indx'], name=name,
#                              Auth=authority, threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                              corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_TCP_UDP_CH':
#         expert = Expert_TCP_UDP_CH(featureIndxs=featureIndxs, srcPort_Indx=Js['srcPort_Indx'],
#                                    portRange=Js['portRange'], name=name, Auth=authority, threshold=threshold,
#                                    hiddenRatio=hiddenRatio, gracePeriod=gracePeriod, corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_dirBW':
#         expert = Expert_dirBW(featureIndxs=featureIndxs, srcIPv4Index=Js['srcIPv4Index'],
#                               dstIPv4Index=Js['srcIPv4Index'], srcIPv6Index=Js['srcIPv6Index'],
#                               dstIPv6Index=Js['srcIPv6Index'], srcMACIndex=Js['srcMACIndex'],
#                               dstMACIndex=Js['srcMACIndex'], maxDirections=Js['maxDirections'], name=name,
#                               Auth=authority, threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                               corruption=corruption)
#         for key, value in json.loads(Js['subExperts']).items():
#             subexp = expertFromJSON(value)
#             expert.subExperts[key] = [subexp]
#     elif expert_type == 'Expert_hostBW':
#         expert = Expert_hostBW(featureIndxs=featureIndxs, srcIndex=Js['srcIndex'], maxHosts=Js['maxHosts'],
#                                name=name, threshold=threshold, Auth=authority, hiddenRatio=hiddenRatio,
#                                gracePeriod=gracePeriod, corruption=corruption)
#         for key, value in json.loads(Js['subExperts']).items():
#             subexp = expertFromJSON(value)
#             expert.subExperts[key] = [subexp]
#     elif expert_type == 'Expert_IPv4':
#         expert = Expert_IPv4(expType=Js['expert_type'], other_featureIndxs=Js['featureIndxs'],
#                              ipsrc_indx=Js['ipsrc_indx'], ipdst_indx=Js['ipdst_indx'], name=name, Auth=authority,
#                              threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                              corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_IPv6':
#         expert = Expert_IPv6(expType=Js['expert_type'], other_featureIndxs=Js['featureIndxs'],
#                              ipsrc_indx=Js['ipsrc_indx'], ipdst_indx=Js['ipdst_indx'], name=name, Auth=authority,
#                              threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                              corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_ICMP':
#         expert = Expert_ICMP(expType=Js['expert_type'], other_featureIndxs=Js['featureIndxs'],
#                              ipsrc_indx=Js['ipsrc_indx'], ipdst_indx=Js['ipdst_indx'],
#                              icmp_type_indx=Js['icmp_type_indx'], icmp_code_indx=Js['icmp_code_indx'], name=name,
#                              Auth=authority, threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                              corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_TCP':
#         expert = Expert_TCP(expType=Js['expert_type'], other_featureIndxs=Js['featureIndxs'],
#                             ipsrc_indx=Js['ipsrc_indx'], port_indx=Js['port_indx'], name=name, Auth=authority,
#                             threshold=threshold, hiddenRatio=hiddenRatio, gracePeriod=gracePeriod,
#                             corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert_RTstream':
#         expert = Expert_RTstream(featureIndxs=featureIndxs, srcIPv4indx=Js['srcIPv4indx'],
#                                  srcTCPindx=Js['srcTCPindx'], srcUDPindx=['srcUDPindx'],
#                                  serverPorts=Js['serverPorts'], maxServers=Js['maxServers'], name=name,
#                                  Auth=authority, threshold=threshold, hiddenRatio=hiddenRatio,
#                                  gracePeriod=gracePeriod, corruption=corruption)
#         expert.AE.loadFromJSON(J_AE)
#     elif expert_type == 'Expert':
#         expert = Expert_Basic(featureIndxs=featureIndxs, name=name, Auth=authority, threshold=threshold,
#                               hiddenRatio=hiddenRatio, gracePeriod=gracePeriod, corruption=corruption,
#                               rm_winsize=rm_winsize)
#         expert.AE.loadFromJSON(J_AE)
#     else:
#         raise Exception("JSON parse Error: " + expert_type + " is not a known Expert")
#
#     J_score_stats = Js['score_stats']
#     norm_max = np.array(Js['norm_max'])
#     norm_min = np.array(Js['norm_min'])
#     expert.score_stats.loadFromJSON(J_score_stats)
#     expert.norm_max = norm_max
#     expert.norm_min = norm_min
#     expert.n = n
#     return expert


#### for python ONLY (in cpp use struct of instance...) ######
class featureIndex():
    def __init__(self):
        ### Coded Indexes (for the given dataset):
        self.frame_time_epoch = 0
        self.frame_len = 1
        self.eth_src = 2
        self.eth_dst = 3
        self.ip_src = 4
        self.ip_dst = 5
        self.ip_hdr_len = 6
        self.ip_len = 7
        self.ip_flags_rb = 8
        self.ip_flags_df = 9
        self.ip_flags_mf = 10
        self.ip_ttl = 11
        self.tcp_srcport = 12
        self.tcp_dstport = 13
        self.tcp_seq = 14
        self.tcp_ack = 15
        self.tcp_flags_res = 16
        self.tcp_flags_ack = 17
        self.tcp_flags_cwr = 18
        self.tcp_flags_ecn = 19
        self.tcp_flags_fin = 20
        self.tcp_flags_ns = 21
        self.tcp_flags_push = 22
        self.tcp_flags_reset = 23
        self.tcp_flags_syn = 24
        self.tcp_flags_urg = 25
        self.tcp_window_size_value = 26
        self.tcp_urgent_pointer = 27
        self.udp_length = 28
        self.udp_srcport = 29
        self.udp_dstport = 30
        self.icmp_type = 31
        self.icmp_code = 32
        self.arp_opcode = 33
        self.arp_src_hw_mac = 34
        self.arp_src_proto_ipv4 = 35
        self.arp_dst_hw_mac = 36
        self.arp_dst_proto_ipv4 = 37
        self.http_request_method = 38
        self.http_request_uri = 39
        self.http_request_version = 40
        self.http_response_code = 41
        self.http_host = 42
        self.http_connection = 43
        self.ipv6_src = 44
        self.ipv6_dst = 45
        self.ipv6_dstopts_nxt = 46
        self.ipv6_flow = 47
        self.ipv6_tclass = 48
        self.ipv6_tclass_dscp = 49
        self.ipv6_tclass_ecn = 50
        self.ipv6_hlim = 51
        self.ipv6_version = 52
        self.ipv6_plen = 53
        self.BW_L5_weight = 54
        self.BW_L5_mean = 55
        self.BW_L5_variance = 56
        self.BW_L3_weight = 57
        self.BW_L3_mean = 58
        self.BW_L3_variance = 59
        self.BW_L1_weight = 60
        self.BW_L1_mean = 61
        self.BW_L1_variance = 62
        self.BW_L0_1_weight = 63
        self.BW_L0_1_mean = 64
        self.BW_L0_1_variance = 65
        self.BW_L0_01_weight = 66
        self.BW_L0_01_mean = 67
        self.BW_L0_01_variance = 68
        self.BW_L0_001_weight = 69
        self.BW_L0_001_mean = 70
        self.BW_L0_001_variance = 71
        self.BW_dir_L1_weight = 72
        self.BW_dir_L1_mean = 73
        self.BW_dir_L1_std = 74
        self.BW_dir_L1_magnitude = 75
        self.BW_dir_L1_radius = 76
        self.BW_dir_L1_covariance = 77
        self.BW_dir_L1_pcc = 78
        self.BW_dir_L0_3_weight = 79
        self.BW_dir_L0_3_mean = 80
        self.BW_dir_L0_3_std = 81
        self.BW_dir_L0_3_magnitude = 82
        self.BW_dir_L0_3_radius = 83
        self.BW_dir_L0_3_covariance = 84
        self.BW_dir_L0_3_pcc = 85
        self.BW_dir_L0_1_weight = 86
        self.BW_dir_L0_1_mean = 87
        self.BW_dir_L0_1_std = 88
        self.BW_dir_L0_1_magnitude = 89
        self.BW_dir_L0_1_radius = 90
        self.BW_dir_L0_1_covariance = 91
        self.BW_dir_L0_1_pcc = 92
        self.MI_dir_L0_01_weight = 93
        self.MI_dir_L0_01_mean = 94
        self.MI_dir_L0_01_variance = 95
        self.H_L5_weight = 96
        self.H_L5_mean = 97
        self.H_L5_variance = 98
        self.H_L3_weight = 99
        self.H_L3_mean = 100
        self.H_L3_variance = 101
        self.H_L1_weight = 102
        self.H_L1_mean = 103
        self.H_L1_variance = 104
        self.H_L0_1_weight = 105
        self.H_L0_1_mean = 106
        self.H_L0_1_variance = 107
        self.H_L0_01_weight = 108
        self.H_L0_01_mean = 109
        self.H_L0_01_variance = 110
        self.H_L0_001_weight = 111
        self.H_L0_001_mean = 112
        self.H_L0_001_variance = 113
        self.HH_L1_weight = 114
        self.HH_L1_mean = 115
        self.HH_L1_std = 116
        self.HH_L1_magnitude = 117
        self.HH_L1_radius = 118
        self.HH_L1_covariance = 119
        self.HH_L1_pcc = 120
        self.HH_L0_1_weight = 121
        self.HH_L0_1_mean = 122
        self.HH_L0_1_std = 123
        self.HH_L0_1_magnitude = 124
        self.HH_L0_1_radius = 125
        self.HH_L0_1_covariance = 126
        self.HH_L0_1_pcc = 127
        self.HH_L0_01_weight = 128
        self.HH_L0_01_mean = 129
        self.HH_L0_01_std = 130
        self.HH_L0_01_magnitude = 131
        self.HH_L0_01_radius = 132
        self.HH_L0_01_covariance = 133
        self.HH_L0_01_pcc = 134
        self.HH_jit_L1_weight = 135
        self.HH_jit_L1_mean = 136
        self.HH_jit_L1_variance = 137
        self.HH_jit_L0_3_weight = 138
        self.HH_jit_L0_3_mean = 139
        self.HH_jit_L0_3_variance = 140
        self.HH_jit_L0_1_weight = 141
        self.HH_jit_L0_1_mean = 142
        self.HH_jit_L0_1_variance = 143
        self.HpHp_L1_weight = 144
        self.HpHp_L1_mean = 145
        self.HpHp_L1_std = 146
        self.HpHp_L1_magnitude = 147
        self.HpHp_L1_radius = 148
        self.HpHp_L1_covariance = 149
        self.HpHp_L1_pcc = 150
        self.HpHp_L0_1_weight = 151
        self.HpHp_L0_1_mean = 152
        self.HpHp_L0_1_std = 153
        self.HpHp_L0_1_magnitude = 154
        self.HpHp_L0_1_radius = 155
        self.HpHp_L0_1_covariance = 156
        self.HpHp_L0_1_pcc = 157
        self.HpHp_L0_01_weight = 158
        self.HpHp_L0_01_mean = 159
        self.HpHp_L0_01_std = 160
        self.HpHp_L0_01_magnitude = 161
        self.HpHp_L0_01_radius = 162
        self.HpHp_L0_01_covariance = 163
        self.HpHp_L0_01_pcc = 164

#TODO: hard set authorities

###############  out of date  ############
#This expert will check if the given conditional_Indx (feature) has content, if not, then it will not proceed
#CPP Implimentation: give name (enum) of protocol
# class Expert_cond(Expert_Basic):
#     def __init__(self,numFeats,conditional_Indx,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         self.init(numFeats,name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.conditional_Indx = conditional_Indx
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_cond'
#         jsonobj['conditional_Indx'] = self.conditional_Indx
#         return jsonobj
#
#     #x_ref is a mutable object [] to the feature vector 'x'
#     def train(self,x_ref):
#         if x_ref[0][self.conditional_Indx] == '':
#             return
#         x = self.prepInstance(x_ref)
#         self.n = self.n + 1
#         error = self.AE.train(x,corruption_level=self.corruption)
#         if self.n > self.gracePeriod:
#             self.score_stats.insert(np.log(error), x_ref[0][len(x_ref[0]) - 1])
#
#     #x_ref is a mutable object [] to the numrical vector 'x'
#     def vote(self,x_ref):
#         if self.n < self.gracePeriod*2:
#             return 0.0
#         if x_ref[0][self.conditional_Indx] == '':
#             return 0.0
#         #std = self.train_stats.std()
#         return self.score(x_ref)
#
#         #consider: histogram of long long unsigned int -then look at >98th percentile
#
# #will only work when a value is supplied at srcPort_Indx (i.e. udp or tcp), and
# #when the portRange is:
# #   0 =: Well-known (0-1023)
# #   1 =: Registered (1024-49151)
# #   2 =: Dynamic (49152-65535)
# class Expert_TCP_UDP_CH(Expert_Basic):
#     def __init__(self,numFeats,srcPort_Indx, portRange,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         self.init(numFeats,name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.srcPort_Indx = srcPort_Indx
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#         if portRange == 0:
#             self.portRange = (0,1023)
#         elif portRange == 1:
#             self.portRange = (1024,49151)
#         else:
#             self.portRange = (49152,65535)
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_TCP_UDP_CH'
#         jsonobj['srcPort_Indx'] = self.srcPort_Indx
#         jsonobj['portRange'] = self.portRange
#         return jsonobj
#
#     def isApplicablePacket(self,x_ref):
#         srcPort = x_ref[0][self.srcPort_Indx]
#         if srcPort != '':
#             if int(srcPort) >= self.portRange[0] & int(srcPort) <= self.portRange[1]:
#                 return True
#         return False
#
#     #x_ref is a mutable object [] to the feature vector 'x'
#     def train(self,x_ref):
#         if not self.isApplicablePacket(x_ref):
#             return
#         x = self.prepInstance(x_ref)
#         self.n = self.n + 1
#         error = self.AE.train(x,corruption_level=self.corruption)
#         if self.n > self.gracePeriod:
#             self.score_stats.insert(np.log(error), x_ref[0][len(x_ref[0]) - 1])
#
#     #x_ref is a mutable object [] to the numrical vector 'x'
#     def vote(self,x_ref):
#         if self.n < self.gracePeriod*2:
#             return 0.0
#         if not self.isApplicablePacket(x_ref):
#             return 0.0
#         #std = self.train_stats.std()
#         return self.score(x_ref)
#
#         #consider: histogram of long long unsigned int -then look at >98th percentile
#
# #builds an AE for each direction to monitor BW. dirIndex is the index to the feature indicating the direction
# class Expert_dirBW(Expert_Complex):
#     def __init__(self, numFeats, srcIPv4Index, dstIPv4Index,srcIPv6Index, dstIPv6Index, srcMACIndex, dstMACIndex, maxDirections=5, name="unknown_expert", Auth=0,threshold=-50, hiddenRatio=0.7, gracePeriod=1000, corruption=0.0, rm_winsize=20): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         self.init(numFeats,name,Auth,hiddenRatio,gracePeriod, corruption,threshold,rm_winsize=rm_winsize)
#         self.maxSubExperts = maxDirections
#         self.srcIPv4Index = srcIPv4Index
#         self.dstIPv4Index = dstIPv4Index
#         self.srcIPv6Index = srcIPv6Index
#         self.dstIPv6Index = dstIPv6Index
#         self.srcMACIndex = srcMACIndex
#         self.dstMACIndex = dstMACIndex
#         self.subExperts = dict()
#         #add the default subExpert
#         Exp = [Expert_Basic(self.featureIndxs, name="defaultHost_directionalBWExp", Auth=0, threshold=-50, hiddenRatio=0.7, gracePeriod=1000, rm_winsize=20, corruption=corruption)]
#         self.subExperts[''] = Exp
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_dirBW'
#         jsonobj['maxDirections'] = self.maxSubExperts
#         jsonobj['srcIPv4Index'] = self.srcIPv4Index
#         jsonobj['dstIPv4Index'] = self.dstIPv4Index
#         jsonobj['srcIPv6Index'] = self.srcIPv6Index
#         jsonobj['dstIPv6Index'] = self.dstIPv6Index
#         jsonobj['srcMACIndex'] = self.srcMACIndex
#         jsonobj['dstMACIndex'] = self.dstMACIndex
#         subExperts = {}
#         for key, value in self.subExperts.items():
#             subExperts[key]= value[0].toJSON()
#         jsonobj['subExperts'] = json.dumps(subExperts)
#         return jsonobj
#
#     def findDirection(self,x_ref):
#         if x_ref[0][self.srcIPv4Index] != '': #is IPv4
#             lstP = x_ref[0][self.srcIPv4Index].rfind('.')
#             src_subnet = x_ref[0][self.srcIPv4Index][0:lstP:]
#             lstP = x_ref[0][self.dstIPv4Index].rfind('.')
#             dst_subnet = x_ref[0][self.dstIPv4Index][0:lstP:]
#         elif x_ref[0][self.srcIPv6Index] != '': #is IPv6
#             src_subnet = x_ref[0][self.srcIPv6Index][0:round(len(x_ref[0][self.srcIPv6Index])/2):]
#             dst_subnet = x_ref[0][self.dstIPv6Index][0:round(len(x_ref[0][self.dstIPv6Index])/2):]
#         else: #no Network layer, use MACs
#             src_subnet = x_ref[0][self.srcMACIndex]
#             dst_subnet = x_ref[0][self.dstMACIndex]
#
#         dir = src_subnet + ">" + dst_subnet
#         return dir
#
#     def getSubExpert(self,x_ref): #****make pure virtual!!
#         dir = self.findDirection(x_ref)
#         return self.subExperts.get(dir)
#
# #builds an AE to learn each IPv$ host's BW. srcIndex is the index to the IPv4 src address
# class Expert_hostBW(Expert_Complex):
#     def __init__(self,numFeats,srcIndex,maxHosts=50,name="unknown_expert",threshold=-50, Auth=0,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0, rm_winsize=20): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         self.init(numFeats,name,Auth,hiddenRatio,gracePeriod, corruption,threshold,rm_winsize=rm_winsize)
#         self.maxSubExperts = maxHosts
#         self.srcIndex = srcIndex #indx to IPv4 src address
#         self.subExperts = dict()
#         #add the default subExpert
#         Exp = [Expert_Basic(self.featureIndxs, name="defaultHost_BWExp", Auth=Auth, hiddenRatio=hiddenRatio, threshold=threshold, corruption=corruption, gracePeriod=gracePeriod, rm_winsize=20)]
#         self.subExperts[''] = Exp
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_hostBW'
#         jsonobj['srcIndex'] = self.srcIndex
#         jsonobj['maxHosts'] = self.maxSubExperts
#         subExperts = {}
#         for key, value in self.subExperts.items():
#             subExperts[key]= value[0].toJSON()
#         jsonobj['subExperts'] = json.dumps(subExperts)
#         return jsonobj
#
#     def getSubExpert(self,x_ref):
#         srcHost = x_ref[0][self.srcIndex]
#         return self.subExperts.get(srcHost)
#
# #This IPv4 Expert is one of 3 types: 0, 1, 2 (expType). This affects which features are taken.
# class Expert_IPv4(Expert_Basic):
#     def __init__(self,expType,other_featureIndxs,ipsrc_indx, ipdst_indx,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         if expType == 0:
#             numFeats = len(other_featureIndxs)+2
#         elif expType == 1:
#             numFeats = len(other_featureIndxs)+1
#         else:
#             numFeats = len(other_featureIndxs)
#         self.init(np.zeros(numFeats),name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.featureIndxs = other_featureIndxs
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#         self.expType=expType
#         self.ipsrc_indx = ipsrc_indx
#         self.ipdst_indx = ipdst_indx
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_IPv4'
#         jsonobj['expsubType'] = self.expType
#         jsonobj['ipsrc_indx'] = self.ipsrc_indx
#         jsonobj['ipdst_indx'] = self.ipdst_indx
#         return jsonobj
#
#     def isApplicablePacket(self,x_ref):
#         ipsrc = x_ref[0][self.ipsrc_indx]
#         if ipsrc != '':
#             return True
#         return False
#
#     def prepInstance(self,x_ref,readOnly=False):
#         x = np.zeros(self.AE.n_visible)
#         if self.expType == 0:
#             x[0] = float(str.split(x_ref[0][self.ipsrc_indx],'.')[3]) #take last byte of IP (e.g. 192.168.1.X)
#             x[1] = float(str.split(x_ref[0][self.ipdst_indx],'.')[3])
#             x[2:] = x_ref[0][self.featureIndxs,].astype(float)
#         elif self.expType == 1:
#             x[0] = float(str.split(x_ref[0][self.ipsrc_indx],'.')[3]) #take last byte of IP (e.g. 192.168.1.X)
#             x[1:] = x_ref[0][self.featureIndxs,].astype(float)
#         else: #==2
#             x = x_ref[0][self.featureIndxs,].astype(float)
#         #update norms?
#         if not readOnly:
#             self.norm_max[x > self.norm_max] = x[x > self.norm_max]
#             self.norm_min[x < self.norm_min] = x[x < self.norm_min]
#         #0-1 normalize
#         x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
#         return x
#
#     #x_ref is a mutable object [] to the feature vector 'x'
#     def train(self,x_ref):
#         if not self.isApplicablePacket(x_ref):
#             return
#         x = self.prepInstance(x_ref)
#         self.n = self.n + 1
#         error = self.AE.train(x,corruption_level=self.corruption)
#         if self.n > self.gracePeriod:
#             self.score_stats.insert(np.log(error), x_ref[0][len(x_ref[0]) - 1])
#
#     #x_ref is a mutable object [] to the numrical vector 'x'
#     def vote(self,x_ref):
#         if self.n < self.gracePeriod*2:
#             return 0.0
#         if not self.isApplicablePacket(x_ref):
#             return 0.0
#         #std = self.train_stats.std()
#         return self.score(x_ref)
#
#         #consider: histogram of long long unsigned int -then look at >98th percentile
#
# #This IPv6 Expert is one of 3 types: 0, 1, 2 (expType). This affects which features are taken.
# class Expert_IPv6(Expert_IPv4):
#     def __init__(self,expType,other_featureIndxs,ipsrc_indx, ipdst_indx,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         if expType == 0 or expType == 1:
#             numFeats = len(other_featureIndxs)+2
#         else:
#             numFeats = len(other_featureIndxs)
#         self.init(np.zeros(numFeats),name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.featureIndxs = other_featureIndxs
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#         self.expType=expType
#         self.ipsrc_indx = ipsrc_indx
#         self.ipdst_indx = ipdst_indx
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_IPv6'
#         jsonobj['expsubType'] = self.expType
#         jsonobj['ipsrc_indx'] = self.ipsrc_indx
#         jsonobj['ipdst_indx'] = self.ipdst_indx
#         return jsonobj
#
#     def prepInstance(self,x_ref,readOnly=False):
#         x = np.zeros(self.AE.n_visible)
#         if self.expType == 0:
#             srcAddr = str.split(x_ref[0][self.ipsrc_indx],':')
#             dstAddr = str.split(x_ref[0][self.ipdst_indx],':')
#             src_o_1 = int(srcAddr[len(srcAddr)-2],16) if srcAddr[len(srcAddr)-2] != '' else 0
#             src_o_2 = int(srcAddr[len(srcAddr)-1],16) if srcAddr[len(srcAddr)-1] != '' else 0
#             dst_o_1 = int(dstAddr[len(dstAddr)-2],16) if dstAddr[len(dstAddr)-2] != '' else 0
#             dst_o_2 = int(dstAddr[len(dstAddr)-1],16) if dstAddr[len(dstAddr)-1] != '' else 0
#             x[0] = src_o_1 + src_o_2    #take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
#             x[1] = dst_o_1 + dst_o_2    #take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
#             x[2] = float(x_ref[0][self.featureIndxs[0],]) #payload length
#             x[3] = float(x_ref[0][self.featureIndxs[1],]) #actual payload length
#             dstopts = x_ref[0][self.featureIndxs[2],] #ipv6.dstopts.nxt
#             if dstopts == '':
#                 dstopts = '0'
#             x[4] = float(dstopts)
#             x[5] = int(x_ref[0][self.featureIndxs[3],],16) #ipv6.flow: convert from hex string '0x0000AF'
#             x[6] = x_ref[0][self.featureIndxs[4],] #HH L0.01 Mean
#         elif self.expType == 1:
#             srcAddr = str.split(x_ref[0][self.ipsrc_indx],':')
#             dstAddr = str.split(x_ref[0][self.ipdst_indx],':')
#             src_o_1 = int(srcAddr[len(srcAddr) - 2], 16) if srcAddr[len(srcAddr) - 2] != '' else 0
#             src_o_2 = int(srcAddr[len(srcAddr) - 1], 16) if srcAddr[len(srcAddr) - 1] != '' else 0
#             dst_o_1 = int(dstAddr[len(dstAddr) - 2], 16) if dstAddr[len(dstAddr) - 2] != '' else 0
#             dst_o_2 = int(dstAddr[len(dstAddr) - 1], 16) if dstAddr[len(dstAddr) - 1] != '' else 0
#             x[0] = src_o_1 + src_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
#             x[1] = dst_o_1 + dst_o_2  # take last two octet sets, voncert to int then sum them. (e.g. 'fe80::e8c4:b026:fa99:5ff1' > 'fe80::e8c4:b026:X:X')
#             x[2] = float(x_ref[0][self.featureIndxs[0],]) #ipv6.tclass.dscp
#             x[3] = float(x_ref[0][self.featureIndxs[1],]) #ipv6.tclass.ecn
#             x[4] = float(x_ref[0][self.featureIndxs[2],]) #ipv6.tclass.hlim
#             x[5] = int(x_ref[0][self.featureIndxs[3],],16) #ipv6.tclass: convert from hex string '0x0000AF'
#             x[6] = float(x_ref[0][self.featureIndxs[4],]) #payload len
#         else: #==2
#             x[0] = int(x_ref[0][self.featureIndxs[0],],16) #ipv6.flow: convert from hex string '0x0000AF'
#             optnext = x_ref[0][self.featureIndxs[1],]
#             x[1] = int(optnext,16) if optnext != '' else 0 #ipv6.dstopts.nxt
#             x[2:] = x_ref[0][self.featureIndxs[2:],].astype(float)
#         #update norms?
#         if not readOnly:
#             self.norm_max[x > self.norm_max] = x[x > self.norm_max]
#             self.norm_min[x < self.norm_min] = x[x < self.norm_min]
#         #0-1 normalize
#         x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
#         return x
#
# #This IPv4 Expert is one of 3 types: 0, 1, 2 (expType). This affects which features are taken.
# class Expert_ICMP(Expert_Basic):
#     def __init__(self,expType,other_featureIndxs,ipsrc_indx, ipdst_indx, icmp_type_indx, icmp_code_indx,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         if expType == 0:
#             numFeats = len(other_featureIndxs)+2
#         elif expType == 1:
#             numFeats = len(other_featureIndxs)+5
#         elif expType == 2:
#             numFeats = len(other_featureIndxs)+3
#         else: #=3
#             numFeats = len(other_featureIndxs)+4
#         self.init(np.zeros(numFeats),name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.featureIndxs = other_featureIndxs
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#         self.expType=expType
#         self.ipsrc_indx = ipsrc_indx
#         self.ipdst_indx = ipdst_indx
#         self.icmp_type_indx = icmp_type_indx
#         self.icmp_code_indx = icmp_code_indx
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_ICMP'
#         jsonobj['expsubType'] = self.expType
#         jsonobj['ipsrc_indx'] = self.ipsrc_indx
#         jsonobj['ipdst_indx'] = self.ipdst_indx
#         jsonobj['icmp_type_indx'] = self.icmp_type_indx
#         jsonobj['icmp_code_indx'] = self.icmp_code_indx
#         return jsonobj
#
#     def isApplicablePacket(self,x_ref):
#         icmp_type = x_ref[0][self.icmp_type_indx]
#         if icmp_type != '':
#             if self.expType == 0:
#                 return True
#             elif self.expType == 1 & (icmp_type == 3 or icmp_type == 5 or icmp_type == 11 or icmp_type == 12): # Destination Unreachable,Redirect Message- Time Exceeded,Parameter Problem: Bad IP header
#                 return True
#             elif self.expType == 2 & (icmp_type == 5 or icmp_type == 9): #Router Advert,Redirect
#                 return True
#             elif self.expType == 3 & (icmp_type == 8 or icmp_type == 10):
#                 return True
#         return False
#
#     def prepInstance(self,x_ref,readOnly=False):
#         x = np.zeros(self.AE.n_visible)
#         if self.expType == 0:
#             ipsrc = str.split(x_ref[0][self.ipsrc_indx],'.')
#             ipdst = str.split(x_ref[0][self.ipdst_indx],'.')
#             x[0] = float(ipsrc[3]) #take last byte of IP (e.g. 192.168.1.X)
#             x[1] = float(ipdst[3])
#             x[2:] = x_ref[0][self.featureIndxs,].astype(float)
#         elif self.expType == 1:
#             ipsrc = str.split(x_ref[0][self.ipsrc_indx],'.')
#             x[0] = float(ipsrc[2]) + float(ipsrc[3]) #take last two bytes of IP (e.g. 192.168.X.X)
#             x[1] = float(x_ref[0][self.icmp_code_indx,]) if float(x_ref[0][self.icmp_type_indx,]) == 3 else -1.0 #the Destination Unreachable code if it exists
#             x[2] = float(x_ref[0][self.icmp_code_indx,]) if float(x_ref[0][self.icmp_type_indx,]) == 11 else -1.0 #the TTL Exceeded Code if it exists
#             x[3] = float(x_ref[0][self.icmp_code_indx,]) if float(x_ref[0][self.icmp_type_indx,]) == 5 else -1.0 #the Redirect Code if it exists
#             x[4] = float(x_ref[0][self.icmp_code_indx,]) if float(x_ref[0][self.icmp_type_indx,]) == 12 else -1.0 #the param problem Code if it exists
#             x[5] = float(x_ref[0][self.featureIndxs,])
#         elif self.expType == 2:
#             ipsrc = str.split(x_ref[0][self.ipsrc_indx],'.')
#             ipdst = str.split(x_ref[0][self.ipdst_indx],'.')
#             x[0] = float(ipsrc[2]) + float(ipsrc[3]) #take last two bytes of IP (e.g. 192.168.X.X)
#             x[1] = float(ipdst[3]) # last  byte of IP (e.g. 192.168.1.X)
#             x[2] = 0.0 if x_ref[0][self.icmp_type_indx,].astype(float) == 5 else 1.0 #router Advert, or Redirect message?
#             x[3:] = x_ref[0][self.featureIndxs,].astype(float)
#         else: #==3
#             x = x_ref[0][self.featureIndxs,].astype(float)
#             ipsrc = str.split(x_ref[0][self.ipsrc_indx], '.')
#             ipdst = str.split(x_ref[0][self.ipdst_indx], '.')
#             x[0] = float(ipsrc[3])  # take last byte of IP (e.g. 192.168.1.X)
#             x[1] = float(ipdst[3])
#             icmp_type = x_ref[0][self.icmp_type_indx,].astype(float)
#             x[2] = icmp_type if (icmp_type == 0 or icmp_type == 8) else -1.0 #is this icmp echo request or reply?
#             x[3] = 1.0 if icmp_type == 10 else 0.0  # is this icmp router solicitation?
#             x[4:] = x_ref[0][self.featureIndxs,].astype(float)
#
#         #update norms?
#         if not readOnly:
#             self.norm_max[x > self.norm_max] = x[x > self.norm_max]
#             self.norm_min[x < self.norm_min] = x[x < self.norm_min]
#         #0-1 normalize
#         x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
#         return x
#
#     #x_ref is a mutable object [] to the feature vector 'x'
#     def train(self,x_ref):
#         if not self.isApplicablePacket(x_ref):
#             return
#         x = self.prepInstance(x_ref)
#         self.n = self.n + 1
#         error = self.AE.train(x,corruption_level=self.corruption)
#         if self.n > self.gracePeriod:
#             self.score_stats.insert(np.log(error), x_ref[0][len(x_ref[0]) - 1])
#
#     #x_ref is a mutable object [] to the numrical vector 'x'
#     def vote(self,x_ref):
#         if self.n < self.gracePeriod*2:
#             return 0.0
#         if not self.isApplicablePacket(x_ref):
#             return 0.0
#         #std = self.train_stats.std()
#         return self.score(x_ref)
#
#         #consider: histogram of long long unsigned int -then look at >98th percentile
#
# #This IPv4 Expert is one of 3 types: 0, 1, 2 (expType). This affects which features are taken.
# class Expert_TCP(Expert_Basic):
#     def __init__(self,expType,other_featureIndxs,ipsrc_indx,port_indx,name="unknown_expert",Auth=0,threshold=-50,hiddenRatio=0.7,gracePeriod=1000, corruption=0.0): #gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         if expType == 0:
#             numFeats = len(other_featureIndxs)+1
#         else: #=1
#             numFeats = len(other_featureIndxs)
#         self.init(np.zeros(numFeats),name,Auth,hiddenRatio,gracePeriod, corruption,threshold)
#         self.featureIndxs = other_featureIndxs
#         self.AE = ae.dA(n_visible=self.numFeats, n_hidden=math.ceil(self.numFeats * hiddenRatio), rng=np.random.RandomState(123))
#         self.expType=expType
#         self.ipsrc_indx = ipsrc_indx
#         self.port_indx = port_indx
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_TCP'
#         jsonobj['expsubType'] = self.expType
#         jsonobj['ipsrc_indx'] = self.ipsrc_indx
#         jsonobj['port_indx'] = self.port_indx
#         return jsonobj
#
#     def isApplicablePacket(self,x_ref):
#         if x_ref[0][self.ipsrc_indx] != '' and x_ref[0][self.port_indx] != '':
#             return True
#         return False
#
#     def prepInstance(self,x_ref, readOnly=False):
#         x = np.zeros(self.AE.n_visible)
#         if self.expType == 0:
#             x[0] = float(str.split(x_ref[0][self.ipsrc_indx],'.')[3]) #take last byte of IP (e.g. 192.168.1.X)
#             x[1:] = x_ref[0][self.featureIndxs,].astype(float)
#         else: #==1
#             x = x_ref[0][self.featureIndxs,].astype(float)
#         #update norms?
#         if not readOnly:
#             self.norm_max[x > self.norm_max] = x[x > self.norm_max]
#             self.norm_min[x < self.norm_min] = x[x < self.norm_min]
#         #0-1 normalize
#         x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
#         return x
#
#     #x_ref is a mutable object [] to the feature vector 'x'
#     def train(self,x_ref):
#         if not self.isApplicablePacket(x_ref):
#             return
#         x = self.prepInstance(x_ref)
#         self.n = self.n + 1
#         error = self.AE.train(x,corruption_level=self.corruption)
#         if self.n > self.gracePeriod:
#             self.score_stats.insert(np.log(error), x_ref[0][len(x_ref[0]) - 1])
#
#     #x_ref is a mutable object [] to the numrical vector 'x'
#     def vote(self,x_ref):
#         if self.n < self.gracePeriod*2:
#             return 0.0
#         if not self.isApplicablePacket(x_ref):
#             return 0.0
#         #std = self.train_stats.std()
#         return self.score(x_ref)
#
#         #consider: histogram of long long unsigned int -then look at >98th percentile
#
# #Builds Expert for each realtime stream server (source socket IP-port where port is one of those provided)
# class Expert_RTstream(Expert_Complex):
#     def __init__(self, numFeats, srcIPv4indx, srcTCPindx, srcUDPindx, serverPorts, maxServers=5, name="unknown_expert", Auth=0,threshold=-50,
#                  hiddenRatio=0.7, gracePeriod=1000,
#                  corruption=0.0):  # gracePeriod is the number of samples ignored before we begin updating the stats (used to normalize predicitons scores)
#         self.init(numFeats, name, Auth, hiddenRatio, gracePeriod, corruption,threshold)
#         self.maxServers = maxServers
#         self.serverPorts = serverPorts
#         self.srcIPv4indx = srcIPv4indx
#         self.srcTCPindx = srcTCPindx
#         self.srcUDPindx = srcUDPindx
#         self.subExperts = dict()
#         Exp = [Expert_Basic(self.featureIndxs, "default RTstream Expert", corruption=corruption, gracePeriod=gracePeriod, threshold=threshold, Auth=Auth)]
#         self.subExperts[''] = Exp
#
#     def toJSONmore(self,jsonobj):
#         jsonobj['expert_type'] = 'Expert_RTstream'
#         jsonobj['maxServers'] = self.maxServers
#         jsonobj['serverPorts'] = self.serverPorts
#         jsonobj['srcIPv4indx'] = self.srcIPv4indx
#         jsonobj['srcTCPindx'] = self.srcTCPindx
#         jsonobj['srcUDPindx'] = self.srcUDPindx
#         subExperts = {}
#         for key, value in self.subExperts.items():
#             subExperts[key]= value[0].toJSON()
#         jsonobj['subExperts'] = json.dumps(subExperts)
#         return jsonobj
#
#     def getServerSocket(self,x_ref):
#         if x_ref[0][self.srcTCPindx,]!='':
#             return x_ref[0][self.srcIPv4indx,] + x_ref[0][self.srcTCPindx,] # IP:port
#         elif x_ref[0][self.srcUDPindx,]!='':
#             return x_ref[0][self.srcIPv4indx,] + x_ref[0][self.srcUDPindx,]  # IP:port
#         #NOTE: if srcPort=='' then its not TCP/UDP...
#         return '' #This packet does not belong to a source stream
#
#     def isApplicable(self,x_ref):
#         socket = self.getServerSocket(x_ref)
#         if socket == '':  # not a legitimate ipv4 subnet-subnet direction
#             return False
#         return True
