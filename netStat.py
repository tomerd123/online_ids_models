import AfterImage as af
import numpy as np

class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    def __init__(self, HostLimit=255,HostSimplexLimit=1000):
        self.HT = af.incStatHT_2D()

        #Lambdas
        self.L_BW_dir = (1,.3,.1) #Directional BW Stats
        self.L_jit = (1,.3,.1) #H-H Jitter Stats
        self.L_BW = (5,3,1,.1,.01,.001) #General BW Stats
        self.L_MI = (0.01,) #MAC-IP relationships
        self.L_H = (5,3,1,.1,.01,.001) #Source Host BW Stats
        self.L_HH = (1,.1,.01) #Source Host BW Stats
        self.L_HpHp = (1,.1,.01) #Source Host BW Stats

        #Records: for memory tracking and cleanup
        self.Rec_Hosts = dict() #IPs
        self.Rec_MAC_Host = dict() #Source MAC-IP combos
        self.Rec_Sessions = dict() #HpHp sessions

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*3

    def checkLimits(self,srcIP,dstIP,srcMAC,srcProtocol, dstProtocol):
        # Maintain Limit on Hosts
        s = self.Rec_Hosts.get(srcIP)
        if s is None:
            if len(self.Rec_Hosts) <= self.HostLimit:
                self.Rec_Hosts[srcIP] = True
            else:
                raise LookupError(
                    'Adding Source Host:\n' + srcIP + '\nwould exceed maxHostLimit. Observation Rejected.')
        d = self.Rec_Hosts.get(dstIP)
        if d is None:
            if len(self.Rec_Hosts) <= self.HostLimit:
                self.Rec_Hosts[dstIP] = True
            else:
                raise LookupError(
                    'Adding Destination Host:\n' + dstIP + '\nwould exceed maxHostLimit. Observation Rejected.')
        mi = self.Rec_MAC_Host.get(srcMAC+srcIP)
        if mi is None:
            if len(self.Rec_MAC_Host) <= self.MAC_HostLimit:
                self.Rec_MAC_Host[srcMAC+srcIP] = True
            else:
                raise LookupError(
                    'Adding Source MAC-IP:\n' + srcMAC+ ' ' +srcIP + '\nwould exceed MAC-HostLimit. Observation Rejected.')
        hphp = self.Rec_Sessions.get(str(srcIP) + ""+str(srcProtocol)+"" + str(dstIP) +""+ str(dstProtocol))
        if hphp is None:
            if len(self.Rec_Sessions) <= self.SessionLimit -2:
                self.Rec_Sessions[str(srcIP) + str(srcProtocol) + str(dstIP) + str(dstProtocol)] = True
                self.Rec_Sessions[str(dstIP) + str(dstProtocol) + str(srcIP) + str(srcProtocol)] = True
            else:
                raise LookupError(
                    'Adding Session:\n' + srcIP +':'+ srcProtocol +'->'+ dstIP +':'+ dstProtocol + '\nwould exceed Session Limit. Observation Rejected.')

    def purgeOldRecords(self,curTime):
        #if len(self.Rec_Sessions)+len(self.Rec_MAC_Host)+len(self.Rec_Hosts) > (self.MAC_HostLimit+self.SessionLimit)*0.85:
        dump = sorted(self.HT.HT.items(),key=lambda tup: tup[1][0].getMaxW(curTime))
        for entry in dump:
            W = entry[1][0].getMaxW(curTime)
            if W <= 0.000001:
                key = entry[0]
                del entry[1][0]
                if key in self.Rec_Sessions:
                    del self.Rec_Sessions[key]
                elif key in self.Rec_MAC_Host:
                    del self.Rec_MAC_Host[key]
                elif key in self.Rec_Hosts:
                    del self.Rec_Hosts[key]
                del self.HT.HT[key]
            elif W > 0.001:
                break
            #else:
            #    break #we found an entry with a larger weight than 0.001 (the rest will be larger)


    def updateGetStats(self, srcMAC, srcIP, srcProtocol, dstIP, dstProtocol, datagramSize, timestamp):
        self.checkLimits(srcIP,dstIP,srcMAC,srcProtocol, dstProtocol) #will throw LookupError if a limit has been reached

        # General Bandwidth
        BWstat = self.HT.updateGet_1D('GeneralBW',datagramSize,timestamp,self.L_BW)

        # Directional Bandwidth
        src_subnet = str.split(srcIP,'.')
        dst_subnet = str.split(dstIP,'.')
        if (len(src_subnet) == 4 and len(dst_subnet) == 4): #both IPv4
            BWstat_dir = self.HT.updateGet_2D("subn_"+src_subnet[2],"subn_"+dst_subnet[2],datagramSize,timestamp,self.L_BW_dir)
        else:
            BWstat_dir = self.HT.updateGet_2D("subn_"+src_subnet[0],"subn_"+dst_subnet[0],datagramSize,timestamp,self.L_BW_dir)

        # Host BW: Stats on the srcIP's general Sender Statistics
        Hstat = self.HT.updateGet_1D(srcIP, datagramSize, timestamp,self.L_H)

        #MAC.IP: Stats on src MAC-IP relationships
        MIstat = self.HT.updateGet_1D(srcMAC+srcIP,datagramSize,timestamp,self.L_MI)

        # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        HHstat = self.HT.updateGet_2D(srcIP, dstIP, datagramSize, timestamp, self.L_HH)

        # Host-Host Jitter:
        HHstat_jit = self.HT.updateGet_1D('jit'+srcIP+dstIP, [], timestamp, self.L_HH, True)

        # HostP-HostP BW: Stats on the dual traffic behavior between srcIP and dstIP individual sessions (if src/dstProtocol is a port number) or protcol traffic (if src/dstProtocol is L3 -e.g. 'ICMP')
        HpHpstat = self.HT.updateGet_2D(str(srcIP) + str(srcProtocol), str(dstIP) + str(dstProtocol), datagramSize, timestamp, self.L_HpHp)

        return np.concatenate((BWstat, BWstat_dir, MIstat, Hstat, HHstat, HHstat_jit, HpHpstat))  # concatenation of stats into one stat vector

    def getNetStatHeaders(self,isDirectional=False):
        BWstat_headers = ["BW_"+h for h in self.HT.getHeaders_1D(self.L_BW)]
        BWdirstat_headers = ["BW_dir_"+h for h in self.HT.getHeaders_1D(self.L_BW_dir)]+["BW_dir10","BW_dir11","BW_dir12","BW_dir13","BW_dir14","BW_dir15","BW_dir16","BW_dir17","BW_dir18","BW_dir19","BW_dir20","BW_dir21"]
        MIstat_headers = ["MI_dir_"+h for h in self.HT.getHeaders_1D(self.L_BW_dir)][:3]
        Hstat_headers = ["H_"+h for h in self.HT.getHeaders_1D(self.L_H)]
        HHstat_headers = ["HH_"+h for h in self.HT.getHeaders_2D(self.L_HH)]
        HHjitstat_headers = ["HH_jit_"+h for h in self.HT.getHeaders_1D(self.L_jit)]
        HpHpstat_headers = ["HpHp_"+h for h in self.HT.getHeaders_2D(self.L_HpHp)]
        return BWstat_headers + BWdirstat_headers + MIstat_headers + Hstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers