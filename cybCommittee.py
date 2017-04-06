import json
import Experts as exp
import numpy as np

class subCommittee:
    #threshold is the number of std (above which is an anom)
    def __init__(self,name="unkown_subComittee",Auth=1,JSONstring=None):
        self.name = name
        self.Exp_authorities = []
        self.authority = Auth
        self.Experts = []

        if JSONstring is not None:
            self.loadFromJSON(JSONstring)

    def toJSON(self):
        subcom_json = {}
        exp_jsons = []
        for exp in self.Experts:
            exp_jsons.append(exp.toJSON())
        subcom_json['Name'] = self.name
        subcom_json['Experts'] = exp_jsons
        subcom_json['Authority'] = self.authority
        subcom_json['ExpAuthorities'] = self.Exp_authorities
        return json.dumps(subcom_json)

    def loadFromJSON(self,JSONstring):
        Js = json.loads(JSONstring)
        self.authority = Js['Authority']
        self.Exp_authorities = Js['ExpAuthorities']
        self.name = Js['Name']
        expList = Js['Experts']
        for expJ in expList:
            expert = exp.expertFromJSON(expJ)
            self.addExpert(expert)

    def size(self):
        return len(self.Experts)

    def name(self):
        return self.name

    def getNames(self):
        names = []
        for exp in self.Experts:
            names.append(exp.name)
        return names

    def addExpert(self,expert):
        self.Experts.append(expert)
        self.Exp_authorities.append(expert.authority)

    def getExpAuths(self):
        return self.Exp_authorities

    def getAuth(self):
        return self.Exp_authorities

    #given the index to an expert, the description of that expert is returned
    def descExpert(self,i):
        if i > len(self.Experts)-1:
            return None
        else:
            return self.Experts[i].description()

    #x_ref is a mutable object [] to the numrical vector 'x'
    def getVotes(self,x_ref):
        votes = np.zeros(len(self.Experts))
        for i in range(0,len(self.Experts)):
            votes[i] = self.Experts[i].vote(x_ref)
        return votes

    #x_ref is a mutable object [] to the numrical vector 'x'
    def train(self,x_ref):
        for exp in range(0,len(self.Experts)):
            self.Experts[exp].train(x_ref)

class Committee:
    #numericIndxs: the indexes to the numeric features in each in each instance
    #threshold: the number of std above the average score which is considered an anomaly
    #defaultCommittee: if True, then a CyberExpert Comm is made, else it is expected that he committees of experts will be added manually.
    def __init__(self,defaultCommittee=True,AEcorruption=0.0,learningRate=0.1,JSONstring=None):
        self.subComittees = []
        #Should we load from a JSON string?
        if JSONstring is not None:
            Js = json.loads(JSONstring)
            subComList = Js['subcomittees']
            for subcomJ in subComList:
                subCom = subCommittee(JSONstring=subcomJ)
                self.addSubComittee(subCom)
        # Should we build the default Cyber Committee?
        elif defaultCommittee:
            self.buildDefaultComittee(AEcorruption,learningRate)

    def size(self):
        return len(self.subComittees)

    def toJSON(self):
        com_model = {}
        subcom_jsons = []
        for subcom in self.subComittees:
            subcom_jsons.append(subcom.toJSON())
        com_model['subcomittees'] = subcom_jsons
        return json.dumps(com_model)

    def addSubComittee(self,subCom):
        self.subComittees.append(subCom)

    def getNames(self):
        names = []
        for sc in self.subComittees:
            names.append(sc.name)
        return names

    def getAllNames(self):
        names = []
        for sc in self.subComittees:
            N = [sc.name + '>' + n for n in sc.getNames()]
            names = names + N
        return names

    #x_ref is a pointer to the instance []
    def execute(self,x_ref):
        sc_votes = [] #list of variable size vectors: committee votes scores
        concensus = np.zeros(len(self.subComittees))
        gen_concensus = 0
        decision = 0 #enum: 0-do nothing, 1-alert only, 2-drop and alert

        #tally the votes: bottom up
        for sc in range(0,len(self.subComittees)):
            exp_votes = self.subComittees[sc].getVotes(x_ref)
            sc_votes.append(exp_votes)
            concensus[sc] = np.linalg.norm(sc_votes[sc],ord=len(sc_votes[sc])) #np.linalg.norm is the euclidean magnitude -by order
        gen_concensus = np.linalg.norm(concensus,ord=len(concensus))
        #Execute Order
        if gen_concensus >= 1:
            for sc in range(0,len(self.subComittees)):
                if concensus[sc] >= 1:
                    for exp in range(0,self.subComittees[sc].size()):
                        auths = self.subComittees[sc].getExpAuths()
                        if sc_votes[sc][exp] >= 1:
                            if auths[exp] == 2:
                                decision = 2
                                #Append to Log ()
                            elif auths[exp] == 1:
                                decision = 1
                                #Append to Log ()
        return decision, gen_concensus

    #for debug
    def execute_full(self,x_ref):
        sc_votes = [] #list of variable size vectors: committee votes scores
        concensus = np.zeros(len(self.subComittees))
        gen_concensus = 0
        decision = 0 #enum: 0-do nothing, 1-alert only, 2-drop and alert
        allScores = np.array([])

        #tally the votes: bottom up
        for sc in range(0,len(self.subComittees)):
            exp_votes = self.subComittees[sc].getVotes(x_ref)
            sc_votes.append(exp_votes)
            allScores = np.concatenate((allScores,exp_votes))
            #concensus[sc] = np.linalg.norm(sc_votes[sc]) #np.linalg.norm is the euclidean magnitude -by order
        #gen_concensus = np.linalg.norm(concensus)
        return allScores


    #x_ref is a pointer to the instance []
    def train(self,x_ref):
        # make instance
        for sc in range(0,len(self.subComittees)):
            self.subComittees[sc].train(x_ref)

    #this function first gets a decision on the instance, then if safe, trains from it
    def execute_train(self,x_ref):
        response = self.execute(x_ref)
        if response[0] == 0: #if the instance is benign
            self.train(x_ref)
        return response

    def buildDefaultComittee(self, AEcorruption=0.0, learningRate = None):
        if learningRate is None:
            learningRate = 0.1

        #### BW subComittee ####
        BW_subCommittee = subCommittee(name="Bandwidth")

        # Total Traffic (directional)
        TT_dir = exp.Expert_dirBW(corruption=AEcorruption, learningRate=learningRate)
        BW_subCommittee.addExpert(TT_dir)

        # Total Traffic (short-term)
        TT_st = exp.Expert_BW(corruption=AEcorruption,learningRate=learningRate)
        BW_subCommittee.addExpert(TT_st)

        # Host Source Traffic (short-term)
        HT_st = exp.Expert_hostBW(corruption=AEcorruption,learningRate=learningRate)
        BW_subCommittee.addExpert(HT_st)

        self.addSubComittee(BW_subCommittee)

        #### Channel subComittee ####
        Channel_subCommittee = subCommittee(name="Channel")

        # IPv4
        CH_ipv4_1 = exp.Expert_IP_Ch(term=0,IPtype=0,corruption=AEcorruption,learningRate=learningRate)
        CH_ipv4_2 = exp.Expert_IP_Ch(term=1,IPtype=0,corruption=AEcorruption,learningRate=learningRate)
        CH_ipv4_3 = exp.Expert_IP_Ch(term=2,IPtype=0,corruption=AEcorruption,learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_ipv4_1)
        Channel_subCommittee.addExpert(CH_ipv4_2)
        Channel_subCommittee.addExpert(CH_ipv4_3)

        # IPv6
        CH_ipv6_1 = exp.Expert_IP_Ch(term=0,IPtype=1,corruption=AEcorruption,learningRate=learningRate)
        CH_ipv6_2 = exp.Expert_IP_Ch(term=1,IPtype=1,corruption=AEcorruption,learningRate=learningRate)
        CH_ipv6_3 = exp.Expert_IP_Ch(term=2,IPtype=1,corruption=AEcorruption,learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_ipv6_1)
        Channel_subCommittee.addExpert(CH_ipv6_2)
        Channel_subCommittee.addExpert(CH_ipv6_3)

        # ICMP
        CH_icmp_1 = exp.Expert_ICMP_Ch(term=0,corruption=AEcorruption,learningRate=learningRate)
        CH_icmp_2 = exp.Expert_ICMP_Ch(term=1,corruption=AEcorruption,learningRate=learningRate)
        CH_icmp_3 = exp.Expert_ICMP_Ch(term=2,corruption=AEcorruption,learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_icmp_1)
        Channel_subCommittee.addExpert(CH_icmp_2)
        Channel_subCommittee.addExpert(CH_icmp_3)

        # TCP - Well-known, Registerd and Dynamic ports
        CH_tcp_wkn_1 = exp.Expert_TCPUDP_Ch(term=0,TCPUDPtype=0,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_wkn_2 = exp.Expert_TCPUDP_Ch(term=1,TCPUDPtype=0,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_wkn_3 = exp.Expert_TCPUDP_Ch(term=2,TCPUDPtype=0,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_reg_1 = exp.Expert_TCPUDP_Ch(term=0,TCPUDPtype=1,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_reg_2 = exp.Expert_TCPUDP_Ch(term=1,TCPUDPtype=1,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_reg_3 = exp.Expert_TCPUDP_Ch(term=2,TCPUDPtype=1,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_dyn_1 = exp.Expert_TCPUDP_Ch(term=0,TCPUDPtype=2,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_dyn_2 = exp.Expert_TCPUDP_Ch(term=1,TCPUDPtype=2,corruption=AEcorruption,learningRate=learningRate)
        CH_tcp_dyn_3 = exp.Expert_TCPUDP_Ch(term=2,TCPUDPtype=2,corruption=AEcorruption,learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_tcp_wkn_1)
        Channel_subCommittee.addExpert(CH_tcp_wkn_2)
        Channel_subCommittee.addExpert(CH_tcp_wkn_3)
        Channel_subCommittee.addExpert(CH_tcp_reg_1)
        Channel_subCommittee.addExpert(CH_tcp_reg_2)
        Channel_subCommittee.addExpert(CH_tcp_reg_3)
        Channel_subCommittee.addExpert(CH_tcp_dyn_1)
        Channel_subCommittee.addExpert(CH_tcp_dyn_2)
        Channel_subCommittee.addExpert(CH_tcp_dyn_3)

        # UDP - Well-known, Registerd and Dynamic ports
        CH_udp_wkn_1 = exp.Expert_TCPUDP_Ch(term=0, TCPUDPtype=3, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_wkn_2 = exp.Expert_TCPUDP_Ch(term=1, TCPUDPtype=3, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_wkn_3 = exp.Expert_TCPUDP_Ch(term=2, TCPUDPtype=3, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_reg_1 = exp.Expert_TCPUDP_Ch(term=0, TCPUDPtype=4, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_reg_2 = exp.Expert_TCPUDP_Ch(term=1, TCPUDPtype=4, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_reg_3 = exp.Expert_TCPUDP_Ch(term=2, TCPUDPtype=4, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_dyn_1 = exp.Expert_TCPUDP_Ch(term=0, TCPUDPtype=5, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_dyn_2 = exp.Expert_TCPUDP_Ch(term=1, TCPUDPtype=5, corruption=AEcorruption, learningRate=learningRate)
        CH_udp_dyn_3 = exp.Expert_TCPUDP_Ch(term=2, TCPUDPtype=5, corruption=AEcorruption, learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_udp_wkn_1)
        Channel_subCommittee.addExpert(CH_udp_wkn_2)
        Channel_subCommittee.addExpert(CH_udp_wkn_3)
        Channel_subCommittee.addExpert(CH_udp_reg_1)
        Channel_subCommittee.addExpert(CH_udp_reg_2)
        Channel_subCommittee.addExpert(CH_udp_reg_3)
        Channel_subCommittee.addExpert(CH_udp_dyn_1)
        Channel_subCommittee.addExpert(CH_udp_dyn_2)
        Channel_subCommittee.addExpert(CH_udp_dyn_3)

        # Host-Host Bandwidth
        CH_HH_1 = exp.Expert_hosthostBW(term=0, corruption=AEcorruption, learningRate=learningRate)
        CH_HH_2 = exp.Expert_hosthostBW(term=1, corruption=AEcorruption, learningRate=learningRate)
        CH_HH_3 = exp.Expert_hosthostBW(term=2, corruption=AEcorruption, learningRate=learningRate)
        Channel_subCommittee.addExpert(CH_HH_1)
        Channel_subCommittee.addExpert(CH_HH_2)
        Channel_subCommittee.addExpert(CH_HH_3)

        self.addSubComittee(Channel_subCommittee)

        #### Protocol subComittee ####
        Protocol_subCommittee = subCommittee(name="Protocol")

        # IPv4
        Pro_ipv4_1 = exp.Expert_IP_Pr(IPtype=0,version=0, corruption=AEcorruption, learningRate=learningRate)
        Pro_ipv4_2 = exp.Expert_IP_Pr(IPtype=0,version=1, corruption=AEcorruption, learningRate=learningRate)
        Pro_ipv4_3 = exp.Expert_IP_Pr(IPtype=0,version=2, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_ipv4_1)
        Protocol_subCommittee.addExpert(Pro_ipv4_2)
        Protocol_subCommittee.addExpert(Pro_ipv4_3)

        # IPv6
        Pro_ipv6_1 = exp.Expert_IP_Pr(IPtype=1,version=0, corruption=AEcorruption, learningRate=learningRate)
        Pro_ipv6_2 = exp.Expert_IP_Pr(IPtype=1,version=1, corruption=AEcorruption, learningRate=learningRate)
        Pro_ipv6_3 = exp.Expert_IP_Pr(IPtype=1,version=2, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_ipv6_1)
        Protocol_subCommittee.addExpert(Pro_ipv6_2)
        Protocol_subCommittee.addExpert(Pro_ipv6_3)

        # ICMP
        Pro_icmp_1 = exp.Expert_ICMP_Pr(version=0, corruption=AEcorruption, learningRate=learningRate)
        Pro_icmp_2 = exp.Expert_ICMP_Pr(version=1, corruption=AEcorruption, learningRate=learningRate)
        Pro_icmp_3 = exp.Expert_ICMP_Pr(version=2, corruption=AEcorruption, learningRate=learningRate)
        Pro_icmp_4 = exp.Expert_ICMP_Pr(version=3, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_icmp_1)
        Protocol_subCommittee.addExpert(Pro_icmp_2)
        Protocol_subCommittee.addExpert(Pro_icmp_3)
        Protocol_subCommittee.addExpert(Pro_icmp_4)

        # TCP
        # note, the enum for TCPUDP type is set to '0' bacuse we just want to signify that its TCP
        Pro_tcp_1 = exp.Expert_TCPUDP_Pr(TCPUDPtype=0, version=0, corruption=AEcorruption, learningRate=learningRate)
        Pro_tcp_2 = exp.Expert_TCPUDP_Pr(TCPUDPtype=0, version=1, corruption=AEcorruption, learningRate=learningRate)
        Pro_tcp_3 = exp.Expert_TCPUDP_Pr(TCPUDPtype=0, version=2, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_tcp_1)
        Protocol_subCommittee.addExpert(Pro_tcp_2)
        Protocol_subCommittee.addExpert(Pro_tcp_3)

        # UDP
        # note, the enum for TCPUDP type is set to '3' bacuse we just want to signify that its TCP
        Pro_tcp_1 = exp.Expert_TCPUDP_Pr(TCPUDPtype=3, version=0, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_tcp_1)

        # ARP
        Pro_arp_1 = exp.Expert_ARP_Pr(term=0, corruption=AEcorruption, learningRate=learningRate)
        Pro_arp_2 = exp.Expert_ARP_Pr(term=1, corruption=AEcorruption, learningRate=learningRate)
        Protocol_subCommittee.addExpert(Pro_arp_1)
        Protocol_subCommittee.addExpert(Pro_arp_2)

        self.addSubComittee(Protocol_subCommittee)

        #### Payload subComittee ####
        Payload_subCommittee = subCommittee(name="Payload")

        # RTSP/RTP
        serverPorts = 80, 554, 5000, 51000
        CH_rts_1 = exp.Expert_RTS(version=0, serverPorts=serverPorts, corruption=AEcorruption, learningRate=learningRate)
        CH_rts_2 = exp.Expert_RTS(version=1, serverPorts=serverPorts, corruption=AEcorruption, learningRate=learningRate)
        Payload_subCommittee.addExpert(CH_rts_1)
        Payload_subCommittee.addExpert(CH_rts_2)

        self.addSubComittee(Payload_subCommittee)

        return

    # def buildDefaultComittee1(self,withWeightFeature=True,AEcorruption=0.0):
    #     ### Coded Indexes (for the given dataset):
    #     frame_time_epoch = 0
    #     frame_len = 1
    #     eth_src = 2
    #     eth_dst = 3
    #     ip_src = 4
    #     ip_dst = 5
    #     ip_hdr_len = 6
    #     ip_len = 7
    #     ip_flags_rb = 8
    #     ip_flags_df = 9
    #     ip_flags_mf = 10
    #     ip_ttl = 11
    #     tcp_srcport = 12
    #     tcp_dstport = 13
    #     tcp_seq = 14
    #     tcp_ack = 15
    #     tcp_flags_res = 16
    #     tcp_flags_ack = 17
    #     tcp_flags_cwr = 18
    #     tcp_flags_ecn = 19
    #     tcp_flags_fin = 20
    #     tcp_flags_ns = 21
    #     tcp_flags_push = 22
    #     tcp_flags_reset = 23
    #     tcp_flags_syn = 24
    #     tcp_flags_urg = 25
    #     tcp_window_size_value = 26
    #     tcp_urgent_pointer = 27
    #     udp_length = 28
    #     udp_srcport = 29
    #     udp_dstport = 30
    #     icmp_type = 31
    #     icmp_code = 32
    #     arp_opcode = 33
    #     arp_src_hw_mac = 34
    #     arp_src_proto_ipv4 = 35
    #     arp_dst_hw_mac = 36
    #     arp_dst_proto_ipv4 = 37
    #     http_request_method = 38
    #     http_request_uri = 39
    #     http_request_version = 40
    #     http_response_code = 41
    #     http_host = 42
    #     http_connection = 43
    #     ipv6_src = 44
    #     ipv6_dst = 45
    #     ipv6_dstopts_nxt = 46
    #     ipv6_flow = 47
    #     ipv6_tclass = 48
    #     ipv6_tclass_dscp = 49
    #     ipv6_tclass_ecn = 50
    #     ipv6_hlim = 51
    #     ipv6_version = 52
    #     ipv6_plen = 53
    #     BW_L5_weight = 54
    #     BW_L5_mean = 55
    #     BW_L5_variance = 56
    #     BW_L3_weight = 57
    #     BW_L3_mean = 58
    #     BW_L3_variance = 59
    #     BW_L1_weight = 60
    #     BW_L1_mean = 61
    #     BW_L1_variance = 62
    #     BW_L0_1_weight = 63
    #     BW_L0_1_mean = 64
    #     BW_L0_1_variance = 65
    #     BW_L0_01_weight = 66
    #     BW_L0_01_mean = 67
    #     BW_L0_01_variance = 68
    #     BW_L0_001_weight = 69
    #     BW_L0_001_mean = 70
    #     BW_L0_001_variance = 71
    #     BW_dir_L1_weight = 72
    #     BW_dir_L1_mean = 73
    #     BW_dir_L1_std = 74
    #     BW_dir_L1_magnitude = 75
    #     BW_dir_L1_radius = 76
    #     BW_dir_L1_covariance = 77
    #     BW_dir_L1_pcc = 78
    #     BW_dir_L0_3_weight = 79
    #     BW_dir_L0_3_mean = 80
    #     BW_dir_L0_3_std = 81
    #     BW_dir_L0_3_magnitude = 82
    #     BW_dir_L0_3_radius = 83
    #     BW_dir_L0_3_covariance = 84
    #     BW_dir_L0_3_pcc = 85
    #     BW_dir_L0_1_weight = 86
    #     BW_dir_L0_1_mean = 87
    #     BW_dir_L0_1_std = 88
    #     BW_dir_L0_1_magnitude = 89
    #     BW_dir_L0_1_radius = 90
    #     BW_dir_L0_1_covariance = 91
    #     BW_dir_L0_1_pcc = 92
    #     MI_dir_L0_01_weight = 93
    #     MI_dir_L0_01_mean = 94
    #     MI_dir_L0_01_variance = 95
    #     H_L5_weight = 96
    #     H_L5_mean = 97
    #     H_L5_variance = 98
    #     H_L3_weight = 99
    #     H_L3_mean = 100
    #     H_L3_variance = 101
    #     H_L1_weight = 102
    #     H_L1_mean = 103
    #     H_L1_variance = 104
    #     H_L0_1_weight = 105
    #     H_L0_1_mean = 106
    #     H_L0_1_variance = 107
    #     H_L0_01_weight = 108
    #     H_L0_01_mean = 109
    #     H_L0_01_variance = 110
    #     H_L0_001_weight = 111
    #     H_L0_001_mean = 112
    #     H_L0_001_variance = 113
    #     HH_L1_weight = 114
    #     HH_L1_mean = 115
    #     HH_L1_std = 116
    #     HH_L1_magnitude = 117
    #     HH_L1_radius = 118
    #     HH_L1_covariance = 119
    #     HH_L1_pcc = 120
    #     HH_L0_1_weight = 121
    #     HH_L0_1_mean = 122
    #     HH_L0_1_std = 123
    #     HH_L0_1_magnitude = 124
    #     HH_L0_1_radius = 125
    #     HH_L0_1_covariance = 126
    #     HH_L0_1_pcc = 127
    #     HH_L0_01_weight = 128
    #     HH_L0_01_mean = 129
    #     HH_L0_01_std = 130
    #     HH_L0_01_magnitude = 131
    #     HH_L0_01_radius = 132
    #     HH_L0_01_covariance = 133
    #     HH_L0_01_pcc = 134
    #     HH_jit_L1_weight = 135
    #     HH_jit_L1_mean = 136
    #     HH_jit_L1_variance = 137
    #     HH_jit_L0_3_weight = 138
    #     HH_jit_L0_3_mean = 139
    #     HH_jit_L0_3_variance = 140
    #     HH_jit_L0_1_weight = 141
    #     HH_jit_L0_1_mean = 142
    #     HH_jit_L0_1_variance = 143
    #     HpHp_L1_weight = 144
    #     HpHp_L1_mean = 145
    #     HpHp_L1_std = 146
    #     HpHp_L1_magnitude = 147
    #     HpHp_L1_radius = 148
    #     HpHp_L1_covariance = 149
    #     HpHp_L1_pcc = 150
    #     HpHp_L0_1_weight = 151
    #     HpHp_L0_1_mean = 152
    #     HpHp_L0_1_std = 153
    #     HpHp_L0_1_magnitude = 154
    #     HpHp_L0_1_radius = 155
    #     HpHp_L0_1_covariance = 156
    #     HpHp_L0_1_pcc = 157
    #     HpHp_L0_01_weight = 158
    #     HpHp_L0_01_mean = 159
    #     HpHp_L0_01_std = 160
    #     HpHp_L0_01_magnitude = 161
    #     HpHp_L0_01_radius = 162
    #     HpHp_L0_01_covariance = 163
    #     HpHp_L0_01_pcc = 164
    #
    #     #### BW subComittee ####
    #     BW_subCommittee = subCommittee(name="Bandwidth")
    #
    #     # Total Traffic (directional)
    #     if withWeightFeature:
    #         fv = BW_dir_L0_3_weight, BW_dir_L0_3_mean, BW_dir_L0_3_std, BW_dir_L0_3_magnitude, BW_dir_L0_3_radius, BW_dir_L0_3_covariance, BW_dir_L0_3_pcc
    #     else:
    #         fv = BW_dir_L0_3_mean, BW_dir_L0_3_std, BW_dir_L0_3_magnitude, BW_dir_L0_3_radius, BW_dir_L0_3_covariance, BW_dir_L0_3_pcc
    #     TT_dir = exp.Expert_dirBW(fv, ip_src, ip_dst, ipv6_src, ipv6_dst, eth_src, eth_dst, maxDirections=25, name="Total Traffic (directional)", Auth=1, threshold=-15, corruption=AEcorruption)
    #     BW_subCommittee.addExpert(TT_dir)
    #
    #     # Total Traffic (short-term)
    #     if withWeightFeature:
    #         fv = BW_L5_mean, BW_L5_variance, BW_L3_mean, BW_L3_variance, BW_L1_mean, BW_L1_variance, BW_L1_weight
    #     else:
    #         fv = BW_L5_mean, BW_L5_variance, BW_L3_mean, BW_L3_variance, BW_L1_mean, BW_L1_variance
    #     TT_st = exp.Expert_Basic(fv, name="Total Traffic (short-term)", Auth=1, threshold=-15, corruption=AEcorruption, rm_winsize=20)
    #     BW_subCommittee.addExpert(TT_st)
    #
    #     #Removed: caused MSE whose distribution was very diffrent than the others (selective thresholds for each scenarios..)
    #     # Total Traffic (long-term)
    #     #fv = BW_L0_1_mean, BW_L0_1_variance, BW_L0_01_mean, BW_L0_01_variance, BW_L0_001_mean, BW_L0_001_variance
    #     #TT_lt = exp.Expert(fv,name="Total Traffic (long-term)", Auth=1,threshold=1e-19)
    #     #BW_subCommittee.addExpert(TT_lt)
    #
    #     # Host Source Traffic (short-term)
    #     if withWeightFeature:
    #         fv = H_L5_mean, H_L5_variance, H_L3_mean, H_L3_variance, H_L1_mean, H_L1_variance, H_L1_weight
    #     else:
    #         fv = H_L5_mean, H_L5_variance, H_L3_mean, H_L3_variance, H_L1_mean, H_L1_variance
    #     HT_st = exp.Expert_hostBW(fv, ip_src, maxHosts=50, name="Host Source Traffic (short-term)", Auth=1, corruption=AEcorruption)
    #     BW_subCommittee.addExpert(HT_st)
    #
    #     #REMOVED: the long-term prediction whas too difficult, raised the error very high
    #     # Host Source Traffic (long-term)
    #     #fv = H_L0_1_mean, H_L0_1_variance, H_L0_01_mean, H_L0_01_variance, H_L0_001_mean, H_L0_001_variance
    #     #HT_st = exp.Expert_hostBW(fv, ip_src, maxHosts=50, name="Host Source Traffic (long-term)", Auth=1)
    #     #BW_subCommittee.addExpert(HT_st)
    #
    #     self.addSubComittee(BW_subCommittee)
    #
    #     #### Channel subComittee ####
    #     Channel_subCommittee = subCommittee(name="Channel")
    #
    #     # IPv4
    #     if withWeightFeature:
    #         fv1 = HH_L1_weight, HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_weight, HH_L1_mean, HH_L1_pcc, HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     else:
    #         fv1 = HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_mean, HH_L1_pcc, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     CH_ipv4_1 = exp.Expert_cond(fv1, conditional_Indx=ip_src, name="IPv4 Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_ipv4_2 = exp.Expert_cond(fv2, conditional_Indx=ip_src, name="IPv4 Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_ipv4_3 = exp.Expert_cond(fv3, conditional_Indx=ip_src, name="IPv4 Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_ipv4_1)
    #     Channel_subCommittee.addExpert(CH_ipv4_2)
    #     Channel_subCommittee.addExpert(CH_ipv4_3)
    #
    #     # IPv6
    #     if withWeightFeature:
    #         fv1 = HH_L1_weight, HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_weight, HH_L1_mean, HH_L1_pcc, HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     else:
    #         fv1 = HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_mean, HH_L1_pcc, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     CH_ipv6_1 = exp.Expert_cond(fv1, conditional_Indx=ipv6_src, name="IPv6 Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_ipv6_2 = exp.Expert_cond(fv2, conditional_Indx=ipv6_src, name="IPv6 Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_ipv6_3 = exp.Expert_cond(fv3, conditional_Indx=ipv6_src, name="IPv6 Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_ipv6_1)
    #     Channel_subCommittee.addExpert(CH_ipv6_2)
    #     Channel_subCommittee.addExpert(CH_ipv6_3)
    #
    #     # ICMP
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_icmp_1 = exp.Expert_cond(fv1, conditional_Indx=icmp_code, name="ICMP Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_icmp_2 = exp.Expert_cond(fv2, conditional_Indx=icmp_code, name="ICMP Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_icmp_3 = exp.Expert_cond(fv3, conditional_Indx=icmp_code, name="ICMP Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_icmp_1)
    #     Channel_subCommittee.addExpert(CH_icmp_2)
    #     Channel_subCommittee.addExpert(CH_icmp_3)
    #
    #     # TCP - Well-known Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_tcp_wkn_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=tcp_srcport, portRange=0,
    #                                          name="TCP Well-known Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_wkn_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=tcp_srcport, portRange=0,
    #                                          name="TCP Well-known Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_wkn_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=tcp_srcport, portRange=0,
    #                                          name="TCP Well-known Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_tcp_wkn_1)
    #     Channel_subCommittee.addExpert(CH_tcp_wkn_2)
    #     Channel_subCommittee.addExpert(CH_tcp_wkn_3)
    #
    #     # TCP - Registered Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_tcp_reg_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=tcp_srcport, portRange=1,
    #                                          name="TCP Registered Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_reg_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=tcp_srcport, portRange=1,
    #                                          name="TCP Registered Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_reg_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=tcp_srcport, portRange=1,
    #                                          name="TCP Registered Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_tcp_reg_1)
    #     Channel_subCommittee.addExpert(CH_tcp_reg_2)
    #     Channel_subCommittee.addExpert(CH_tcp_reg_3)
    #
    #     # TCP - Dynamic Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_tcp_dyn_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=tcp_srcport, portRange=2,
    #                                          name="TCP Dynamic Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_dyn_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=tcp_srcport, portRange=2,
    #                                          name="TCP Dynamic Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_tcp_dyn_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=tcp_srcport, portRange=2,
    #                                          name="TCP Dynamic Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_tcp_dyn_1)
    #     Channel_subCommittee.addExpert(CH_tcp_dyn_2)
    #     Channel_subCommittee.addExpert(CH_tcp_dyn_3)
    #
    #     # UDP - Well-known Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_udp_wkn_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=udp_srcport, portRange=0,
    #                                          name="UDP Well-known Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_wkn_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=udp_srcport, portRange=0,
    #                                          name="UDP Well-known Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_wkn_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=udp_srcport, portRange=0,
    #                                          name="UDP Well-known Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_udp_wkn_1)
    #     Channel_subCommittee.addExpert(CH_udp_wkn_2)
    #     Channel_subCommittee.addExpert(CH_udp_wkn_3)
    #
    #     # UDP - Registered Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #
    #     CH_udp_reg_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=udp_srcport, portRange=1,
    #                                          name="UDP Registered Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_reg_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=udp_srcport, portRange=1,
    #                                          name="UDP Registered Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_reg_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=udp_srcport, portRange=1,
    #                                          name="UDP Registered Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_udp_reg_1)
    #     Channel_subCommittee.addExpert(CH_udp_reg_2)
    #     Channel_subCommittee.addExpert(CH_udp_reg_3)
    #
    #     # UDP - Dynamic Ports
    #     if withWeightFeature:
    #         fv1 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_weight, HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_weight, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     else:
    #         fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_magnitude, HpHp_L1_radius, HpHp_L1_covariance, HpHp_L1_pcc
    #         fv2 = HpHp_L0_1_mean, HpHp_L0_1_std, HpHp_L0_1_magnitude, HpHp_L0_1_radius, HpHp_L0_1_covariance, HpHp_L0_1_pcc
    #         fv3 = HpHp_L1_mean, HpHp_L1_pcc, HpHp_L0_1_mean, HpHp_L0_1_pcc, HpHp_L0_1_pcc
    #     CH_udp_dyn_1 = exp.Expert_TCP_UDP_CH(fv1, srcPort_Indx=udp_srcport, portRange=2,
    #                                          name="UDP Dynamic Port Channel (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_dyn_2 = exp.Expert_TCP_UDP_CH(fv2, srcPort_Indx=udp_srcport, portRange=2,
    #                                          name="UDP Dynamic Port Channel (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_udp_dyn_3 = exp.Expert_TCP_UDP_CH(fv3, srcPort_Indx=udp_srcport, portRange=2,
    #                                          name="UDP Dynamic Port Channel (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_udp_dyn_1)
    #     Channel_subCommittee.addExpert(CH_udp_dyn_2)
    #     Channel_subCommittee.addExpert(CH_udp_dyn_3)
    #
    #     # Host-Host Bandwidth
    #     if withWeightFeature:
    #         fv1 = HH_L1_weight, HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_weight, HH_L1_mean, HH_L1_pcc, HH_L0_1_weight, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     else:
    #         fv1 = HH_L1_mean, HH_L1_std, HH_L1_magnitude, HH_L1_radius, HH_L1_covariance, HH_L1_pcc
    #         fv2 = HH_L0_1_mean, HH_L0_1_std, HH_L0_1_magnitude, HH_L0_1_radius, HH_L0_1_covariance, HH_L0_1_pcc
    #         fv3 = HH_L1_mean, HH_L1_pcc, HH_L0_1_mean, HH_L0_1_pcc, HH_L0_1_pcc
    #     CH_HH_1 = exp.Expert_Basic(fv1, name="General Host-Host Traffic (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_HH_2 = exp.Expert_Basic(fv2, name="General Host-Host Traffic (long-term)", Auth=2, corruption=AEcorruption)
    #     CH_HH_3 = exp.Expert_Basic(fv3, name="General Host-Host Traffic (multi-term)", Auth=2, corruption=AEcorruption)
    #     Channel_subCommittee.addExpert(CH_HH_1)
    #     Channel_subCommittee.addExpert(CH_HH_2)
    #     Channel_subCommittee.addExpert(CH_HH_3)
    #
    #     #LEAVE OUT BECAUSE WAS TOO NOISEY (TOO MANY FPs)
    #     # Host-Host Jitter (MITM)
    #     #fv1 = HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_3_mean, HH_jit_L0_3_variance
    #     #fv2 = HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_3_mean, HH_jit_L0_3_variance, HH_jit_L0_1_mean, HH_jit_L0_1_variance
    #     #CH_HHjit_1 = exp.Expert(fv1, name="Host-Host Jitter 1", Auth=2)
    #     #CH_HHjit_2 = exp.Expert(fv2, name="Host-Host Jitter 2", Auth=2)
    #     #Channel_subCommittee.addExpert(CH_HHjit_1)
    #     #Channel_subCommittee.addExpert(CH_HHjit_2)
    #
    #     self.addSubComittee(Channel_subCommittee)
    #
    #     #### Protocol subComittee ####
    #     Protocol_subCommittee = subCommittee(name="Protocol")
    #
    #     # IPv4
    #     fv1 = ip_hdr_len, ip_len, frame_len, ip_ttl, HH_L0_1_mean
    #     fv2 = ip_flags_rb, ip_flags_df, ip_flags_mf, ip_ttl, ip_hdr_len
    #     fv3 = ip_hdr_len, ip_len, ip_flags_rb, ip_flags_df, ip_flags_mf, ip_ttl, HH_L0_1_mean
    #     Pro_ipv4_0 = exp.Expert_IPv4(0, fv1, ip_src, ip_dst, "IPv4 Protocol 1", Auth=1, corruption=AEcorruption)
    #     Pro_ipv4_1 = exp.Expert_IPv4(1, fv2, ip_src, ip_dst, "IPv4 Protocol 2", Auth=1, corruption=AEcorruption)
    #     Pro_ipv4_2 = exp.Expert_IPv4(2, fv3, ip_src, ip_dst, "IPv4 Protocol 3", Auth=1, corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_ipv4_0)
    #     Protocol_subCommittee.addExpert(Pro_ipv4_1)
    #     Protocol_subCommittee.addExpert(Pro_ipv4_2)
    #
    #     # IPv6
    #     fv1 = ipv6_plen, frame_len, ipv6_dstopts_nxt, ipv6_flow, HH_L0_1_mean
    #     fv2 = ipv6_tclass_dscp, ipv6_tclass_ecn, ipv6_hlim, ipv6_tclass, ipv6_plen
    #     fv3 = ipv6_flow, ipv6_dstopts_nxt, ipv6_hlim, ipv6_plen, frame_len, ipv6_version
    #     Pro_ipv6_0 = exp.Expert_IPv6(0, fv1, ipv6_src, ipv6_dst, "IPv6 Protocol 1", Auth=2,corruption=AEcorruption)
    #     Pro_ipv6_1 = exp.Expert_IPv6(1, fv2, ipv6_src, ipv6_dst, "IPv6 Protocol 2", Auth=2,corruption=AEcorruption)
    #     Pro_ipv6_2 = exp.Expert_IPv6(2, fv3, ipv6_src, ipv6_dst, "IPv6 Protocol 3", Auth=2,corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_ipv6_0)
    #     Protocol_subCommittee.addExpert(Pro_ipv6_1)
    #     Protocol_subCommittee.addExpert(Pro_ipv6_2)
    #
    #     # ICMP
    #     fv1 = icmp_type, icmp_code, HpHp_L0_1_mean
    #     fv2 = (HpHp_L0_1_mean,)
    #     fv3 = HpHp_L0_1_mean, H_L0_1_mean
    #     fv4 = HpHp_L0_1_mean, H_L0_1_mean
    #     Pro_icmp_0 = exp.Expert_ICMP(0, fv1, ip_src, ip_dst, icmp_type, icmp_code, "ICMP Genreal", Auth=2, corruption=AEcorruption)
    #     Pro_icmp_1 = exp.Expert_ICMP(1, fv2, ip_src, ip_dst, icmp_type, icmp_code, "ICMP Error Msg", Auth=2, corruption=AEcorruption)
    #     Pro_icmp_2 = exp.Expert_ICMP(2, fv3, ip_src, ip_dst, icmp_type, icmp_code, "ICMP Router Msg", Auth=2, corruption=AEcorruption)
    #     Pro_icmp_3 = exp.Expert_ICMP(3, fv4, ip_src, ip_dst, icmp_type, icmp_code, "ICMP Query", Auth=2, corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_icmp_0)
    #     Protocol_subCommittee.addExpert(Pro_icmp_1)
    #     Protocol_subCommittee.addExpert(Pro_icmp_2)
    #     Protocol_subCommittee.addExpert(Pro_icmp_3)
    #
    #     #TCP
    #     fv1 = tcp_ack, tcp_flags_ack, tcp_flags_syn, HH_L1_mean, HH_L1_radius, HH_L1_pcc
    #     fv2 = tcp_window_size_value, tcp_urgent_pointer, tcp_flags_ack, tcp_flags_cwr, tcp_flags_ecn, tcp_flags_syn, tcp_flags_urg
    #     fv3 = tcp_seq, tcp_ack, frame_len, tcp_window_size_value, HH_L0_1_mean, HpHp_L0_1_mean
    #     Pro_tcp_0 = exp.Expert_TCP(0, fv1, ip_src, tcp_srcport, "TCP Scanning", Auth=2, corruption=AEcorruption)
    #     Pro_tcp_1 = exp.Expert_TCP(1, fv2, ip_src, tcp_srcport, "TCP Fuzzing", Auth=2, corruption=AEcorruption)
    #     Pro_tcp_2 = exp.Expert_TCP(1, fv3, ip_src, tcp_srcport, "TCP Payload Inconsistency", Auth=1, corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_tcp_0)
    #     Protocol_subCommittee.addExpert(Pro_tcp_1)
    #     Protocol_subCommittee.addExpert(Pro_tcp_2)
    #
    #     #UDP
    #     fv1 = udp_length, frame_len, HH_L1_mean, HH_L1_std, HH_L1_radius, HH_L1_pcc
    #     Pro_udp = exp.Expert_TCP(1, fv1, ip_src, udp_srcport, "UDP Protocol", Auth=2, corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_udp)
    #
    #     #ARP
    #     if withWeightFeature:
    #         fv1 = MI_dir_L0_01_weight, MI_dir_L0_01_mean, MI_dir_L0_01_variance, HH_L1_mean, HH_L1_std, HH_L1_radius, HH_L1_pcc
    #         fv2 = MI_dir_L0_01_weight, MI_dir_L0_01_mean, MI_dir_L0_01_variance, HH_L0_01_mean, HH_L0_01_std, HH_L0_01_radius, HH_L0_01_pcc
    #     else:
    #         fv1 = MI_dir_L0_01_mean, MI_dir_L0_01_variance, HH_L1_mean, HH_L1_std, HH_L1_radius, HH_L1_pcc
    #         fv2 = MI_dir_L0_01_mean, MI_dir_L0_01_variance, HH_L0_01_mean, HH_L0_01_std, HH_L0_01_radius, HH_L0_01_pcc
    #     Pro_arp_0 = exp.Expert_cond(fv1, conditional_Indx=arp_opcode, name="ARP short-term", Auth=2, gracePeriod=200, corruption=AEcorruption)
    #     Pro_arp_1 = exp.Expert_cond(fv2, conditional_Indx=arp_opcode, name="ARP long-term", Auth=2, gracePeriod=200, corruption=AEcorruption)
    #     Protocol_subCommittee.addExpert(Pro_arp_0)
    #     Protocol_subCommittee.addExpert(Pro_arp_1)
    #
    #     self.addSubComittee(Protocol_subCommittee)
    #
    #     #### Payload subComittee ####
    #     Payload_subCommittee = subCommittee(name="Payload")
    #
    #     # RTSP/RTP
    #     fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_pcc, HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_1_mean, HH_jit_L0_1_variance
    #     fv2 = HpHp_L0_1_mean, HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_3_mean, HH_jit_L0_3_variance, HH_jit_L0_1_mean, HH_jit_L0_1_variance
    #     serverPorts = 80, 554, 5000
    #     CH_rts_1 = exp.Expert_RTstream(fv1, ip_src, tcp_srcport, udp_srcport, serverPorts, maxServers=30, name="RT Stream Flow 1 (short-term)", Auth=2, corruption=AEcorruption)
    #     CH_rts_2 = exp.Expert_RTstream(fv2, ip_src, tcp_srcport, udp_srcport, serverPorts, maxServers=30, name="RT Stream Flow 2 (short-term)", Auth=2, corruption=AEcorruption)
    #     Payload_subCommittee.addExpert(CH_rts_1)
    #     Payload_subCommittee.addExpert(CH_rts_2)
    #
    #     self.addSubComittee(Payload_subCommittee)
    #
    #     #### Identifier subComittee ####
    #     #Identifier_subCommittee = subCommittee(name="Identifier")
    #
    #     # RTSP/RTP
    #     #fv1 = HpHp_L1_mean, HpHp_L1_std, HpHp_L1_pcc, HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_1_mean, HH_jit_L0_1_variance
    #     #fv2 = HpHp_L0_1_mean, HH_jit_L1_mean, HH_jit_L1_variance, HH_jit_L0_3_mean, HH_jit_L0_3_variance, HH_jit_L0_1_mean, HH_jit_L0_1_variance
    #     #serverPorts = 80, 554, 5000
    #     #CH_rts_1 = exp.Expert_RTstream(fv1, ip_src, tcp_srcport, udp_srcport, serverPorts, maxServers=30,
    #     #                               name="RT Stream Flow 1 (short-term)", Auth=2)
    #     #CH_rts_2 = exp.Expert_RTstream(fv2, ip_src, tcp_srcport, udp_srcport, serverPorts, maxServers=30,
    #     #                               name="RTS Stream Flow 2 (short-term)", Auth=2)
    #     #Payload_subCommittee.addExpert(CH_rts_1)
    #     #Payload_subCommittee.addExpert(CH_rts_2)
    #     #self.addSubComittee(Identifier_subCommittee)
    #
    #     return


# NOTES TODO:
# Try without weights -time and accuracy
# Try Expert_TCP_UDP that consideres if EITHER srcport or dstport are the selected range (not just srcport)


