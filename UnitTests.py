import unittest
import netStat as ns
import AfterImage as af
import time
import random
import math
import numpy as np
import csv
import sys
import gc


class computeWinStats2D(unittest.TestCase):
    def setUp(self):
        self.data1 = [324,45,437,8,3,37,4,234,89,3]*1000
        self.data2 = [3324,45,47,8,300,37,-14,234,890,3]*1000
        #from "netStat test - no Decay.xls" : n, mean, std, mag, radius, cov, pcc
        self.expected1 = [10000, 118.4,148.9726149,501.5748399,961867.6979,65871.04,0.450908294]
        self.expected2 = [10000, 487.4,980.6179888,501.5748399,961867.6979,65871.04,0.450908294]
        Ls = (1,1,1)
        self.winStat1 = af.windowed_incStat_2D(Ls)
        self.winStat2 = af.windowed_incStat_2D(Ls)
        self.winStat1.join_with_winStat([self.winStat2])

        #fill with data
        t = 0  # time should not have an impact as long as the timestamps are interleaved
        for i in range(0, len(self.data1)):
            self.winStat1.updateStats(self.data1[i], t)
            self.winStat2.updateStats(self.data2[i], t)
        self.win1results = self.winStat1.getStats()
        self.win2results = self.winStat2.getStats()

    def test_mean_noDecay(self):
        indx = 1
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.0001)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.0001)

    def test_std_noDecay(self):
        indx = 2
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.0001)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.0001)

    def test_magnitude_noDecay(self):
        indx = 3
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.0001)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.0001)

    def test_radius_noDecay(self):
        indx = 4
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.0001)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.0001)

    def test_cov_noDecay(self):
        indx = 5
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.57)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.57)

    def test_pcc_noDecay(self):
        indx = 6
        self.assertLess(abs((self.win1results[indx] - self.expected1[indx])) / self.expected1[indx], 0.57)
        self.assertLess(abs((self.win2results[indx] - self.expected2[indx])) / self.expected2[indx], 0.57)

    def test_allStats_wDecay(self):
        Ls = (2,1,.5)
        winStat1 = af.windowed_incStat_2D(Ls)
        winStat2 = af.windowed_incStat_2D(Ls)
        winStat1.join_with_winStat([winStat2])

        t=0
        t0 = time.time()
        for i in range(0,len(self.data1)):
            out1 = winStat1.updateAndGetStats(self.data1[i],t)
            out2 = winStat2.updateAndGetStats(self.data2[i],t+0.36)
            t = t + 1.1
        t1 = time.time()
        print("It took: "+str(t1-t0)+" seconds to process "+str(len(self.data1))+" instances with two winStats")
        print(winStat1.getHeaders())
        print(out1)
        print(out2)

class checkNetstatFramework(unittest.TestCase):
    def test_sessionLimit(self):
        maxHost=10
        maxSess=10
        nstat = ns.netStat(maxHost,maxSess)
        t=0
        sessionCount = 0
        try:
            for src_h in range(0,maxHost):
                for sid in range(0,maxSess+1):
                    nstat.updateGetStats('MAC','10.0.0.'+str(src_h),str(sid),'10.0.1.1',str(sid),1,t)
                    t=t+0.001
                    sessionCount = sessionCount + 1
        except LookupError:
            self.assertEquals(sessionCount - maxSess*maxHost, 0)  #If fails, this means that LookupError was throw correctly (more or less sessions were allowed)
            return
        self.assertLessEqual(maxSess*maxHost,sessionCount)  #LookupError wasn't raised but it should have

    def test_HostLimit(self):
        maxHost = 10
        maxSess = 10
        nstat = ns.netStat(maxHost, maxSess)
        t = 0
        hostCount = 0
        try:
            for src_h in range(0, maxHost+1):
                for sid in range(0, maxSess):
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.1.1', str(sid), 1, t)
                    t = t + 0.001
                hostCount = hostCount + 1
        except LookupError:
            self.assertEquals(hostCount-maxHost,0) #if Fails, this means that LookupError was not throw correctly (more or less sessions were allowed)
            return
        self.assertLessEqual(maxHost,hostCount) #LookupError wasn't raised but it should have

    def test_MACIPLimit(self):
        maxHost = 10
        maxSess = 10
        nstat = ns.netStat(maxHost, maxSess)
        t = 0
        MACIPCount = 0
        try:
            for src_h in range(0, maxHost):
                for MAC in range(0, 3+1):
                    nstat.updateGetStats(str(MAC), '10.0.0.' + str(src_h), str(1), '10.0.1.1', str(2), 1, t)
                    t = t + 0.001
                    MACIPCount = MACIPCount + 1
        except LookupError:
            self.assertEquals(MACIPCount-maxHost*3,1) #if Fails, this means that LookupError was not thrown correctly (more or less sessions were allowed)
            return
        self.assertLessEqual(maxHost*3,MACIPCount) #LookupError wasn't raised but it should have

    def test_purgeOldData(self):
        maxHost = 255
        maxSess = 80000
        nstat = ns.netStat(maxHost, maxSess)
        t = 0
        print("Adding Before Sessions")
        for src_h in range(0, 5):
            for dst_h in range(0, 10):
                for sid in range(0, 5):
                    ssid=sid
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.0.'+str(dst_h), str(ssid), 1, t)
        print("Adding After Sessions")
        t = 1000000000
        for src_h in range(0, 5):
            for dst_h in range(0, 5):
                for sid in range(0, 2):
                    ssid=sid
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.0.'+str(dst_h), str(ssid), 1, t)
        print("Begin Purge")
        before = len(nstat.HT.HT)
        memb4 = sys.getsizeof(nstat.HT.HT)+sys.getsizeof(nstat.Rec_MAC_Host)+sys.getsizeof(nstat.Rec_Hosts)+sys.getsizeof(nstat.Rec_Sessions)
        tic = time.time()
        nstat.purgeOldRecords(t)
        toc = time.time() - tic
        gc.collect()
        memAft = sys.getsizeof(nstat.HT.HT)+sys.getsizeof(nstat.Rec_MAC_Host)+sys.getsizeof(nstat.Rec_Hosts)+sys.getsizeof(nstat.Rec_Sessions)
        after = len(nstat.HT.HT)
        print('Purge: Before '+str(before)+' After '+str(after)+'Time: '+str(toc)+' seconds.\nMem Before: '+str(memb4/(1024*1024))+'MB, Mem After: '+str(memAft/(1024*1024))+' MB')
        self.assertEqual(before-after,400) #there should be 4 less entries

    def test_purgeOldDataMulti(self):
        maxHost = 255
        maxSess = 80000
        nstat = ns.netStat(maxHost, maxSess)
        t = 0
        print("Adding Before Sessions")
        for src_h in range(0, 5):
            for dst_h in range(0, 10):
                for sid in range(0, 5):
                    ssid=sid
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.0.'+str(dst_h), str(ssid), 1, t)
        print("Adding After Sessions 1")
        t = 1000000000
        for src_h in range(0, 5):
            for dst_h in range(0, 8):
                for sid in range(0, 4):
                    ssid=sid
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.0.'+str(dst_h), str(ssid), 1, t)
        print("Begin Purge 1")
        before = len(nstat.HT.HT)
        memb4 = sys.getsizeof(nstat.HT.HT)+sys.getsizeof(nstat.Rec_MAC_Host)+sys.getsizeof(nstat.Rec_Hosts)+sys.getsizeof(nstat.Rec_Sessions)
        tic = time.time()
        nstat.purgeOldRecords(t)
        toc = time.time() - tic
        gc.collect()
        memAft = sys.getsizeof(nstat.HT.HT)+sys.getsizeof(nstat.Rec_MAC_Host)+sys.getsizeof(nstat.Rec_Hosts)+sys.getsizeof(nstat.Rec_Sessions)
        after = len(nstat.HT.HT)
        print('Purge: Before '+str(before)+' After '+str(after)+' Time: '+str(toc)+' seconds.\nMem Before: '+str(memb4/(1024*1024))+'MB, Mem After: '+str(memAft/(1024*1024))+' MB')
        print("Adding After Sessions 2")
        t = 2000000000
        for src_h in range(0, 5):
            for dst_h in range(0, 5):
                for sid in range(0, 2):
                    ssid = sid
                    nstat.updateGetStats('MAC', '10.0.0.' + str(src_h), str(sid), '10.0.0.' + str(dst_h), str(ssid), 1,
                                         t)
        print("Begin Purge 2")
        memb4 = sys.getsizeof(nstat.HT.HT) + sys.getsizeof(nstat.Rec_MAC_Host) + sys.getsizeof(
            nstat.Rec_Hosts) + sys.getsizeof(nstat.Rec_Sessions)
        tic = time.time()
        nstat.purgeOldRecords(t)
        toc = time.time() - tic
        gc.collect()
        memAft = sys.getsizeof(nstat.HT.HT) + sys.getsizeof(nstat.Rec_MAC_Host) + sys.getsizeof(
            nstat.Rec_Hosts) + sys.getsizeof(nstat.Rec_Sessions)
        after = len(nstat.HT.HT)
        print('Purge 2: Before ' + str(before) + ' After ' + str(after) + ' Time: ' + str(
            toc) + ' seconds.\nMem Before: ' + str(memb4 / (1024 * 1024)) + 'MB, Mem After: ' + str(
            memAft / (1024 * 1024)) + ' MB')
        self.assertEqual(before-after,400) #there should be 4 less entries


    def test_run(self):
        maxHost = 50
        maxSess = 50
        nstat = ns.netStat(maxHost, maxSess)
        with open('D:\datasets\\SYN.tsv', 'rt', encoding="utf8") as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            count = 0
            timestats = []

            for row in tsvin:
                count = count + 1
                if count % 10000 == 0:
                    print(count)
                if count > 1:
                    if count == 10000:
                        print((srcMAC, srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp)))
                        #print(stats)
                        print('Mean packet processing time: '+str(np.mean(timestats)))
                        break
                    else:
                        timestamp = row[0]
                        framelen = row[1]
                        srcIP = row[5] + row[50]  # ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                        dstIP = row[6] + row[51]  # ipv4 or ipv6 address
                        srcMAC = row[2]
                        srcproto = row[14] + row[
                            32]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                        dstproto = row[15] + row[33]  # UDP or TCP port
                        if srcproto == '':  # it's a L2/L1 level protocol
                            if row[37] != '':  # is ARP
                                srcproto = 'arp'
                                dstproto = 'arp'
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                            elif row[36] != '':  # is IGMP
                                srcproto = 'igmp'
                                dstproto = 'igmp'
                            elif row[34] != '':  # is ICMP
                                srcproto = 'icmp'
                                dstproto = 'icmp'
                            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                        tic = time.time()
                        stats = nstat.updateGetStats(srcMAC,srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp))
                        toc = time.time() - tic
                        timestats.append(toc)

    def test_run_affectOfOneSided_winstats(self): #should have no affect on results
        maxHost = 50
        maxSess = 50
        nstat = ns.netStat(maxHost, maxSess)
        with open('D:\datasets\\SYN.tsv', 'rt', encoding="utf8") as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            count = 0
            timestats = []

            for row in tsvin:
                count = count + 1
                if count % 10000 == 0:
                    print(count)
                if count > 1:
                    if count == 100000:
                        #print((srcMAC, srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp)))
                        #print(stats1)
                        print('Mean packet processing time: '+str(np.mean(timestats)))
                        break
                    else:
                        timestamp = row[0]
                        framelen = row[1]
                        srcIP = row[5] + row[50]  # ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                        dstIP = row[6] + row[51]  # ipv4 or ipv6 address
                        srcMAC = row[2]
                        srcproto = row[14] + row[
                            32]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                        dstproto = row[15] + row[33]  # UDP or TCP port
                        if srcproto == '':  # it's a L2/L1 level protocol
                            if row[37] != '':  # is ARP
                                srcproto = 'arp'
                                dstproto = 'arp'
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                            elif row[36] != '':  # is IGMP
                                srcproto = 'igmp'
                                dstproto = 'igmp'
                            elif row[34] != '':  # is ICMP
                                srcproto = 'icmp'
                                dstproto = 'icmp'
                            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                        tic = time.time()
                        stats1 = nstat.updateGetStats(srcMAC,srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp))
                        toc = time.time() - tic
                        timestats.append(toc)
        #reset, with purges
        nstat = ns.netStat(maxHost, maxSess)
        with open('D:\datasets\\SYN.tsv', 'rt', encoding="utf8") as tsvin:
            tsvin = csv.reader(tsvin, delimiter='\t')
            count = 0
            timestats = []

            for row in tsvin:
                count = count + 1
                if count % 10000 == 0:
                    print(count)
                if count > 1:
                    if count == 100000:
                        print((srcMAC, srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp)))
                        print(stats1[0:10])
                        print(stats2[0:10])
                        print('Relative error')
                        print(np.absolute(np.array(stats1)-np.array(stats2))/np.array(stats1))
                        print('percent error:')
                        print((np.array(stats2)/np.array(stats1)-1)*100)
                        print('Mean packet processing time: ' + str(np.mean(timestats)))
                        break
                    else:
                        timestamp = row[0]
                        framelen = row[1]
                        srcIP = row[5] + row[50]  # ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                        dstIP = row[6] + row[51]  # ipv4 or ipv6 address
                        srcMAC = row[2]
                        srcproto = row[14] + row[
                            32]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                        dstproto = row[15] + row[33]  # UDP or TCP port
                        if srcproto == '':  # it's a L2/L1 level protocol
                            if row[37] != '':  # is ARP
                                srcproto = 'arp'
                                dstproto = 'arp'
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                            elif row[36] != '':  # is IGMP
                                srcproto = 'igmp'
                                dstproto = 'igmp'
                            elif row[34] != '':  # is ICMP
                                srcproto = 'icmp'
                                dstproto = 'icmp'
                            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                                srcIP = row[2]  # src MAC
                                dstIP = row[3]  # dst MAC
                        if count % 10000 == 0:
                            nstat.purgeOldRecords(float(timestamp))
                        tic = time.time()
                        stats2 = nstat.updateGetStats(srcMAC, srcIP, srcproto, dstIP, dstproto, int(framelen),
                                                     float(timestamp))
                        toc = time.time() - tic
                        timestats.append(toc)
        self.assertGreaterEqual(np.mean((np.array(stats2)/np.array(stats1)-1)*100),0.000001) #average percent error with purges

suite = unittest.TestLoader().loadTestsFromTestCase(computeWinStats2D)
#unittest.TextTestRunner(verbosity=2).run(suite)
#suite = unittest.TestLoader().loadTestsFromTestCase(checkNetstatFramework)
#unittest.TextTestRunner(verbosity=2).run(suite)


