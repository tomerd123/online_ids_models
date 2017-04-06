import csv
import time
import unittest
import cybCommittee as cy
import netStat as ns
import numpy as np


def run(args):
    type = args[0]
    aeNoise = args[1]

    if type == "SYN":
        infile = 'D:\datasets\SYN.tsv'
        benign = range(250000,536268)
        mal = range(536268,624505)
    if type == "RTSP":
        infile = 'D:\datasets\RTSPMITM.tsv'
        benign = range(500000,747648)
        mal = range(747648,946987)

    maxHost = 255
    maxSess = 1000
    nstat = ns.netStat(maxHost, maxSess)
    CC = cy.Committee(defaultCommittee=True,AEcorruption=aeNoise)
    Res = []
    T1 = []
    T2 = []
    T3 = []
    with open(infile, 'rt', encoding="utf8") as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0
        for row in tsvin:
            count = count + 1
            if count > 1:
                timestamp = row[0]
                framelen = row[1]
                if row[4] != '':  # IPv4
                    srcIP = row[4]
                    dstIP = row[5]
                    IPtype = 0
                else:
                    srcIP = row[44]
                    dstIP = row[45]
                    IPtype = 1
                srcproto = row[12] + row[
                    29]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                dstproto = row[13] + row[30]  # UDP or TCP port
                srcMAC = row[2]
                dstMAC = row[3]
                if srcproto == '':  # it's a L2/L1 level protocol
                    if row[33] != '':  # is ARP
                        srcproto = 'arp'
                        dstproto = 'arp'
                        srcIP = row[35]  # src IP (ARP)
                        dstIP = row[37]  # dst IP (ARP)
                        IPtype = 0
                    elif row[31] != '':  # is ICMP
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                        IPtype = 0
                    elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                        srcIP = row[2]  # src MAC
                        dstIP = row[3]  # dst MAC
                try:
                    tic1 = time.time()
                    stats = nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto,
                                                 int(framelen),
                                                 float(timestamp))
                    toc1 = time.time() - tic1
                    x_ref = [np.concatenate((np.array(row, dtype=object), stats, (float(timestamp),)))]
                    T1.append(toc1)
                    if count <= 1000000:  # just train
                        tic2 = time.time()
                        CC.train(x_ref)
                        toc2 = time.time() - tic2
                        T2.append(toc2)
                    if count > 1000000:  # execute, then train (if bening)          #1536268
                        tic3 = time.time()
                        response = CC.execute_train(x_ref)
                        toc3 = time.time() - tic3
                        Res = Res + [np.concatenate(((response[0],),(response[1],)))]
                        T3.append(toc3)
                except Exception as e:
                    print(e)
                    if count > 1000000:
                        Res = Res + [np.concatenate(((0,),(0,)))]
    BenignRoof = -np.Inf
    for i in benign:
        if Res[i][1] > BenignRoof:
            BenignRoof = Res[i][1]
    TP = 0
    for i in mal:
        if Res[i][1] > BenignRoof:
            TP = TP + 1
    return type, aeNoise, BenignRoof, TP, np.mean(T1), np.mean(T2), np.mean(T3)


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import sys

if __name__ == '__main__':
    trials = []
    for t in range(0,20):
        trials = trials + [('SYN',(0.5/20)*t)]
    for t in range(20, 40):
        trials = trials + [('RTSP', (0.5 / 20) * t)]
    pool = multiprocessing.Pool(42)
    out1 = zip(*pool.map(run, trials))
    print(out1)
    with open('D:\datasets\\par_corruptionTest_v2.csv', 'wt', newline='') as csvout:
        csvout = csv.writer(csvout)
        for r in out1:
            csvout.writerow(r)

