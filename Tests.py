import csv
import time
import unittest
import cybCommittee as cy
import netStat as ns
import numpy as np


def run(type=None):
    if type == "SYN":
        infile = 'C:\datasets\SYN.tsv'
        outfile1 = 'C:\datasets\SYN_results_refactoredUML7_fixedLR.csv'
        outfile2 = 'C:\datasets\SYN_results_refactoredUML_COM7_fixedLR.csv'
        startOffset = 0
        trainExecStart = startOffset + 1000000
    if type == "RTSP":
        infile = 'C:\datasets\RTSPMITM.tsv'
        outfile1 = 'C:\datasets\RTSPMITM_results_refactoredUML7_fixedLR.csv'
        outfile2 = 'C:\datasets\RTSPMITM_results_refactoredUML_COM7_fixedLR.csv'
        startOffset = 0
        trainExecStart = startOffset + 1000000
    if type == "RTSP2":
        infile = 'C:\datasets\RTSPMITM2.tsv'
        outfile1 = 'C:\datasets\RTSPMITM2_results_short.csv'
        outfile2 = 'C:\datasets\RTSPMITM2_results_COM_short.csv'
        startOffset = 1259815
        trainExecStart = startOffset + 800000
    if type == None:
        infile = 'C:\datasets\RTSPMITM.tsv'
        outfile1 = 'C:\datasets\\tmp.csv'
        outfile2 = 'C:\datasets\\tmp_COM.csv'
        startOffset = 0
        trainExecStart = startOffset + 1000000

    maxHost = 255
    maxSess = 10000
    nstat = ns.netStat(maxHost, maxSess)
    CC = cy.Committee(defaultCommittee=True)
    with open(infile, 'rt', encoding="utf8") as tsvin, open(outfile1, 'wt', newline='') as csvout1,  open(outfile2, 'wt', newline='') as csvout2:
        csvout1 = csv.writer(csvout1)
        csvout2 = csv.writer(csvout2)
        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0
        T1=[]
        T2=[]
        T3=[]
        for row in tsvin:
            count = count + 1
            if count >= startOffset:
                if count % 10000 == 0:
                    print(count)
                    print('FE time: ' + str(np.mean(T1)))
                    print('ML Train time: ' + str(np.mean(T2)))
                    if len(T3) > 0:
                        print('ML Exec+Train time: ' + str(np.mean(T3)))
                if count == 1:
                    csvout1.writerow(('Decision','Severity'))
                if count > 1:
                    IPtype = np.nan
                    timestamp = row[0]
                    framelen = row[1]
                    if row[4] != '': #IPv4
                        srcIP = row[4]
                        dstIP = row[5]
                        IPtype=0
                    elif row[44] != '': #ipv6
                        srcIP = row[44]
                        dstIP = row[45]
                        IPtype=1
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
                        stats = nstat.updateGetStats(IPtype, srcMAC, dstMAC,srcIP, srcproto, dstIP, dstproto, int(framelen),
                                                     float(timestamp))
                        toc1 = time.time() - tic1
                        x_ref = [np.concatenate((np.array(row,dtype=object),stats,(float(timestamp),)))]
                        T1.append(toc1)
                        if count <= trainExecStart:  # just train
                            tic2 = time.time()
                            CC.train(x_ref)
                            toc2 = time.time() - tic2
                            T2.append(toc2)
                        if count > trainExecStart: #  execute, then train (if bening)          #1536268
                            csvout2.writerow([srcIP]+CC.execute_full(x_ref).tolist())
                            tic3 = time.time()
                            response = CC.execute_train(x_ref)
                            toc3 = time.time() - tic3
                            T3.append(toc3)
                            csvout1.writerow(np.concatenate(((response[0],),(response[1],))))
                    except Exception as e:
                        print(e)
                        if count > trainExecStart:
                            csvout1.writerow(np.concatenate(((np.nan,),(np.nan,))))
                            csvout2.writerow([srcIP]+[np.nan]*len(CC.getAllNames()))
        # print('HH')
        # print(nstat.HT_HH.HT.keys())
        # print('dir')
        # print(nstat.HT_BW_dir.HT.keys())
        # print('hphp')
        # print(nstat.HT_HpHp.HT.keys())



import sys
#type=sys.argv[1]
#infile=sys.argv[2]
#outfile=sys.argv[3]
run()