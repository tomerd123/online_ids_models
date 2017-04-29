#import netStat
import sys
import time
# s = [324,45,437,8,3,37,4,234,89,3]*1000
# ht = netStat.incStatHT(1,.1,.01,.001)
# t=0
# start= time.clock()
# for x in s:
#     out1 = ht.updateGet_2D("some_src","some_dst",x,t)
#     out2 = ht.updateGet_2D("some_dst","some_src",-x,t+3)
#     t = t + 6
#
# print(time.clock()-start)
# print(out1)
# print(out2)
# print(t)

from itertools import repeat
import io
import csv
import netStat as ns
import numpy as np

def RTSP_videoJak_Dataset_Gen():
    ht = ns.netStat()
    with io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/RTSP_record_parsed.tsv','rt',encoding="utf8") as tsvin, io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/videoJak_full.csv', 'wt', newline='') as csvout:
        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0

        for row in tsvin:
            count= count + 1
            if count%10000==0:
                print(count)

            if count == 1:
                #print (str(len(row))+str (" num of original headers"))

                #csvout.writerow(str(row) + str(ht.getNetStatHeaders())+["Class"])
                for f in row:
                    csvout.write(unicode(str(f)+",","utf-8"))
                for f in ht.getNetStatHeaders():

                    csvout.write(unicode(str(f)+",","utf-8"))
                csvout.write(unicode("Class","utf-8"))
                csvout.write(unicode("\n","utf-8"))
                #print (str(len(ht.getNetStatHeaders()))+str(" are the stats headers"))
                #csvout = csv.writer(csvout)

                """
                counter = 0
                for x in row:
                    print(str(x) + ", " + str(counter))
                    counter += 1
                """
            else:
                #print (str(len(row))+str(" num of original features"))
                try:

                    timestamp = row[53]
                    framelen = row[54]
                    srcIP = row[15] #ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                    dstIP = row[16] #ipv4 or ipv6 address
                    srcproto = row[17] + row[33] #UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                    dstproto = row[18] + row[34] #UDP or TCP port


                    if srcproto == '': #it's a L2/L1 level protocol
                        if row[48] != '': #is ARP
                            srcproto = 'arp'
                            dstproto = 'arp'
                            srcIP = row[49] #src MAC
                            dstIP = row[51] #dst MAC
                        elif row[40] != '': #is IGMP
                            srcproto = 'igmp'
                            dstproto = 'igmp'
                        elif row[37] != '': #is ICMP
                            srcproto = 'icmp'
                            dstproto = 'icmp'
                        elif srcIP+srcproto+dstIP+dstproto == '': #some other protocol
                            srcIP = row[1]  # src MAC
                            dstIP = row[0]  # dst MAC
                    stats = ht.updateGetStats(srcIP,srcIP,srcproto,dstIP,dstproto,int(framelen),float(timestamp))
                    #print (str(len(stats))+ str(" num of stat features"))
                    Label = "0"
                    if float(timestamp)>=2874.460763: #1750648 frame.no
                        print("reached")
                        if row[15] != '': #row[5] is srcIPv4
                            if row[15].split(".")[3] == "13":
                                Label = "1"
                    #replace missing values with -1
                    for index, item in enumerate(row):
                        if item == '':
                            row[index] = '-1'
                    m=map(str,row)
                    m2=map(str,list(stats))
                    j2=', '.join(m2)
                    j=', '.join(m)
                    j+=","+j2
                    j+=","+Label
                    #csvout.writerow(row + list(stats) + [Label],"utf-8")
                    csvout.write(unicode(str(j),"utf-8"))
                    csvout.write(unicode("\n","utf-8"))

                except:
                    count+=1
                    print("observation "+str(count)+" was rejected")
                    continue

def SYN_Dataset_Gen():
    ht = ns.netStat()
    with io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/syn_record_parsed.tsv','rt',encoding="utf8") as tsvin, io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/SYN_full.csv', 'wt', newline='') as csvout:
        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0
        #csvout = csv.writer(csvout)

        for row in tsvin:
            count= count + 1
            if count%10000==0:
                print(count)

            if count == 1:
                directional=True
                for f in row:
                    csvout.write(unicode(str(f)+",","utf-8"))
                for f in ht.getNetStatHeaders():

                    csvout.write(unicode(str(f)+",","utf-8"))
                csvout.write(unicode("Class","utf-8"))
                csvout.write(unicode("\n","utf-8"))

                #csvout.writerow(row + ht.getNetStatHeaders(directional)+["Class"])
            else:
                timestamp = row[53]
                framelen = row[54]
                srcIP = row[15] #ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                dstIP = row[16] #ipv4 or ipv6 address
                srcproto = row[17] + row[33]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                dstproto = row[18] + row[34]  # UDP or TCP port
                if srcproto == '': #it's a L2/L1 level protocol
                    if row[48] != '': #is ARP
                        srcproto = 'arp'
                        dstproto = 'arp'
                        srcIP = row[49] #src MAC
                        dstIP = row[51] #dst MAC
                    elif row[40] != '': #is IGMP
                        srcproto = 'igmp'
                        dstproto = 'igmp'
                    elif row[37] != '': #is ICMP
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                    elif srcIP+srcproto+dstIP+dstproto == '': #some other protocol
                        srcIP = row[1]  # src MAC
                        dstIP = row[0]  # dst MAC
                if row[1] == '00:a0:de:f1:88:6e': #the source is the yamaha gateway
                    direction = "in"
                else:
                    direction = "out"

                stats = ht.updateGetStats(direction,str(srcIP),srcproto,dstIP,dstproto,int(framelen),float(timestamp))

                Label = "0"
                if count >= 1536268:
                    if row[15] != '':  # row[5] is srcIPv4
                        if row[15].split(".")[3] == "13":
                            Label = "1"

                #replace missing values with -1
                for index, item in enumerate(row):
                    if item == '':
                        row[index] = '-1'
                #csvout.writerow(row + list(stats) + [Label])
                m = map(str, row)
                m2 = map(str, list(stats))
                j2 = ', '.join(m2)
                j = ', '.join(m)
                j += "," + j2
                j += "," + Label
                # csvout.writerow(row + list(stats) + [Label],"utf-8")
                csvout.write(unicode(str(j), "utf-8"))
                csvout.write(unicode("\n", "utf-8"))


def physicalMIM_Dataset_Gen():
    ht = ns.netStat()
    with io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/piddle_record_parsed.tsv','rt',encoding="utf8") as tsvin, io.open('/media/root/66fff5fd-de78-45b0-880a-d2e8104242b5/datasets/piddle_FULL.csv', 'wt', newline='') as csvout:

        tsvin = csv.reader(tsvin, delimiter='\t')
        count = 0
        #csvout = csv.writer(csvout)

        for row in tsvin:
            count = count + 1
            if count % 10000 == 0:
                print(count)

            if count == 1:
                directional = True
                for f in row:
                    csvout.write(unicode(str(f)+",","utf-8"))
                for f in ht.getNetStatHeaders():

                    csvout.write(unicode(str(f)+",","utf-8"))
                csvout.write(unicode("Class","utf-8"))
                csvout.write(unicode("\n","utf-8"))

                #csvout.writerow(row + ht.getNetStatHeaders(directional) + ["Class"])
            else:

                timestamp = row[53]
                framelen = row[54]
                srcIP = row[15]  # ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                dstIP = row[16]  # ipv4 or ipv6 address
                srcproto = row[17] + row[
                    33]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                dstproto = row[18] + row[34]  # UDP or TCP port
                if srcproto == '':  # it's a L2/L1 level protocol

                    if row[48] != '':  # is ARP
                        srcproto = 'arp'
                        dstproto = 'arp'
                        srcIP = row[49]  # src MAC
                        dstIP = row[51]  # dst MAC
                    elif row[40] != '':  # is IGMP
                        srcproto = 'igmp'
                        dstproto = 'igmp'
                    elif row[37] != '':  # is ICMP
                        srcproto = 'icmp'
                        dstproto = 'icmp'
                    elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                        srcIP = row[1]  # src MAC
                        if srcIP=='':
                            srcIP=row[49]
                        dstIP = row[0]  # dst MAC
                        if dstIP=='':
                            dstIP=row[51]
                        srcproto="other"
                        dstproto="other"
                elif srcIP+dstIP=='':
                    srcIP = row[1]
                    dstIP = row[0]
                if row[1] == '00:a0:de:f1:88:6e':  # the source is the yamaha gateway
                    direction = "in"
                else:
                    direction = "out"

                try:

                    stats = ht.updateGetStats(direction,srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp))

                except:
                    count+=1
                    print ("skipped netstat")
                    continue
                Label = "0"
                if count >= 5179941:

                    Label = "1"

                # replace missing values with -1
                for index, item in enumerate(row):
                    if item == '':
                        row[index] = '-1'

                m = map(str, row)
                m2 = map(str, list(stats))
                j2 = ', '.join(m2)
                j = ', '.join(m)
                j += "," + j2
                j += "," + Label
                # csvout.writerow(row + list(stats) + [Label],"utf-8")
                csvout.write(unicode(str(j), "utf-8"))
                csvout.write(unicode("\n", "utf-8"))

                        #csvout.writerow(row + list(stats) + [Label])


def CTU52818_Dataset_Gen():
    ht = ns.netStat(50000,50000)
    with io.open('E:/thesis_data/datasets/ctu-818-52.csv','rt',encoding="utf8") as tsvin, io.open('E:/thesis_data/datasets/ctu_818_52_Full.csv', 'wt', newline='') as csvout:
        tsvin = csv.reader(tsvin, delimiter=',')
        count = 0

        for row in tsvin:
            count= count + 1
            if count%10000==0:
                print(count)

            if count == 1:
                #print (str(len(row))+str (" num of original headers"))

                #csvout.writerow(str(row) + str(ht.getNetStatHeaders())+["Class"])
                for f in row:
                    csvout.write(unicode(str(f)+",","utf-8"))
                for f in ht.getNetStatHeaders():

                    csvout.write(unicode(str(f)+",","utf-8"))
                csvout.write(unicode("Class","utf-8"))
                csvout.write(unicode("\n","utf-8"))
                #print (str(len(ht.getNetStatHeaders()))+str(" are the stats headers"))
                #csvout = csv.writer(csvout)

                """
                counter = 0
                for x in row:
                    print(str(x) + ", " + str(counter))
                    counter += 1
                """
            else:
                #print (str(len(row))+str(" num of original features"))
                try:
                    if count==2:
                        dateAr=row[0].split(' ')[1].split(':')
                        for m in range(len(dateAr)):
                            dateAr[m]=float(dateAr[m])
                        startTS=dateAr[0]*3600*1000+dateAr[1]*60*1000+dateAr[2]*1000
                    dateAr = row[0].split(' ')[1].split(':')
                    for m in range(len(dateAr)):
                        dateAr[m] = float(dateAr[m])
                    timestamp = dateAr[0]*3600*1000+dateAr[1]*60*1000+dateAr[2]*1000-startTS #change format
                    framelen = row[12]
                    srcIP = row[3] #ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                    dstIP = row[6] #ipv4 or ipv6 address
                    srcproto = row[4] #UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                    dstproto = row[7] #UDP or TCP port


                    if srcproto == '': #it's a L2/L1 level protocol
                        if row[2]=="arp": #is ARP
                            srcproto = 'arp'
                            dstproto = 'arp'
                        elif row[2] =='igmp': #is IGMP
                            srcproto = 'igmp'
                            dstproto = 'igmp'
                        elif row[2] == 'icmp': #is ICMP
                            srcproto = 'icmp'
                            dstproto = 'icmp'

                        elif srcIP+srcproto+dstIP+dstproto == '': #some other protocol
                            srcIP = row[3]  # src MAC
                            dstIP = row[6]  # dst MAC
                    stats = ht.updateGetStats(srcIP,srcIP,srcproto,dstIP,dstproto,int(framelen),float(timestamp))
                    #print (str(len(stats))+ str(" num of stat features"))
                    Label = "0"
                    if row[14].find("Botnet")!=-1: #1750648 frame.no
                        #print("reached")
                        Label = "1"
                    #replace missing values with -1
                    for index, item in enumerate(row):
                        if item == '':
                            row[index] = '-1'
                    m=map(str,row)
                    m2=map(str,list(stats))
                    j2=', '.join(m2)
                    j=', '.join(m)
                    j+=","+j2
                    j+=","+Label
                    #csvout.writerow(row + list(stats) + [Label],"utf-8")
                    csvout.write(unicode(str(j),"utf-8"))
                    csvout.write(unicode("\n","utf-8"))


                except Exception as ex:
                    print (ex.message)
                    count+=1
                    print("observation "+str(count)+" was rejected")
                    continue
def sortCTU52818File ():

    with io.open('E:/thesis_data/datasets/ctu52818_400.txt', 'rt', encoding="utf8") as tsvinFullLine:
        with io.open('E:/thesis_data/datasets/ctu52818_400.txt', 'rt', encoding="utf8") as tsvin:

            with io.open('E:/thesis_data/datasets/ctu52818_400_sortedTS.csv', 'wt', encoding="utf8") as tswrite:
                tsvin = csv.reader(tsvin, delimiter='\t')
                count = 0
                ht = ns.netStat(50000, 50000)
                fullLineOrig = tsvinFullLine.readline()

                rowListByTS = []
                for row in tsvin:
                    count = count + 1
                    if count % 10000 == 0:
                        print(count)

                    if count == 1:
                        # print (str(len(row))+str (" num of original headers"))

                        # csvout.writerow(str(row) + str(ht.getNetStatHeaders())+["Class"])
                        editedLine=''
                        for f in row:
                            editedLine+=str(f) + ","

                        #for f in ht.getNetStatHeaders():
                            #editedLine += str(f) + ","
                        #editedLine+="Class"

                        tswrite.write(unicode(str(editedLine)+"\n", "utf-8"))
                        continue

                    if count==2:
                        dateAr = row[0].split(' ')[1].split(':')
                        for m in range(len(dateAr)):
                            dateAr[m] = float(dateAr[m])
                        startTS = dateAr[0] * 3600 * 1000 + dateAr[1] * 60 * 1000 + dateAr[2] * 1000   # change format
                        timestamp=0
                    else:
                        dateAr = row[0].split(' ')[1].split(':')
                        for m in range(len(dateAr)):
                            dateAr[m] = float(dateAr[m])
                        timestamp= dateAr[0] * 3600 * 1000 + dateAr[1] * 60 * 1000 + dateAr[2] * 1000  # change format
                        #timestamp -=startTS
                    fullLineOrig=tsvinFullLine.readline()
                    # sort the file
                    fullLine = ''

                    for r in range(len(row)):

                        if row[r]=='':
                            continue
                        if r<len(row)-1:
                            fullLine += str(row[r]) + ","
                        else:
                            fullLine += str(row[r])
                    rowListByTS.append((timestamp, fullLineOrig))
                    #if row[3].__contains__(':')==False or row[5].__contains__(':')==False:
                     #   print("fs")


                    if fullLine.__contains__(',,')==True:
                        print("here")

                rowListByTS.sort(key=lambda tup: tup[0])

                for row in rowListByTS:

                    tswrite.write(unicode(str(row[1]) + "\n", "utf-8"))

                print (rowListByTS[:30])
                print("finished sort")


def CTU52818_Desaset_Gen_V2_400():

    ht = ns.netStat(50000, 50000)
    with io.open('E:/thesis_data/datasets/ctu52818_400_sortedTS.txt', 'rt', encoding="utf8") as tsvin, io.open(
            'E:/thesis_data/datasets/ctu52818_400_full.csv', 'wt', newline='') as csvout:
        tsvin = csv.reader(tsvin, delimiter=',')
        count = 0

        rowListByTS=[]
        for row in tsvin:
            count = count + 1
            if count % 10000 == 0:
                print(count)

            if count == 1:
                # print (str(len(row))+str (" num of original headers"))

                # csvout.writerow(str(row) + str(ht.getNetStatHeaders())+["Class"])
                for f in row:
                    csvout.write(unicode(str(f) + ",", "utf-8"))
                for f in ht.getNetStatHeaders():
                    csvout.write(unicode(str(f) + ",", "utf-8"))
                csvout.write(unicode("Class", "utf-8"))
                csvout.write(unicode("\n", "utf-8"))



            else:
                # print (str(len(row))+str(" num of original features"))
                try:
                    if count == 2:
                        dateAr = row[0].split(' ')[1].split(':')
                        for m in range(len(dateAr)):
                            dateAr[m] = float(dateAr[m])
                        startTS = dateAr[0] * 3600 * 1000 + dateAr[1] * 60 * 1000 + dateAr[2] * 1000
                    dateAr = row[0].split(' ')[1].split(':')
                    for m in range(len(dateAr)):
                        dateAr[m] = float(dateAr[m])
                    timestamp = dateAr[0] * 3600 * 1000 + dateAr[1] * 60 * 1000 + dateAr[
                                                                                      2] * 1000 - startTS  # change format



                    framelen = row[9]
                    srcIP = row[3].split(':')[0]  # ipv4 or ipv6 address: ipv4 or ipv6 (one will be '')
                    dstIP = row[5].split(':')[0]  # ipv4 or ipv6 address
                    if row[3].__contains__(':')==True:
                        srcproto = row[3].split(':')[1]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
                    else:
                        srcproto='NoProto'
                    if row[3].__contains__(':') == True:
                        dstproto = row[5].split(':')[1]  # UDP or TCP port
                    else:
                        dstproto='NoProto'

                    if srcproto == '':  # it's a L2/L1 level protocol
                        if row[2] == "ARP":  # is ARP
                            srcproto = 'ARP'
                            dstproto = 'ARP'
                        elif row[2] == 'IGMP':  # is IGMP
                            srcproto = 'IGMP'
                            dstproto = 'IGMP'
                        elif row[2] == 'ICMP':  # is ICMP
                            srcproto = 'ICMP'
                            dstproto = 'ICMP'

                        elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                            srcIP = row[3].split(':')[0]  # src MAC
                            dstIP = row[5].split(':')[0]  # dst MAC
                    stats = ht.updateGetStats(srcIP, srcIP, srcproto, dstIP, dstproto, int(framelen),
                                              float(timestamp))
                    # print (str(len(stats))+ str(" num of stat features"))
                    Label = "0"
                    if row[11].find("Botnet") != -1:  # 1750648 frame.no
                        # print("reached")
                        Label = "1"
                    # replace missing values with -1
                    for index, item in enumerate(row):
                        if item == '':
                            row[index] = '-1'
                    m = map(str, row)
                    m2 = map(str, list(stats))
                    j2 = ', '.join(m2)
                    j = ', '.join(m)
                    j += "," + j2
                    j += "," + Label
                    # csvout.writerow(row + list(stats) + [Label],"utf-8")
                    csvout.write(unicode(str(j), "utf-8"))
                    csvout.write(unicode("\n", "utf-8"))


                except Exception as ex:
                    print (ex.message)
                    count += 1
                    print("observation " + str(count) + " was rejected")
                    continue


#SYN_Dataset_Gen()
#physicalMIM_Dataset_Gen()
#SYN_Dataset_Gen()
#physicalMIM_Dataset_Gen()
CTU52818_Desaset_Gen_V2_400()
#sortCTU52818File()
#CTU52818_Dataset_Gen()
#CTU52818_Desaset_Gen_V2_400()
# with open('D:\datasets\\videoJak.tsv', 'rb') as f:
#     temp_lines = f.readline() + '\n'.encode('ascii') + f.readline()
#     dialect = csv.Sniffer().sniff(f.read(1024), delimiters='\t'.encode('ascii'))
#     f.seek(0)
#     reader = csv.reader(f,dialect)
#     data_as_list = list(reader)

