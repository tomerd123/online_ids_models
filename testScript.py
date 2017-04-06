import netStat
import time
s = [324,45,437,8,3,37,4,234,89,3]*1000
ht = netStat.incStatHT(1,.1,.01,.001)
t=0
start= time.clock()
for x in s:
    out1 = ht.updateGet_2D("some_src","some_dst",x,t)
    out2 = ht.updateGet_2D("some_dst","some_src",-x,t+3)
    t = t + 6

print(time.clock()-start)
print(out1)
print(out2)
print(t)