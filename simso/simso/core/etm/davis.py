#!/usr/bin/env python
import math
import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt

import time

# polynomial
from numpy.polynomial import Polynomial as Poly

with open('cacheConfig.txt') as f:
    line = f.readline()
    size = int(line)
    line = f.readline()
    way = int(line)
    line = f.readline()
    block = int(line)
    cacheF = f.readline()
    cacheF = cacheF[:-1]
    cacheFData = f.readline()
    cacheFData = cacheFData[:-1]

bit = 32 # 32-bit address
ADDR = 0xffffffff

# miss cycle number
nMiss = 100
# hit cycle number
nHit = 1

# used to calculate number of misses
MISS = 1
HIT = 0

# used to convert number of misses to cycles
def calc_cycles(nMem, cycles):
    #print nMiss, nHit, nMem
    #print cycles
    for i in range(len(cycles)):
        cycles[i] = cycles[i]*nMiss + (nMem - cycles[i])*nHit

    return cycles


# get reuse distance
def getReuse(ins, N):
    # reuse distance
    reuse = []
    # previous instruction
    prevIns = []
    # previous instruction index number
    prevNum = []

    # prob. for instuctions
    for i in range(len(ins)):
        w = ins[i]
        if w in prevIns:
            K = i-prevNum[prevIns.index(w)]-1
            reuse.append(K)
            prevNum[prevIns.index(w)] = i
        else: # first time use
            prevIns.append(w)
            prevNum.append(i)
            reuse.append('inf')

    return reuse

# get hit prob. using reuse distance
def getHit(reuse, N):
    Phit = []
    for K in reuse:
        if K<N:
            P = (1-1.0/N)**K
            Phit.append(P)
        else:
            Phit.append(0)
    return Phit

# speedup algorithm
# techniques applied:
#   discretization
def calc_speedup(Phit):
    if Phit==[]:
        return [0], [1]
    
    # sort prob. and then divide them into groups   
    Phit.sort()
    
    tmp = [Phit[0]]
    cycle = [0]
    prob = [1]

    #print 'Phit:', Phit
    for i in range(len(Phit)-1):
        if Phit[i]==Phit[i+1]:
            tmp.append(Phit[i+1])
        else:
            cycle0, prob0 = conv_same(tmp)
            cycle, prob = add_time_prob(cycle, cycle0, prob, prob0)
            
            # next group of prob.
            tmp = [Phit[i+1]]

    # last group of prob.
    cycle0, prob0 = conv_same(tmp)
    cycle, prob = add_time_prob(cycle, cycle0, prob, prob0)

    return cycle, prob


# convolution of the same ETP
def conv_same(Phit):
    val = math.log(len(Phit), 2)
    if val % 1 == 0:
        cycle, prob = conv_p2(Phit, int(val))
    else:
        # find index for power-of-two
        ind = 2**int(val)
        
        cycle, prob = conv_p2(Phit[0:ind], int(val))
        cycle0, prob0 = conv_same(Phit[ind:])
        cycle, prob = add_time_prob(cycle, cycle0, prob, prob0)

    return cycle, prob

# convolution of power-of-two ETPs 
def conv_p2(Phit, power):
    cycle = [HIT, MISS]
    prob = [Phit[0], 1-Phit[0]]
    for i in range(power):
        cycle, prob = add_time_prob(cycle, cycle, prob, prob)   

    return cycle, prob

# polynomial multiplication is used for convolution
def add_time_prob(time, cycles, prob, cyclesP):
    p1 = [0]*(time[-1]+1)
    p1 = [0]*(max(time)+1)
    for i in range(len(time)):
        p1[time[i]] = prob[i]

    p2 = [0]*(max(cycles)+1)
    for i in range(len(cycles)):
        p2[cycles[i]] = cyclesP[i]

    # polynomial mult.
    p1 = Poly(p1)
    p2 = Poly(p2)
    p = p1*p2
    p = p.coef.tolist()

    # get cycles and prob.
    time = []
    prob = []
    for i in range(len(p)):
        #if p[i]>0:
        time.append(i)
        prob.append(p[i])

    return time, prob



def get_data(enableIns, enableData):
    
    nMem = 0
    
    if enableIns:
        start_time = time.time()
        cycle, prob, n = get_data_real(cacheF)
        nMem = nMem + n
        print 'davis ins. time:', time.time() - start_time
    else:
        cycle = prob = []

    cycleID = cycle
    probID = prob
    
    if enableData:
        start_time = time.time()
        cycleData, probData, n = get_data_real(cacheFData)
        nMem = nMem + n
        print 'davis data time:', time.time() - start_time
        cycleID, probID = add_time_prob(cycle, cycleData, prob, probData)
    else:
        cycleData = probData = []



    cycleID = calc_cycles(nMem, cycleID)    
    
    return cycle, prob, cycleData, probData, cycleID, probID

def get_data_real(fileStr): 
    index = size/way/block
    
    print 'cache size:', size, 'way:', way, 'block:', block, 'sets:', index
    print 'instruction trace:', cacheF 
    
    offsetBit = int(round(math.log(block, 2)))
    indexBit = int(round(math.log(index, 2)))
    tagBit = bit - offsetBit - indexBit
    
    addrAll = []
    for i in range(index):
        addrAll.append([])

    # number of memory addresses
    nMem = 0
    
    # get addresses in cache blocks
    with open(fileStr, 'r') as f:
        for line in f:
            nMem = nMem + 1

            val = int(line)
            ind = val
        
            # remove tag bit
            ind &= ADDR >> tagBit
            # remove and shift offset bit
            ind &= ADDR << offsetBit
            ind >>= offsetBit
    
            val &= ADDR << offsetBit
            addrAll[ind].append(val)
   
    time, prob = getTimeProb(addrAll)
    return time, prob, nMem


def getResult():
    ######
    # two task files: f1 and f2
    f1 = 'address/arm/hf_no_sys/cnt.txt'
    f2 = 'address/arm/hf_no_sys/cover.txt'

    with open(f1) as f:
        addr1 = [int(x) for x in f.readlines()]

    with open(f2) as f:
        addr2 = [int(x) for x in f.readlines()]    

    offset = 0
    addr = combine(addr1, addr2, offset)

    ######
    # cache config
    size = 1024
    way = 16
    block = 4

    index = size/way/block

    print 'cache size:', size, 'way:', way, 'block:', block, 'sets:', index
    print 'instruction traces:', f1, f2

    offsetBit = int(round(math.log(block, 2)))
    indexBit = int(round(math.log(index, 2)))
    tagBit = bit - offsetBit - indexBit
   
    ######
    addrAll = []
    for i in range(index):
        addrAll.append([])

    # number of memory addresses
    nMem = len(addr)
    
    # get addresses in cache blocks
    for val in addr:
        ind = val
        
        # remove tag bit
        ind &= ADDR >> tagBit
        # remove and shift offset bit
        ind &= ADDR << offsetBit
        ind >>= offsetBit
    
        val &= ADDR << offsetBit
        addrAll[ind].append(val)
   
    time, prob = getTimeProb(addrAll)
    time = calc_cycles(nMem, time) 
    
    return time, prob   

def combine(addr1, addr2, offset):
    """
    Combine two memory traces together.
    FIFO policy is applied.
    
    a, b, c, d, e, ...
    offset.. f, g, ...
    """
    addr = addr1 + addr2

    if offset < len(addr1):
        i1 = offset
        i2 = 0

        if offset % 2 == 0:
            val = 0
        else:
            val = 1

        for i in range(offset, len(addr)):
            if i%2 == val:
                addr[i] = addr1[i1]
                i1 += 1
            else:
                addr[i] = addr2[i2]
                i2 += 1

            if i1 == len(addr1) or i2 == len(addr2):
                break

        if len(addr1) - offset > len(addr2):
            for i in range(offset + len(addr2)*2, len(addr)):
                addr[i] = addr1[i1]
                i1 += 1


    return addr

def getTimeProb(addrAll):
    """
    Obtain cycles and prob. for given addresses.
    """
    time = [0]
    prob = [1]
    

    for i in range(len(addrAll)):
        #print 'index:', i, 'len', len(addrAll[i])
        # all addresses for each index
        addrI = addrAll[i]
        if not addrI:
            continue

        ins = addrI
        # number of cache entries
        N = way 

        # reuse distance
        reuse = getReuse(ins, N)

        # hit prob.
        Phit = getHit(reuse, N) 

        Nc, Pdf = calc_speedup(Phit)

        #print Nc, Pdf
        
        time, prob = add_time_prob(time, Nc, prob, Pdf)

    # sort time
	time = np.array(time)
	prob = np.array(prob)
	time = time[prob.nonzero()]
	prob = prob[prob.nonzero()]
	ind = np.argsort(time)
	time = time[ind]
	prob = prob[ind]

	time = time.tolist()
	prob = prob.tolist()

    return time, prob

def main():
    #time, prob, timeData, probData, timeID, probID = get_data(1, 0)
    #f = open('data_davis_cycles_ID.dat', 'w')
    #for i in timeID:
    #    f.write('%g\n' % (i))
    #f.close()
    #f = open('data_davis_weights_ID.dat', 'w')
    #for i in probID:
    #    f.write('%g\n' % (i))
    #f.close()

    timeID, probID = getResult() 

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #plt.hist(time, weights=prob, bins=100, normed=1, histtype='step', cumulative=-1, label='fig')
    #plt.hist(timeData, weights=probData, bins=100, normed=1, histtype='step', cumulative=-1, label='fig')
    plt.hist(timeID, weights=probID, bins=100, normed=1, histtype='step', cumulative=-1, label='fig')
    plt.ylim([1e-15, 1.05])
    #plt.xlim([8000, 15000])
    ax.set_yscale('log')
    plt.grid()
    plt.savefig('fig_davis.eps')
    #plt.show()

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main()
