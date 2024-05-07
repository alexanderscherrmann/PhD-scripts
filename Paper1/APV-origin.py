import numpy as np
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import argparse
import pickle

parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
parser.add_argument('deltaPSP',default='',type=int,help='difference between surface pressure and pressure that should be evaluated as orographical influence')

parser.add_argument('ZBB',default='',type=int,help='evelation in m at which PV changes should be evaluated as orographic')

args = parser.parse_args()
rdis = int(args.rdis)
deltaPSP = int(args.deltaPSP)
zbb = int(args.ZBB)

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'

oro = 'oro'
env = 'env'
cyc = 'cyc'
noro= 'noro'

split=[cyc,env]
NORO = dict()


f = open(pload + 'PV-data-' + 'dPSP-' + str(deltaPSP) + '-ZB-' + str(zbb) + '-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

dit=PVdata['dit']
datadi=PVdata['rawdata']

for cycID in dit.keys():

    dit[cycID][noro] = (dit[cycID][oro]+1)%2

    NORO[cycID]=dict()
    NORO[cycID][cyc]=dict()
    NORO[cycID][env]=dict()

    deltaPV = np.zeros(datadi[cycID]['time'].shape)
    deltaPV[:,1:] = datadi[cycID]['PV'][:,:-1]-datadi[cycID]['PV'][:,1:]
    ttmp = dict()
    ttmp[noro] = (dit[cycID][noro][:,:-1]+dit[cycID][noro][:,1:])/2.
    for key in split:
        ttmp[key] = (dit[cycID][key][:,:-1]+dit[cycID][key][:,1:])/2.
        NORO[cycID][key]= np.zeros(datadi[cycID]['time'].shape)
        NORO[cycID][key][:,:-1] = np.flip(np.cumsum(np.flip(deltaPV[:,1:]*ttmp[key][:,:]*ttmp[noro][:,:],axis=1),axis=1),axis=1)

PVdata[noro] = NORO
f = open(pload + 'PV-data-' + 'dPSP-' + str(deltaPSP) + '-ZB-' + str(zbb) + '-2-%d-correct-distance-noro.txt'%rdis,'wb')
pickle.dump(PVdata,f)
f.close()
