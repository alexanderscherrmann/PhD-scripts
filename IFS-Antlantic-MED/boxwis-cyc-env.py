# coding: utf-8
import numpy as np
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper


import itertools
from matplotlib.cbook import _reshape_2D
import matplotlib.pyplot as plt
import numpy as np

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800.txt','rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]
avaID = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    
aa = 0
c = 'cyc'
e = 'env'
pvloc = dict()
pvloe = dict()
pvl6c = dict()
pvl6e = dict()
for h in range(0,49):
    pvloc[h] =np.array([])
    pvloe[h] =np.array([])
    pvl6e[h] =np.array([])
    pvl6c[h] =np.array([])
    
for qq,k in enumerate(dipv.keys()):
    d = k
    q = np.where(avaID==int(k))[0][0]
    i = np.where(PVdata['rawdata'][k]['PV'][:,0]>=0.75)[0]
    if hourstoSLPmin[q][0]>-6:
        for h in range(0,49):
         pvloc[h] = np.append(pvloc[h],dipv[k][c][i,h])
         pvloe[h] = np.append(pvloe[h],dipv[k][e][i,h])        
        continue

    cy = np.mean(dipv[d][c][i,0])
    if cy<0.05:
        continue

    for h in range(0,49):
        pvl6c[h] = np.append(pvloc[h],dipv[k][c][i,h])
        pvl6e[h] = np.append(pvloc[h],dipv[k][e][i,h])
        
t = np.flip(np.arange(-48,1))
fig,ax = plt.subplots()
boxpvc = []
boxpve = []
boxpv6c = []
boxpv6e = []
for h in np.flip(np.arange(0,49)):
    boxpvc.append(np.sort(pvloc[h]))
    boxpve.append(np.sort(pvloe[h]))
    boxpv6c.append(np.sort(pvl6c[h]))
    boxpv6e.append(np.sort(pvl6e[h]))
    
flier = dict(marker='+',markerfacecolor='grey',markersize=1,linestyle=' ',markeredgecolor='grey')
meanline = dict(linestyle='-',linewidth=1,color='red')

meanline2 = dict(linestyle=':',linewidth=1,color='navy')
capprops = dict(linestyle=':',linewidth=1,color='dodgerblue')
medianprops = dict(linestyle=':',linewidth=1,color='purple')
boxprops = dict(linestyle=':',linewidth=1.,color='slategrey')
whiskerprops= dict(linestyle=':',linewidth=1,color='dodgerblue')


fig,ax = plt.subplots()
ax.set_ylabel(r'PV [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-.25,2.0)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(boxpvc,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
ax.set_xticks(ticks=range(1,len(t)+1,6))
bp2 = ax.boxplot(boxpve,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])
fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-env-cyc-contribution-all.png',dpi=300,bbox_inches="tight")
plt.close('all')
fig,ax = plt.subplots()
ax.set_ylabel(r'PV [PVU]')
ax.set_xlabel(r'time to mature stage [h]')
ax.set_ylim(-.25,2.0)
ax.set_xlim(0,50)
t = np.arange(-48,1)
bp = ax.boxplot(boxpv6c,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline,meanline=True,showmeans=True,showfliers=False)
ax.set_xticks(ticks=range(1,len(t)+1,6))

#ax.text(0.05, 0.95, 'a)', transform=ax.transAxes,fontsize=16, fontweight='bold', va='top')
ax.tick_params(labelright=False,right=True)
ax.set_xticklabels(labels=t[0::6])

bp2 = ax.boxplot(boxpv6e,whis=(10,90),labels=t,flierprops=flier,meanprops=meanline2,meanline=True,showmeans=True,showbox=True,showcaps=True,showfliers=False,medianprops=medianprops,capprops=capprops,whiskerprops=whiskerprops,boxprops=boxprops)
fig.savefig('/home/ascherrmann/009-ERA-5/MED/boxwis-env-cyc-contribution-6h-0.05PVUbond.png',dpi=300,bbox_inches="tight")
plt.close('all')
