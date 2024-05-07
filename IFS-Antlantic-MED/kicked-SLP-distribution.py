import numpy as np
import pickle
import os
import matplotlib
from matplotlib import pyplot as plt
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

rdis = 400
pload = '/home/ascherrmann/010-IFS/traj/MED/use/'
CT = 'MED'
f = open(pload + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-%d.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

slpr = np.array([])

for u,k in enumerate(PVdata['rawdata'].keys()):
    mon = PVdata['mons'][u]
    q = int(k[-3:])
    slpr = np.append(slpr,locdata[mon][q]['SLP'][abs(locdata[mon][q]['hSLP'][0]).astype(int)])

fig,ax = plt.subplots()

ax.set_ylabel(r'number of cyclones')
ax.set_xlabel(r'minimum SLP [hPa]')
slm = 970
slma = 1010
#ax.hist(slpkh,bins=32,range=[slm,slma],facecolor='k',alpha=0.5)
#ax.hist(slpkc,bins=32,range=[slm,slma],facecolor='r',alpha=0.5)
da = ax.hist(slpr,bins=32,range=[slm,slma],facecolor='k',edgecolor='grey',alpha=1)
ax.set_xlim(970,1010)
ax2 = ax.twinx()
ax2.set_ylim(0,60)

ax2.plot(da[1],np.append(0,np.cumsum(da[0])),color='b')
ax2.set_ylabel('number of cyclones')
fig.savefig('/home/ascherrmann/010-IFS/SLP-distribution-IFS.png',dpi=300,bbox_inches="tight")
plt.close('all')
