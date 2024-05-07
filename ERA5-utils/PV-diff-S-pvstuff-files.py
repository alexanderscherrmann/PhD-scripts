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

p = '/home/ascherrmann/009-ERA-5/'

traced = np.array([])
traj = np.array([])

for d in os.listdir(p):
    if(d.startswith('traced-vars-2full')):
            traced = np.append(traced,d)
    if(d.startswith('traced-vars-S-2full')):
        traj = np.append(traj,d)

traced = np.sort(traced)
traj = np.sort(traj)


datadi = dict() ####raw data of traced vars
datadi2 = dict()
H = 47
labs1 = ['time','lon','lat','P','PS','pvt','pvf','PV'] 
labs2 = ['time','lon','lat','P','PS','PV']

fig, ax = plt.subplots()

av = np.zeros((len(traced),47))
for uyt, txt in enumerate(traced[:]):
    cycID=txt[-10:-4]
    date=txt[-25:-14]
    cycID2 = traj[uyt][-10:-4]
    date2 = traj[uyt][-25:-14]

    datadi[cycID]=dict() #raw data
    datadi2[cycID] = dict()

    tt = np.loadtxt(p + txt)
    tt2 = np.loadtxt(p + traj[uyt])

    for k, el in enumerate(labs1):
        datadi[cycID][el] = tt[:,k].reshape(-1,H+1)
    for k, el in enumerate(labs2):
        datadi2[cycID][el] = tt2[:,k].reshape(-1,H+1)

    deltaPV = datadi2[cycID]['PV'][:,1:] - datadi[cycID]['PV'][:,1:]
    
    avdelta = np.mean(deltaPV,axis=0)
    if (~np.any(avdelta>2)):
        av[uyt,:] = avdelta

        ax.plot(np.flip(np.arange(-47,0)),avdelta,color='grey')

ax.plot(np.flip(np.arange(-47,0)),np.mean(av,axis=0),color='k')
ax.set_xlim(-47,1)
ax.set_ylim(0,0.7)
ax.set_xlabel('time to mature stage [h]')
ax.set_ylabel(r'$\Delta$ PV (S-PVstuff) [PVU]')
fig.savefig(p + 'delta-PV-pvstuff-S-files.png',dpi=300,bbox_inches="tight")
plt.close('all')

    
