import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

sim,at,med=wrfsims.cesm_ids()

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
sea='DJF'

colors=['k','blue','yellow','orange','red']
x0,y0,x1,y1=70,30,181,101

ofsetfac=[0,0.5,1,2]
xof=[0,-8,8,0,0]
yof=[0,0,0,-8,8]
names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']
refx=dict()
refy=dict()

fig,ax = plt.subplots(figsize=(8,6))

for si,a,m in zip(sim,at,med):
    # load
    if not 'ERA5' in si:
        continue

    m=np.array(m)
    a=np.array(a)

    if np.any(m==None):
        continue
    tra = np.loadtxt(tracks + si + '-new-tracks.txt')

    # store
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    
    loc = np.where(IDs==2)[0]
    ax.plot(t[loc],slp[loc])


fig.savefig('/atmosdyn2/ascherrmann/015-CESM-WRF/images/SLP-evolution.png',dpi=300,bbox_inches='tight')
