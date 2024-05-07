import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from scipy.stats import pearsonr as pr

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[93,55],[95,56],[140,73],[125,72]]
steps = [[0,0],[4,0],[-4,0],[0,4],[0,-4],[8,0],[-8,0],[0,8],[0,-8]]

names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

colors=['grey','b','dodgerblue','b','k','k','k','k']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['1','d','s','x']
markers=['o','o','o','o','1','d','s','x']
ls=' '
fig,ax=plt.subplots(figsize=(8,6))
for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)


legend=['NA cyclone','Med cyclone','200 km','400 km','east','north','south','west']
errorbars = dict()
base = dict()
amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']
pappath='/atmosdyn2/ascherrmann/paper/NA-MED-link/'

for simid, sim in enumerate(SIMS):

    if not sim.startswith('DJF-'):
        continue

    if sim=='DJF-clim':
        continue
    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-1])
    sea = sim[:3]
    if np.any(dist==int(strings[0])):
        dis = int(strings[0])

    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    if len(medid)==0 or len(atid)==0:
        continue

    print(sim)
    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos)*len(steps))
    for q,l in enumerate(PVpos):
        for qq,st in enumerate(steps):
            maxpv[q*len(PVpos)+qq] = pv[l[1]+st[1],l[0]+st[0]]

    maxpv = np.max(maxpv)
    loc = np.where(IDs==2)[0] 

    deepening = (slp[loc[0]]-slp[loc[4]])*2/(24*np.sin(np.mean(tlat[np.argmin(slp[loc])])/180*np.pi)/np.sin(np.pi/3))
    mark='o'
    if sim[4:7]=='200':
       col = col200km[0]
    if sim[4:7]=='400':
        col='b'
    if sim.startswith('DJF-clim'):
        col='k'

    if np.any(ENSW==sim[11:16]):
        qq = np.where(ENSW==sim[11:16])[0][0]
        mark = enswmarker[qq]
 
    ax.scatter(maxpv,deepening,color='grey') 

SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
SIMS = np.array(SIMS)

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
for simid, sim in enumerate(SIMS):

    if not sim.startswith('DJF-'):
        continue

    if sim[-3]!='0':
        continue
    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-2])
    sea = sim[:3]
    if np.any(dist==int(strings[0])):
        dis = int(strings[0])

    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    if len(medid)==0 or len(atid)==0:
        continue

    print(sim)
    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos)*len(steps))
    for q,l in enumerate(PVpos):
        for qq,st in enumerate(steps):
            maxpv[q*len(PVpos)+qq] = pv[l[1]+st[1],l[0]+st[0]]

    maxpv = np.max(maxpv)
    loc = np.where(IDs==2)[0]

    deepening = (slp[loc[0]]-slp[loc[4]])*2/(24*np.sin(np.mean(tlat[np.argmin(slp[loc])])/180*np.pi)/np.sin(np.pi/3))
    mark='o'

    if sim[4:7]=='200':
       col = col200km[0]
    if sim[4:7]=='400':
        col='b'
    if sim.startswith('DJF-clim'):
        col='k'

    if np.any(ENSW==sim[11:16]):
        qq = np.where(ENSW==sim[11:16])[0][0]
        mark = enswmarker[qq]

    ax.scatter(maxpv,deepening,color='k')

ax.set_ylabel('max Med cyclone deepening [hPa]')
ax.set_xlabel('max PV at 300 hPa [PVU]')

ax.legend(legend,loc='upper right')

name = pappath + 'DJF-MED-cyclone-deepening-all-sppt.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig)

