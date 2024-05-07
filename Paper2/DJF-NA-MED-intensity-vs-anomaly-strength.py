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

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','rb')
prec = pickle.load(f)
f.close()

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[93,55],[95,56],[140,73],[125,72]]
steps = [[0,0],[4,0],[-4,0],[0,4],[0,-4],[8,0],[-8,0],[0,8],[0,-8]]

names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig,ax = plt.subplots(figsize=(8,6))
legend=np.array(['DJF','MAM','JJA','SON'])
colors=['b','darkgreen','r','saddlebrown','dodgerblue','dodgerblue','royalblue']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['>','^','v','<']

markers=['o','o','o','o','+','x','d']
ls=' '
#for co,ma in zip(colors,markers):
#    ax.plot([],[],ls=ls,marker=ma,color=co)


errorbars = dict()
base = dict()
amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']
pappath='/atmosdyn2/ascherrmann/paper/NA-MED-link/'
for simid, sim in enumerate(SIMS):

    if not sim.startswith('DJF-clim-max-U-at'):
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
    loc = np.where(IDs==1)[0] 

    aminslp = np.min(slp[loc])
    mark='o'
    ax.scatter(maxpv,aminslp,color='grey',marker=mark)
    ax.scatter(maxpv,slpmin,color='blue',marker=mark)

ax.set_ylabel('Cyclone mininmal SLP [hPa]')
ax.set_xlabel('PV anomaly [PVU]')
ax.legend(['NA cyclone','MED cyclone'],loc='upper right')
name = pappath + 'DJF-NA-MED-cyclone-intensity-scatter.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig)


