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
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[93,55],[95,56],[140,73],[125,72]]
steps = [[0,0],[4,0],[-4,0],[0,4],[0,-4],[8,0],[-8,0],[0,8],[0,-8]]

names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))
colors=['grey','k','dodgerblue','b','k','k','k','k']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['1','d','s','x']
markers=['o','o','o','o','1','d','s','x']
ls=' '


legend=['NA cyclone','Med cyclone','200 km','400 km','east','north','south','west']
amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']
pappath='/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ATSLP = np.array([])
MEDSLP = np.array([])
corratslp = np.array([])
corrmedslp = np.array([])
corru = np.array([])
for simid, sim in enumerate(SIMS):

    if not sim.startswith('DJF-') or 'no-' in sim:
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

#    print(sim)
    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])

    U = wrf.getvar(ic,'U')
    U = (U[:,:,:-1]+U[:,:,1:])/2
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(U,p,300,meta=False)

    maxpv = np.max(pv[20:100,60:130])
    loc = np.where(IDs==1)[0] 

    aminslp = np.min(slp[loc])
    mark='o'



    if sim[4:7]=='200':
       col = col200km[0]
    if sim[4:7]=='400':
        col='b'
    if sim.startswith('DJF-clim'):
        col='k'

    t5=ds(dwrf + sim + '/wrfout_d01_2000-12-05_00:00:00')
    PV=wrf.getvar(t5,'pvo')
    p=wrf.getvar(t5,'pressure')
    pv300=wrf.interplevel(PV,p,300,meta=False)
    lon=wrf.getvar(t5,'lon')[0]
    lat=wrf.getvar(t5,'lat')[:,0]

    lons,lats=np.where((lon>=-10) & (lon<=10))[0],np.where((lat>=30) & (lat<=48))[0]
    lo0,lo1,la0,la1 = lons[0],lons[-1],lats[0],lats[-1]

    avpv = np.mean(pv300[la0:la1+1,lo0:lo1+1][pv300[la0:la1+1,lo0:lo1+1]>=2])


    ax[0].scatter(maxpv,avpv,color='k')
    ax[1].scatter(avpv,slpmin,color='k')
    corru = np.append(corru,maxpv)
    corratslp = np.append(corratslp,avpv)
    corrmedslp = np.append(corrmedslp,slpmin)

#SIMS,ATIDS,MEDIDS = wrfsims.sppt_ids()
#SIMS = np.array(SIMS)
#
#tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
#for simid, sim in enumerate(SIMS):
#
#    if not sim.startswith('DJF-'):
#        continue
#
#    if sim[-3]!='0':
#        continue
#    strings=np.array([])
#    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
#        strings = np.append(strings,float(l[1:]))
#
#    amp = float(strings[-2])
#    sea = sim[:3]
#    if np.any(dist==int(strings[0])):
#        dis = int(strings[0])
#
#    medid = MEDIDS[simid]
#    atid = ATIDS[simid]
#
#    if len(medid)==0 or len(atid)==0:
#        continue
#
#    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
#    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
#    t = tra[:,0]
#    tlon,tlat = tra[:,1],tra[:,2]
#    slp = tra[:,3]
#    IDs = tra[:,-1]
#
#    loc = np.where(IDs==2)[0]
#    slpmin = np.min(slp[loc])
#
#    U = wrf.getvar(ic,'U')
#    U = (U[:,:,:-1]+U[:,:,1:])/2
#    p = wrf.getvar(ic,'pressure')
#
#    pv = wrf.interplevel(U,p,300,meta=False)
#    
#    maxpv = np.max(pv[20:100,60:130])
#    loc = np.where(IDs==1)[0]
#
#    aminslp = np.min(slp[loc])
#    mark='o'
#
#    if sim[4:7]=='200':
#       col = col200km[0]
#    if sim[4:7]=='400':
#        col='b'
#    if sim.startswith('DJF-clim'):
#        col='k'
#
#    if np.any(ENSW==sim[11:16]):
#        qq = np.where(ENSW==sim[11:16])[0][0]
#        mark = enswmarker[qq]
#    
#    ax.scatter(maxpv,aminslp,color='grey')
#    ax.scatter(maxpv,slpmin,color='k')
#
#    corru = np.append(corru,maxpv)
#    corratslp = np.append(corratslp,aminslp)
#    corrmedslp = np.append(corrmedslp,slpmin)

print(pr(corru,corratslp))
print(pr(corru,corrmedslp))
ax[0].set_xticks(ticks=np.arange(40,80,10))
ax[0].set_xticklabels(labels=np.arange(40,80,10),fontsize=14)
#ax[1].set_yticklabels(labels=np.arange(950,1020,10),fontsize=14)
ax[1].set_ylabel('min SLP [hPa]',fontsize=14)
ax[0].set_xlabel('Maximal zonal jet velocity [m s$^{-1}$]',fontsize=14)
legend=['Atlantic','Mediterranean']
plt.subplots_adjust(top=0.8)
#ax.legend(legend,loc='upper right')
name = pappath + 'review-1-PV-streamer-av-PV-scatter.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig)
