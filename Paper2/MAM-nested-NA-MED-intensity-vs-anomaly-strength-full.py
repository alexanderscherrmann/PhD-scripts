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

fig,ax = plt.subplots(figsize=(8,6))
figg,axx = plt.subplots(figsize=(8,6))

colors=['grey','k']#,'dodgerblue','b','k','k','k','k']
col200km = ['deepskyblue','limegreen','lightcoral','peru']
ENSW = np.array(['east-','north','south','west-'])
enswmarker = ['1','d','s','x']
markers=['o','o','o','o','1','d','s','x']
ls=' '
for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)

legend=['NA cyclone','Med cyclone']#,'200 km','400 km','east','north','south','west']
errorbars = dict()
base = dict()
amps = [0.7, 1.4, 2.1, 2.8, 3.5, 4.2,0.9,1.7,1.1,0.3,0.5]
dist = np.array([200, 400])
var = ['slp','t-genesis','pvampl','precip']
pappath='/atmosdyn2/ascherrmann/paper/NA-MED-link/'
ATSLP = np.array([])
MEDSLP = np.array([])
bigcorrmslp = np.array([])
bigcorraslp = np.array([])
bigcorrpv = np.array([])
for simid, sim in enumerate(SIMS):

    if not sim.startswith('MAM-'):
        continue
    if 'not' in sim:
        continue

    if sim=='MAM-clim':
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

    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    if loc.size==0:
        continue
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
    deepening = np.max(slp[loc[:-4]]-slp[loc[4:]])*2/(24*np.sin(np.mean(tlat[np.argmin(slp[loc])])/180*np.pi)/np.sin(np.pi/3))
    aminslp = np.min(slp[loc])
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
 
    ax.scatter(maxpv,aminslp,color='grey')
    axx.scatter(deepening,slpmin)
#    ax.scatter(maxpv,aminslp,color='grey',marker=mark)
#    ATSLP  =np.append(ATSLP,aminslp)
#    MEDSLP = np.append(MEDSLP,slpmin)
    ax.scatter(maxpv,slpmin,color='k')
#    ax.scatter(maxpv,slpmin,color=col,marker=mark)

    bigcorrmslp = np.append(bigcorrmslp,slpmin)
    bigcorraslp = np.append(bigcorraslp,aminslp)
    bigcorrpv = np.append(bigcorrpv,maxpv)
    if sim=='DJF-redo-clim-1.4':
        ax.scatter(maxpv,slpmin,color='k',marker=mark)
#        ax.scatter(maxpv,aminslp,color='orange',marker=mark,s=2)
#        ax.scatter(maxpv,slpmin,color='cyan',marker=mark,s=2)
    if sim=='DJF-clim-max-U-at-300-hPa-1.4-QGPV' or sim=='DJF-redo-clim-1.4':
        print(sim,maxpv,slpmin)


SIMS,ATIDS,MEDIDS01,MEDIDS02 = wrfsims.nested_ids()
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'
for dom,MEDIDS in zip(['01','02'],[MEDIDS01,MEDIDS02]):
 for simid, sim in enumerate(SIMS):
    if sim=='nested-test':
        continue
    if 'DJF-nested' in sim:
        continue
    if 'DJF' in sim:
        continue
    
    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])
    if medid.size==0 or atid.size==0:
        continue

    tra = np.loadtxt(tracks + sim + '-' + dom + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    loc = np.where(IDs==2)[0]
    if loc.size==0:
        continue
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
    bigcorrmslp = np.append(bigcorrmslp,slpmin)
    bigcorraslp = np.append(bigcorraslp,aminslp)
    bigcorrpv = np.append(bigcorrpv,maxpv)
    if dom=='01':
        ax.scatter(maxpv,slpmin,color='k',marker=mark,zorder=100)
#        ax.scatter(maxpv,slpmin,color='purple',marker=mark,zorder=100)
        print(maxpv,slpmin)
    else:
        ax.scatter(maxpv,slpmin,color='k',marker=mark)
#        ax.scatter(maxpv,slpmin,color='red',marker=mark)

ax.set_ylabel('Cyclone mininmal SLP [hPa]')
ax.set_xlabel('max PV at 300 hPa [PVU]')

#print(pr(ATSLP,MEDSLP))
ax.legend(legend,loc='upper right')

print('all correlation')
print(pr(bigcorrpv,bigcorrmslp))
print(pr(bigcorrpv,bigcorraslp))
print(pr(bigcorraslp,bigcorrmslp))

#ax.legend(['NA cyclone','MED cyclone'],loc='upper right')
name = pappath + 'nested-MAM-NA-MED-cyclone-intensity-max-PV-full-onecolor.png'
#name = pappath + 'nested-DJF-NA-MED-cyclone-intensity-max-PV-full.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig)


axx.set_xlabel('max deepening in 12 h [bergeron]')
axx.set_ylabel('Med cyclone intensity [hPa]')
figg.savefig(pappath + 'nested-MAM-NA-deepning-MED-cyclone-intensity.png',dpi=300,bbox_inches='tight')
plt.close(figg)


fig,ax = plt.subplots(figsize=(8,6))
corrslp = np.array([])
corrpv = np.array([])
corraslp =np.array([])
ax.plot([],[],color='grey',marker='o',ls=' ')
ax.plot([],[],color='k',marker='o',ls=' ')
legend=['NA cyclone','MED cyclone']
for dom,MEDIDS in zip(['01','02'],[MEDIDS01,MEDIDS02]):
 for simid, sim in enumerate(SIMS):
    if sim=='nested-test':
        continue
    if 'DJF-nested' in sim:
        continue
    if 'DJF' in sim:
        continue

    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])
    if medid.size==0:
        continue
    tra = np.loadtxt(tracks + sim + '-' + dom + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    loc = np.where(IDs==2)[0]
    if loc.size==0:
        continue
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
    if dom=='01':
        continue
    else:
        corraslp = np.append(corraslp,aminslp)
        corrslp = np.append(corrslp,slpmin)
        corrpv = np.append(corrpv,maxpv)
        ax.scatter(maxpv,slpmin,color='red',marker=mark)

ax.set_ylabel('Cyclone mininmal SLP [hPa]')
ax.set_xlabel('max PV at 300 hPa [PVU]')

print(pr(corrpv,corrslp))
print(pr(corrpv,corraslp))
print(pr(corraslp,corrslp))
#print(pr(ATSLP,MEDSLP))
ax.legend(legend,loc='upper right')

#ax.legend(['NA cyclone','MED cyclone'],loc='upper right')
name = pappath + 'nested-only-MAM-NA-MED-cyclone-intensity-max-PV-full.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig)
