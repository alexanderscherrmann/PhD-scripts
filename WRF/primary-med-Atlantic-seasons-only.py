import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt

SIMS,ATIDS,MEDIDS = wrfsims.primary_cyclone_ids()
SIMS = np.array(SIMS)

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

lon1,lat1,lon2,lat2 = -8,20,1.5,40
lon3,lat3,lon4,lat4 = 1.5,20,50,48

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[52,28],[145,84],[93,55],[55,46],[84,59],[73,59]]
names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])

fig1,ax = plt.subplots(figsize=(8,6))
fig2,ax2 = plt.subplots(figsize=(8,6))
fig3,ax3 = plt.subplots(figsize=(8,6))
fig4,ax4 = plt.subplots(figsize=(8,6))
fig5,ax5 = plt.subplots(figsize=(8,6))
legend=['DJF','MAM','JJA','SON']
colors=['b','seagreen','r','saddlebrown','dodgerblue','dodgerblue','royalblue']
markers=['o','o','o','o','+','x','d']
ls=' '
for co,ma in zip(colors,markers):
    ax.plot([],[],ls=ls,marker=ma,color=co)
    ax2.plot([],[],ls=ls,marker=ma,color=co)
    ax3.plot([],[],ls=ls,marker=ma,color=co)
    ax4.plot([],[],ls=ls,marker=ma,color=co)
    ax5.plot([],[],ls=ls,marker=ma,color=co)
import re

for simid, sim in enumerate(SIMS):

    if sim[4:12]!='clim-max':
        continue

    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))
    if strings.size==2 and strings[0]<300:
        strings = np.append(200.,strings)
    if strings.size==2 and strings[0]==300:
        strings = np.append(strings,0.0)

    uppano = strings[1]
    lowano = strings[2]
    lowsize = strings[0]

    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    if len(medid)==0:
        continue

    ic = ds(dwrf + sim + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + sim + '-filter.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    deepening = tra[:,-2]
    IDs = tra[:,-1]
    minslp=np.array([])
    isd = np.array([])
    genesis = np.array([]) 
    medmt = np.array([])
    for mei in medid:
        loc = np.where((IDs==mei) & (t<192))[0]
        minslp = np.append(minslp,np.min(slp[loc]))
        genesis = np.append(genesis,np.min(t[loc]))
        medmt = np.append(medmt,t[loc[np.argmin(slp[loc])]])

    slpmin = np.min(minslp)
    genesis = np.min(genesis)
    medmt = medmt[np.argmin(minslp)]

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(len(PVpos))
    for q,l in enumerate(PVpos):
        maxpv[q] = pv[l[1],l[0]]

    maxpv = np.max(maxpv)
    mark='o'
    col = 'k'
    if sim.startswith('DJF-clim'):
        col = 'b'
    if sim.startswith('MAM-clim'):
        col = 'seagreen'
    if sim.startswith('JJA-clim'):
        col = 'r'
    if sim.startswith('SON-clim'):
        col = 'saddlebrown'
    if sim.startswith('DJF-clim-double'):
        mark='+'
        col='dodgerblue'
    if sim.startswith('DJF-L-300'):
        mark='x'
        col='dodgerblue'
    if sim.startswith('DJF-L-500'):
        mark='d'
        col='royalblue'
    if sim.startswith('DJF-L-500-double-0.0') or sim.startswith('DJF-L-800-double-0.0'):
        mark='s'
        col='grey'

    ax.scatter(maxpv,slpmin,color=col,marker=mark)
    ax.text(maxpv,slpmin,'%.1f'%(uppano))
    

    aslp = np.array([])
    atdeep = np.array([])
    atmt = np.array([])
    for ai in atid:
        loc = np.where(IDs==ai)[0]
        aslp = np.append(aslp,np.min(slp[loc]))
        atdeep = np.append(atdeep,np.nanmin(deepening))
        atmt = np.append(atmt,t[loc[np.argmin(slp[loc])]])

    
    atdeep = np.min(atdeep)
    aminslp = np.min(aslp)
    atmt = atmt[np.argmin(aslp)]
    print(medmt,atmt)
    ax3.scatter(aminslp,genesis,color=col,marker=mark)

    ax2.scatter(aminslp,slpmin,color=col,marker=mark)
    ax2.text(aminslp,slpmin,'%.1f'%(uppano))
    
    ax4.scatter(atdeep,slpmin,color=col,marker=mark)

    ax5.scatter(atmt,medmt,color=col,marker=mark)
    ax5.text(atmt,medmt,'%.1f'%(uppano))
    
ax.set_xlabel('PV anomaly [PVU]')
ax.set_ylabel('MED cyclone min SLP [hPa]')
ax.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-MED-cyclone-PV-anomaly-SLP-scatter-min-of-all-tracks-in-MED.png'
fig1.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig1)


ax2.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax2.set_ylabel('MED cyclone min SLP [hPa]')
ax2.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-Atlantic-MED-cyclone-SLP-scatter-min-of-all-tracks-in-MED.png'
fig2.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig2)


#ax3.set_xlabel('MED cyclone genesis lon [$^{\circ}$]')
#ax3.set_ylabel('PV anomaly [PVU]')
ax3.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax3.set_ylabel('Med cyclone genesis time [h]')
ax3.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-MED-cyclone-genesis-time-scatter-min-of-all-tracks-in-MED.png'
fig3.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig3)


ax4.set_xlabel('Atlantic cyclone strongest deepening [hPa (6 h)$^{-1}$]')
ax4.set_ylabel('Med cyclone min SLP [hPa]')
ax4.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-MED-cyclone-Atlantic-deepening-6h-scatter-min-of-all-tracks-in-MED.png'
fig4.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig4)


ax5.set_xlabel('time of mature stage of Atlantic cyclone [h]')
ax5.set_ylabel('time of mature stage of Mediterranean cyclone [h]')
ax5.legend(legend,loc='upper right')
name = dwrf + 'image-output/seasons-MED-cyclone-mature-time-min-of-all-tracks-in-MED.png'
fig5.savefig(name,dpi=300,bbox_inches='tight')
plt.close(fig5)

