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
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'


fig,ax=plt.subplots(figsize=(8,6))
fig1=plt.figure(figsize=(8,6))
fig2=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax1=fig1.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax2=fig2.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax1.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
ax1.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax2.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
ax2.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
ref = dict()
ref['lat'] = dict()
ref['lon'] = dict()
AMPs = np.array([0.7,1.4,2.1])
cols = ['navy','r','purple']
for simid, sim in enumerate(SIMS):

    if sim!='SON-clim-max-U-at-300-hPa-0.7-QGPV' and sim!='SON-clim-max-U-at-300-hPa-1.4-QGPV' and sim!='SON-clim-max-U-at-300-hPa-2.1-QGPV':
        continue

    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-1])
    
    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])
#    ref['lat'][amp] = tlat[loc[np.argmin(slp[loc])]]
#    ref['lon'][amp] = tlon[loc[np.argmin(slp[loc])]]
    col=cols[np.where(AMPs==amp)[0][0]]
    ax1.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color=col)
    loc = np.where(IDs==1)[0]

    aminslp = np.min(slp[loc])
    ax1.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color=col)
    mark='o'

    ax.scatter(aminslp,slpmin,color='r')
    #ax1.scatter(0,0,marker='o',color='r',zorder=10)

sppt,ATIDS,MEDIDS=wrfsims.sppt_ids()
AMPs = np.array([0.7,1.4,2.1])
cols = ['dodgerblue','lightcoral','orchid']
for simid, sim in enumerate(sppt):
    if not 'SON' in sim:
        continue

    strings=np.array([])
    for l in re.findall(r"[-+]?(?:\d*\.\d+|\d+)",sim):
        strings = np.append(strings,float(l[1:]))

    amp = float(strings[-2])
    sea = sim[:3]

    medid = MEDIDS[simid]
    atid = ATIDS[simid]

    print(sim)
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])
    
    col=cols[np.where(AMPs==amp)[0][0]]
    ax1.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color=col)
    loc = np.where(IDs==1)[0]

    aminslp = np.min(slp[loc])
    ax1.scatter(tlon[loc[np.argmin(slp[loc])]],tlat[loc[np.argmin(slp[loc])]],color=col)
    mark='o'
    
    ax.scatter(aminslp,slpmin,color='grey')


ax.set_xlabel('Atlantic cyclone min SLP [hPa]')
ax.set_ylabel('Mediterranean cyclone min SLP [hPa]')
ax.set_xlim(960,990)
ax.set_ylim(985,1010)
fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/sppt-SON-NA-MED-SLP.png',dpi=300,bbox_inches='tight')
plt.close(fig)


#ax1.set_xlabel('relative mature stage longitude [$^{\circ}$]')
#ax1.set_ylabel('relative mature stage latitude [$^{\circ}$]')
ax1.set_xlim(-80,50)
ax1.set_ylim(20,80)
fig1.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/sppt-SON-MED-mature-position.png',dpi=300,bbox_inches='tight')
plt.close(fig1)

#ax2.set_xlim(-120,10)
#ax2.set_ylim(20,80)
#fig2.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/sppt-NA-mature-position.png',dpi=300,bbox_inches='tight')
#plt.close(fig2)
