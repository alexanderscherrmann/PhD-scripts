import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

track20 = 0


colors = ['dodgerblue','darkgreen','r','saddlebrown']
seas = np.array(['DJF','MAM','JJA','SON'])
pos = np.array(['east','north','south','west'])
amps = np.array([0.7,1.4,2.1])

alpha = [0.333,0.66,1]
marks = ['s','d','o','|']

for se,col in zip(seas,colors):

    fig1=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax1 = fig1.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax1.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    
    fig2=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax2 = fig2.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax2.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    for simid2,sim in enumerate(SIMS):
        simid = simid2+track20
        if "-not" in sim:
            continue
    
        if sim[-4:]=='clim':
            continue
        medid = np.array(MEDIDS[simid])
        atid = np.array(ATIDS[simid])
    
        if medid.size==0:
            continue
    
        sea = sim[:3]
        if se!=sea:
            continue

        tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]
    
        locat,locmed = np.where(IDs==1)[0],np.where(IDs==2)[0] 
        loc = [locat,locmed]
        mark = None
        if '-east-' in sim:
            mark = marks[0]
        if '-north-' in sim:
            mark = marks[1]
        if '-south-' in sim:
            mark = marks[2]
        if '-west-' in sim:
            mark = marks[3]
    
        alp = 1
    
        if '-0.7-QGPV' in sim:
            alp = alpha[0]
        if '-1.4-QGPV' in sim:
            alp = alpha[1]
    
        for q,l in enumerate(loc):
            if q==0:
                ax1.plot(tlon[l],tlat[l],color=col,alpha=alp)
                ax1.scatter(tlon[l],tlat[l],color=col,alpha=alp,marker=mark,s=5)
            else:
                ax2.plot(tlon[l],tlat[l],color=col,alpha=alp)
                ax2.scatter(tlon[l],tlat[l],color=col,alpha=alp,marker=mark,s=5)
    
    
    ax1.set_xlim(-80,30)
    ax1.set_ylim(30,80)
    ax2.set_xlim(-5,50)
    ax2.set_ylim(25,55)
    #ax.set_extent([np.min(LON),np.max(LON),np.min(LAT),np.max(LAT)])
    fig1.savefig(dwrf + 'image-output/at-%s-cyclone-tracks.png'%se,dpi=300,bbox_inches='tight')
    fig2.savefig(dwrf + 'image-output/med-%s-cyclone-tracks.png'%se,dpi=300,bbox_inches='tight')
    plt.close('all')
    print(se)
