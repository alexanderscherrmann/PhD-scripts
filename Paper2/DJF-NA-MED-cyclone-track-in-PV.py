import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'


# for t=0512

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-05_12:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]


Pres = wrf.getvar(ref,'pressure')
PV = wrf.getvar(ref,'pvo')
MSLP = ref.variables['MSLP'][0,:]
PSlevel = np.arange(975,1031,5)

PV300=wrf.interplevel(PV,Pres,300,meta=False)

track20 = 0


colors = ['dodgerblue','darkgreen','r','saddlebrown']
seas = np.array(['DJF','MAM','JJA','SON'])
pos = np.array(['east','north','south','west'])
amps = np.array([0.7,1.4,2.1])

alpha = [0.333,0.66,1]
marks = ['s','d','o','|']

fig=plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(nrows=1, ncols=3)
ax = fig.add_subplot(gs[0,:2],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
h2 = ax.contourf(LON,LAT,PV300,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)

hc=ax.contour(LON,LAT,MSLP,levels=PSlevel,colors='purple',linewidths=0.5)
plt.clabel(hc,inline=True,fmt='%d',fontsize=6)
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h2, ticks=pvlevels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.005)
fig.canvas.mpl_connect('draw_event', func)

col='dodgerblue'
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'

for simid2,sim in enumerate(SIMS):
    simid = simid2+track20
    if "-not" in sim:
        continue

    if sim!='DJF-clim':
        continue

    name = 'DJF-clim-t0512-PV-cyclone-tracks'

    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])

    if medid.size==0:
        continue

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    locat,locmed = np.where(IDs==1)[0],np.where(IDs==2)[0] 
    loc = [locat,locmed]
#    mark = None
    mark='o'
#    if '-east-' in sim:
#        mark = marks[0]
#    if '-north-' in sim:
#        mark = marks[1]
#    if '-south-' in sim:
#        mark = marks[2]
#    if '-west-' in sim:
#        mark = marks[3]

    alp = 1

#    try:
#        alp = float(sim[-8:-5])/2.8
#    except:
#        alp = 0.05

    for q,l in enumerate(loc):
        if q==0:
            col='grey'
            ax.plot(tlon[l],tlat[l],color=col,alpha=alp)
            ax.scatter(tlon[l],tlat[l],color=col,alpha=alp,marker=mark,s=5)
        else:
            col='dodgerblue'
            ax.plot(tlon[l],tlat[l],color=col,alpha=alp)
            ax.scatter(tlon[l],tlat[l],color=col,alpha=alp,marker=mark,s=5)

    ax2 = fig.add_subplot(gs[0,2])
    ax2.plot(1 + t[locat]/24,slp[locat],color='k')
    ax2.set_yticks(ticks=np.arange(1000,1020,2))

ax.set_xlim(-80,80)
ax.set_ylim(20,80)
ax2.set_aspect('auto')
ax2.set_ylabel('SLP [hPa]')
ax2.set_xlabel('simulation time [d]')
ax2.axvline(132/24,color='grey')
ax2.set_xticks(np.arange(1,10,1))
ax2.set_xlim(1 + t[locat[0]]/24,1 + t[locat[-1]]/24)
plt.subplots_adjust(top=0.4)
#ax2.set_xlim(-10,50)
#ax2.set_ylim(25,55)
#ax.set_extent([np.min(LON),np.max(LON),np.min(LAT),np.max(LAT)])


fig.savefig(pappath + '%s.png'%name,dpi=300,bbox_inches='tight')
plt.close('all')
