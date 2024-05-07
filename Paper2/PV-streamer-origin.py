import numpy as np
import argparse
from netCDF4 import Dataset as ds
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
from wrf import getvar, interplevel
import sys
sys.path.append('/home/ascherrmann/scripts/')

import helper
import wrfsims

SIMS,ATIDS,MEDIDS=wrfsims.upper_ano_only()

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1,ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

simc = ['200-km-west','400-km-west','200-km-east','400-km-east','200-km-north','400-km-north','200-km-south','400-km-south','clim-max']
ticklabels=['2W','4W','2E','4E','2N','4N','2S','4S','C']
lcol = ['cyan','navy','orange','red','grey','k','plum','purple','saddlebrown']
pappath='/atmosdyn2/ascherrmann/paper/NA-MED-link/'
for sim in SIMS:
    if not 'DJF' in sim:
        continue
    if not '1.4' in sim:
        continue
    if 'check' in sim:
        continue
    if sim[-4:]=='clim' or 'AT' in sim:
        continue

    for q,si in enumerate(simc):
        if si in sim:
            col=lcol[q]

    p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

    d = np.loadtxt(p + 'trajectories.ll',skiprows=5)
#    d = np.loadtxt(p + 'trace.ll',skiprows=5)

#    with open(p + 'trace.ll') as f:
    with open(p + 'trajectories.ll') as f:
        fl = next(f)
    f.close()

    H = int(int(fl[-9:-5])/60/3)+1
    #H = int(int(fl[-9:-5])/60)+1
    
    t = d[:,0].reshape(-1,H)
    lon = d[:,1].reshape(-1,H)
    lat = d[:,2].reshape(-1,H)
#    pv = d[:,4].reshape(-1,H)

#    sel=  np.where(pv[:,0]>=4)[0]
#    t=t[sel]
#    lon=lon[sel]
#    lat=lat[sel]
#    pv=pv[sel]

    ax.scatter(lon[:,-1],lat[:,-1],color=col,s=1)

lonticks=np.arange(-120,81,40)
latticks=np.arange(10,81,10)

ax.set_xticks(ticks=lonticks, crs=ccrs.PlateCarree())
ax.set_yticks(ticks=latticks, crs=ccrs.PlateCarree())

ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)

ax.set_extent([-120,80,10,80], ccrs.PlateCarree())

figname = pappath + 'streamer-origin.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')



