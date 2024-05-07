import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

import xarray as xr
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

CT = 'MED'
pload = '/home/ascherrmann/010-IFS/data/DEC17/'
psave = '/home/ascherrmann/010-IFS/'
maxv = 3000
minv = 800

pvr_levels = np.arange(minv,maxv,200)
data = xr.open_dataset(pload + 'IFSORO')
ap = plt.cm.BrBG
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))

alpha=1.
linewidth=.2
ticklabels=pvr_levels

minpltlatc = 25 
minpltlonc = -10

maxpltlatc = 50
maxpltlonc = 50

lon = data['lon']
lat = data['lat']
ZB = data['ZB'].values[0,0]

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()

lc=ax.contourf(lon,lat,ZB,levels=pvr_levels,cmap=cmap,extend='both',norm=norm)
lc.cmap.set_under('white')

lonticks=np.arange(minpltlonc, maxpltlonc,5)
latticks=np.arange(minpltlatc, maxpltlatc,5)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('m',fontsize=8)
cbar.ax.set_xticklabels(ticklabels)

figname = psave + 'elevation-contours.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close()


