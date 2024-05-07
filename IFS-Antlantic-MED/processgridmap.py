import pickle
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


pload = '/home/ascherrmann/010-IFS/data/DEC17/'
f = open(pload + 'process-di.txt','rb')
prodi = pickle.load(f)
f.close()

LAT = np.round(np.linspace(0,90,226),1)
LON = np.round(np.linspace(-180,180,901),1)
bounds = 1./3.
promap = dict()
for date in prodi.keys():
    for ll in prodi[date].keys():
        if ll not in promap.keys():
            promap[ll] = np.array([])
        da = prodi[date][ll]
        
        app = 0
        for q in da:
            if (len(np.where(da==q)[0])/len(da) > bounds):
                app = q
        
        promap[ll] = np.append(promap[ll],app)
        
pmap = np.zeros((LAT.size,LON.size))-100
for ll in promap.keys():
    da = promap[ll]
    lat = int(ll[:3])
    lon = int(ll[-3:])
    app = 0
    for q in da:
        if (len(np.where(da==q)[0])/len(da) > bounds):
            app = q
            
    pmap[lat,lon] = app
    
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
cmap = ListedColormap(['grey','orange','green','saddlebrown','dodgerblue','blue','magenta','salmon','red'])
norm = BoundaryNorm([0,1 ,2, 3, 4, 5, 6,7,8,9], cmap.N)
levels = np.arange(-0.5,9,1)

labs = helper.traced_vars_IFS()
domlabs = labs[8:]    
ticklabels  = np.array([])
for q in np.append('PVRnone',domlabs):
    ticklabels = np.append(ticklabels,q[3:])
    
lc=ax.contourf(LON,LAT,pmap,levels=levels,cmap=cmap,extend='both',norm=norm)
lc.cmap.set_under('white')

maxv = 3000
minv = 800
elv_levels = np.arange(minv,maxv,400)

pload = '/home/ascherrmann/010-IFS/data/DEC17/'
psave = '/home/ascherrmann/010-IFS/'

data = xr.open_dataset(pload + 'IFSORO')
lon = data['lon']
lat = data['lat']
ZB = data['ZB'].values[0,0]
ax.contour(lon,lat,ZB,levels=elv_levels,linewidths=0.5,colors='black')
minpltlatc = 15
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50
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
PVedge = 0.75
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(lc, ticks=levels,cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(' ',fontsize=8)
cbar.ax.set_yticklabels(ticklabels)

figname = psave + 'process-gridmap-test' + '-PVedge-' + str(PVedge) + '.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close()
