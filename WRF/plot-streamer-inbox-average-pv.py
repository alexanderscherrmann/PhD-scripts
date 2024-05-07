import pickle
import numpy as np

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/all-dates-streamer-boxes-pv.txt','rb')
da = pickle.load(f)
f.close()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from wrf import interplevel as intp
from netCDF4 import Dataset as ds

import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

lon = np.linspace(-120,80,401)
lat = np.linspace(10,80,141)

minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
cmap,pv_levels,norm,ticklabels=PV_cmap2()

for k in da.keys():
    if k=='n':
        continue
    djf = np.zeros((len(lat),len(lon)))
    counter=0
    for d in da[k]:
        if d=='n':
            continue
        if int(d[4:6])==12 or int(d[4:6])==1 or int(d[4:6])==2:
            djf+=da[k][d]
            counter+=1

    djf/=counter

    fig = plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
    ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

    h = ax.contourf(lon,lat,djf,cmap=cmap,levels=pv_levels,norm=norm,extend='both')
    lonticks=np.arange(minlon, maxlon+1,20)
    latticks=np.arange(minlat, maxlat+1,10)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks[:-1].astype(int),fontsize=10)
    ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=pv_levels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_xlabel('PVU',fontsize=10)
    cbar.ax.set_xticklabels(ticklabels)

    name = 'PV-315K-' + k + '.png'
    fig.savefig(pi + name,dpi=300,bbox_inches='tight')
    plt.close('all')

