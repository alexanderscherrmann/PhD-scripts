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

LON = np.round(np.linspace(-180,180,721),1)
LAT = np.round(np.linspace(-90,90,361),1)

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


CT = 'MED'
pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
psave = '/home/ascherrmann/009-ERA-5/MED/'

f = open(pload+'PV-data-' + 'dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

maxv = 3000
minv = 800
elv_levels = np.arange(minv,maxv,400)

pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/'
elv = xr.open_dataset(pload2 + 'NORO')

ZB = elv['ZB'].values[0]
lon3 = elv['lon']
lat3 = elv['lat']



datadi = data['rawdata']
gridmap = dict()
labs = helper.traced_vars_ERA5MED()
pvrs = np.append('DELTAPV','PV')

for pv in pvrs:
    gridmap[pv] = np.zeros((LAT.size,LON.size))

counter = np.zeros((LAT.size,LON.size))
PVedge=0.75

for q, date in enumerate(datadi.keys()):
    idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
    tralon = datadi[date]['lon'][idp]
    tralat = datadi[date]['lat'][idp]
    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]

    for k in range(len(idp)):
        for l in range(len(tralon[0])):
            lon = tralon[k,l]
            lat = tralat[k,l]
            lonid,latid = find_nearest_grid_point(lon,lat)

            for pv in pvrs[:-1]:
                gridmap[pv][latid,lonid]+=datadi[date][pv][idp[k],l]

            gridmap['PV'][latid,lonid]+=datadi[date]['PV'][idp[k],-1]

            counter[latid,lonid]+=1

gridmap['counter']=counter
loc = np.where(counter!=0)

for pv in pvrs:
    gridmap[pv][loc]/=counter[loc]

loc=np.where(counter==0)
for pv in pvrs:
    gridmap[pv][loc]-=100

alpha=1.
linewidth=.2

minpltlatc = 15 
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50

for pv in np.append('counter',pvrs):
    minv=-0.35
    maxv=0.35
    steps=0.05
    pvr_levels = np.arange(minv,maxv+0.0001,steps)
    ap = plt.cm.BrBG
    cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
    ticklabels=pvr_levels
    if pv=='PV':
        minv=-0.3
        maxv=1.5
        steps=0.1
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='counter':
        minv = 0
        maxv = 800
        steps= 20
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()
    
    lc=ax.contourf(LON,LAT,gridmap[pv],levels=pvr_levels,cmap=cmap,extend='both',norm=norm)
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
    ax.contour(lon3,lat3,ZB,levels=elv_levels,linewidths=0.5,colors='black',alpha=0.4)
    func=resize_colorbar_vert(cbax, ax, pad=0.01, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU/h',fontsize=8)
    if pv=='PV' or pv=='RES' or pv=='locres':
        cbar.ax.set_xlabel('PVU',fontsize=8)
    if pv=='counter':
        cbar.ax.set_xlabel(' ',fontsize=8)
    cbar.ax.set_xticklabels(ticklabels)
    
    figname = psave + 'ERA5-' + pv +'-gridmap' + '-PVedge-' + str(PVedge) + '.png'
    
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


