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
import cartopy
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

LON = np.round(np.linspace(-180,180,721),1)
LAT = np.round(np.linspace(0,90,361),1)

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/'
psave = '/atmosdyn2/ascherrmann/paper/cyc-env-PV/'

f = open(pload+'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
gridmap = dict()
pvrs = ['DELTAPV']

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Eminv = 800
#elevation_levels = np.arange(Eminv,Emaxv,400)
elevation_levels = np.array([Eminv,1600,2400])

for pv in pvrs:
    gridmap[pv] = np.zeros((LAT.size,LON.size))

gridmap['nPV'] = np.zeros((LAT.size,LON.size))


counter = np.zeros((LAT.size,LON.size))
ncounter = np.zeros((LAT.size,LON.size))
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

            if datadi[date]['DELTAPV'][idp[k],l] >=0.15:
                gridmap['DELTAPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                counter[latid,lonid]+=1

            if datadi[date]['DELTAPV'][idp[k],l] <=-0.15:
                gridmap['nPV'][latid,lonid]+=datadi[date]['DELTAPV'][idp[k],l]
                ncounter[latid,lonid]+=1

gridmap['ncounter'] = ncounter
gridmap['counter'] = counter

f = open(psave + 'frequency-PV-data-0.15.txt','wb')
pickle.dump(gridmap,f)
f.close()

f = open(psave + 'frequency-PV-data-0.15.txt','rb')
gridmap = pickle.load(f)
f.close()

counter = gridmap['counter']
ncounter = gridmap['ncounter']
loc = np.where(counter!=0)

gridmap['DELTAPV'][loc]/=np.sum(counter[loc])
gridmap['counter'][loc]=counter[loc]/np.sum(counter[loc])

loc = np.where(counter==0)
gridmap['DELTAPV'][loc]=np.nan
gridmap['counter'][loc]=np.nan

loc = np.where(ncounter!=0)
gridmap['nPV'][loc]/=np.sum(ncounter[loc])
gridmap['ncounter'][loc]=ncounter[loc]/np.sum(ncounter[loc])

loc = np.where(ncounter==0)
gridmap['nPV'][loc]=np.nan
gridmap['ncounter'][loc] = np.nan

alpha=1.
linewidth=.2

minpltlatc = 15 
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50


#for pv in np.append('DETLAPV','nPV'):
for pv in np.append('counter','ncounter'):
    minv=np.nanmin(gridmap[pv])*100
    maxv=np.nanmax(gridmap[pv])*100
    print(minv,maxv,(maxv-minv)/10)
    
    steps=(maxv-minv)/10
    pvr_levels = np.arange(minv,maxv+0.0001,steps)
    gridmap[pv][np.where(gridmap[pv]*100<=pvr_levels[1])]=np.nan
    ap = plt.cm.nipy_spectral
    cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
    ticklabels=pvr_levels

    fig = plt.figure(figsize=(6,3))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7) 
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='k',linewidths=0.35,alpha=1)

    lc=ax.contourf(LON,LAT,gridmap[pv]*100,levels=pvr_levels,cmap=cmap,norm=norm,zorder=1)
    
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
    
    func=resize_colorbar_vert(cbax, ax, pad=0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
#    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_yticklabels(np.append(r'%',np.round(ticklabels[1:],3)))
    cbar.ax.set_xlabel('\%')
    figname = psave + pv +'-gridmap' + '-PVedge-' + str(PVedge) + '.png'
    
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


