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

import cartopy
import matplotlib.gridspec as gridspec

def colbar(cmap,minval,maxval,nlevels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(pvr_levels,cmap.N)
    return newmap, norm

LON = np.round(np.linspace(-180,180,901),1)
LAT = np.round(np.linspace(0,90,226),1)

def find_nearest_grid_point(lon,lat):

    dlon = LON-lon
    dlat = LAT-lat

    lonid = np.where(abs(dlon)==np.min(abs(dlon)))[0][0]
    latid = np.where(abs(dlat)==np.min(abs(dlat)))[0][0]

    return lonid,latid


CT = 'MED'
pload = '/home/ascherrmann/010-IFS/ctraj/MED/use/'
psave = '/home/ascherrmann/010-IFS/'

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

f = open(pload+'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dit = data['dit']

gridmap = dict()
labs = helper.traced_vars_IFS()
pvrs = np.append(np.append(np.append(np.append(np.append(labs[8:],'PVRTOT'),'DELTAPV'),'RES'),'locres'),'PV')

counter = dict()
PVedge=0.75
casecounter = np.zeros((LAT.size,LON.size))
gc = np.zeros((LAT.size,LON.size))
for q, date in enumerate(datadi.keys()):
    idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
    tralon = datadi[date]['lon'][idp]
    tralat = datadi[date]['lat'][idp]
    datadi[date]['DELTAPV'] = np.zeros(datadi[date]['PV'].shape)

    datadi[date]['DELTAPV'][:,1:] = datadi[date]['PV'][:,:-1]-datadi[date]['PV'][:,1:]
    datadi[date]['RES'] = np.zeros(datadi[date]['PV'].shape)

    datadi[date]['PVRTOT'] = np.zeros(datadi[date]['PV'].shape)
    for pv in labs[8:]:
        datadi[date]['PVRTOT']+=datadi[date][pv]

    datadi[date]['RES'][:,1:] = np.cumsum(datadi[date]['PVRTOT'][:,1:],axis=1)-np.cumsum(datadi[date]['DELTAPV'][:,1:],axis=1)
    datadi[date]['locres'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['locres'][:,1:] = datadi[date]['PVRTOT'][:,1:] - datadi[date]['DELTAPV'][:,1:]
    datadi[date]['locres2'] = np.zeros(datadi[date]['PV'].shape)
    datadi[date]['locres2'][:,1:] = abs(datadi[date]['PVRTOT'][:,1:] - datadi[date]['DELTAPV'][:,1:])

    gridmap[date] = dict()
    for pv in np.append('locres2',pvrs):
        gridmap[date][pv] = np.zeros((LAT.size,LON.size))
    counter[date] = np.zeros((LAT.size,LON.size))
    wh ='env'
    for k in range(len(idp)):
        # this is for environmental PV changes only!
        #for l in np.where(dit[date][wh][idp[k]]==1)[0]:#range(len(tralon[0])):
        for l in range(len(tralon[0])):
            lon = tralon[k,l]
            lat = tralat[k,l]
            lonid,latid = find_nearest_grid_point(lon,lat)

            for pv in np.append(pvrs[:-1],'locres2'):
                gridmap[date][pv][latid,lonid]+=datadi[date][pv][idp[k],l]

            gridmap[date]['PV'][latid,lonid]+=datadi[date]['PV'][idp[k],-1]

            counter[date][latid,lonid]+=1
            gc[latid,lonid]+=1

    gridmap[date]['counter'] = counter[date]
    
#    loc = np.where(counter[date]!=0)

#    casecounter[loc]+=1

#    for pv in pvrs:
#        gridmap[date][pv][loc]/=counter[date][loc]

    gridmap[date]['PVRT'] = gridmap[date]['PVRCONVT'] + gridmap[date]['PVRTURBT']
    gridmap[date]['LW'] = gridmap[date]['PVRLWH'] + gridmap[date]['PVRLWC']


alpha=1.
linewidth=.2

minpltlatc = 15 
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50


for pv in np.append(np.append(np.append(np.append('PVRT','counter'),pvrs),'locres2'),'LW'):

    gridmap[pv] = np.zeros((LAT.size,LON.size))

    for q, date in enumerate(datadi.keys()):
        gridmap[pv] += gridmap[date][pv]

#    loc2 = np.where(casecounter!=0)
    loc2 = np.where(gc>10)
#    gridmap[pv][loc2] = gridmap[pv][loc2]/casecounter[loc2]
    gridmap[pv][loc2] = gridmap[pv][loc2]/gc[loc2]

#    loc = np.where(casecounter==0)
    loc = np.where(gc<10)
    gridmap[pv][loc]=np.nan#100
    minv=-0.35
    maxv=0.35
    steps=0.05
    pvr_levels = np.arange(minv,maxv+0.0001,steps)
    ap = plt.cm.BrBG
    cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
    ticklabels=pvr_levels
    if pv=='PV':
        minv=-1.6
        maxv=1.6
        steps=0.2
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='RES':
        minv=-2.0
        maxv=2.0
        steps=0.25
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='counter':
        minv = 0
        maxv = 200
        steps= 10
        pvr_levels = np.arange(minv,maxv+0.0001,steps)
        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels
    if pv=='locres2':
        ap = plt.cm.PuBuGn
        minv = 0
        maxv = 0.5
        steps= 0.05
        pvr_levels = np.arange(minv,maxv+0.0001,steps)

        cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
        ticklabels=pvr_levels


    fig=plt.figure(figsize=(6,4))
    gs = gridspec.GridSpec(ncols=1, nrows=1)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree()) 
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)

    lc=ax.contourf(LON,LAT,gridmap[pv],levels=pvr_levels,cmap=cmap,extend='both',norm=norm)
#    lc.cmap.set_under('saddlebrown')
    if pv=='RES' or pv=='locres':
        lc.cmap.set_over('red')
    
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
    cbar.ax.set_xlabel('PVU/h',fontsize=8)
    if pv=='PV' or pv=='RES' or pv=='locres' or pv=='locres2':
        cbar.ax.set_xlabel('PVU',fontsize=8)
    if pv=='counter':
        cbar.ax.set_xlabel(' ',fontsize=8)
    cbar.ax.set_xticklabels(ticklabels)
    
    figname = psave + pv +'-gridmap-' + '-PVedge-' + str(PVedge) + '-' + 'total' + '.png'
    
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close()


