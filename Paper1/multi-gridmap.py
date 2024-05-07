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
pload = '/atmosdyn2/ascherrmann/010-IFS/ctraj/MED/use/'
psave = '/atmosdyn2/ascherrmann/010-IFS/'

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 1200
Eminv = 800
#elevation_levels = np.arange(Eminv,Emaxv,400)
elevation_levels = np.array([Eminv,1600,2400])

f = open(pload+'PV-data-' + CT + 'dPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

datadi = data['rawdata']
dit = data['dit']

gridmap = dict()
labs = helper.traced_vars_IFS()

counter = dict()
PVedge=0.75
casecounter = np.zeros((LAT.size,LON.size))
gc = np.zeros((LAT.size,LON.size))
pvrs = ['PVRCONVT','PVRLS','PVRTURBT','PVRTURBM','PVRLWH','PVRLWC']

abscounter =0
localcounter = np.zeros((LAT.size,LON.size))
for q, date in enumerate(datadi.keys()):
    idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
    tralon = datadi[date]['lon'][idp]
    tralat = datadi[date]['lat'][idp]

    abscounter+= datadi[date]['PV'].shape[0] * (datadi[date]['PV'].shape[1]-1)
    gridmap[date] = dict()
    for pv in pvrs:
        gridmap[date][pv] = np.zeros((LAT.size,LON.size))

    counter[date] = np.zeros((LAT.size,LON.size))
    for k in range(len(idp)):
        for l in range(len(tralon[0])):
            lon = tralon[k,l]
            lat = tralat[k,l]
            lonid,latid = find_nearest_grid_point(lon,lat)

            localcounter[latid,lonid]+=1
            for pv in pvrs:
                if datadi[date][pv][k,l]>=0.15 or datadi[date][pv][k,l]<=-0.15:
                    gridmap[date][pv][latid,lonid]+=1

alpha=1.
linewidth=.2

minpltlatc = 25 
minpltlonc = -20

maxpltlatc = 60
maxpltlonc = 50

fig=plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(nrows=2, ncols=2)

gax = fig.add_subplot(111, frameon=False)
gax.set_xticks(ticks=[])
gax.set_yticks(ticks=[])

axes = []
for k in range(2):
  for l in range(2):
    axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))

maxv=0
grid = np.zeros((LAT.size,LON.size))
for pv in pvrs[:]:
    gridmap[pv] = np.zeros((LAT.size,LON.size))
    for q, date in enumerate(datadi.keys()):
        gridmap[pv] += gridmap[date][pv]

    grid+=gridmap[pv]

    loc = np.where(localcounter!=0)
#    gridmap[pv][loc] /= localcounter[loc]
#    gridmap[pv] *= 100
    loc = np.where(gridmap[pv]==0)
    gridmap[pv][loc] = np.nan
    tmpmax = np.nanmax(gridmap[pv])
    if tmpmax>maxv:
        maxv=tmpmax

loc = np.where(localcounter!=0)


loc = np.where(grid==0)
grid[loc] = np.nan

pvr_levels = np.array([0,50,100,150,200,250,300,350,400])
cmap= plt.cm.YlGnBu
norm = BoundaryNorm(pvr_levels,cmap.N)
ticklabels=pvr_levels
for ax,pv,labels in zip(axes,pvrs[:6],['(a)','(b)','(c)','(d)','(e)','(f)']):

    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)

    lc=ax.contourf(LON,LAT,gridmap[pv],levels=pvr_levels,cmap=cmap,extend='max',norm=norm,zorder=1)

    lonticks=np.arange(minpltlonc, maxpltlonc,10)
    latticks=np.arange(minpltlatc, maxpltlatc,10)

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#    ax.set_aspect('auto')
    if labels=='(b)':
        ax.text(0.75,0.90,'MP',transform=ax.transAxes,fontsize=12)
    else:
        ax.text(0.75,0.90,pv[3:],transform=ax.transAxes,fontsize=12)
    ax.text(0.02,0.90,labels,transform=ax.transAxes,fontsize=14,)

cax=plt.axes([0.775,0.1,0.01,0.405])
plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.505)
cbar = fig.colorbar(lc,ticks=pvr_levels,cax=cax)
cbar.ax.set_yticklabels(labels=ticklabels)
psave='/home/ascherrmann/publications/cyclonic-environmental-pv/'

figname = psave + 'fig13.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')

#grid[loc] /= localcounter[loc]
#grid*=100
#loc = np.where(grid==0)
#grid[loc] = np.nan
#
#
#fig=plt.figure(figsize=(10,8))
#gs = gridspec.GridSpec(nrows=2, ncols=2)
#
#gax = fig.add_subplot(111, frameon=False)
#gax.set_xticks(ticks=[])
#gax.set_yticks(ticks=[])
#
#axes = []
#for k in range(2):
#  for l in range(2):
#    axes.append(fig.add_subplot(gs[k,l],projection=ccrs.PlateCarree()))
#
#maxv=70
#minv = 0
#steps =maxv/7
#print(minv,maxv,steps)
#pvr_levels = np.arange(minv,maxv+0.0001,steps)
#ap = plt.cm.YlGnBu
#cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels))
#ticklabels=np.append(r'%',np.round(pvr_levels[1:],1))
#
#for ax,pv,labels in zip(axes,pvrs[:6],['(a)','(b)','(c)','(d)','(e)','(f)']):
#
#    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#
#    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)
#
#    lc=ax.contourf(LON,LAT,gridmap[pv],levels=pvr_levels,cmap=cmap,extend='max',norm=norm,zorder=1)
#    
#    lonticks=np.arange(minpltlonc, maxpltlonc,10)
#    latticks=np.arange(minpltlatc, maxpltlatc,10)
#    
#    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
##    ax.set_aspect('auto')
#    if labels=='(b)':
#        ax.text(0.75,0.90,'MP',transform=ax.transAxes,fontsize=12)
#    else:
#        ax.text(0.75,0.90,pv[3:],transform=ax.transAxes,fontsize=12)
#    ax.text(0.02,0.90,labels,transform=ax.transAxes,fontsize=14)
#
#cax=plt.axes([0.775,0.1,0.01,0.4])
#plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
#cbar = fig.colorbar(lc,ticks=pvr_levels,cax=cax)
#cbar.ax.set_yticklabels(labels=ticklabels)
#psave='/atmosdyn2/ascherrmann/paper/cyc-env-PV/'
#figname = psave + 'important-positive-frequency-multi-gridmap-' + '-PVedge-' + str(PVedge) + '-' + 'total' + '.png'
##fig.savefig(figname,dpi=300,bbox_inches="tight")
#plt.close('all')
#
#fig=plt.figure(figsize=(5,4))
#gs = gridspec.GridSpec(nrows=1, ncols=1)
#
#ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
#ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)
#lc=ax.contourf(LON,LAT,grid,levels=pvr_levels,cmap=cmap,extend='max',norm=norm,zorder=1)
#ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
#cax=plt.axes([0.775,0.1,0.01,0.4])
#plt.subplots_adjust(wspace=0,hspace=0,right=0.775,bottom=0.1,top=0.5)
#cbar = fig.colorbar(lc,ticks=pvr_levels,cax=cax)
#cbar.ax.set_yticklabels(labels=ticklabels)
#psave='/atmosdyn2/ascherrmann/paper/cyc-env-PV/'
#figname=psave + 'all-positive-changes-map.png'
##fig.savefig(figname,dpi=300,bbox_inches="tight")
#plt.close('all')
#
