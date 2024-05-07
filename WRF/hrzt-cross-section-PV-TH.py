import numpy as np
import netCDF4
import argparse
import cartopy
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

import matplotlib.pyplot as plt
import matplotlib
import wrf

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('day',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')

args = parser.parse_args()
d=str(args.day)

pl = '/atmosdyn/era5/cdf/'+ d[:4] + '/' + d[4:6] + '/'
ps = '/atmosdyn2/ascherrmann/012-WRF-cyclones/'

PVlvl = np.arange(-1,10,0.5)
THlvl = np.arange(280,380.1,5)

cmap = matplotlib.cm.jet
norm = plt.Normalize(np.min(PVlvl),np.max(PVlvl))
ticklabels=PVlvl

f = pl + 'S' + d

lon = readcdf(f,'lon')
lat = readcdf(f,'lat')
PV = readcdf(f,'PV')[0]
TH = readcdf(f,'TH')[0]
GHT_300hPa = readcdf(pl + 'H' + d,'Z')[0,17]/9.81


PV_325K = wrf.interplevel(PV,TH,325.0,meta=False)
TH_2PVU = wrf.interplevel(np.flip(TH,axis=0),np.flip(PV,axis=0),2.0,meta=False)


fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
hc=ax.contourf(lon,lat,PV_325K,levels=PVlvl,cmap=cmap,norm=norm,extend='both')
ax.contour(lon,lat,PV_325K,levels=[2,1e5],colors='k',linewidths=0.75,animated='False',alpha=1,zorder=1)

## axis
lonticks=np.arange(np.min(lon),np.max(lon)+0.5,40)
latticks=np.arange(np.min(lat),np.max(lat)+0.5,10)

ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
ax.set_xlim(np.min(lon),np.max(lon))
ax.set_ylim(np.min(lat),np.max(lat))
ax.set_extent([-120,80,10,80], ccrs.PlateCarree())


cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=PVlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
#cbar.ax.set_xticks(ticks=PVlvl[::2])
cbar.ax.set_xlabel('PVU',fontsize=8)
#cbar.ax.set_xticklabels(PVlvl[::2])

sa = ps + d + '-PV-at-325K.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)


####
####  TH @ 2PVU
####
fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

cmap = matplotlib.cm.seismic
norm = plt.Normalize(np.min(THlvl),np.max(THlvl))

hc=ax.contourf(lon,lat,TH_2PVU,levels=THlvl,cmap=cmap,norm=norm,extend='both')

## colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=THlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
#cbar.ax.set_xticks(ticks=THlvl[::2])
cbar.ax.set_xlabel('K',fontsize=8)
#cbar.ax.set_xticklabels(labels=THlvl[::2])
## axis
lonticks=np.arange(np.min(lon),np.max(lon)+0.5,40)
latticks=np.arange(np.min(lat),np.max(lat)+0.5,10)

ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
ax.set_xlim(np.min(lon),np.max(lon))
ax.set_ylim(np.min(lat),np.max(lat))

ax.set_extent([-120,80,10,80], ccrs.PlateCarree())

sa = ps + d + '-TH-at-2PVU.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)


####
####  GHT @ 300 hPa
####

GHTlvl = np.arange(7500,10000,200)
cmap = matplotlib.cm.seismic
norm = plt.Normalize(np.min(GHTlvl),np.max(GHTlvl))

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

hc=ax.contourf(lon,lat,GHT_300hPa,levels=THlvl,cmap=cmap,norm=norm,extend='both')

## colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=THlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
#cbar.ax.set_xticks(ticks=THlvl[::2])
cbar.ax.set_xlabel('K',fontsize=8)
#cbar.ax.set_xticklabels(labels=THlvl[::2])
## axis
lonticks=np.arange(np.min(lon),np.max(lon)+0.5,40)
latticks=np.arange(np.min(lat),np.max(lat)+0.5,10)

ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
ax.set_xlim(np.min(lon),np.max(lon))
ax.set_ylim(np.min(lat),np.max(lat))

ax.set_extent([-120,80,10,80], ccrs.PlateCarree())

sa = ps + d + '-GHT-at-300hPa.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)
