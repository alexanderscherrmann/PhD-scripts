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

PVlvl = np.arange(-2,2.1,0.25)
THlvl = np.arange(-20,20.1,2)
GHTlvl = np.arange(-300,300.1,50)

cmap = matplotlib.cm.BrBG
norm = plt.Normalize(np.min(PVlvl),np.max(PVlvl))
ticklabels=PVlvl

### wrf reference data
data = netCDF4.Dataset('/home/ascherrmann/scripts/WRF/wrf-mean-reference')
PVwrf = wrf.getvar(data,'pvo',meta=False) #in PVU already
lonwrf = wrf.getvar(data,'lon',meta=False)[0]-0.25
latwrf = wrf.getvar(data,'lat',meta=False)[:,0]-0.25


THwrf = wrf.getvar(data,'T',meta=False) + 300
GHTwrf = wrf.getvar(data,'geopotential',meta=False)/9.81
Pwrf = wrf.getvar(data,'pressure',meta=False)


f = pl + 'S' + d

lon = readcdf(f,'lon')
lat = readcdf(f,'lat')

loi = np.where((lon>=np.min(lonwrf)) & (lon<=np.max(lonwrf)))[0]
lai = np.where((lat>=np.min(latwrf)) & (lat<=np.max(latwrf)))[0]
lo0,lo1,la0,la1 = loi[0],loi[-1],lai[0],lai[-1]

PV = readcdf(f,'PV')[0]
TH = readcdf(f,'TH')[0]

GHT_250hPa = readcdf(pl + 'H' + d,'Z')[0,16,la0:la1+1,lo0:lo1+1]/9.81
GHT_500hPa = readcdf(pl + 'H' + d,'Z')[0,21,la0:la1+1,lo0:lo1+1]/9.81
GHT_850hPa = readcdf(pl + 'H' + d,'Z')[0,30,la0:la1+1,lo0:lo1+1]/9.81

PV_325K = wrf.interplevel(PV,TH,325.0,meta=False)[la0:la1+1,lo0:lo1+1]
TH_2PVU = wrf.interplevel(np.flip(TH,axis=0),np.flip(PV,axis=0),2.0,meta=False)[la0:la1+1,lo0:lo1+1]

PVwrf_325K = wrf.interplevel(PVwrf,THwrf,325.0,meta=False)
THwrf_2PVU = wrf.interplevel(THwrf,PVwrf,2.0,meta=False)
GHTwrf_250hPa = wrf.interplevel(GHTwrf,Pwrf,250,meta=False)
GHTwrf_500hPa = wrf.interplevel(GHTwrf,Pwrf,500,meta=False)
GHTwrf_850hPa = wrf.interplevel(GHTwrf,Pwrf,850,meta=False)

#
# PV difference at 325K
#

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')


hc=ax.contourf(lonwrf,latwrf,PVwrf_325K-PV_325K,levels=PVlvl,cmap=cmap,norm=norm,extend='both')

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

sa = ps + d + '-delta-PV-at-325K.png'
#fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)


####
####  TH @ 2PVU
####
fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

norm = plt.Normalize(np.min(THlvl),np.max(THlvl))

hc=ax.contourf(lonwrf,latwrf,THwrf_2PVU-TH_2PVU,levels=THlvl,cmap=cmap,norm=norm,extend='both')

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

sa = ps + d + '-delta-TH-at-2PVU.png'
#fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)


####
####  GHT @ 250 hPa
####

norm = plt.Normalize(np.min(GHTlvl),np.max(GHTlvl))

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

hc=ax.contourf(lonwrf,latwrf,GHTwrf_250hPa-GHT_250hPa,levels=GHTlvl,cmap=cmap,norm=norm,extend='both')

## colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=GHTlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('m',fontsize=8)
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

sa = ps + d + '-delta-GHT-at-250hPa.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)


####
####  GHT @ 500 hPa
####

norm = plt.Normalize(np.min(GHTlvl),np.max(GHTlvl))

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

hc=ax.contourf(lonwrf,latwrf,GHTwrf_500hPa-GHT_500hPa,levels=GHTlvl,cmap=cmap,norm=norm,extend='both')

## colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=GHTlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('m',fontsize=8)
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

sa = ps + d + '-delta-GHT-at-500hPa.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)



####
####  GHT @ 850 hPa
####

norm = plt.Normalize(np.min(GHTlvl),np.max(GHTlvl))

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')

hc=ax.contourf(lonwrf,latwrf,GHTwrf_850hPa-GHT_850hPa,levels=GHTlvl,cmap=cmap,norm=norm,extend='both')

## colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=GHTlvl[::2],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('m',fontsize=8)
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

sa = ps + d + '-delta-GHT-at-850hPa.png'
fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)



