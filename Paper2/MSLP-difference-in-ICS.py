import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
from scipy.stats import pearsonr as pr
import matplotlib
import wrf

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

cmap,levels,norm,ticklabels=PV_cmap2()
minlon = -15
maxlon = 50
minlat = 20
maxlat = 60
djfic = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
    
DJFMSLP = djfic.variables['MSLP'][0,:]
mamic = ds(dwrf + 'MAM-clim/wrfout_d01_2000-12-01_00:00:00')
MAMMSLP = mamic.variables['MSLP'][0,:]
dMSLP = DJFMSLP-MAMMSLP

DJFTH = wrf.getvar(djfic,'th')
MAMTH = wrf.getvar(mamic,'th')
DJFP = wrf.getvar(djfic,'pressure')
MAMP = wrf.getvar(mamic,'pressure')

DJFT = wrf.getvar(djfic,'tk')
MAMT = wrf.getvar(mamic,'tk')

thp=900
DJFth=wrf.interplevel(DJFTH,DJFP,thp,meta=False)
MAMth=wrf.interplevel(MAMTH,MAMP,thp,meta=False)

DJFt=wrf.interplevel(DJFT,DJFP,thp,meta=False)
MAMt=wrf.interplevel(MAMT,MAMP,thp,meta=False)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
h = ax.contourf(LON,LAT,dMSLP,cmap=matplotlib.cm.BrBG,levels=np.arange(-10,11,2),extend='both')
ax.set_extent([minlon,maxlon,minlat,maxlat])

cbax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=np.arange(-10,11,2),cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.015)
fig.canvas.mpl_connect('draw_event', func)

fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/DJF-MAM-MSLP-ic-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')

mamic = ds(dwrf + 'SON-clim/wrfout_d01_2000-12-01_00:00:00')
SONMSLP = mamic.variables['MSLP'][0,:]
dMSLP = DJFMSLP-SONMSLP

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
h = ax.contourf(LON,LAT,dMSLP,cmap=matplotlib.cm.BrBG,levels=np.arange(-10,11,2),extend='both')
ax.set_extent([minlon,maxlon,minlat,maxlat])

cbax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=np.arange(-10,11,2),cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.015)
fig.canvas.mpl_connect('draw_event', func)

fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/DJF-SON-MSLP-ic-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
h = ax.contourf(LON,LAT,MAMth-DJFth,cmap=matplotlib.cm.coolwarm,levels=np.arange(0,13,1),extend='both')
ax.set_extent([minlon,maxlon,minlat,maxlat])

cbax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=np.arange(0,13,1),cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.015)
fig.canvas.mpl_connect('draw_event', func)

fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/DJF-MAM-TH-%d-ic-diff.png'%thp,dpi=300,bbox_inches='tight')
plt.close('all')

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=2, edgecolor='black')
h = ax.contourf(LON,LAT,MAMt-DJFt,cmap=matplotlib.cm.coolwarm,levels=np.arange(0,13,1),extend='both')
ax.set_extent([minlon,maxlon,minlat,maxlat])

cbax = fig.add_axes([0.0, 0.0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=np.arange(0,13,1),cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.015)
fig.canvas.mpl_connect('draw_event', func)

fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/DJF-MAM-T-%d-ic-diff.png'%thp,dpi=300,bbox_inches='tight')
plt.close('all')
