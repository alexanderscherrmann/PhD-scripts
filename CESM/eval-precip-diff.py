from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
import numpy as np
import wrf
import os
import pickle

pappath = '/atmosdyn2/ascherrmann/015-CESM-WRF/'

f = open(pappath + 'precipitation-UV10M-data-DJF.txt','rb')
d = pickle.load(f)
f.close()

tmp = ds('/atmosdyn2/ascherrmann/013-WRF-sim/DJF-clim/wrfout_d01_2000-12-01_00:00:00','r')
lon = wrf.getvar(tmp,'lon')[0]
lat = wrf.getvar(tmp,'lat')[:,0]

k=np.array([])
for l in np.array(list(d.keys())):
    if '2010' in l:
        k=np.append(k,l)

kk=k[4]
k[1:5]=k[:4]
k[0]=kk
cmap=matplotlib.cm.BrBG

fig = plt.figure(figsize=(7.5,6))
gs = gridspec.GridSpec(nrows=7, ncols=6)

for q,l in enumerate(k):
    ax = fig.add_subplot(gs[int(q/6),q%6],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    cn=ax.contourf(lon,lat,d[l[:5] + '2100' + l[9:]]['total']-d[l]['total'],levels=np.arange(-50,51,10),cmap=cmap,extend='both')
    if q==0:
        av=d[l[:5] + '2100' + l[9:]]['total']-d[l]['total']
    else:
        av+=d[l[:5] + '2100' + l[9:]]['total']-d[l]['total']

    ax.set_extent([-10,50,20,60], ccrs.PlateCarree())
    
    ax.text(0.,1.02,l[10:],transform=ax.transAxes,fontsize=4)

av/=(q+1)

plt.subplots_adjust(wspace=0,hspace=0.1)
pos = ax.get_position()
cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.005,pos.height])
cbar=plt.colorbar(cn,cax=cbax)
fig.savefig(pappath + 'precipitation-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')    

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.set_extent([-10,50,20,60], ccrs.PlateCarree())

cn=ax.contourf(lon,lat,av,levels=np.arange(-50,51,10),cmap=cmap,extend='both')
pos = ax.get_position()
cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
cbar=plt.colorbar(cn,cax=cbax,ticks=np.arange(-50,51,10))
cbax.set_yticklabels(np.arange(-50,51,10))
cbax.set_ylabel('[mm (9 d)$^{-1}$]')
fig.savefig(pappath + 'av-precipitation-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')

fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.set_extent([-10,50,20,60], ccrs.PlateCarree())

cn=ax.contourf(lon,lat,av/9,levels=np.arange(-2,2.1,0.25),cmap=cmap,extend='both')
pos = ax.get_position()
cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
cbar=plt.colorbar(cn,cax=cbax,ticks=np.arange(-2,2.1,0.25))
cbax.set_yticklabels(np.arange(-2,2.1,0.25))
cbax.set_ylabel('[mm d$^{-1}$]')
fig.savefig(pappath + 'daily-av-precipitation-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig = plt.figure(figsize=(6,3))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.set_extent([-10,50,20,60], ccrs.PlateCarree())

cn=ax.contourf(lon,lat,av-(d[l[:5] + '2100' + l[9:]]['total']-d[l]['total']),levels=np.arange(-50,51,10),cmap=cmap,extend='both')
pos = ax.get_position()
cbax = fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
cbar=plt.colorbar(cn,cax=cbax,ticks=np.arange(-50,51,10))
cbax.set_yticklabels(np.arange(-50,51,10))
cbax.set_ylabel('[mm (9 d)$^{-1}$]')
fig.savefig(pappath + 'av-precipitation-diff-minus-climatology-diff.png',dpi=300,bbox_inches='tight')
plt.close('all')
