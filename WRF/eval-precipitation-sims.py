import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
sys.path.append('/home/raphaelp/phd/scripts/basics/')
import wrfsims
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import BoundaryNorm
import wrf
import sys
import cmocean
import matplotlib.gridspec as gridspec
import matplotlib

SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','rb')
di = pickle.load(f)
f.close()

lon = np.arange(-9.75,50.,0.5)
lat = np.arange(25.25,55,0.5)
preciplevels = np.arange(10,100,10)
preciplevels2 = np.delete(np.arange(-50,51,10),[5])

for sim in di.keys():
    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    totalprecip = di[sim]['totalprecip']
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    loc = np.where(IDs==2)[0]

    sea = sim[:3]
    climprecip = di[sea + '-clim']['MED']

    fig=plt.figure(figsize=(8,6)) 
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.plot(tlon[loc],tlat[loc],color='grey',linewidth=2)
    ax.text(0.45,0.95,'%d m'%(totalprecip/1000),transform=ax.transAxes,fontsize=8)
    h=ax.contourf(lon,lat,di[sim]['MED'],levels=preciplevels,cmap=cmocean.cm.tempo,extend='max')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=preciplevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    fig.savefig(dwrf + 'image-output/precipitation/%s-within-MED-precip.png'%sim,dpi=300,bbox_inches='tight')
    plt.close('all')

    fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(gs[0,0])
    h = ax.contourf(np.arange(-10,10.5,0.5),np.arange(-10,10.5,0.5),di[sim]['track'],levels=preciplevels,cmap=cmocean.cm.tempo,extend='max')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=preciplevels,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    fig.savefig(dwrf + 'image-output/precipitation/%s-precip-along-track-center.png'%sim,dpi=300,bbox_inches='tight')
    plt.close('all')


    
    fig=plt.figure(figsize=(8,6))
    gs = gridspec.GridSpec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.plot(tlon[loc],tlat[loc],color='grey',linewidth=2)

    h=ax.contourf(lon,lat,di[sim]['MED']-climprecip,levels=preciplevels2,cmap=matplotlib.cm.BrBG,extend='both')
    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(h, ticks=preciplevels2,cax=cbax)
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)

    fig.savefig(dwrf + 'image-output/precipitation/%s-precip-by-cyclone-in-MED.png'%sim,dpi=300,bbox_inches='tight')
    plt.close('all')


    



