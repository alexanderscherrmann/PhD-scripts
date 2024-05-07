import numpy as np
import netCDF4
import argparse

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy

import wrf

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

sim1='DJF-clim-max-U-at-300-hPa-1.4-QGPV/'
sim2='CESM-ERA5-DJF-0-km-max-1.4-QGPV/'

p = '/atmosdyn2/ascherrmann/013-WRF-sim/'
fb = 'wrfout_d01_2000-12-'
sb = 'deltaMSLP-2000-'
fe = ':00:00'

PSlevel = np.arange(-5,6,1)
cmap = matplotlib.cm.seismic
norm = plt.Normalize(np.min(PSlevel),np.max(PSlevel))
ticklabels=PSlevel
levels=PSlevel
PVcon = np.array([2])
dt = 0
try:
    for d in range(1,32):
        for h in ['00','06','12','18']:
            
    
#            fig = plt.figure(figsize=(6,4))
#            gs = gridspec.GridSpec(nrows=1, ncols=1)
#            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
#            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
            
#            f = p + sim1 + fb + '%02d_'%d + h + fe
#            sa = p + sim1 + sb + '%02d_'%d + h + '.png'
            
            data = netCDF4.Dataset(f)
#            SLP = wrf.getvar(data,'slp')
#            lon = wrf.getvar(data,'lon')[0]
#            lat = wrf.getvar(data,'lat')[:,0]
#            f2 = p + sim2 + fb + '%02d_'%d + h + fe 
#            data2 = netCDF4.Dataset(f2)
#    
#            SLP2 = wrf.getvar(data2,'slp')
#            hc=ax.contourf(lon,lat,SLP-SLP2,levels=PSlevel,cmap=cmap,norm=norm,extend='both')
#            
#            cbax = fig.add_axes([0, 0, 0.1, 0.1])
#            cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
#            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
#            fig.canvas.mpl_connect('draw_event', func)
#            cbar.ax.set_xlabel('hPa')
#    
#            ## axis
#            lonticks=np.arange(np.min(lon)-0.25,np.max(lon)+0.5,40)
#            latticks=np.arange(np.min(lat)-0.25,np.max(lat)+0.5,10)
#            
#            ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
#            ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
#            ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
#            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
#            ax.set_xlim(np.min(lon),np.max(lon))
#            ax.set_ylim(np.min(lat),np.max(lat))
#            #ax.text(0.02,0.95,'%02d_'%d + h,transform=ax.transAxes,fontsize=10,)
#            ax.text(-42.5,72.5,'%02d_'%d + h,fontsize=10)
#            
#            fig.savefig(sa,dpi=300,bbox_inches="tight")
#            plt.close(fig)
#            dt+=6
except:
    sim1='DJF-clim/'
    sim2='CESM-ERA5-DJF-clim/'
    
    for d in range(1,32):
        for h in ['00','06','12','18']:
    
    
            fig = plt.figure(figsize=(6,4))
            gs = gridspec.GridSpec(nrows=1, ncols=1)
            ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
            ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    
            f = p + sim1 + fb + '%02d_'%d + h + fe
            sa = p + sim1 + sb + '%02d_'%d + h + '.png'
    
            data = netCDF4.Dataset(f)
            SLP = wrf.getvar(data,'slp')
            lon = wrf.getvar(data,'lon')[0]
            lat = wrf.getvar(data,'lat')[:,0]
            f2 = p + sim2 + fb + '%02d_'%d + h + fe
            data2 = netCDF4.Dataset(f2)
    
            SLP2 = wrf.getvar(data2,'slp')
            hc=ax.contourf(lon,lat,SLP-SLP2,levels=PSlevel,cmap=cmap,norm=norm,extend='both')
    
            cbax = fig.add_axes([0, 0, 0.1, 0.1])
            cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
            func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
            fig.canvas.mpl_connect('draw_event', func)
            cbar.ax.set_xlabel('hPa')
    
            ## axis
            lonticks=np.arange(np.min(lon)-0.25,np.max(lon)+0.5,40)
            latticks=np.arange(np.min(lat)-0.25,np.max(lat)+0.5,10)
    
            ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
            ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
            ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
            ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
            ax.set_xlim(np.min(lon),np.max(lon))
            ax.set_ylim(np.min(lat),np.max(lat))
            #ax.text(0.02,0.95,'%02d_'%d + h,transform=ax.transAxes,fontsize=10,)
            ax.text(-42.5,72.5,'%02d_'%d + h,fontsize=10)
    
            fig.savefig(sa,dpi=300,bbox_inches="tight")
            plt.close(fig)
            dt+=6
