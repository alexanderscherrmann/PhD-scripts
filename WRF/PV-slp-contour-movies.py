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
from colormaps import PV_cmap2

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('sim',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')


args = parser.parse_args()
sim=str(args.sim)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
fb = 'wrfout_d01_2000-12-'
sb = 'PV-300-2000-'
fe = ':00:00'

PSlevel = np.arange(920,1021,5)
cmap = matplotlib.cm.jet
norm = plt.Normalize(np.min(PSlevel),np.max(PSlevel))
ticklabels=PSlevel
levels=PSlevel
PVcon = np.array([2])
cmap,pvlevels,norm,pvticklabels=PV_cmap2()
dt = 0 
for d in range(1,32):
    for h in ['00','06','12','18']:
        

        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        
        f = p + fb + '%02d_'%d + h + fe
        sa = p + sb + '%02d_'%d + h + '.png'
        
        data = netCDF4.Dataset(f)
        SLP = wrf.getvar(data,'slp')
        PV = wrf.getvar(data,'pvo') #in PVU already
        pres = wrf.getvar(data,'pressure')
        pv300 = wrf.interplevel(PV,pres,300,meta=False)

        lon = wrf.getvar(data,'lon')[0]
        lat = wrf.getvar(data,'lat')[:,0]
        
        hc=ax.contourf(lon,lat,pv300,levels=pvlevels,cmap=cmap,norm=norm,extend='both')
        h2=ax.contour(lon,lat,SLP,levels=PSlevel,colors='purple',animated=False,linewidths=0.75,alpha=1,zorder=3)
        plt.clabel(h2, inline=1, fontsize=6, fmt='%d')


        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
        fig.canvas.mpl_connect('draw_event', func)
        cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))


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
		
