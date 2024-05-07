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

p = '/atmosdyn2/era5/cdf/2018/09/'

PSlevel = np.arange(920,1021,5)
cmap = matplotlib.cm.jet
norm = plt.Normalize(np.min(PSlevel),np.max(PSlevel))
ticklabels=PSlevel
levels=PSlevel
PVcon = np.array([2])
cmap,pvlevels,norm,pvticklabels=PV_cmap2()
dt = 0
fb='S200001'

p = '/atmosdyn2/era5/cdf/%s/%s/'%(fb[1:5],fb[-2:])
sb='PV-%s-%s'%(fb[1:5],fb[-2:])
sp='/home/ascherrmann/defense/'
for d in range(1,32):
    for h in ['00','06','12','18']:
        
        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        
        f = p + fb + '%02d_'%d + h
        sa = sp + sb + '%02d_'%d + h + '.png'
        
        s = netCDF4.Dataset(f)

        PS = s.variables['PS'][0]
        pv = s.variables['PV'][0]
        M=netCDF4.Dataset(p + 'B' + fb[1:] + '%02d_'%d + h)
        slp = M.variables['MSL'][0]/100
        ak=s.variables['hyam'][137-98:]
        bk=s.variables['hybm'][137-98:]

        ps3d=np.tile(PS[:,:],(len(ak),1,1))
        p3d=(ak/100.+bk*ps3d.T).T
        
        pv300=wrf.interplevel(pv,p3d,300,meta=False)
        lon = M.variables['lon'][:]
        lat = M.variables['lat'][:]
        s.close() 
        print(lon,lat,pv300.shape)
        hc=ax.contourf(lon,lat,pv300,levels=pvlevels,cmap=cmap,norm=norm,extend='both')
        h2=ax.contour(lon,lat,slp,levels=PSlevel,colors='purple',animated=False,linewidths=0.75,alpha=1,zorder=3)
        plt.clabel(h2, inline=1, fontsize=6, fmt='%d')


        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
        fig.canvas.mpl_connect('draw_event', func)
        cbar.ax.set_yticklabels(labels=np.append(pvticklabels[:5],np.append(np.array(pvticklabels[5:-1]).astype(int),'PVU')))


        ## axis
        lonticks=np.arange(-100,60,40)
        latticks=np.arange(10,80,10)
        
        ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
        ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
        ax.set_xlim(-100,60)
        ax.set_ylim(10,80)
        ax.text(-42.5,72.5,'%02d_'%d + h,fontsize=10)
        
        fig.savefig(sa,dpi=300,bbox_inches="tight")
        plt.close(fig)
        dt+=6
		
