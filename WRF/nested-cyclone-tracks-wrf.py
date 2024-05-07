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

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('sim',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')
parser.add_argument('dom',default='',type=str,help='')


args = parser.parse_args()
sim=str(args.sim)
dom=str(args.dom)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
fb = 'wrfout_d%s_2000-12-'%dom
sb = 'PS-PV-2000-'
fe = ':00:00'

#trackdata = np.loadtxt('/atmosdyn2/ascherrmann/scripts/WRF/nest-test-tracking/out/' + sim + '-' + dom + '-filter.txt')
trackdata = np.loadtxt('/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/' + sim + '-' + dom + '-filter.txt')
t = trackdata[:,0]
lont = trackdata[:,1]
latt = trackdata[:,2]
IDS = trackdata[:,-1]

PSlevel = np.arange(975,1031,3)
cmap = matplotlib.cm.jet
norm = plt.Normalize(np.min(PSlevel),np.max(PSlevel))
ticklabels=PSlevel
levels=PSlevel
PVcon = np.array([2])
dt = 0 
for d in range(1,10):
    for h in ['00','06','12','18']:
        dt = (d-1)*24 + int(h)

        fig = plt.figure(figsize=(6,4))
        gs = gridspec.GridSpec(nrows=1, ncols=1)
        ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
        
        f = p + fb + '%02d_'%d + h + fe
        sa = p + sb + dom +  '-%02d_'%d + h + '.png'
        
        data = netCDF4.Dataset(f)
        SLP = wrf.getvar(data,'slp')
        PV = wrf.getvar(data,'pvo')[32] #in PVU already
        lon = wrf.getvar(data,'lon')[0]
        lat = wrf.getvar(data,'lat')[:,0]
        
        hc=ax.contourf(lon,lat,SLP,levels=PSlevel,cmap=cmap,norm=norm,extend='both')
        h2=ax.contour(lon,lat,PV,levels=PVcon,colors='purple',animated=False,linewidths=0.75,alpha=1,zorder=3)
        
        ## colorbar
        if np.any(t==dt):
            locs = np.where(t==dt)[0]
            if dom=='01':
                ax.scatter(lont[locs],latt[locs],marker='x',color='grey')
                for ul in locs:
                    ax.text(lont[ul],latt[ul],'%02d'%IDS[ul],fontsize=8,zorder=2000,color='white',fontweight='bold')
            else:
                for ul in locs:
                    if lont[ul]<np.min(lon) or lont[ul]>np.max(lon) or latt[ul]<np.min(lat) or latt[ul]>np.max(lat):
                        continue
                    ax.scatter(lont[ul],latt[ul],marker='x',color='grey')
                    ax.text(lont[ul],latt[ul],'%02d'%IDS[ul],fontsize=8,zorder=2000,color='white',fontweight='bold')

        cbax = fig.add_axes([0, 0, 0.1, 0.1])
        cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
        func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
        fig.canvas.mpl_connect('draw_event', func)
        cbar.ax.set_xlabel('hPa')

        ## axis
        if d=='01':
            lonticks=np.arange(np.min(lon)-0.25,np.max(lon)+0.5,40)
            latticks=np.arange(np.min(lat)-0.25,np.max(lat)+0.5,10)
        else:
            lonticks=np.arange(np.min(lon)-0.05,np.max(lon)+0.05,10)
            latticks=np.arange(np.min(lat)-0.05,np.max(lat)+0.05,5)
        
        ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
        ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
        ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
        ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
        ax.set_xlim(np.min(lon),np.max(lon))
        ax.set_ylim(np.min(lat),np.max(lat))
#        ax.set_extent([np.min(lon),np.max(lon),np.min(lat),np.max(lat)])
        #ax.text(0.02,0.95,'%02d_'%d + h,transform=ax.transAxes,fontsize=10,)
        if dom=='01':
            ax.text(-42.5,72.5,'%02d_'%d + h,fontsize=10)
        else:
            ax.text(19,21,'%02d_'%d + h,fontsize=10)
        
        fig.savefig(sa,dpi=300,bbox_inches="tight")
        plt.close(fig)
        dt+=6
		
