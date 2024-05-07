import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import BoundaryNorm
import wrf
import sys
import cmocean

sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import helper

seasons = ['DJF','SON','JJA','SON']
pvpos = [[93,55],[95,56],[140,73],[125,72]]
amps = [0.7,1.4,2.1,2.8]

pvcmap,pvlevels,pvnorm,pvticklabels=PV_cmap2()
ucmap,ulevels = cmocean.cm.tempo,np.arange(10,65,5)
unorm = BoundaryNorm(ulevels,ucmap.N)
PSlevel = np.arange(975,1031,5)
dis = 1500

dlat = helper.convert_radial_distance_to_lon_lat_dis_new(dis,0)

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/atmosdyn2/ascherrmann/paper/NA-MED-link/'


fig=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=5, ncols=2)
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
times=['01_00','03_00','05_00','07_00','09_00']

sims=['SON-clim','SON-clim-max-U-at-300-hPa-1.4-QGPV']

q = 0
for k,sim in enumerate(sims):
    for l,t in enumerate(times):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        ref=ds(wrfd + sim + '/wrfout_d01_2000-12-%s:00:00'%t)

        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)

        lon = wrf.getvar(ref,'lon')[0]
        lat = wrf.getvar(ref,'lat')[:,0]


        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
        
        if k==0 and l==3:
            pos =ax.get_position()
            x0 = pos.x0
            
        t=ax.text(-115,68,labels[q],zorder=15,fontsize=8)
        t.set_bbox(dict(facecolor='white',edgecolor='white'))
        ax.set_aspect('auto')
        q+=1

pvticklabels[-1]='PVU'
pos = ax.get_position()
cbax = fig.add_axes([x0+pos.width/4, pos.y0-0.02, pos.x0+pos.width-(x0+pos.width/4)-pos.width/4, 0.02])
cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax,orientation='horizontal')
cbar.ax.set_xticklabels(pvticklabels)
#func=resize_colorbar_hor(cbax, ax, pad=0.0, size=0.005)
#fig.canvas.mpl_connect('draw_event', func)
plt.subplots_adjust(wspace=0.0,hspace=0)

fig.savefig(pappath + 'SON-ref-1.4-dynamics.png',dpi=300,bbox_inches='tight')
plt.close('all')


fig=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=5, ncols=2)
labels=['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)']
times=['01_00','03_00','05_00','07_00','09_00']
sims=['SON-clim-max-U-at-300-hPa-0.7-QGPV','SON-clim-max-U-at-300-hPa-2.1-QGPV']

q = 0
for k,sim in enumerate(sims):
    for l,t in enumerate(times):
        ax=fig.add_subplot(gs[l,k],projection=ccrs.PlateCarree())
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

        ref=ds(wrfd + sim + '/wrfout_d01_2000-12-%s:00:00'%t)

        PVref=wrf.getvar(ref,'pvo')
        Pref = wrf.getvar(ref,'pressure')
        MSLPref = ref.variables['MSLP'][0,:]
        PV300ref = wrf.interplevel(PVref,Pref,300,meta=False)

        lon = wrf.getvar(ref,'lon')[0]
        lat = wrf.getvar(ref,'lat')[:,0]


        h2=ax.contour(lon,lat,MSLPref,levels=PSlevel,colors='purple',linewidths=0.5)
        hc=ax.contourf(lon,lat,PV300ref,cmap=pvcmap,norm=pvnorm,extend='both',levels=pvlevels)
        plt.clabel(h2,inline=True,fmt='%d',fontsize=6)

        if k==0 and l==3:
            pos =ax.get_position()
            x0 = pos.x0

        t=ax.text(-115,70,labels[q],zorder=15,fontsize=8)
        t.set_bbox(dict(facecolor='white',edgecolor='white'))
        ax.set_aspect('auto')
        q+=1

pos = ax.get_position()
cbax = fig.add_axes([x0+pos.width/4, pos.y0-0.02, pos.x0+pos.width-(x0+pos.width/4)-pos.width/4, 0.02])
cbar=plt.colorbar(hc, ticks=pvlevels,cax=cbax,orientation='horizontal')
cbar.ax.set_xticklabels(pvticklabels)
#func=resize_colorbar_hor(cbax, ax, pad=0.0, size=0.005)
#fig.canvas.mpl_connect('draw_event', func)
plt.subplots_adjust(wspace=0.0,hspace=0)

fig.savefig(pappath + 'SON-0.7-2.1-dynamics.png',dpi=300,bbox_inches='tight')
plt.close('all')

