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

period = ['ERA5','2010','2040','2070','2100']

seasons = ['DJF','MAM','SON']

wrfd = '/atmosdyn2/ascherrmann/013-WRF-sim/'
pappath = '/home/ascherrmann/thesis-images/'
Fig = plt.figure(figsize=(8,8))
Gs = gridspec.GridSpec(nrows=1, ncols=1)

for sea in seasons[:1]:
 for q,perio in enumerate(period[:1]):
    ### figure
    per = ds(wrfd + 'CESM-%s-DJF-clim/wrfout_d01_2000-12-01_00:00:00'%perio,'r')
    lon = wrf.getvar(per,'lon')[0]
    lat = wrf.getvar(per,'lat')[:,0]
    print(np.min(lat),np.min(lon)) 
    ax = Fig.add_subplot(Gs[q,:2],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

pos = ax.get_position()
ax.set_yticks([20,40,60,80])
ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
ax.set_xticks([-120,-90,-60,-30,0,30,60])
ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
ax.set_extent([-120,80,10,80], ccrs.PlateCarree())
Fig.savefig(pappath + 'domain-overview.png',dpi=300,bbox_inches='tight')
plt.close('all')
Fig = plt.figure(figsize=(8,8))
Gs = gridspec.GridSpec(nrows=1, ncols=1)

for sea in seasons[:1]:
 for q,perio in enumerate(period[:1]):
    ### figure
    per = ds(wrfd + 'CESM-%s-DJF-clim/wrfout_d01_2000-12-01_00:00:00'%perio,'r')
    lon = wrf.getvar(per,'lon')[0]
    lat = wrf.getvar(per,'lat')[:,0]
    print(np.min(lat),np.min(lon))
    ax = Fig.add_subplot(Gs[q,:2],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
    ax.plot([-10,50],[20,20],color='b')
    ax.plot([-10,50],[48,48],color='b')
    ax.plot([-10,-10],[20,48],color='b')
    ax.plot([50,50],[20,48],color='b')
    
    ax.plot([-85,-25],[25,25],color='b')
    ax.plot([-85,-25],[65,65],color='b')
    ax.plot([-85,-85],[25,65],color='b')
    ax.plot([-25,-25],[25,65],color='b')

pos = ax.get_position()
ax.set_yticks([20,40,60,80])
ax.set_yticklabels([r'20$^{\circ}$N',r'40$^{\circ}$N',r'60$^{\circ}$N',r'80$^{\circ}$N'])
ax.set_xticks([-120,-90,-60,-30,0,30,60])
ax.set_xticklabels([r'120$^{\circ}$W',r'90$^{\circ}$W',r'60$^{\circ}$W',r'30$^{\circ}$W',r'0$^{\circ}$E',r'30$^{\circ}$E',r'60$^{\circ}$E'])
ax.set_extent([-120,80,10,80], ccrs.PlateCarree())
Fig.savefig(pappath + 'nested-domain-overview.png',dpi=300,bbox_inches='tight')
plt.close('all')
