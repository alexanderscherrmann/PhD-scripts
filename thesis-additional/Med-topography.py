import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy
from matplotlib.colors import BoundaryNorm
import matplotlib.cbook as cbook


Fig = plt.figure(figsize=(8,8))
Gs = gridspec.GridSpec(nrows=1, ncols=1)
NORO = ds('/atmosdyn2/ascherrmann/009-ERA-5/MED/data/ORO','r')

lon = NORO.variables['lon'][:]
lat = NORO.variables['lat'][:]
ZB = NORO.variables['ZB'][0]

pappath = '/home/ascherrmann/thesis-images/'

ax = Fig.add_subplot(Gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)

#dem = cbook.get_sample_data('topobathy.npz', np_load=True)
#topo = dem['topo']
#longitude = dem['longitude']
#latitude = dem['latitude']
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
all_colors = np.vstack((colors_undersea, colors_land))
terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=4000)
pcm = ax.contourf(lon, lat, ZB, norm=divnorm,cmap=terrain_map)

ax.set_xlim(-10,50)
ax.set_ylim(25,55)

pos = ax.get_position()
cbax = Fig.add_axes([pos.x0+pos.width,pos.y0,0.01,pos.height])
cbar = plt.colorbar(pcm,cax=cbax)


Fig.savefig(pappath + 'MED-domain.png',dpi=300,bbox_inches='tight')
plt.close('all')
