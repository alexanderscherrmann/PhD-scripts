import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair,
                 get_basemap, latlon_coords)

from dypy.intergrid import Intergrid
from mpl_toolkits.basemap import Basemap
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper
from colormaps import PV_cmap2
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert


import argparse
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('sim',default='',type=str,help='')
parser.add_argument('day_hour',default='',type=str,help='')
parser.add_argument('var',default='',type=str,help='')
parser.add_argument('lvlmin',default=0,type=float,help='')
parser.add_argument('lvlmax',default=0,type=float,help='')
parser.add_argument('steps',default=0,type=float,help='')
parser.add_argument('lon',default=0,type=float,help='')
parser.add_argument('lat',default=0,type=float,help='')


args = parser.parse_args()
sim=str(args.sim)
day_hour=str(args.day_hour)
var=str(args.var)
lvlmin=float(args.lvlmin)
lvlmax=float(args.lvlmax)
steps=float(args.steps)

lonc = float(args.lon)
latc = float(args.lat)
dis = 1000 #km

dlon = helper.convert_radial_distance_to_lon_lat_dis_new(dis,latc)

# Define cross section line for each date (tini)
lon_start = lonc-dlon
lon_end   = lonc+dlon
lat_start = latc
lat_end   = latc

ymin = 100.
ymax = 1000.

ds = 5.
mvcross    = Basemap()
line,      = mvcross.drawgreatcircle(lon_start, lat_start, lon_end, lat_end, del_s=ds)
path       = line.get_path()
lonp, latp = mvcross(path.vertices[:,0], path.vertices[:,1], inverse=True)
dimpath    = len(lonp)



### data
p='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
ncfile = Dataset(p + 'wrfout_d01_2000-12-' + day_hour + ':00:00')


### get vars
Var = getvar(ncfile,var)
lons = getvar(ncfile,'lon')[0]
lats = getvar(ncfile,'lat')[:,0]
P = getvar(ncfile,"P") + getvar(ncfile,"PB")
P /= 100.

### prepare cross
vcross = np.zeros(shape=(Var.shape[0],dimpath))
vcross_p= np.zeros(shape=(Var.shape[0],dimpath))
bottomleft = np.array([lats[0], lons[0]])
topright   = np.array([lats[-1], lons[-1]])

for k in range(Var.shape[0]):
    f_vcross     = Intergrid(Var[k,:,:], lo=bottomleft, hi=topright, verbose=0)
    f_p3d_vcross   = Intergrid(P[k,:,:], lo=bottomleft, hi=topright, verbose=0)
    for i in range(dimpath):
        vcross[k,i]     = f_vcross.at([latp[i],lonp[i]])
        vcross_p[k,i]   = f_p3d_vcross.at([latp[i],lonp[i]])






### plot
cmap = matplotlib.cm.nipy_spectral
levels = np.arange(lvlmin,lvlmax+1e-5,steps)
norm = BoundaryNorm(levels,256)

if var=='pvo':
    cmap,levels,norm,ticklabels=PV_cmap2()
    unit='PVU'

fig = plt.figure(figsize=(6,4))
ax = plt.axes()

xcoord = np.zeros(shape=(Var.shape[0],dimpath))
for x in range(Var.shape[0]):
    xcoord[x,:] = np.array([ i*ds-dis for i in range(dimpath) ])

h = ax.contourf(xcoord, vcross_p, vcross, levels = levels, cmap = cmap, norm=norm, extend = 'both')


#ax.text(0.03, 0.95, 'd)', transform=ax.transAxes,fontsize=12, fontweight='bold',va='top')
#ax.set_xlabel('Distance from center [km]', fontsize=12)
ax.set_ylabel('Pressure [hPa]', fontsize=12)
ax.set_ylim(bottom=ymin, top=ymax)
#ax.set_xlim(-500,500)
#ax.set_xticks(ticks=np.arange(-500,500,250))
ax.set_xlim(-1000,1000)
ax.set_xticks(ticks=np.arange(-1000,1000,250))
# Invert y-axis
plt.gca().invert_yaxis()

# Add colorbar
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(h, ticks=levels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)
cbar.ax.set_xlabel(unit)

name=p + var + '-' + day_hour + '-%.1f-%.1f-%.1f-%.1f'%(lon_start,lon_end,lat_start,lat_end) + '.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close('all')

