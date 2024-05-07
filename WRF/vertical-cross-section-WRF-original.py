import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from netCDF4 import Dataset
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair,
                 get_basemap, latlon_coords)

import argparse
parser = argparse.ArgumentParser(description=' ')
parser.add_argument('sim',default='',type=str,help='')
parser.add_argument('day_hour',default='',type=str,help='')
parser.add_argument('var',default='',type=str,help='')
parser.add_argument('lvlmin',default=0,type=float,help='')
parser.add_argument('lvlmax',default=0,type=float,help='')
parser.add_argument('steps',default=0,type=float,help='')
parser.add_argument('lonstart',default=0,type=float,help='')
parser.add_argument('lonend',default=0,type=float,help='')
parser.add_argument('latstart',default=0,type=float,help='')
parser.add_argument('latend',default=0,type=float,help='')


args = parser.parse_args()
sim=str(args.sim)
day_hour=str(args.day_hour)
var=str(args.var)
lvlmin=float(args.lvlmin)
lvlmax=float(args.lvlmax)
steps=float(args.steps)
lonstart=float(args.lonstart)
lonend=float(args.lonend)
latend=float(args.latend)
latstart=float(args.latstart)

p='/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
ncfile = Dataset(p + 'wrfout_d01_2000-12-' + day_hour + ':00:00')

Var = getvar(ncfile,var)
z = getvar(ncfile,"P") + getvar(ncfile,"PB")
z = z/100.

fig = plt.figure(figsize=(6,4))
ax = plt.axes()

start_point = CoordPair(lat=latstart, lon=lonstart)
end_point = CoordPair(lat=latend, lon=lonend)

levels = np.arange(lvlmin,lvlmax+1e-5,steps)
norm = BoundaryNorm(levels,256)

fig = plt.figure(figsize=(6,4))
ax = plt.axes()

Var_cross = vertcross(Var, z, wrfin=ncfile, start_point=start_point,
                       end_point=end_point, latlon=True, meta=True)

z_cross = vertcross(z,z,wrfin=ncfile,start_point=start_point,end_point=end_point,latlon=True,meta=True)

Var_contours = ax.contourf(to_np(Var_cross), cmap=matplotlib.cm.nipy_spectral,norm=norm,levels=levels,extend='both')
plt.colorbar(Var_contours, ax=ax,pad=0.01)
#vert_vals = to_np(Var_cross.coords["vertical"])
#v_ticks = np.arange(vert_vals.shape[0])
ax.set_yticks(np.arange(0,len(to_np(z_cross[:,0])),20))#z_cross[2::20,0])#v_ticks[::20])
ax.set_yticklabels(to_np(z_cross[:,0])[::20])#vert_vals[::20], fontsize=8)

name=p + var + '-' + day_hour + '-%.1f-%.1f-%.1f-%.1f'%(lonstart,lonend,latstart,latend) + '.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close('all')

