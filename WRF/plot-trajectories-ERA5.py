import numpy as np
import argparse
from netCDF4 import Dataset as ds
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
from wrf import getvar, interplevel
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
from colormaps import PV_cmap2
intp=interplevel

parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('tf',default='',type=str,help='which reference state to perturb: mean or overlap')
args = parser.parse_args()
tf=str(args.tf)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/ERA5-streamer/'
date = tf[-15:-4]
y=date[:4]
m=date[4:6]
e5 = '/atmosdyn2/era5/cdf/%s/%s/'%(y,m)

d = np.loadtxt(p + tf,skiprows=5)

LON=np.linspace(-180,180,721)[:-1]
LAT=np.linspace(-90,90,361)

with open(p + tf) as f:
    fl = next(f)
f.close()
H = int(int(fl[-9:-5])/60)+1
HH=0
date0 = helper.change_date_by_hours(date,-(H-1)+(H+HH))

s=ds(e5+'S'+date0)
PS = s.variables['PS'][0]#,la0:la1,lo0:lo1]
PV = s.variables['PV'][0]#,:,la0:la1,lo0:lo1]
hyam = s.variables['hyam'][137-98:]
hybm = s.variables['hybm'][137-98:]
ps3d=np.tile(PS,(PV.shape[0],1,1))
P=(hyam/100.+hybm*ps3d.T).T
PV300 = intp(PV,P,300,meta=False)


cmap,levels,norm,ticklabels=PV_cmap2()

t = d[:,0].reshape(-1,H)
lon = d[:,1].reshape(-1,H)
lat = d[:,2].reshape(-1,H)
pv = d[:,-1].reshape(-1,H)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1,ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

linewidth=1.0
alpha=1.0

lon = lon[:,:HH]
lat = lat[:,:HH]
pv = pv[:,:HH]
for q in range(len(lon[:,0])):
    seg = helper.make_segments(lon[q,:],lat[q,:])
    z = pv[q,:]
    lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax=plt.gca()
    ax.add_collection(lc)

ax.contour(LON,LAT,PV300,colors='purple',linewidths=1,levels=[2,3,5,7,9,11])

lonticks=np.arange(-120,80.1,40)
latticks=np.arange(10,80.1,10)

ax.set_xticks(ticks=lonticks, crs=ccrs.PlateCarree())
ax.set_yticks(ticks=latticks, crs=ccrs.PlateCarree())

ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)

ax.set_extent([-120,80,10,80], ccrs.PlateCarree())
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(lc, ticks=levels,cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('PVU',fontsize=8)
cbar.ax.set_yticklabels(labels=levels)

figname = p + 'traj-%s.png'%date
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')



