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


parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')
parser.add_argument('time',default='',type=str,help='time when trajectories start')
args = parser.parse_args()
sim=str(args.sim)
time=str(args.time)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

d = np.loadtxt(p + 'trace.ll',skiprows=5)
data = ds(p + 'wrfout_d01_2000-12-%s:00:00'%time,'r')
d0 =  ds(p + 'wrfout_d01_2000-12-01_00:00:00','r')

pv0 = d0.variables['PV'][0,:]
p0 = getvar(d0,'pressure',meta=False)
LON=getvar(d0,'lon',meta=False)[0]
LAT=getvar(d0,'lat',meta=False)[:,0]

pv0300 = interplevel(pv0,p0,300,meta=False)

with open(p + 'trace.ll') as f:
    fl = next(f)

f.close()
cmap,levels,norm,ticklabels=PV_cmap2()
H = int(int(fl[-9:-5])/60)+1

t = d[:,0].reshape(-1,H)
lon = d[:,1].reshape(-1,H)
lat = d[:,2].reshape(-1,H)
pv = d[:,4].reshape(-1,H)

#if t[0,1]<0:
#    t=np.flip(t,axis=1)
#    lon=np.flip(lon,axis=1)
#    lat=np.flip(lat,axis=1)

fig = plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(nrows=1,ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

linewidth=1.0
alpha=1.0

for q in range(len(lon[:,0])):
    seg = helper.make_segments(lon[q,:],lat[q,:])
    z = pv[q,:]
    lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax=plt.gca()
    ax.add_collection(lc)


ax.contour(LON,LAT,pv0300,colors='purple',linewidths=1,levels=[2,3,5,7,9,11])

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

figname = p + 'streamer-traj.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')



