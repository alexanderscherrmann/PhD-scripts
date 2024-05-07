import numpy as np
import argparse

import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm

import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper



parser = argparse.ArgumentParser(description="perturb unperturbed reference domain and write new output field")
parser.add_argument('sim',default='',type=str,help='which reference state to perturb: mean or overlap')
args = parser.parse_args()
sim=str(args.sim)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

d = np.loadtxt(p + 'trace.ll',skiprows=5)
t = d[:,0].reshape(-1,144)

lon = d[:,1].reshape(-1,144)
lat = d[:,2].reshape(-1,144)
lat = d[:,2].reshape(-1,144)
z = d[:,3].reshape(-1,144)
PV = d[:,4].reshape(-1,144)
TH = d[:,5].reshape(-1,144)
P = (d[:,6] + d[:,7]).reshape(-1,144)


fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1,ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)



cmap = matplotlib.cm.jet
norm = BoundaryNorm(t[0],cmap.N)
linewidth=1.0
alpha=1.0

for q in range(len(lon[:,0])):
    seg = helper.make_segments(lon[q,:],lat[q,:])
    z = t[q,:]
    lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    ax=plt.gca()
    ax.add_collection(lc)


lonticks=np.arange(-120,80.1,40)
latticks=np.arange(10,80.1,10)

ax.set_xticks(ticks=lonticks, crs=ccrs.PlateCarree())
ax.set_yticks(ticks=latticks, crs=ccrs.PlateCarree())

ax.set_xticklabels(labels=lonticks,fontsize=8)
ax.set_yticklabels(labels=latticks,fontsize=8)

ax.set_extent([-120,80,10,80], ccrs.PlateCarree())
cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(lc, ticks=t[0,::12],cax=cbax)

func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
fig.canvas.mpl_connect('draw_event', func)

cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('h',fontsize=8)
cbar.ax.set_xticklabels(labels=t[0,::12])

figname = p + 'forward-traj.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')

t = t[0]
fig,axes = plt.subplots(2,1,sharex=True)
axes = axes.flatten()
ax = axes[0]
ax.plot(t,np.mean(TH,axis=0),color='k')

ax = axes[1]
ax.plot(t,np.mean(PV,axis=0),color='k')


figname = p + 'PVevolution-forward-traj.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')


