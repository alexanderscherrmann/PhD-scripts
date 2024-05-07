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
args = parser.parse_args()
sim=str(args.sim)

p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'

d = np.loadtxt(p + 'trace.ll',skiprows=5)

with open(p + 'trace.ll') as f:
    fl = next(f)
f.close()

#H = int(int(fl[-9:-5])/60/3)+1
H = int(int(fl[-9:-5])/60)+1

t = d[:,0].reshape(-1,H)
lon = d[:,1].reshape(-1,H)
lat = d[:,2].reshape(-1,H)
pv = d[:,4].reshape(-1,H)
pres=d[:,-1].reshape(-1,H)

sel=  np.where(pv[:,0]>=4)[0]
t=t[sel]
lon=lon[sel]
lat=lat[sel]
pv=pv[sel]
pres=pres[sel]
fig,axes=plt.subplots(figsize=(8,6),nrows=1,ncols=2)
ax=axes[0]
ax.scatter(pres[:,0]-pres[:,-1],pv[:,0]-pv[:,-1],color='k')
ax.set_xlabel('$\Delta$ P [hPa]')
ax.set_ylabel('$\Delta$ PV[PVU]')
ax=axes[1]
print(np.min(pres[:,0]),np.max(pres[:,0]))
#hc=ax.scatter(pv[:,-1],pv[:,0],c=pres[:,0]-pres[:,-1],cmap=matplotlib.cm.coolwarm)
hc=ax.scatter(pres[:,-1],pres[:,0],c=pv[:,0]-pv[:,-1],cmap=matplotlib.cm.coolwarm)
#ax.set_xlabel('PV start [PVU]')
#ax.set_ylabel('PV end [PVU]')
ax.set_xlabel('Pres start [hPa]')
ax.set_ylabel('Pres end [hPa]')
pos=ax.get_position()
cax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
plt.colorbar(hc,cax=cax)

figname = p + 'delta-pressure-delta-PV-traj.png'
fig.savefig(figname,dpi=300,bbox_inches="tight")
plt.close('all')



