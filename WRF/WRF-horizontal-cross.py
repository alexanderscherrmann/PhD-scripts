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
from colormaps import PV_cmap2

parser = argparse.ArgumentParser(description="composite vertical cross section of XX ocean below XXX hPa")
parser.add_argument('sim',default='',type=str,help='folder/simulation for which to evaluate surface pressure and PV at 300 hPa')
parser.add_argument('day_hour',default='',type=str,help='')
parser.add_argument('var',default='',type=str,help='')


args = parser.parse_args()
sim=str(args.sim)
day=str(args.day_hour)
var=str(args.var)


p = '/atmosdyn2/ascherrmann/013-WRF-sim/' + sim + '/'
fb = 'wrfout_d01_2000-12-'
sb = 'PV-300hPa-2000-'
fe = ':00:00'

if var=='pvo':
    cmap,levels,norm,ticklabels=PV_cmap2()
    unit='PVU'

PSlevel = np.arange(975,1031,5)
#cmap = matplotlib.cm.jet
#norm = plt.Normalize(np.min(PSlevel),np.max(PSlevel))
#ticklabels=PSlevel
#levels=PSlevel
#PVcon = np.array([2])

fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1, ncols=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=10, edgecolor='black')

f = p + fb + day + fe
sa = p + sb + day + '.png'

data = netCDF4.Dataset(f)
SLP = wrf.getvar(data,'slp')
PV = wrf.getvar(data,'pvo') #in PVU already
PB = wrf.getvar(data,'PB')
P = wrf.getvar(data,'P')
U = wrf.getvar(data,'U')
U = (U[:,:,1:] + U[:,:,:-1])/2
V = wrf.getvar(data,'V')
V = (V[:,1:,:] + V[:,:-1,:])/2

PV = wrf.interplevel(PV,(PB+P)/100,300,meta=False)
U = wrf.interplevel(U,(PB+P)/100,300,meta=False)
V = wrf.interplevel(V,(PB+P)/100,300,meta=False)
lon = wrf.getvar(data,'lon')[0]
lat = wrf.getvar(data,'lat')[:,0]

LON,LAT = np.meshgrid(lon,lat)

h2=ax.contour(lon,lat,SLP,levels=PSlevel,colors='purple',linewidths=0.5)
hc=ax.contourf(lon,lat,PV,cmap=cmap,norm=norm,extend='both',levels=levels)
l=10
Q = ax.quiver(LON[::l,::l],LAT[::l,::l],U[::l,::l],V[::l,::l],scale=None,scale_units='width')
Qk = ax.quiverkey(Q,0.9,1.05,20,r'$20 \mathrm{m s}^{-1}$',labelpos='E',coordinates='axes')
plt.clabel(h2,inline=True,fmt='%d',fontsize=6)
## colorbar

cbax = fig.add_axes([0, 0, 0.1, 0.1])
cbar=plt.colorbar(hc, ticks=levels,cax=cbax)
func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.01)
fig.canvas.mpl_connect('draw_event', func)

## axis
lonticks=np.arange(np.min(lon)-0.25,np.max(lon)+0.5,40)
latticks=np.arange(np.min(lat)-0.25,np.max(lat)+0.5,10)

ax.set_xticks(lonticks)#, crs=ccrs.PlateCarree());
ax.set_yticks(latticks)#, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks.astype(int),fontsize=10)
ax.set_yticklabels(labels=latticks.astype(int),fontsize=10)
ax.set_xlim(np.min(lon),np.max(lon))
ax.set_ylim(np.min(lat),np.max(lat))
#ax.text(0.02,0.95,'%02d_'%d + h,transform=ax.transAxes,fontsize=10,)
ax.text(-42.5,82.5,day,fontsize=10)

fig.savefig(sa,dpi=300,bbox_inches="tight")
plt.close(fig)
		
		
