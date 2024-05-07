import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr
import pandas as pd


def colbar(cmap,minval,maxval,nlevels,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/'

df = pd.read_csv('/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv')
thresh='075'
df = df.loc[df['ntraj%s'%thresh]>=200]
ID = df['ID'].values
lon = df['lon'].values
lat = df['lat'].values
SLP = df['minSLP'].values

f = open('/atmosdyn2/ascherrmann/009-ERA-5/MED/ctraj/use/PV-data-dPSP-100-ZB-800-2-400-correct-distance-noro.txt','rb')
data = pickle.load(f)
f.close()

NORO = xr.open_dataset('/home/ascherrmann/scripts/ERA5-utils/NORO')
ZB = NORO['ZB'].values[0]
Zlon = NORO['lon']
Zlat = NORO['lat']
Emaxv = 3000
Eminv = 800
elevation_levels = np.arange(Eminv,Emaxv,400)

dipv = data['dipv']
oro = data['oro']
datadi = data['rawdata']
noro = data['noro']

rdis = 400
H = 48
a = 1


maxv = 0.61
minv =-0.6
pvr_levels = np.arange(minv,maxv,0.15)

ap = plt.cm.BrBG
cmap ,norm = colbar(ap,minv,maxv,len(pvr_levels),pvr_levels)
for k in range(75,150):
    cmap.colors[k] = np.array([189/256, 195/256, 199/256, 1.0])

alpha=1.
linewidth=.2
ticklabels=pvr_levels

labs = helper.traced_vars_ERA5MED()

save = np.zeros((len(ID),9))
co = 0
for q,date in enumerate(datadi.keys()):
  if date!='119896' and date!='108215':
      continue
  ids = np.where(ID==int(date))[0][0]
  minsl = np.where(SLP[ids]==np.min(SLP[ids]))[0][0]

  fig = plt.figure(figsize=(8,6))
  gs = gridspec.GridSpec(nrows=1,ncols=1)
  ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
  ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

  idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]

  if True:

    tralon = datadi[date]['lon'][idp,:] 
    tralat = datadi[date]['lat'][idp,:]

    deltaPV = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPV[:,1:] = datadi[date]['PV'][idp,:-1]-datadi[date]['PV'][idp,1:]

    PVoro = oro[date]['env'][idp,:]
    deltaPVoro = np.zeros(datadi[date]['time'][idp,:].shape)
    deltaPVoro[:,1:] = PVoro[:,:-1]-PVoro[:,1:]
    pvr = deltaPV   

    latc = lat[ids]
    lonc = lon[ids]

    minpltlatc = np.round(latc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2000)),0)
    minpltlonc = np.round(lonc-np.floor(helper.convert_radial_distance_to_lon_lat_dis(2000)),0)

    maxpltlatc = np.round(latc+np.round(helper.convert_radial_distance_to_lon_lat_dis(1000),0),0)
    maxpltlonc = np.round(lonc+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)

    ax.contour(Zlon,Zlat,ZB,levels = elevation_levels,colors='purple',linewidths=0.35,alpha=1)

    for q in range(len(tralon[:,0])):
        seg = helper.make_segments(tralon[q,:],tralat[q,:])
        z = pvr[q,:]
        lc = mcoll.LineCollection(seg, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
        ax=plt.gca()
        ax.add_collection(lc)

#    ax.plot(lon[ids],lat[ids],color='k')
    ax.scatter(lon[ids],lat[ids],marker='o',color='k',s=20,zorder=10)
    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)

    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

    cbax = fig.add_axes([0, 0, 0.1, 0.1])
    cbar=plt.colorbar(lc, ticks=pvr_levels,cax=cbax)
    
    func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
    fig.canvas.mpl_connect('draw_event', func)
    
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xlabel('PVU h$^{-1}$',fontsize=10)
#    cbar.ax.set_xticklabels(ticklabels)
    ax.text(-0.07,0.875,'(a)',transform=ax.transAxes,fontsize=14)

    if date=='119896':
        n=9
    else:
        n=10
    figname = '/home/ascherrmann/publications/cyclonic-environmental-pv/fig%02da.png'%n
    fig.savefig(figname,dpi=300,bbox_inches="tight")
    plt.close('all')
    co+=1


