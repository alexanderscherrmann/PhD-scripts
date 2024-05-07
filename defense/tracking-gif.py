import numpy as np
import os
import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')

import helper
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patch
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pickle
import xarray as xr

import cartopy
import matplotlib.gridspec as gridspec
import functools
from netCDF4 import Dataset as ds
def colbar(cmap,minval,maxval,nlevels,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/cases/'

f = open(pload + 'zorbas-data.txt','rb')
cdata = pickle.load(f)
f.close()

print(cdata)

tdata = np.loadtxt(pload + 'trajectories-mature-20180928_05-ID-524945.txt')

latc = cdata['lat']
lonc = cdata['lon']

slp = cdata['SLP']

htmin=cdata['hourstoSLPmin']

latm = cdata['lat'][abs(cdata['hourstoSLPmin'][0]).astype(int)]
lonm = cdata['lon'][abs(cdata['hourstoSLPmin'][0]).astype(int)]

minpltlatc = np.round(latm-np.floor(helper.convert_radial_distance_to_lon_lat_dis(800)),0)
minpltlonc = np.round(lonm-np.floor(helper.convert_radial_distance_to_lon_lat_dis(800)),0)

maxpltlatc = np.round(latm+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)
maxpltlonc = np.round(lonm+np.round(helper.convert_radial_distance_to_lon_lat_dis(2000),0),0)

import cartopy.feature as cfeature
fig,ax=plt.subplots()
ax.plot(htmin[:96],slp[:96],color='k')
ax.set_xlabel('Time to min. SLP [h]')
ax.set_ylabel('SLP [hPa]')
fig.savefig('/home/ascherrmann/FSO-interview-SLP-evo.png',dpi=300,bbox_inches='tight')


d0='20180927_01'
t = np.arange(-27,100)
i0 = np.where(htmin==t[0])[0][0]

for hours in t:
    i = np.where(htmin==hours)[0][0]
    print(d0)
    B=ds('/atmosdyn2/era5/cdf/2018/09/B'+d0)
    slp=B.variables['MSL'][0]/100
    B.close()
    d0=helper.change_date_by_hours(d0,1)
    fig=plt.figure(figsize=(8,6))
    titles=['cyclogenesis', '48 h after cyclogenesis']
    gs = gridspec.GridSpec(ncols=1, nrows=1)# figure=fig)
    ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
    ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    ax.contour(np.arange(-180,180,0.5),np.arange(-90,90.1,0.5),slp,levels=np.arange(985,1021,5),colors='purple')

    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)
    
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=8)
    ax.set_yticklabels(labels=latticks,fontsize=8)
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
   
    ax.plot(lonc[i0:i+1],latc[i0:i+1],color='k')
    ax.scatter(lonc[(i).astype(int)],latc[(i).astype(int)],marker='o',color='k',s=40,zorder=100)

    figname = 'tracking-zorbas_mature-%02d.png'%(hours+27)
    fig.savefig('/home/ascherrmann/FSO-interview/' + figname,dpi=300,bbox_inches="tight")
    plt.close('all')

