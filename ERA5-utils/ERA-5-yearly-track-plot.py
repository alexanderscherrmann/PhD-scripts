import sys
sys.path.append('/home/raphaelp/phd/scripts/basics/')
sys.path.append('/home/ascherrmann/scripts/')
import helper as h
import colorbars
import os
import pickle
#raphaels modules
from useful_functions import get_field_at_level,resize_colorbar_horz,resize_colorbar_vert
from colormaps import PV_cmap2

import matplotlib

#cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

p = '/home/ascherrmann/009-ERA-5/'


savings = ['SLP-','lon-','lat-','ID-','hourstoSLPmin-','dates-']
flist = dict()
for sv in savings:
 flist[sv[:-1]] = []
 for d in os.listdir(p):
     if(d.startswith(sv[:-1])):
         flist[sv[:-1]].append(d)

 flist[sv[:-1]] = sorted(flist[sv[:-1]])

pltlat = np.linspace(0,90,226)[70:121]
pltlon = np.linspace(-180,180,901)[440:541]

minpltlatc = pltlat[0]
maxpltlatc = pltlat[-1]

minpltlonc = pltlon[0]
maxpltlonc = pltlon[-1]

for k in range(len(flist['SLP'][:])):

    fig, axes = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax=axes
    ax.coastlines()
    
    LON = np.linspace(-180,180,901)
    LAT = np.linspace(0,90,226)
    d = open(p + flist['lat'][k],"rb") 
    lats = pickle.load(d)
    d.close()
    d = open(p + flist['lon'][k],"rb")
    lons = pickle.load(d)
    d.close()

    for u in range(len(lats)):
        ax.plot(lons[u],lats[u])
    
    
    lonticks=np.arange(minpltlonc, maxpltlonc,5)
    latticks=np.arange(minpltlatc, maxpltlatc,5)
    
    ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
    ax.set_yticks(latticks, crs=ccrs.PlateCarree());
    ax.set_xticklabels(labels=lonticks,fontsize=6)
    ax.set_yticklabels(labels=latticks,fontsize=6)
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())
    fig.savefig(p + 'cyclone-tracks-'+ str(flist['lat'][k][-8:-4]) + '.png',dpi=300,bbox_inches="tight")
    plt.close('all')
