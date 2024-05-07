import numpy as np
import pickle
import pandas as pd
CT = 'MED'

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
pload2 ='/atmosdyn2/ascherrmann/010-IFS/traj/MED/use/'

df = pd.read_csv(pload[:-4] + 'pandas-all-data.csv')
thresh='075'
df = df.loc[df['ntraj%s'%thresh]>=200]
print(df.columns)

SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP']
maturedates = df['date']
months = df['mon'].values

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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import cm
import cartopy
import matplotlib.gridspec as gridspec
import functools


if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5


LON=np.arange(-180,180.1,0.5)
LAT=np.arange(-90,90.1,0.5)
ILON=np.arange(-180,180,0.4)
ILAT=np.arange(0,90,0.4)

DLON = np.arange(-180,180.1,1)
DLAT = np.arange(-90,90.1,1)
counter = np.zeros((len(LAT),len(LON)))

fig=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
colors = ['b','g','r','saddlebrown']

for lo,la,mo in zip(lon,lat,months):
    if mo=='JAN' or mo=='FEB' or mo=='DEC':
        col='b'
    if mo=='MAR' or mo=='APR' or mo=='MAY':
        col='g'
    if mo=='JUN' or mo=='JUL' or mo=='AUG':
        col='r'
    if mo=='SEP' or mo=='OCT' or mo=='NOV':
        col='saddlebrown'

    ax.scatter(lo,la,color=col,s=2)

ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

fig.savefig('/atmosdyn2/ascherrmann/paper/NA-MED-link/mature-seasonality.png',dpi=300,bbox_inches="tight")
plt.close('all')
