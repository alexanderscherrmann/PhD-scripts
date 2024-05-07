import numpy as np
import pickle

CT = 'MED'

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'

f = open(pload + 'PV-data-dPSP-100-ZB-800.txt','rb')
data = pickle.load(f)
f.close()

f = open(pload[:-4] + 'trackdata-ERA5.txt','rb')
td = pickle.load(f)
f.close()

oro = data['oro']
datadi = data['rawdata']

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

LON=np.linspace(-180,180,901)
LAT=np.linspace(0,90,226)

resPVp = np.array(['217927','225140','313864','390733'])

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
for ul, date in enumerate(oro.keys()):
    idp = np.where(datadi[date]['PV'][:,0]>=0.75)[0]
    sq = np.ones(49)*len(idp)
    if ((np.any(resPVp==date))&(np.mean(datadi[date]['OL'][idp,0])<0.5)):
            print(date)
#    if (idp.size):
#     for pl, key in enumerate(['cyc','env']):
#        ll = len(np.where((oro[date][key][idp,0]>0.3)&(datadi[date]['OL'][idp,0]<0.2))[0])
            
            #meantmp = np.array([])
            #for xx in range(len(sq)):
            #    meantmp = np.append(meantmp,np.sum(oro[date][key][ru][idp,xx]/sq[xx]))
            lon = np.mean(datadi[date]['lon'][:,0])
            lat = np.mean(datadi[date]['lat'][:,0])
#        if (meantmp[0]>0.3):
#            lat = td[int(date)]['lat']
#            lon = td[int(date)]['lon']
#            ax.plot(lon,lat)
            ax.scatter(lon,lat,marker='x',zorder=100,s=10)
    else:
            lon = np.mean(datadi[date]['lon'][:,0])
            lat = np.mean(datadi[date]['lat'][:,0])
            ax.scatter(lon,lat,marker='x',color='k',s=0.2)

if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5

lonticks=np.arange(minpltlonc, maxpltlonc,steps)
latticks=np.arange(minpltlatc, maxpltlatc,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

fig.savefig(pload + 'highly-orographically-influenced-cyclones-scatter.png',dpi=300,bbox_inches="tight")
plt.close('all')
