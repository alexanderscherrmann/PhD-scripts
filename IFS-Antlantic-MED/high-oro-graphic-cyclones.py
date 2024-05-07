import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('type',default='',type=str,help='MED, TRO or ETA')

parser.add_argument('PVtrajedge',default='',type=float,help='PV value that traj have to exceed')

parser.add_argument('oroPVedge',default='',type=float,help='APV by orogrophy to identify cyclones with large contribution')
parser.add_argument('LSMedge',default='',type=float,help='evelation boundary for mautre stage, to make sure above water')

args = parser.parse_args()
CT = str(args.type)
oroPVedge = float(args.oroPVedge)
PVedge = float(args.PVtrajedge)
LSMedge = float(args.LSMedge)

pload = '/home/ascherrmann/010-IFS/traj/' + CT + '/use/'

f = open(pload + 'PV-data-' + CT + 'dPSP-100-ZB-800.txt','rb')
data = pickle.load(f)
f.close()

oro = data['oro']
datadi = data['rawdata']

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
td = pickle.load(f)
f.close()

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


fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines()
for ul, date in enumerate(oro.keys()):
      idp = np.where(datadi[date]['PV'][:,0]>=PVedge)[0]
      sq = np.ones(49)*len(idp)
      for pl, key in enumerate(['env']):
          for wt, ru in enumerate(['APVTOT']):
              meantmp = np.array([])
              for xx in range(len(sq)):
                  meantmp = np.append(meantmp,np.sum(oro[date][key][ru][idp,xx]/sq[xx]))

          if ((abs(meantmp[0])>oroPVedge)&((meantmp[12]/meantmp[0])>0.85) & (meantmp[0]!=0)):
              mon = data['mons'][ul]
              ids = data['ids'][ul]
              maturepoint = np.where(td[mon][ids]['hzeta']==0)[0][0]
              #lat = LAT[np.mean(td[mon][ids]['clat'],axis=1).astype(int)]
              #lon = LON[np.mean(td[mon][ids]['clon'],axis=1).astype(int)]
              lat = LAT[np.mean(td[mon][ids]['clat'][maturepoint]).astype(int)]
              lon = LON[np.mean(td[mon][ids]['clon'][maturepoint]).astype(int)]
              #ax.plot(lon,lat,zorder=100)
              ax.scatter(lon,lat,marker='x',zorder=100)
          else:
              mon = data['mons'][ul]
              ids = data['ids'][ul]
              maturepoint = np.where(td[mon][ids]['hzeta']==0)[0][0]
              lat = LAT[np.mean(td[mon][ids]['clat'][maturepoint]).astype(int)]
              lon = LON[np.mean(td[mon][ids]['clon'][maturepoint]).astype(int)]
              #lat = LAT[np.mean(td[mon][ids]['clat'],axis=1).astype(int)]
              #lon = LON[np.mean(td[mon][ids]['clon'],axis=1).astype(int)]
#              ax.plot(lon,lat,color='k',zorder=1.)
              ax.scatter(lon,lat,color='k',zorder=1.,marker='x',s=0.5)
  
  
if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5
else:
    minpltlonc = -100
    maxpltlonc = 50
    minpltlatc = 0
    maxpltlatc = 90
    steps = 20

lonticks=np.arange(minpltlonc, maxpltlonc,steps)
latticks=np.arange(minpltlatc, maxpltlatc,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

fig.savefig(pload + 'highly-orographically-influenced-' + CT +'-trajPV-' + str(PVedge) + '-oroPV-'+ str(oroPVedge) + '-LSM-' + str(LSMedge) +'-cyclones-scatter.png',dpi=300,bbox_inches="tight")
plt.close('all')
