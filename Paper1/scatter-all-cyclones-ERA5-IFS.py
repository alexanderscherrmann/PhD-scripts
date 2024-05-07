import numpy as np
import pickle
import pandas as pd
CT = 'MED'

pload = '/atmosdyn2/ascherrmann/009-ERA-5/MED/traj/use/'
pload2 ='/atmosdyn2/ascherrmann/010-IFS/traj/MED/use/'

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

f = open('/atmosdyn2/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

f = open(pload2 + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data2 = pickle.load(f)
f.close()

df = pd.read_csv(pload[:-4] + 'pandas-all-data.csv')
thresh='075PVU'
#df = df.loc[df['ntrajgt%s'%thresh]>=200]
df=df.loc[df['ntraj075']>=200]
print(df)

SLP = df['minSLP'].values
lon = df['lon'].values
lat = df['lat'].values
ID = df['ID'].values
hourstoSLPmin = df['htminSLP']
maturedates = df['date']

datadi = data['rawdata']
datadi2 = data2['rawdata']

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

import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

if CT=='MED':
    minpltlonc = -10
    maxpltlonc = 45
    minpltlatc = 25
    maxpltlatc = 50
    steps = 5


def colbar(cmap,levels):
    maplist = [cmap(i) for i in range(cmap.N)]
    newmap = ListedColormap(maplist)
    norm = BoundaryNorm(levels,cmap.N)
    return newmap, norm

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

for ul, date in enumerate(datadi.keys()):
    if np.all(ID!=int(date)):
        continue
    q = np.where(ID==int(date))[0][0]
    if lat[q]<30 or lat[q]>48:
        continue
    if lon[q]<-5 or lon[q]>42:
        continue
    if lon[q]<2 and lat[q]>42:
        continue

    lo = np.where(abs(LON-lon[q])==np.min(abs(LON-lon[q])))[0][0]
    la = np.where(abs(LAT-lat[q])==np.min(abs(LAT-lat[q])))[0][0]
    counter[la,lo]+=1

nonlat,nonlon = np.where(counter!=0)
ncounter = np.zeros(counter.shape)
pm = 1
step = [[0,0],[0,1],[1,0],[1,1]]
for l in range(0,361):
    for a in range(0,721):
        tmp=[]
        for z,s in step:
            tmp.append(np.sum(counter[l-1+z:l+z+2,a+s-1:a+s+2]))
        ncounter[l,a] = np.mean(tmp)/4


cmap=ListedColormap(['darkcyan','lightseagreen','mediumturquoise','tan','peru','sienna','saddlebrown'])#[matplotlib.cm.BrBG[0],matplotlib.cm.BrBG[64],matplotlib.cm.BrBG[128],matplotlib.cm.BrBG[-1]])
cmap.set_over('brown')
cmap.set_under('white')
ax.plot([],[],color='k',ls=' ',marker='.',markersize=2)
ax.plot([],[],color='k',ls=' ',marker='.',markersize=np.max(counter)/12)
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='coastline',scale='50m'),zorder=4, edgecolor='black')
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=100, edgecolor='black')
ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

levels = np.array([1,3,5,10,15,20,25,30,40,50])
cmap=truncate_colormap(matplotlib.cm.Blues,0.3,1.0)
#cf = ax.contour(LON,LAT,ncounter,linewidths=[0.5,0.5,1.5,0.5,0.5,1.5,0.5,0.5],colors='blue',levels=levels)
cf=ax.contourf(LON,LAT,ncounter,levels=levels,cmap=cmap,norm=BoundaryNorm(levels,cmap.N))
pos = ax.get_position()
cbax=fig.add_axes([pos.x0+pos.width,pos.y0,0.02,pos.height])
plt.colorbar(cf,cax=cbax,ticks=levels,boundaries=levels)
#plt.clabel(cf,inline=True,fontsize=6,fmt='%d',manual=True)

#for ul,date in enumerate(datadi2.keys()):
#    mon = data2['mons'][ul]
#    q = int(date[-3:])
#
#    ax.scatter(ILON[np.mean(locdata[mon][q]['clon'][abs(locdata[mon][q]['hzeta'][0]).astype(int)]).astype(int)],ILAT[np.mean(locdata[mon][q]['clat'][abs(locdata[mon][q]['hzeta'][0]).astype(int)]).astype(int)],marker='o',color='r',zorder=100,s=5)

longrids=np.arange(-180,180,5)
latgrids=np.arange(-90,90,5)

ax.plot([-5,42],[30,30],color='k',zorder=1)
ax.plot([-5,-5],[30,42],color='k',zorder=1)
ax.plot([-5,2],[42,42],color='k',zorder=1)
ax.plot([2,2],[42,48],color='k',zorder=1)
ax.plot([2,42],[48,48],color='k',zorder=1)
ax.plot([42,42],[30,48],color='k',zorder=1)

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

fig.savefig('/atmosdyn2/ascherrmann/paper/cyc-env-PV/review/' + 'allcyclone-IFS-ERA5-NEW-%s.png'%thresh,dpi=300,bbox_inches="tight")
plt.close('all')
