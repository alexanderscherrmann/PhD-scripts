import numpy as np
import pickle
import pandas as pd
CT = 'MED'

pload = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
pload2 ='/home/ascherrmann/010-IFS/traj/MED/use/'


#savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
#var = []
#for u,x in enumerate(savings):
#    f = open(pload[:-4] + x + 'furthersel.txt',"rb")
#    var.append(pickle.load(f))
#    f.close()
#
#SLP = var[0]
#lon = var[1]
#lat = var[2]
#ID = var[3]
#hourstoSLPmin = var[4]
#avaID = np.array([])
#minSLP = np.array([])
#for k in range(len(ID)):
#    avaID=np.append(avaID,ID[k][0].astype(int))
#    minSLP = np.append(minSLP,SLP[k][abs(hourstoSLPmin[k][0]).astype(int)])

f = open(pload + 'PV-data-dPSP-100-ZB-800-2-400-correct-distance.txt','rb')
data = pickle.load(f)
f.close()

f = open('/home/ascherrmann/010-IFS/data/All-CYC-entire-year-NEW-correct.txt','rb')
locdata =  pickle.load(f)
f.close()

f = open(pload2 + 'PV-data-MEDdPSP-100-ZB-800PVedge-0.3-400-correct-distance.txt','rb')
data2 = pickle.load(f)
f.close()


df = pd.read_csv(pload[:-4] + 'pandas-all-data.csv')
df = df.loc[df['ntraj075']>=200]

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
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='coastline',scale='50m'),zorder=4, edgecolor='black')
#ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
#ax.coastlines()

#fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()))
#ax.coastlines()

for ul, date in enumerate(datadi.keys()):
    if np.all(ID!=int(date)):
        continue
    q = np.where(ID==int(date))[0][0]
#    if lat[q][abs(hourstoSLPmin[q][0]).astype(int)]<30 or lat[q][abs(hourstoSLPmin[q][0]).astype(int)]>48:
    if lat[q]<30 or lat[q]>48:
        continue
#    if lon[q][abs(hourstoSLPmin[q][0]).astype(int)]<-5 or lon[q][abs(hourstoSLPmin[q][0]).astype(int)]>42:
    if lon[q]<-5 or lon[q]>42:
        continue
#    if lon[q][abs(hourstoSLPmin[q][0]).astype(int)]<2 and lat[q][abs(hourstoSLPmin[q][0]).astype(int)]>42:
    if lon[q]<2 and lat[q]>42:
        continue

#        lat[q][abs(hourstoSLPmin[q][0]).astype(int)]=np.round(lat[q][abs(hourstoSLPmin[q][0]).astype(int)],0)
#    if lat[q][abs(hourstoSLPmin[q][0]).astype(int)]<30:
#        continue
#    if lon[q][abs(hourstoSLPmin[q][0]).astype(int)]%2!=0:
#        lon[q][abs(hourstoSLPmin[q][0]).astype(int)]=lon[q][abs(hourstoSLPmin[q][0]).astype(int)]-lon[q][abs(hourstoSLPmin[q][0]).astype(int)]%2
#    if lat[q][abs(hourstoSLPmin[q][0]).astype(int)]%2!=0:
#        lat[q][abs(hourstoSLPmin[q][0]).astype(int)]=lat[q][abs(hourstoSLPmin[q][0]).astype(int)]-lat[q][abs(hourstoSLPmin[q][0]).astype(int)]%2
#    lo = np.where((LON[:-1]<=lon[q][abs(hourstoSLPmin[q][0]).astype(int)])&(LON[1:]>lon[q][abs(hourstoSLPmin[q][0]).astype(int)]))[0][0]
    lo = np.where(abs(LON-lon[q])==np.min(abs(LON-lon[q])))[0][0]
#    la = np.where((LAT[:-1]<=lat[q][abs(hourstoSLPmin[q][0]).astype(int)])&(LAT[1:]>lat[q][abs(hourstoSLPmin[q][0]).astype(int)]))[0][0]
    la = np.where(abs(LAT-lat[q])==np.min(abs(LAT-lat[q])))[0][0]
    counter[la,lo]+=1

#    ax.scatter(lon[q][abs(hourstoSLPmin[q][0]).astype(int)],lat[q][abs(hourstoSLPmin[q][0]).astype(int)],marker='.',color='k',zorder=1,s=3)


#f = open('/home/ascherrmann/009-ERA-5/MED/counter5deg.txt','wb')
#pickle.dump(counter,f)
#f.close()

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

#for lo,la in zip(nonlon,nonlat):
#    ncounter[la,lo]=np.mean(counter[la-pm:la+pm+1,lo-pm:lo+pm+1])

cmap=ListedColormap(['darkcyan','lightseagreen','mediumturquoise','tan','peru','sienna','saddlebrown'])#[matplotlib.cm.BrBG[0],matplotlib.cm.BrBG[64],matplotlib.cm.BrBG[128],matplotlib.cm.BrBG[-1]])
cmap.set_over('brown')
cmap.set_under('white')
#ap = plt.cm.BrBG
levels = np.array([5,10,20,30,50,100,250,500,1000])
#cmap, norm = colbar(ap,levels)
#cmap.set_under('white')
#cmap.set_over('red')
norm = BoundaryNorm(levels,cmap.N)
ax.plot([],[],color='k',ls=' ',marker='.',markersize=2)
ax.plot([],[],color='k',ls=' ',marker='.',markersize=np.max(counter)/12)

#lati,loni = np.where(counter!=0)
#print(len(loni))
#ax.scatter(DLON[loni],DLAT[lati],marker='.',color='k',s=counter[lati,loni],zorder=150)
#ax.legend(['1 cyclone','150cyclones'],loc='upper left')#%d cyclones'%np.max(counter)],loc='upper left')
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
#ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='coastline',scale='50m'),zorder=4, edgecolor='black')
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
ax.set_extent([minpltlonc, maxpltlonc, minpltlatc, maxpltlatc], ccrs.PlateCarree())

#levels=np.array([1,2,4,6,8,10,12,14,16,18])
levels = np.array([1,3,5,10,15,20,25,30])
colors=['blue','dodgerblue','lightseagreen','orange','red','peru','saddlebrown','k']
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.append(levels,30),cmap.N)
cf = ax.contour(LON,LAT,ncounter,linewidths=0.5,colors='k',levels=levels)
plt.clabel(cf,inline=True,fontsize=6,fmt='%d',manual=True)

#for ul,date in enumerate(datadi2.keys()):
#    mon = data2['mons'][ul]
#    q = int(date[-3:])
#    ax.scatter(ILON[np.mean(locdata[mon][q]['clon'][abs(locdata[mon][q]['hzeta'][0]).astype(int)]).astype(int)],ILAT[np.mean(locdata[mon][q]['clat'][abs(locdata[mon][q]['hzeta'][0]).astype(int)]).astype(int)],marker='o',color='r',zorder=100,s=5)
#cf = ax.contourf(DLON,DLAT,counter,cmap=cmap,levels=levels,norm=norm,zorder=1,alpha=1,extend='max')

longrids=np.arange(-180,180,5)
latgrids=np.arange(-90,90,5)
#ax.gridlines(xlocs=longrids, ylocs=latgrids, linestyle='--',color='grey',zorder=1)

ax.plot([-5,42],[30,30],color='blue',zorder=1)
ax.plot([-5,-5],[30,42],color='blue',zorder=1)
ax.plot([-5,2],[42,42],color='blue',zorder=1)
ax.plot([2,2],[42,48],color='blue',zorder=1)
ax.plot([2,42],[48,48],color='blue',zorder=1)
ax.plot([42,42],[30,48],color='blue',zorder=1)

lonticks=np.arange(minpltlonc, maxpltlonc+1,steps)
latticks=np.arange(minpltlatc, maxpltlatc+1,steps)

ax.set_xticks(lonticks, crs=ccrs.PlateCarree());
ax.set_yticks(latticks, crs=ccrs.PlateCarree());
ax.set_xticklabels(labels=lonticks,fontsize=10)
ax.set_yticklabels(labels=latticks,fontsize=10)

ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())


#cbax = fig.add_axes([0, 0, 0.1, 0.1])
#cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=levels,cax=cbax)
#func=resize_colorbar_vert(cbax, ax, pad=0.0, size=0.02)
#fig.canvas.mpl_connect('draw_event', func)

fig.savefig('/home/ascherrmann/010-IFS/' + 'allcyclone-IFS-ERA5-NEW.png',dpi=300,bbox_inches="tight")
plt.close('all')
