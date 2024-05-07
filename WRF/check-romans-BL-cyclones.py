# coding: utf-8
import numpy as np
import pickle
pload = '/atmosdyn2/ascherrmann/011-all-ERA5/data/'
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
import xarray as xr
import cartopy
import matplotlib.gridspec as gridspec
import pandas as pd

trf = np.loadtxt('FEB18-TRACKED_CYCLONES',skiprows=1)
trm = np.loadtxt('MAR18-TRACKED_CYCLONES',skiprows=1)
tra = np.loadtxt('APR18-TRACKED_CYCLONES',skiprows=1)

id1 = 15
id2 = 18
id3 = 10
id4 = 36

idf1 = np.where(trf[:,1]==id1)[0]
idf2 = np.where(trf[:,1]==id2)[0]
idm = np.where(trm[:,1]==id3)[0]
ida = np.where(trm[:,1]==id4)[0]

f15 = trf[idf1]
f17 = trf[idf2]
m20 = trm[idm]
a25 = tra[ida]




from datetime import datetime, date, timedelta
ft = date.toordinal(date(1950,1,1))

fig=plt.figure(figsize=(8,6))
gs = gridspec.GridSpec(ncols=1, nrows=1)
ax=fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.5)

DATE = [['20180206_01','20180208_08'],['20180205_14','20180208_09'],['20180304_19','20180309_14'],['20180406_10','20180407_12']]

trs = [trf,trf,trm,tra]
IDS2 = [id1,id2,id3,id4]


#for d,tr,idr in zip(DATE,trs,IDS2):
#    di = d[0]
#    de = d[1]
#    for tmpi in range(1,idr+10):
#        tmpids = np.where(tr[:,1]==tmpi)[0]
#        k = str(helper.datenum_to_datetime(ft+tr[tmpids[0],0]/24))
#        dic = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
#        k = str(helper.datenum_to_datetime(ft+tr[tmpids[-1],0]/24))
#        dec = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
#        if dic==di and dec==de:
#            print(d,tmpi)
#


ax.plot(f15[:,2],f15[:,3],color='red')
ax.scatter(f15[0,2],f15[0,3],color='red',s=40)
ids = np.where(f15[:,6]==np.min(f15[:,6]))[0][0]
ax.scatter(f15[ids,2],f15[ids,3],color='k',s=40)
ax.plot(f17[:,2],f17[:,3],color='k')
ax.plot(m20[:,2],m20[:,3],color='g')
ax.plot(a25[:,2],a25[:,3],color='b')
ax.axvline(-90)
ax.axhline(25)
ax.axvline(0)

fig.savefig('/atmosdyn2/ascherrmann/012-WRF-cyclones/track.png',dpi=300,bbox_inches="tight")
plt.close('all')

lons=[]
lats=[]
dates=[]

LON = np.round(np.linspace(-180,180,901),1)
LAT = np.round(np.linspace(0,90,226),1)

ids = np.where(f15[:,6]==np.min(f15[:,6]))[0][0]
k = str(helper.datenum_to_datetime(ft+f15[0,0]/24))
Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
print(Date)
lon,lat =f15[ids,2], f15[ids,3]
dates.append(Date)
lons.append(lon)
lats.append(lat)

ids = np.where(f17[:,6]==np.min(f17[:,6]))[0][0]
k = str(helper.datenum_to_datetime(ft+f17[ids,0]/24))
Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
lon,lat =f17[ids,2], f17[ids,3]
dates.append(Date)
lons.append(lon)
lats.append(lat)

ids =np.where(m20[:,6]==np.min(m20[:,6]))[0][0]
k = str(helper.datenum_to_datetime(ft+m20[ids,0]/24))
Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
lon,lat =m20[ids,2], m20[ids,3]
dates.append(Date)
lons.append(lon)
lats.append(lat)

ids = np.where(a25[:,6]==np.min(a25[:,6]))[0][0]
k = str(helper.datenum_to_datetime(ft+a25[ids,0]/24))
Date = k[0:4]+k[5:7]+k[8:10]+'_'+k[11:13]
lon,lat =a25[ids,2], a25[ids,3]
dates.append(Date)
lons.append(lon)
lats.append(lat)

months = ['FEB18/cdf/','FEB18/cdf/','MAR18/cdf/','APR18/cdf/']
IDS = [id1,id2,id3,id4]
paths = ['/atmosdyn2/atroman/phd/','/atmosdyn2/atroman/phd/','/atmosdyn2/atroman/phd/','/atmosdyn/atroman/phd/']
a = 0
for da,lo,la,j in zip(dates,lons,lats,IDS):
    CLONIDS, CLATIDS = helper.IFS_radial_ids_correct(200,LAT[np.where(LAT==np.round(la,1))[0][0]])
    addlon = CLONIDS + np.where(LON==np.round(lo,1))[0][0]
    addlon[np.where((addlon-900)>0)] = addlon[np.where((addlon-900)>0)]-900
    clon = addlon.astype(int)
    clat = CLATIDS.astype(int) + np.where(LAT==np.round(la,1))[0][0]

    s = xr.open_dataset(paths[a] + months[a] + 'S' + da)
    pt = np.array([])
    plat = np.array([])
    plon = np.array([])
    PS = s.PS.values[0,0,clat,clon]
    pv = s.PV.values[0]
    for l in range(len(clat)):
        P = helper.modellevel_to_pressure(PS[l])
        pid = np.where((P>=700) & (P<=975) & (pv[:,clat[l],clon[l]]>=0.75))[0]
        for i in pid:
               pt = np.append(pt,P[i])
               plat = np.append(plat,LAT[clat[l]])
               plon = np.append(plon,LON[clon[l]])

    save = np.zeros((len(pt),4))
    save[:,1] = plon
    save[:,2] = plat
    save[:,3] = pt
    a+=1
    np.savetxt('trastart-mature-' + da + '-ID-' + '%06d'%j + '.txt',save,fmt='%f', delimiter=' ', newline='\n')
