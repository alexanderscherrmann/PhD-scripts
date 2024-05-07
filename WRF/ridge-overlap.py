from netCDF4 import Dataset as ds
import numpy as np
import os
import pandas as pd
import pickle


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import os
import cartopy.crs as ccrs
import cartopy
import matplotlib.gridspec as gridspec


minlon = -70
minlat = 30
maxlat = 75
maxlon = 45

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
checkval=0.8
minn=300
f = open(ps + 'same-ridge-streamer-%.1f-%d.txt'%(checkval,minn),'rb')
d = pickle.load(f)
f.close()

p ='300'
s = 'streamer'
r = 'ridge'

overlap = np.zeros((len(lats),len(lons)))

refridge = ds(ps + '325832/300/ridge-mask.nc','r')
refridge = refridge.variables['mask'][56,la0:la1,lo0:lo1]

refstream = ds(ps + '325832/300/old-mask.nc','r')
refstream = refstream.variables['mask'][56,la0:la1,lo0:lo1]

overlap[refstream==3777]+=1
overlap[refridge==4992]-=1

counter=0
for k in d.keys():
    if len(d[k][p][s])==0 or len(d[k][p][r])==0:
        continue

    tmp = ds(ps + '%06d/300/streamer-mask.nc'%k,'r')
    overlapstream = np.array([])
    for l in d[k][p][s]:
        sov = np.zeros_like(overlap)
        f = l[0]
        sov[refstream==3777]+=1
        sov[tmp.variables['mask'][f,la0:la1,lo0:lo1]==l[1]]+=1
        overlapstream = np.append(overlapstream,len(np.where(sov==2)[0]))

    f,l = d[k][p][s][np.argmax(overlapstream)]
    mask = tmp.variables['mask'][f,la0:la1,lo0:lo1]
    overlap[mask==l]+=1

    ridgeoverlap=np.array([])
    tmp = ds(ps + '%06d/300/ridge-mask.nc'%k,'r')
    for l in d[k][p][r]:
        rov = np.zeros_like(overlap)
        f = l[0]
        rov[refridge==4992]+=1
        rov[tmp.variables['mask'][f,la0:la1,lo0:lo1]==l[1]]+=1
        ridgeoverlap = np.append(ridgeoverlap,len(np.where(rov==2)[0]))
    

    f,l = d[k][p][r][np.argmax(ridgeoverlap)]
    mask = tmp.variables['mask'][f,la0:la1,lo0:lo1]
    overlap[mask==l]-=1
    counter+=1
   

print(counter)
fig = plt.figure(figsize=(6,4))
gs = gridspec.GridSpec(nrows=1,ncols=1)
ax = fig.add_subplot(gs[0,0],projection=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',name='land',scale='50m'),zorder=0, edgecolor='black',facecolor='lightgrey',alpha=0.7)
ax.add_feature(cartopy.feature.GSHHSFeature(scale='low'),zorder=1, edgecolor='black')
ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())

print(np.max(overlap),np.min(overlap))
overlap[overlap>0] /=counter
overlap[overlap<0] /=(-1*counter)

ax.contour(LON[lons],LAT[lats],overlap * 100, levels=np.arange(20,101,10),colors=['k','green','lime','b','purple','r','orange','cyan'],linewdiths=1)
ax.set_xticks([])
ax.set_yticks([])

name = 'max-ridge-streamer-density-%.1f-%d.png'%(checkval,minn)
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'

fig.savefig(pi + name,dpi=300,bbox_inches='tight')
plt.close('all')

