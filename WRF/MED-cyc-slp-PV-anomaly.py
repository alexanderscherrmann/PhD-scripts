import wrf
from netCDF4 import Dataset as ds
import os
import numpy as np
import matplotlib.pyplot as plt

dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

lon1,lat1,lon2,lat2 = -5,25,2,42
lon3,lat3,lon4,lat4 = 2,25,50,48

#lons1,lats1 = np.where((LON>=lon1) & (LON<=lon2))[0], np.where((LAT>=lat1) & (LAT<=lat2))[0]
#lons2,lats2 = np.where((LON>=lon3) & (LON<=lon4))[0], np.where((LAT>=lat3) & (LAT<=lat4))[0]

#lo10,lo11,la10,la11 = lons1[0],lons1[-1],lats1[0],lats1[-1]
#lo20,lo21,la20,la21 = lons2[0],lons2[-1],lats2[0],lats2[-1]

PVpos = [[52,28],[145,84],[94,56]]
names = np.array([])
counter = 1
MINslp = np.array([])
MAXpv = np.array([])
for d in os.listdir(dwrf):
    if not d.startswith('DJF-clim-'):
        continue

    print(counter-1,d)

    ic = ds(dwrf + d + '/wrfout_d01_2000-12-01_00:00:00')
    tra = np.loadtxt(tracks + d + '-filter.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]

    locs = np.where( (((tlon>=lon1) & (tlon<=lon2) & (tlat>=lat1) & (tlat<=lat2)) | ((tlon>=lon3) & (tlon<=lon4) & (tlat>=lat3) & (tlat<=lat4))) & (t<=156))[0]

    PV = wrf.getvar(ic,'pvo')
    p = wrf.getvar(ic,'pressure')

    pv = wrf.interplevel(PV,p,300,meta=False)
    maxpv = np.zeros(3)
    for q,l in enumerate(PVpos):
        maxpv[q] = pv[l[1],l[0]]

    maxpv = np.max(maxpv)
    minslp = np.min(slp[locs])


    MINslp = np.append(MINslp,minslp)
    MAXpv = np.append(MAXpv,maxpv)
    counter+=1

fig,ax = plt.subplots()

ax.scatter(MAXpv,MINslp,color='k')
for q,pv,slp in zip(range(MAXpv.size),MAXpv,MINslp):
    ax.text(pv,slp,'%d'%q)

ax.set_xlabel('PV anomaly [PVU]')
ax.set_ylabel('MED cyclone min SLP [hPa]')

name = dwrf + 'image-output/MED-cyclone-PV-anomaly-SLP-scatter.png'
fig.savefig(name,dpi=300,bbox_inches='tight')
plt.close('all')

