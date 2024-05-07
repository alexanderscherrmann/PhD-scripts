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

lon1,lat1,lon2,lat2 = -8,20,1.5,40
lon3,lat3,lon4,lat4 = 1.5,20,50,48

lons1,lats1 = np.where((LON>=lon1)&(LON<=lon2))[0],np.where((LAT>=lat1)&(LAT<=lat2))[0]
lons2,lats2 = np.where((LON>=lon3)&(LON<=lon4))[0],np.where((LAT>=lat3)&(LAT<=lat4))[0]
lo1,lo2,lo3,lo4,la1,la2,la3,la4 = lons1[0],lons1[-1]+1,lons2[0],lons2[-1]+1,lats1[0],lats1[-1]+1,lats2[0],lats2[-1]+1

### time of slp measure of atlantic cyclone
hac = 12

PVpos = [[52,28],[145,84],[93,55],[55,46],[84,59],[73,59]]
names = np.array([])
MINslp = np.array([])
MAXpv = np.array([])
t = np.arange(0,192,3)
for d in os.listdir(dwrf):
    if d=='DJF-clim-max-U-at-300hPa-hourly-2.1QG' or d=='sat-DJF-clim-max-U-at-300hPa-hourly-2.1QG':
        continue
    if not d.startswith('DJF-clim-max') and not d.startswith('DJF-clim-double') and not d.startswith('MAM-clim-max') and not d.startswith('JJA-clim-max') and not d.startswith('SON-clim-max') and not d.startswith('DJF-L300') and not d.startswith('DJF-L500'):
        continue

    minSLPmed = np.array([])

    for day in range(1,9):
        for hour in range(0,24,3):

            tmpslp = np.zeros(2)
            data = ds(dwrf + d + '/wrfout_d01_2000-12-%02d_%02d:00:00'%(day,hour))
            slp = data.variables['MSLP'][0,la1:la2,lo1:lo2]
            tmpslp[0] = np.min(slp)
            slp = data.variables['MSLP'][0,la3:la4,lo3:lo4]
            tmpslp[1] = np.min(slp)

            minSLPmed = np.append(minSLPmed,np.min(tmpslp))
    
    np.savetxt(tracks + d + '-minMedSLP.txt',np.stack((t,minSLPmed),axis=1),fmt='%.2f',delimiter=' ',newline='\n')
