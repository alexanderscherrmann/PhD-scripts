import numpy as np
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import os
import pickle

box = [-2.5,-2.5,2.5,2.5]
boxclon = np.arange(2.5,20,5)
boxclat = np.arange(37.5,45,5)

LON = np.linspace(-180,179.5,720)
LAT = np.linspace(-90,90,361)

era5 = '/atmosdyn/era5/cdf/'

boxdi = dict()
for blon in boxclon:
    for blat in boxclat:
        boxdi['%.1f%.1f'%(blon,blat)] = np.array([])

for y in range(1979,2021):
    for m in range(1,13):
        yy = str(y)
        mm = '%02d'%m
        cdfp = era5 + yy + '/' + mm + '/'

        for f in os.listdir(cdfp):
          if f.startswith('S') and (f.endswith('_00') or f.endswith('_12')):
            da = ds(cdfp + f,mode='r')
            PV = da.variables['PV'][:]
            TH = da.variables['TH'][:]
            PV315 = intp(PV,TH,315,meta=False)

            for blon in boxclon:
                for blat in boxclat:
                    lonmin,lonmax,latmin,latmax = blon+box[0],blon+box[2],blat+box[1],blat+box[3]
                    loi,lai = np.where((LON>=lonmin) & (LON<=lonmax))[0], np.where((LAT>=latmin) & (LAT<=latmax))[0]
                    lo0,lo1,la0,la1 = loi[0],loi[-1]+1,lai[0],lai[-1]+1
                    if len(np.where(PV315[la0:la1,lo0:lo1]>=6)[0])>=10:
                        boxdi['%.1f%.1f'%(blon,blat)] = np.append(boxdi['%.1f%.1f'%(blon,blat)],f[1:])

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
f = open(ps + 'streamer-in-boxes.txt','wb')
pickle.dump(boxdi,f)
f.close()
