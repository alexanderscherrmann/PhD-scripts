import pandas as pd
import os
import numpy as np
from netCDF4 import Dataset as ds

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'

minlon = -10
minlat = 30
maxlat = 50
maxlon = 45

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

pressures= ['250','300','350','400','450']
f = 'streamer-mask.nc'
for dirs in os.listdir(ps):
  if dirs[-1]=='c' or dirs[-1]=='t':
      continue
  for q,pr in enumerate(pressures[:-1]):
    ref = ds(ps + dirs + '/%s/'%pr + f)
    mr = ref.variables['mask'][:,la0:la1,lo0:lo1]

    d = ds(ps + dirs + '/' + pressures[q+1] + '/' + f,'r+')
    mc = d.variables['mask'][:,la0:la1,lo0:lo1]
    mcf = d.variables['mask'][:]
    for k in range(mr.shape[0]):
        vr = np.unique(mr[k])[1:]
        vc = np.unique(mc[k])[1:]
        for v in vr:
            for vv in vc:
                ov = np.zeros_like(mr[k])
                ov[mr[k]==v]+=1
                ov[mc[k]==vv]+=1
                if np.any(ov==2):
                    if len(np.where(ov==2))/len(np.where(mc[k]==vv))>=0.6:
                        mc[k][mc[k]==vv]=v
                        mcf[k][mcf[k]==vv]=v
    d['mask'][:] = mcf[:]
    d.close()
    ref.close()

            

            






