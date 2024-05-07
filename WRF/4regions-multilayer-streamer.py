import pandas as pd
import os
import numpy as np
from netCDF4 import Dataset as ds

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'

minlon = -10
minlat = 30
maxlat = 65
maxlon = 60

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

pressures= ['250','300']
f = 'streamer-mask.nc'
for dirs in os.listdir(ps):
  if dirs!='321336' and dirs!='321738' and dirs!='062511' and dirs!='348396' and dirs!='078357' and dirs!='469480' and dirs!='498732' and dirs!='512666' and dirs!='103938' and dirs!='117648' and dirs!='119013' and dirs!='156879' and dirs!='185794' and dirs!='252628':
        continue
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
                    ### if the overlapping area is at least 60% of the 
                    ### mask of the layer below than assign it to the upper level
                    if len(np.where(ov==2)[0])/len(np.where(mc[k]==vv)[0])>=0.6:
                        mc[k][mc[k]==vv]=v
                        mcf[k][mcf[k]==vv]=v

    d['mask'][:] = mcf[:]
    d.close()
    ref.close()

            

            






