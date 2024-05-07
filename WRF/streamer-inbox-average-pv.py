import pickle
from netCDF4 import Dataset as ds
import numpy as np
from wrf import interplevel as intp

f = open('/atmosdyn2/ascherrmann/013-WRF-sim/data/streamer-in-boxes.txt','rb')
d = pickle.load(f)
f.close()
    
era5 = '/atmosdyn/era5/cdf/'
LON = np.linspace(-180,179.5,720)
LAT = np.linspace(-90,90,361)

avpv = dict()
lonmin,lonmax,latmin,latmax = -120,80,10,80
loi,lai = np.where((LON>=lonmin) & (LON<=lonmax))[0], np.where((LAT>=latmin) & (LAT<=latmax))[0]
lo0,lo1,la0,la1 = loi[0],loi[-1]+1,lai[0],lai[-1]+1

for k in d.keys():
    avpv[k] = dict()
    
    for da in d[k]:
        avpv[k][da] = np.zeros((len(lai),len(loi)))
        f = era5 + da[:4] + '/' + da[4:6] + '/S' + da
        dt = ds(f,mode='r')
        PV = dt.variables['PV'][0,:,la0:la1,lo0:lo1]
        TH = dt.variables['TH'][0,:,la0:la1,lo0:lo1]
        PV315 = intp(PV,TH,315,meta=False)
        avpv[k][da] = PV315

    avpv[k]['n'] = len(da)
    
f = open('all-dates-streamer-boxes-avpv.txt','wb')
pickle.dump(avpv,f)
f.close()
