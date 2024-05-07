import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt


SIMS,ATIDS,MEDIDS = wrfsims.upper_ano_only()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/home/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'

ref = ds(dwrf + 'DJF-clim/wrfout_d01_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')[0]
LAT = wrf.getvar(ref,'lat')[:,0]

lons,lats = np.where((LON>=-10)&(LON<=50))[0],np.where((LAT>=25)&(LAT<=55))[0]
lo0,lo1,la0,la1 = lons[0],lons[-1],lats[0],lats[-1]


slons1,slats1 = np.where((LON>=-5)&(LON<2))[0],np.where((LAT>=25)&(LAT<=42))[0]
slo0,slo1,sla0,sla1 = slons1[0],slons1[-1],slats1[0],slats1[-1]
slons2 = np.where((LON>=2)&(LON<50))[0]
slo20,slo21 = slons2[0],slons2[-1]
name = ['MED','track']

di = dict()
for n in name:
    di[n] = dict()

track20=0
counter=0
SONclimc=1
di = dict()
for simid2,sim in enumerate(SIMS):
    simid = simid2+track20
    if "-not" in sim:
        continue
    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])

    if medid.size==0:
        continue
    if SONclimc==2 and sim=='SON-clim':
        sim+='_2'

    di[sim] = dict()
    di[sim]['track'] = np.zeros((41,41))
    di[sim]['MED'] = np.zeros((lats.size,lons.size))

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    locat,locmed = np.where(IDs==1)[0],np.where(IDs==2)[0] 
    loc = [locat,locmed]
    for q,l in enumerate(loc):
        # loop over track time select cyc center in lon,lat
        # add precip +-25 gridpoints to 2D zeros
        # save summed precip in figure
        # make med region precipiation, around track time and plot cyclone track into it.

        if q==0:
            continue
        for qq2,T in enumerate(t[l]):
            qq=qq2
            dd = 1 + int(int(T)/24)
            hh = int(T)%24

            print(sim,T/24,"%02d"%dd)

            if dd>=10:
                print('fuck')
            
            if SONclimc==2 and sim=='SON-clim_2':
                sim = 'SON-clim'

            out = ds(dwrf + sim + '/wrfout_d01_2000-12-%02d_%02d:00:00'%(dd,hh))
            rainc = wrf.getvar(out,'RAINC')
            rainnc = wrf.getvar(out,'RAINNC')

            if qq==0:
                rainold = rainc + rainnc

            rain = (rainc + rainnc)
            lon = tlon[l[qq]]
            lat = tlat[l[qq]]
            loloc,laloc = np.where(abs(LON-lon)==np.min(abs(LON-lon)))[0][0],np.where(abs(LAT-lat)==np.min(abs(LAT-lat)))[0][0]
            lifetime = t[l[-1]]-t[l[0]]
            if loloc>=380:
                continue
            di[sim]['track'] += (rain-rainold)[laloc-20:laloc+21,loloc-20:loloc+21]
            di[sim]['lifetime'] = lifetime
            rainold = rain

        out = ds(dwrf + sim + '/wrfout_d01_2000-12-03_00:00:00')
        rainc3 = wrf.getvar(out,'RAINC')
        rainnc3 = wrf.getvar(out,'RAINNC')

        out = ds(dwrf + sim + '/wrfout_d01_2000-12-09_00:00:00')
        rainc9 = wrf.getvar(out,'RAINC')
        rainnc9 = wrf.getvar(out,'RAINNC')

        di[sim]['MED'] = (rainc9 + rainnc9 - rainc3 -rainnc3)[la0:la1+1,lo0:lo1+1]
        di[sim]['totalprecip'] = np.sum((rainc9 + rainnc9 - rainc3 -rainnc3)[sla0:sla1+1,slo0:slo1+1]) + np.sum((rainc9 + rainnc9 - rainc3 -rainnc3)[la0:la1+1,slo20:slo21+1])
            
import pickle
f = open('/atmosdyn2/ascherrmann/013-WRF-sim/precipitation-dict.txt','wb')
pickle.dump(di,f)
f.close()
