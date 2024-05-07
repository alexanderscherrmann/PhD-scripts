import wrf
from netCDF4 import Dataset as ds
import os
import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import helper
import numpy as np
import matplotlib.pyplot as plt


SIMS,ATIDS,MEDIDS,MEDIDS2 = wrfsims.nested_ids()
SIMS = np.array(SIMS)
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
tracks = '/atmosdyn2/ascherrmann/scripts/WRF/nested-cyclone-tracking/out/'

ref = ds(dwrf + 'nested-DJF-clim-max-U-at-300hPa-0.3QG/wrfout_d02_2000-12-01_00:00:00')
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

name = ['AT','MED']
var=['t','slp','PV850','PV300','PV400-200','PV950-700']

di = dict()
for n in name:
    di[n] = dict()

### connect tracks
#track20=180
#for simid2, sim in enumerate(SIMS[track20:track20+20]):
for simid,sim in enumerate(SIMS):
    print(sim)

    medid = np.array(MEDIDS2[simid])
    atid = np.array(ATIDS[simid])
    if sim=='nested-test' or sim.startswith('DJF-nested'):
        continue
    if medid.size==0:
        continue

    if not 'DJF' in sim:
        continue
    tra = np.loadtxt(tracks + sim + '-02-new-tracks.txt')
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]

    locat,locmed = np.where(IDs==1)[0],np.where(IDs==2)[0] 
    loc = [locat,locmed]
    for q,l in enumerate(loc):
        for v in var:
            di[name[q]][v] = np.zeros_like(t[l])

        di[name[q]]['t'] = t[l]
        di[name[q]]['slp'] = slp[l]
        if q==0:
            continue

        for v in var:
            di[name[q]][v] = np.zeros_like(t[l])

        di[name[q]]['t'] = t[l]
        di[name[q]]['slp'] = slp[l]
        
        for qq2,T in enumerate(di[name[q]]['t']):
            qq=qq2
            dd = 1 + int(int(T)/24)
            hh = int(T)%24
            
            out = ds(dwrf + sim + '/wrfout_d02_2000-12-%02d_%02d:00:00'%(dd,hh))
            PV = wrf.getvar(out,'pvo')
            P = wrf.getvar(out,'pressure')
            lon = tlon[l[qq]]
            lat = tlat[l[qq]]

            dlon = LON-lon
            dlat = LAT-lat

            distance = helper.convert_dlon_dlat_to_radial_dis_new(dlon,dlat,LAT)
            lai200,loi200 = np.where(distance<=100)
            lai400,loi400 = np.where(distance<=400)


            PV850 = wrf.interplevel(PV,P,850,meta=False)
            pv850 = np.zeros(lai200.size)
            delids = np.array([])
            for qqq,la2,lo2 in zip(range(pv850.size),lai200,loi200):
                if not PV850.mask[la2,lo2]:
                    pv850[qqq] = PV850[la2,lo2]
                else: 
                    delids= np.append(delids,qqq)

            if delids.size!=0:
                pv850 = np.delete(pv850,delids.astype(int))
            PV850 = np.mean(pv850)

            print(sim,name[q],T,'done PV850')

            ##PV @ 300

            PV300 = wrf.interplevel(PV,P,300,meta=False)            
            pv300 = np.zeros(lai400.size)
            for qqq,la4,lo4 in zip(range(pv300.size),lai400,loi400):
                pv300[qqq] = PV300[la4,lo4]

#            PV300 = np.sum(np.sort(pv300)[int(pv300.size/2):])/(pv300.size/2)
            PV300 = np.mean(pv300)

            print(sim,name[q],T,'done PV300')
            ### get indeces around track within certain distance

            di[name[q]]['PV850'][qq] = PV850
            di[name[q]]['PV300'][qq] = PV300

            print(sim,name[q],T,'next T')

    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    pv300 = np.append(di['AT']['PV300'],di['MED']['PV300'])
    pv850 = np.append(di['AT']['PV850'],di['MED']['PV850'])

    np.savetxt(tracks + sim + '-test-PV-tracks.txt',np.stack((t,tlon,tlat,slp,pv300,pv850,IDs),axis=1),fmt='%.2f',delimiter=' ',newline='\n')




