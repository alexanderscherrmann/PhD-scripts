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
LON = wrf.getvar(ref,'lon')
LAT = wrf.getvar(ref,'lat')

#PVwithin 200km at 850
#PVwithin 400,600km at 300
#AVPVwithin 400-200hPa within 400km 
#AVPVwithin 950-700hPa within 200km
name = ['AT','MED']
var=['t','slp','PV850','PV300','PV400-200','PV950-700']

di = dict()
for n in name:
    di[n] = dict()

### connect tracks
#track20=180
track20=0
counter=0
SONclimc=1
#for simid2, sim in enumerate(SIMS[track20:track20+20]):
for simid2,sim in enumerate(SIMS):
#    if sim!='MAM-200-km-south-from-max-300-hPa-2.1-QGPV' and sim!='MAM-200-km-west-from-max-300-hPa-2.1-QGPV':
#    if sim!='JJA-400-km-west-from-max-300-hPa-0.7-QGPV':
#        continue
    print(sim)
    simid = simid2+track20
    if "-not" in sim:
        continue
    
    if sim!='DJF-clim' and sim!='DJF-clim-max-U-at-300-hPa-0.3-QGPV' and sim!='DJF-clim-max-U-at-300-hPa-0.5-QGPV':
        continue

    medid = np.array(MEDIDS[simid])
    atid = np.array(ATIDS[simid])

    if medid.size==0:
        continue
    if SONclimc==2 and sim=='SON-clim':
        sim+='_2'

    tra = np.loadtxt(tracks + sim + '-new-tracks.txt')
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
        for qq2,T in enumerate(di[name[q]]['t']):
            qq=qq2
            dd = 1 + int(int(T)/24)
            hh = int(T)%24
            print(sim,T/24,"%02d"%dd)
            if dd>=10:
                print('fuck')
            
            if SONclimc==2 and sim=='SON-clim_2':
                sim = 'SON-clim'
            out = ds(dwrf + sim + '/wrfout_d01_2000-12-%02d_%02d:00:00'%(dd,hh))
            PV = wrf.getvar(out,'pvo')
            P = wrf.getvar(out,'pressure')
            lon = tlon[l[qq]]
            lat = tlat[l[qq]]

            dlon = LON-lon
            dlat = LAT-lat


            distance = helper.convert_dlon_dlat_to_radial_dis_new(dlon,dlat,LAT)
            lai200,loi200 = np.where(distance<=100)
            lai400,loi400 = np.where(distance<=400)

            lo200_0,lo200_1,la200_0,la200_1 = int(np.min(loi200)),int(np.max(loi200))+1,int(np.min(lai200)),int(np.max(lai200))+1
            lo400_0,lo400_1,la400_0,la400_1 = int(np.min(loi400)),int(np.max(loi400))+1,int(np.min(lai400)),int(np.max(lai400))+1
   #         lo600_0,lo600_1,la600_0,la600_1 = int(np.min(loi600)),int(np.max(loi600))+1,int(np.min(lai600)),int(np.max(lai600))+1
            print(sim,name[q],T,'done distance')
            ## PV @ 850
            PV850 = wrf.interplevel(PV[:,la200_0:la200_1,lo200_0:lo200_1],P[:,la200_0:la200_1,lo200_0:lo200_1],850,meta=False)
            PV850 = np.mean(PV850[lai200-la200_0,loi200-lo200_0])

            print(sim,name[q],T,'done PV850')
            ## PV between 950 and 700 hPa

#            dis3d = np.tile(distance,(P.shape[0],1,1))
#            z,y,x = np.where((P[:,la200_0:la200_1,lo200_0:lo200_1]<=950)&(P[:,la200_0:la200_1,lo200_0:lo200_1]>=700) & (dis3d[:,la200_0:la200_1,lo200_0:lo200_1]<=200))
#            PV950_700 = PV[:,la200_0:la200_1,lo200_0:lo200_1]
#            PV950_700 = np.mean(PV950_700[z,y,x])

            print(sim,name[q],T,'done PV950-700')
            ##PV @ 300

            PV300 = wrf.interplevel(PV[:,la400_0:la400_1,lo400_0:lo400_1],P[:,la400_0:la400_1,lo400_0:lo400_1],300,meta=False)
            PV300 = PV300[lai400-la400_0,loi400-lo400_0]
            PV300 = np.mean(np.sort(PV300)[int(PV300.size/2):])

            print(sim,name[q],T,'done PV300')
            ## PV between 400 and 200 hPa

#            z,y,x = np.where((P[:,la400_0:la400_1,lo400_0:lo400_1]<=400)&(P[:,la400_0:la400_1,lo400_0:lo400_1]>=200) & (dis3d[:,la400_0:la400_1,lo400_0:lo400_1]<=400))
#            PV400_200 = PV[:,la400_0:la400_1,lo400_0:lo400_1]
#            PV400_200 = np.mean(PV400_200[z,y,x])

            print(sim,name[q],T,'done PV400-200')
            ### get indeces around track within certain distance

            di[name[q]]['PV850'][qq] =  PV850
            di[name[q]]['PV300'][qq] =  PV300
#            di[name[q]]['PV950-700'][qq] =  PV950_700
#            di[name[q]]['PV400-200'][qq] =  PV400_200

            print(sim,name[q],T,'next T')

    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    pv300 = np.append(di['AT']['PV300'],di['MED']['PV300'])
    pv850 = np.append(di['AT']['PV850'],di['MED']['PV850'])
    pv400_200 = np.append(di['AT']['PV400-200'],di['MED']['PV400-200'])
    pv950_700 = np.append(di['AT']['PV950-700'],di['MED']['PV950-700'])
    print(sim,'presave')
    if SONclimc==2 and sim=='SON-clim':
        sim+='_2'
    np.savetxt(tracks + sim + '-PV-tracks-2.txt',np.stack((t,tlon,tlat,slp,pv300,pv850,pv400_200,pv950_700,IDs),axis=1),fmt='%.2f',delimiter=' ',newline='\n')
    print(sim,'saved')
    if SONclimc==1 and sim=='SON-clim':
        SONclimc+=1
    counter=1




