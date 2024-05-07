import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

sim,at,med=wrfsims.cesm_ids()

tracks = '/atmosdyn2/ascherrmann/scripts/WRF/cyclone-tracking-wrf/out/'
ics='/atmosdyn2/ascherrmann/013-WRF-sim/ics/'
dwrf = '/atmosdyn2/ascherrmann/013-WRF-sim/'
sea='DJF'

colors=['k','blue','yellow','orange','red']
x0,y0,x1,y1=70,30,181,101

ofsetfac=[0,0.5,1,2]
xof=[0,-8,8,0,0]
yof=[0,0,0,-8,8]
names = ['-0-km','west','east','south','north']
km=['-0-km','200','400','800']
period=['ERA5','2010','2040','2070','2100']
refx=dict()
refy=dict()

pvdi=dict()
atslpdi=dict()
medslpdi=dict()

for perio in period:
    atslpdi[perio]=[]
    medslpdi[perio]=[]


if not os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data-no-weak-experiments.txt'):
    for si,a,m in zip(sim,at,med):
    #    if not '2070' in si and not '2100' in si:
    #        continue
        if si[-4:]=='clim':
            continue
        if '0.7' in si:
            continue
        a=np.array(a)
        m=np.array(m)
        if np.any(m==None):
            continue
            
        # position ofset
        ic = ds(dwrf + si + '/wrfout_d01_2000-12-01_00:00:00')
        tra = np.loadtxt(tracks + si + '-new-tracks.txt')
    
        # store
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]
    
        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        loc = np.where(IDs==1)[0]
        aminslp = np.min(slp[loc])
        if aminslp>1000:
            print(aminslp,si)
    
        atslpdi[perio].append(aminslp)
        medslpdi[perio].append(slpmin)


save=dict()
save['at']=atslpdi
save['med']=medslpdi

f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/CESM-slp-PV-data-no-weak-experiments.txt','wb')
pickle.dump(save,f)
f.close()


sim,at,med=wrfsims.upper_ano_only()
if not os.path.isfile('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data-no-weak-experiments.txt'):
    e5pv=[]
    e5atslp=[]
    e5medslp=[]
    for si,a,m in zip(sim,at,med):
        if si[-4:]=='clim' or 'nested' in si or '0.5' in si or '0.3' in si or 'not' in si or '0.9' in si or '1.1' in si or '1.7' in si or '2.8' in si or 'DJF' not in si or 'check' in si or '0.7' in si:
            continue

        a=np.array(a)
        m=np.array(m)
        # get correct position
        tra = np.loadtxt(tracks + si + '-new-tracks.txt')

        # store
        t = tra[:,0]
        tlon,tlat = tra[:,1],tra[:,2]
        slp = tra[:,3]
        IDs = tra[:,-1]

        loc = np.where(IDs==2)[0]
        slpmin = np.min(slp[loc])
        loc = np.where(IDs==1)[0]
        aminslp = np.min(slp[loc])
        e5atslp.append(aminslp)
        e5medslp.append(slpmin)

    era5di=dict()
    era5di['at']=e5atslp
    era5di['med']=e5medslp

    f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/paper-ERA5-slp-pv-data-no-weak-experiments.txt','wb')
    pickle.dump(era5di,f)
    f.close()

