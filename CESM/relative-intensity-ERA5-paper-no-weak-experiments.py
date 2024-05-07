import sys
sys.path.append('/home/ascherrmann/scripts/')
import wrfsims
import numpy as np
from netCDF4 import Dataset as ds
import matplotlib.pyplot as plt
import wrf
import pickle
import os

sim,at,med=wrfsims.upper_ano_only()

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
period=['DJF']

refx=dict()
refy=dict()

pvdi=dict()
atslpdi=dict()
medslpdi=dict()

for perio in period:
    pvdi[perio]=[]
    atslpdi[perio]=[]
    medslpdi[perio]=[]

clim = dict()
for perio in period:
    clim[perio]=ds('/atmosdyn2/ascherrmann/013-WRF-sim/%s-clim/wrfout_d01_2000-12-01_00:00:00'%perio)
LON = clim[perio].variables['XLONG'][0,0]
LAT = clim[perio].variables['XLAT'][0,:,0]

save = dict()
check=np.array(['0.3','0.5','0.9','1.1','1.7','2.8'])
for si,a,m in zip(sim,at,med):
#    if not '2070' in si and not '2100' in si:
#        continue
    if si[-4:]=='clim' or 'not' in si or not perio in si or 'AT' in si or 'check' in si or '0.7' in si:
        continue
    

    for c in check:
        if c in si:
            break
    if c in si:
        continue

    a=np.array(a)
    m=np.array(m)
    if np.any(m==None):
        continue
        
    # load
    tra = np.loadtxt(tracks + si + '-new-tracks.txt')

    # store
    t = tra[:,0]
    tlon,tlat = tra[:,1],tra[:,2]
    slp = tra[:,3]
    IDs = tra[:,-1]
    
    loc = np.where(IDs==2)[0]
    slpmin = np.min(slp[loc])

    mstage = np.where(slp[loc]==np.min(slp[loc]))[0][0]
    mlon,mlat = tlon[loc[mstage]],tlat[loc[mstage]]
    lo,la = np.where(abs(LON-mlon)==np.min(abs(LON-mlon)))[0][0],np.where(abs(LAT-mlat)==np.min(abs(LAT-mlat)))[0][0]
    for perio in period:
        if perio in si:
            slpclim = clim[perio].variables['MSLP'][0,np.where(abs(LAT-mlat)==np.min(abs(LAT-mlat)))[0][0],np.where(abs(LON-mlon)==np.min(abs(LON-mlon)))[0][0]]
            print(si,perio,slpclim,slpmin,mlon,lo,mlat,la)
            medslpdi[perio].append(slpmin-slpclim)

    loc = np.where(IDs==1)[0]
    slpmin = np.min(slp[loc])

    mstage = np.where(slp[loc]==np.min(slp[loc]))[0][0]
    mlon,mlat = tlon[loc[mstage]],tlat[loc[mstage]]
    lo,la = np.where(abs(LON-mlon)==np.min(abs(LON-mlon)))[0][0],np.where(abs(LAT-mlat)==np.min(abs(LAT-mlat)))[0][0]
    for perio in period:
        if perio in si:
            slpclim = clim[perio].variables['MSLP'][0,np.where(abs(LAT-mlat)==np.min(abs(LAT-mlat)))[0][0],np.where(abs(LON-mlon)==np.min(abs(LON-mlon)))[0][0]]
            print(si,perio,slpclim,slpmin,mlon,lo,mlat,la)
            atslpdi[perio].append(slpmin-slpclim)

for perio in medslpdi.keys():
    print(perio,len(medslpdi[perio]))
save['med'] = medslpdi
save['at'] = atslpdi

f=open('/atmosdyn2/ascherrmann/015-CESM-WRF/delta-SLP-track-ERA5-paper-no-weak-experiments.txt','wb')
pickle.dump(save,f)
f.close()
