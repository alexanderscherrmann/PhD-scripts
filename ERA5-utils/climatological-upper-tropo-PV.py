import numpy as np
import netCDF4
import os
import pickle

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)

### all ids in box we regard as Mediterranean
###
LON=np.arange(-180,180,0.5)
LAT=np.arange(-90,90.1,0.5)

LSM = readcdf('/home/ascherrmann/009-ERA-5/MED/data/NORO','OL')


xx,yy=np.meshgrid(LON,LAT)

latids1,lonids1 = np.where((xx>=-5) & (xx<2) & (yy>=30) & (yy<=42))
latids2,lonids2 = np.where((xx>=2) & (xx<=42) & (yy>=30) & (yy<48))

lonids = np.append(lonids1,lonids2)
latids = np.append(latids1,latids2)


c = dict()
ca = dict()
PV = dict()
PVa = dict()

### initialize summed PV and counter for every month 1-12 and 0 for total
###
for m in range(0,13):
    c[m] = 0
    ca[m] = 0
    PV[m] = 0
    PVa[m] = 0

fp = '/atmosdyn/era5/cdf/2018/09/P20180928_04'
### save only up to around 600 hPa to speed up the process
###
sav = 0
hyam=readcdf(fp,'hyam')
hybm=readcdf(fp,'hybm')
ak=hyam[hyam.shape[0]-98 + sav:]
bk=hybm[hybm.shape[0]-98 + sav:]

for y in range(1979,2021):
    for m in range(1,13):
        fp='/home/era5/cdf/%d/%02d/'%(y,m)
        for d in os.listdir(fp):
            if d.startswith('S%d'%y):
                pv = readcdf(fp+d,'PV')[0,sav:,latids,lonids]
                PS = readcdf(fp+d,'PS')[0,latids,lonids]
                PSM = np.mean(PS)
                p = ak/100.+bk*PSM
                idp = np.where((p<=350) & (p>=150))[0]
                ca[m]+=len(idp)*len(lonids)
                ca[0]+=len(idp)*len(lonids)
                PVa[m]+=np.sum(pv[:,idp])
                PVa[0]+=np.sum(pv[:,idp])
                for q, ps in enumerate(PS):
                    p = ak/100.+bk*ps
                    idp = np.where((p<=350) & (p>=150))[0]
                    c[m]+=len(idp)
                    c[0]+=len(idp)
                    PV[m]+=np.sum(pv[q,idp])
                    PV[0]+=np.sum(pv[q,idp])

SAVE = dict()
SAVE['PV'] = PV
SAVE['count']= c
SAVE['avPV'] = PVa
SAVE['avcount']= ca


f = open('/home/ascherrmann/009-ERA-5/MED/data/climatology-upper-tropo-PV.txt','wb')
pickle.dump(SAVE,f)
f.close()


