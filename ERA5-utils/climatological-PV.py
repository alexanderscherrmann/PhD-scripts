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

latids1,lonids1 = np.where((xx>=-5) & (xx<2) & (yy>=30) & (yy<=42))# & (LSM[0]==0))
latids2,lonids2 = np.where((xx>=2) & (xx<=42) & (yy>=30) & (yy<48))# & (LSM[0]==0))

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
    PV[m] = 0
    ca[m] = dict()
    PVa[m] = dict()
    for q in range(len(lonids)):
        ca[m]['%.1f,%.1f'%(LON[lonids[q]],LAT[latids[q]])]=0
        PVa[m]['%.1f,%.1f'%(LON[lonids[q]],LAT[latids[q]])]=0


fp = '/atmosdyn/era5/cdf/2018/09/P20180928_04'
### save only up to around 600 hPa to speed up the process
###
sav = 0
hyam=readcdf(fp,'hyam')
hybm=readcdf(fp,'hybm')
ak=hyam[hyam.shape[0]-98 + sav:]
bk=hybm[hybm.shape[0]-98 + sav:]
y0 = 2015
y1 = 2020
for y in range(y0,y1+1):
    for m in range(1,13):
        if y==2020 and m>=7:
            continue
        fp='/home/era5/cdf/%d/%02d/'%(y,m)
        maskp = '/atmosdyn/michaesp/mincl.era-5/cdf.final/%d/%02d/'%(y,m)
        for d in os.listdir(fp):
            if d.startswith('S%d'%y):
                pv = readcdf(fp+d,'PV')[0,sav:,latids,lonids]
                PS = readcdf(fp+d,'PS')[0,latids,lonids]
#                mask = readcdf(maskp+'C' + d[1:],'LABEL')[0,0,latids,lonids] 
                for q, ps in enumerate(PS):
#                    if mask[q]!=0:
#                        continue
                    p = ak/100.+bk*ps
                    idp = np.where((p<=975) & (p>=700))[0]
                    c[m]+=len(idp)
                    c[0]+=len(idp)
                    PV[m]+=np.sum(pv[q,idp])
                    PV[0]+=np.sum(pv[q,idp])
                    # 'LON,LAT'
                    ca[m]['%.1f,%.1f'%(LON[lonids[q]],LAT[latids[q]])]+=len(idp)
                    PVa[m]['%.1f,%.1f'%(LON[lonids[q]],LAT[latids[q]])]+=np.sum(pv[q,idp])

SAVE = dict()
SAVE['PV'] = PV
SAVE['count']= c
SAVE['locPV']=PVa
SAVE['loccount']=ca

f = open('/home/ascherrmann/009-ERA-5/MED/climatologyPV-w-cyclones-%d-%d.txt'%(y0,y1),'wb')
pickle.dump(SAVE,f)
f.close()


