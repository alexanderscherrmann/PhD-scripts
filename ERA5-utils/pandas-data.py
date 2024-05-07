import pandas as pd
import numpy as np
import pickle
import os
import fnmatch
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper
import xarray as xr


pload = '/home/ascherrmann/009-ERA-5/MED/traj/'
psave =pload

### raw data
###
savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()
SLP = var[0]
Clon = var[1]
Clat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
HourstoSLPmin = var[4]
dates = var[5]

### extract mature stage data
###
avaID = np.array([])
maturedates = np.array([])
htSLPmin = np.array([])
mlon = np.array([])
mlat = np.array([])
minSLP = np.array([])
for k in range(len(ID)):
    avaID=np.append(avaID,ID[k][0].astype(int))
    loc = np.where(hourstoSLPmin[k]==0)[0][0]
    maturedates = np.append(maturedates,dates[k][loc])
    htSLPmin = np.append(htSLPmin,abs(hourstoSLPmin[k][0]).astype(int))
    mlon = np.append(mlon,Clon[k][loc])
    mlat = np.append(mlat,Clat[k][loc])
    minSLP = np.append(minSLP,SLP[k][loc])

### create early arrray
###
df = pd.DataFrame(columns= ['ID','dates','lon','lat','htminSLP','minSLP'])
df['ID'] = avaID
df['dates'] = maturedates
df['lon'] = mlon
df['lat'] = mlat
df['htminSLP'] = htSLPmin
df['minSLP'] = minSLP

LON = np.arange(-180,180,0.5)
LAT = np.arange(-90,90.1,0.5)

###
PVs = np.array([])
PVs075 = np.array([])
tras = np.array([])
tras075 =np.array([])

for dat, lo, la in zip(maturedates,mlon,mlat):

    y = dat[:4]
    m = dat[4:6]

    lonids,latids = helper.IFS_radial_ids_correct(200,la)
    clon = (np.where(abs(LON-lo)==np.min(abs(LON-lo)))[0][0] + lonids).astype(int)
    clat = (np.where(abs(LAT-la)==np.min(abs(LAT-la)))[0][0] + latids).astype(int)

    pa = '/home/era5/cdf/' + y + '/' + m  + '/'
    sf = pa +'S' + dat
    s = xr.open_dataset(sf)

    PS = s.PS.values[0,clat,clon]
    pv = s.PV.values[0]

    PVsum = 0
    PVsum075 = 0
    ntra = 0
    ntra075 =0
    for l,cla,clo in zip(np.arange(len(clat)),clat,clon):
        P = 0.01 * s.hyam.values[137-98:] + s.hybm.values[137-98:] * PS[l]
        pid = np.where((P>=700) & (P<=975) & (pv[:,cla,clo]>=0.75))[0]
        pid2= np.where((P>=700) & (P<=975))[0]
        for i in pid2:
            PVsum+=pv[i,cla,clo]
            ntra+=1
            if pv[i,cla,clo]>=0.75:
                PVsum075+=pv[i,cla,clo]
                ntra075+=1

    PVs = np.append(PVs,PVsum)
    PVs075 = np.append(PVs075,PVsum075)
    tras = np.append(tras,ntra)
    tras075 = np.append(tras075,ntra075)

df['PVsum'] = PVs
df['PV075sum'] = PVs075
df['ntraj'] = tras
df['ntraj075'] = tras075

df['ID'] = df['ID'].astype('float32').astype('int32')
df['htminSLP'] = df['htminSLP'].astype('float32').astype('int32')
df['lon'] = df['lon'].astype('float32')
df['lat'] = df['lat'].astype('float32')
df['minSLP'] = df['minSLP'].astype('float32')
df['PVsum'] = df['PVsum'].astype('float32')
df['PV075sum'] = df['PV075sum'].astype('float32')
df['ntraj'] = df['ntraj'].astype('float32').astype('int32')
df['ntraj075'] = df['ntraj075'].astype('float32').astype('int32')

df.to_csv(psave + 'pandas-ERA5-basic-data.csv',index=False)

