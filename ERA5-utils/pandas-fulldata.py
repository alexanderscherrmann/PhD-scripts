import numpy as np
import pickle
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

import argparse
import pandas as pd
import netCDF4

def readcdf(ncfile,varnam):
    infile = netCDF4.Dataset(ncfile, mode='r')
    var = infile.variables[varnam][:]
    return(var)


parser = argparse.ArgumentParser(description="plot accumulated average PV gain that is associated with the cyclone and the environment")
parser.add_argument('rdis',default='',type=int,help='distance from center in km that should be considered as cyclonic')
args = parser.parse_args()
rdis = int(args.rdis)

pload = '/home/ascherrmann/009-ERA-5/MED/ctraj/use/'
pload2 = '/home/ascherrmann/009-ERA-5/MED/traj/use/'
f = open(pload + 'PV-data-dPSP-100-ZB-800-2-%d-correct-distance.txt'%rdis,'rb')
PVdata = pickle.load(f)
f.close()

dipv = PVdata['dipv']
datadi = PVdata['rawdata']
dit = PVdata['dit']

savings = ['SLP-', 'lon-', 'lat-', 'ID-', 'hourstoSLPmin-', 'dates-']
var = []

for u,x in enumerate(savings):
    f = open(pload2[:-4] + x + 'furthersel.txt',"rb")
    var.append(pickle.load(f))
    f.close()

SLP = var[0]
lon = var[1]
lat = var[2]
ID = var[3]
hourstoSLPmin = var[4]
dates = var[5]

c = 'cyc'
e = 'env'
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
    mlon = np.append(mlon,lon[k][loc])
    mlat = np.append(mlat,lat[k][loc])
    minSLP = np.append(minSLP,SLP[k][loc])

    
pvgain = np.array([])
pvgainc = np.array([])
pvgaine = np.array([])
pvadv = np.array([])

cycper = np.array([])
envper = np.array([])
advper = np.array([])

saveID = np.array([])
slon= np.array([])
slat= np.array([])
sdate= np.array([])
shtSLPmin= np.array([])
sminSLP= np.array([])

PVs = np.array([])
PVs075 = np.array([])
tras = np.array([])
tras075 =np.array([])

LON = np.arange(-180,180,0.5)
LAT = np.arange(-90,90.1,0.5)

for ll,k in enumerate(dipv.keys()):

    q = np.where(avaID==int(k))[0][0]
    saveID = np.append(saveID,avaID[q])
    slon = np.append(slon,mlon[q])
    slat = np.append(slat,mlat[q])
    sdate = np.append(sdate,maturedates[q])
    shtSLPmin = np.append(shtSLPmin,htSLPmin[q])
    sminSLP = np.append(sminSLP,minSLP[q])

    d = k
    PV = PVdata['rawdata'][d]['PV']
    i = np.where(PV[:,0]>=0.75)[0]

    pvend = PV[i,0]
    pvstart = PV[i,-1]

    pvadv = np.append(pvadv,np.mean(pvstart))
    cypv = dipv[d][c][i,0]
    pvgainc = np.append(pvgainc,np.mean(cypv))
    cycper = np.append(cycper,np.mean(cypv/pvend))
    enpv = dipv[d][e][i,0]
    pvgaine = np.append(pvgaine,np.mean(enpv))
    envper = np.append(envper,np.mean(enpv/pvend))
    advper = np.append(advper,np.mean(pvstart/pvend))

    pvgain = np.append(pvgain,np.mean(pvend-pvstart))

    dat = maturedates[q]
    lo = mlon[q]
    la = mlat[q]

    y = dat[:4]
    m = dat[4:6]

    lonids,latids = helper.IFS_radial_ids_correct(200,la)
    clon = (np.where(abs(LON-lo)==np.min(abs(LON-lo)))[0][0] + lonids).astype(int)
    clat = (np.where(abs(LAT-la)==np.min(abs(LAT-la)))[0][0] + latids).astype(int)

    pa = '/home/era5/cdf/' + y + '/' + m  + '/'
    sf = pa +'S' + dat

    PS=readcdf(sf,'PS')
    PS = PS[0,clat,clon]
    pv = readcdf(sf,'PV')
    pv = pv[0]
    hyam=readcdf(sf,'hyam')  # 137 levels  #für G-file ohne levels bis
    hybm=readcdf(sf,'hybm')

    PVsum = 0
    PVsum075 = 0
    ntra = 0
    ntra075 =0
    for l,cla,clo in zip(np.arange(len(clat)),clat,clon):
        P = 0.01 * hyam[137-98:] + hybm[137-98:] * PS[l]
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


    
df = pd.DataFrame(columns = ['ID','date','lon','lat','htminSLP','minSLP','PVsum','PV075sum','ntraj','ntraj075','PVgain','cycPV','envPV','advPV','cycper','envper','advper'])

df['ID'] = saveID
df['date'] = sdate.astype(str)
df['lon'] = slon
df['lat'] = slat
df['htminSLP'] = shtSLPmin
df['minSLP'] = sminSLP

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

df['PVgain'] = pvgain
df['cycPV'] = pvgainc
df['envPV'] = pvgaine
df['advPV'] = pvadv

df['PVgain'] = df['PVgain'].astype('float32')
df['cycPV'] = df['cycPV'].astype('float32')
df['envPV'] = df['envPV'].astype('float32')
df['advPV'] = df['advPV'].astype('float32')

df['cycper'] = cycper
df['envper'] = envper
df['advper'] = advper

df['cycper'] = df['cycper'].astype('float32')
df['envper'] = df['envper'].astype('float32')
df['advper'] = df['advper'].astype('float32')

df.to_csv('/home/ascherrmann/009-ERA-5/MED/traj/pandas-all-data.csv',index=False)

