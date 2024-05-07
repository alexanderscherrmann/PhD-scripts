import numpy as np
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
from numpy.random import randint as ran
import pandas as pd
import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
PD = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'

datesDJF = np.loadtxt(ps + 'draw-dates-from-DJF.txt',dtype=str)
datesSON = np.loadtxt(ps + 'draw-dates-from-SON.txt',dtype=str)

dfD = pd.read_csv(ps + 'DJF-intense-cyclones.csv')
dfS = pd.read_csv(ps + 'SON-intense-cyclones.csv')

ncD = np.array([])
ncS = np.array([])

#### get the 6 most abundant clusters in terms of number of cyclones
for cl in np.unique(dfD['region'].values):
    ncD = np.append(ncD,len(np.where(dfD['region'].values==cl)[0]))

for cl in np.unique(dfS['region'].values):
    ncS = np.append(ncS,len(np.where(dfS['region'].values==cl)[0]))

DJFcluster = np.unique(dfD['region'].values)[np.argsort(ncD)[-6:]]
SONcluster = np.unique(dfS['region'].values)[np.argsort(ncS)[-6:]]
ncD = ncD[np.argsort(ncD)[-6:]]
ncS = ncS[np.argsort(ncS)[-6:]]


### plotting region
minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1


### pre do some interpolation stuff
P = ds(PD + 'P20101010_10',mode='r')
hyam=P.variables['hyam']  # 137 levels  #für G-file ohne levels bis
hybm=P.variables['hybm']  #   ''
ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
bk=hybm[hybm.shape[0]-98:]


### dictionaries to save
avDJFU = dict()
avDJFT = dict()
avSONU = dict()
avSONT = dict()

### for reproducing the random number stuff
IDS = dict()
IDs = dict()


for q,l in enumerate(DJFcluster):
    avDJFU[l] = np.zeros((1000,len(lats),len(lons)))
    avDJFT[l] = np.zeros((1000,len(lats),len(lons)))

    IDS[l] = dict()
    nD = ncD[q].astype(int)
    
    for k in range(1000):
        IDS[l][k] = np.zeros(nD,dtype=int)
        avDU = np.zeros((len(lats),len(lons)))
        avDT = np.zeros((len(lats),len(lons)))

        for n in range(nD):
            ind = ran(len(datesDJF))
            IDS[l][k][n] = ind
            ### respect sample size

            P = ds(PD + 'P' + datesDJF[ind],mode='r')
            S = ds(PD + 'S' + datesDJF[ind],mode='r')

            PS = S.variables['PS'][0,la0:la1,lo0:lo1]
            TH = S.variables['TH'][0,:,la0:la1,lo0:lo1]
            U = P.variables['U'][0,:,la0:la1,lo0:lo1]
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            Pr = (ak/100.+bk*ps3d.T).T
            
            U = intp(U,Pr,300,meta=False)
            TH = intp(TH,Pr,850,meta=False)

            avDU += U
            avDT += TH

        avDU/=nD
        avDT/=nD
        avDJFU[l][k] = avDU
        avDJFT[l][k] = avDT

### repeat for SON

for q,l in enumerate(SONcluster):
    avSONU[l] = np.zeros((1000,len(lats),len(lons)))
    avSONT[l] = np.zeros((1000,len(lats),len(lons)))
    IDs[l] = dict()

    nS = ncS[q].astype(int)

    for k in range(1000):

        IDs[l][k] = np.zeros(nS,dtype=int)
        avSU = np.zeros((len(lats),len(lons)))
        avST = np.zeros((len(lats),len(lons)))

        for n in range(nS):

            ins = ran(len(datesSON))
            IDs[l][k][n] = ins
            ### respect sample size

            P = ds(PD + 'P' + datesSON[ins],mode='r')
            S = ds(PD + 'S' + datesSON[ins],mode='r')

            PS = S.variables['PS'][0,la0:la1,lo0:lo1]
            TH = S.variables['TH'][0,:,la0:la1,lo0:lo1]
            U = P.variables['U'][0,:,la0:la1,lo0:lo1]
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            Pr = (ak/100.+bk*ps3d.T).T

            U = intp(U,Pr,300,meta=False)
            TH = intp(TH,Pr,850,meta=False)

            avSU += U
            avST += TH

        avSU/=nS
        avST/=nS
        avSONU[l][k] = avSU
        avSONT[l][k] = avST



f = open(ps + 'DJF-boot-strap-U-data.txt','wb')
pickle.dump(avDJFU,f)
f.close()

f = open(ps + 'DJF-boot-strap-T-data.txt','wb')
pickle.dump(avDJFT,f)
f.close()

f = open(ps + 'SON-boot-strap-U-data.txt','wb')
pickle.dump(avSONU,f)
f.close()

f = open(ps + 'SON-boot-strap-T-data.txt','wb')
pickle.dump(avSONT,f)
f.close()

f = open(ps + 'DJF-boot-strap-randomids.txt','wb')
pickle.dump(IDS,f)
f.close()

f = open(ps + 'SON-boot-strap-randomids.txt','wb')
pickle.dump(IDs,f)
f.close()
