import numpy as np
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
from numpy.random import randint as ran
import pandas as pd
import pickle


ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
PD = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'
cd = '/atmosdyn/michaesp/mincl.era-5/cdf.final/'


datesDJF = np.loadtxt(ps + 'draw-dates-from-DJF.txt',dtype=str)
datesSON = np.loadtxt(ps + 'draw-dates-from-SON.txt',dtype=str)

dfD = pd.read_csv(ps + 'DJF-intense-cyclones.csv')
dfS = pd.read_csv(ps + 'SON-intense-cyclones.csv')

f = open(ps + 'DJF-boot-strap-randomids.txt','rb')
IDSD = pickle.load(f)
f.close()

f = open(ps + 'SON-boot-strap-randomids.txt','rb')
IDSS = pickle.load(f)
f.close()


avcycD = dict()
avcycS = dict()

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


minlon = -120
minlat = 10
maxlat = 80
maxlon = 80

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)

lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1


for q,l in enumerate(DJFcluster):
    nc = ncD[q]
    avcycD[l] = np.zeros((1000,len(lats),len(lons)))

    for k in range(1000):
        avC = np.zeros((len(lats),len(lons)))
        for n in IDSD[l][k]:
            d = datesDJF[n]
            y = d[:4]
            m = d[4:6]

            cyc = ds(cd + y +'/' + m + '/C' + d,mode='r')
            mask = cyc.variables['LABEL'][0,0,la0:la1,lo0:lo1]
            avC[mask!=0] +=1

        avcycD[l][k] += avC/nc


for q,l in enumerate(SONcluster):
    nc = ncS[q]
    avcycS[l] = np.zeros((1000,len(lats),len(lons)))

    for k in range(1000):
        avC = np.zeros((len(lats),len(lons)))
        for n in IDSS[l][k]:
            d = datesSON[n]
            y = d[:4]
            m = d[4:6]

            cyc = ds(cd + y +'/' + m + '/C' + d,mode='r')
            mask = cyc.variables['LABEL'][0,0,la0:la1,lo0:lo1]
            avC[mask!=0] +=1

        avcycS[l][k] += avC/nc


f = open(ps + 'DJF-boot-strap-cycmask-data.txt','wb')
pickle.dump(avcycD,f)
f.close()

f = open(ps + 'SON-boot-strap-cycmask-data.txt','wb')
pickle.dump(avcycS,f)
f.close()




