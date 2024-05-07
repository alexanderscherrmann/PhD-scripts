import numpy as np
from netCDF4 import Dataset as ds
from wrf import interplevel as intp
import pickle

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
pi = '/atmosdyn2/ascherrmann/013-WRF-sim/image-output/'
era5 = '/atmosdyn2/era5/cdf/'

f = open(ps + 'WR-days-24h-separated-for-average.txt','rb')
data = pickle.load(f)
f.close()
lon = np.linspace(-180,180,721)
lat = np.linspace(-90,90,361)

LON = np.linspace(-150,80,461)
LAT = np.linspace(-20,80,201)

lo = np.where((lon>=LON[0]) & (lon<=LON[-1]))[0]
la = np.where((lat>=LAT[0]) & (lat<=LAT[-1]))[0]
lo0,lo1,la0,la1 = lo[0],lo[-1]+1,la[0],la[-1]+1

pres = pres = np.array(['1', '2', '3',
            '5', '7', '10',
            '20', '30', '50',
            '70', '100', '125',
            '150', '175', '200',
            '225', '250', '300',
            '350', '400', '450',
            '500', '550', '600',
            '650', '700', '750',
            '775', '800', '825',
            '850', '875', '900',
            '925', '950', '975',
            '1000'])
pres = pres.astype(int)

varP = ['U','T','V']
varS = ['RH']
varH = ['Z']
varB = ['U10M','V10M','T2M','PS','MSL','D2M','SSTK']
fi = ['B','P','S','H']
var = [varB,varP,varS,varH]
sdi = dict()

for r in data.keys():
  sdi[r] = dict()
  for f,Var in zip(fi,var):
    for va in Var:
        if f=='B':
            sdi[r][va] = np.zeros((len(LAT),len(LON)))
        else:
            sdi[r][va] = np.zeros((len(pres),len(LAT),len(LON)))

PF = '/atmosdyn/era5/cdf/2018/09/P20180928_04'
PFD = ds(PF,mode='r')
hyam=PFD.variables['hyam'][:]
hybm=PFD.variables['hybm'][:]
ak=hyam[hyam.shape[0]-98:]
bk=hybm[hybm.shape[0]-98:]

for r in data.keys():
    for d in data[r]:
        path = era5 + d[:4] + '/' + d[4:6] + '/'
        for f,Var in zip(fi,var):
            if f=='B':
                B = ds(path + f + d)
                for va in Var:
                    tmpb = B.variables[va][0]
                    sdi[r][va] +=tmpb[la0:la1,lo0:lo1]

                PS = B.variables['PS'][0]

            elif f=='P' or f=='S':
                ps3d=np.tile(PS[:,:],(len(ak),1,1)) # write/repete ps to each level of dim 0
                p3d=(ak/100.+bk*ps3d.T).T
                P = ds(path + f + d)

                for va in Var:
                    for q,pre in enumerate(pres):
                        tmps = intp(P.variables[va][0],p3d,pre,meta=False)
                        sdi[r][va][q] += tmps[la0:la1,lo0:lo1]

            else:
                for va in Var:
                    H = ds(path + f + d)
                    sdi[r][va] += H.variables[va][0,:,la0:la1,lo0:lo1]

f = open(ps + 'average-conditions-WR.txt','wb')
pickle.dump(sdi,f)
f.close()
