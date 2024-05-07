import numpy as np
import pandas as pd
from wrf import interplevel as intp
from netCDF4 import Dataset as ds
import pickle
import os

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
ep = '/atmosdyn2/ascherrmann/009-ERA-5/MED/data/'
we = 'dates'

minlon = -10
minlat = 30
maxlat = 50
maxlon = 45

LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1

npo = len(lons)*len(lats)
seasons = ['DJF','SON']
save = dict()

wi = 'intense-cyclones.csv'
if not os.path.isfile(ps + 'PV-streamer-masks-DJF-SON.txt'):
    for sea in seasons:
        save[sea] = dict()
        sel = pd.read_csv(ps + sea + '-' + wi)
        
        for q,d,ID in zip(range(200),sel[we].values,sel['ID'].values):
            save[sea][ID] = np.zeros((len(lats),len(lons)))
            
            S = ds(ep + 'S' + d,mode='r')
            
            PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
            PS = S.variables['PS'][0,la0:la1,lo0:lo1]
            hyam=S.variables['hyam']  # 137 levels  #für G-file ohne levels bis
            hybm=S.variables['hybm']  #   ''
            ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
            bk=hybm[hybm.shape[0]-98:]
            ps3d=np.tile(PS[:,:],(len(ak),1,1))
            Pr=(ak/100.+bk*ps3d.T).T
            PV = intp(PV,Pr,300,meta=False)
            save[sea][ID][PV>=5] +=1
    
    f = open(ps + 'PV-streamer-masks-DJF-SON.txt','wb')
    pickle.dump(save,f)
    f.close()

f = open(ps + 'PV-streamer-masks-DJF-SON.txt','rb')
data = pickle.load(f)
f.close()

streamercat = dict()
streamercount = dict()
for sea in seasons[:1]:
    streamercat[sea] = dict()
    streamercount[sea] = dict()
    df = pd.read_csv(ps + sea + '-intense-cyclones.csv')
    nc = np.array([])
    for cl in np.unique(df['region'].values):
        nc = np.append(nc,len(np.where(df['region'].values==cl)[0]))
    clus = np.unique(df['region'].values)[np.argsort(nc)[-6:]]
    nc = nc[np.argsort(nc)[-6:]].astype(int)

    for wl,l in enumerate(clus):
        streamercat[sea][l]=dict()
        streamercount[sea][l] = dict()
        for q,ID in zip(range(nc[wl]),df['ID'].values[np.where(df['region'].values==l)[0]]):
            if q == 0:
                streamercat[sea][l][0] = data[sea][ID]
                streamercount[sea][l][0] = 1
                continue

            ov = np.array([])
            for k in list(streamercat[sea][l].keys()):
                tmpa = data[sea][ID]+streamercat[sea][l][k]
                ov = np.append(ov,len(np.where(tmpa==2)[0]))

            if np.any(ov/npo>=0.4):
                wq = np.where(ov/npo>=0.6)[0][0]
                streamercount[sea][l][wq] +=1

            else:
                streamercat[sea][l][len(list(streamercat[sea][l].keys()))] = data[sea][ID]
                streamercount[sea][l][len(list(streamercat[sea][l].keys()))] = 1

        print(nc[wl],len(list(streamercount[sea][l])))
saven = dict()
saven['cat'] = streamercat
saven['count'] = streamercount

f = open(ps + 'streamer-categories-counts.txt','wb')
pickle.dump(saven,f)
f.close()











