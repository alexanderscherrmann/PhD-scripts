import pandas as pd
import numpy as np

import pickle

ds = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
cyc = 'DJF-intense-cyclones.csv'

df = pd.read_csv(ds + cyc)
sb = '/atmosdyn/michaesp/pvstreamer.era-5/asc/'
when = []
for k in df.columns:
    if k[-2:]=='re' or k=='dates':
        when.append(k)

# streamer mask
sm = dict()
for we in when:
    sm[we] = dict()
    for ID,d in zip(df['ID'].values,df[we].values):
        sm[we][ID] = dict()
        y = d[:4]
        if y=='1979' or y=='2019' or y=='2020':
            for th in [305, 315, 325, 335]:
                sm[we][ID][th] = np.zeros((361,720))
            continue

        m = d[4:6]
        sdata = np.loadtxt(sb + y + '/' + m + '/STR_' + d,usecols=[5,7,8])
        lat = sdata[:,2]
        lon = sdata[:,1]
        TH = sdata[:,0]

        ### only Norther hemisphere
        ids = np.where(lat>=0)[0]
        lat = lat[ids]; lon = lon[ids]; TH = TH[ids]

        ### only euro atlantic domain
        ids = np.where((lon>=60) & (lon<=240))[0]
        lon = lon[ids]; lat = lat[ids]; TH = TH[ids]

        for th in [305, 315, 325, 335]:
            sm[we][ID][th] = np.zeros((181,720))
            ids = np.where(TH==th)[0]
            loi = (lon[ids]*2).astype(int)
            lai = (lat[ids]*2).astype(int)
            for lo,la in zip(loi,lai):
                sm[we][ID][th][la,lo] +=1

f = open(ds + 'streamer-masks' + cyc,'wb')
pickle.dump(sm,f)
f.close()



