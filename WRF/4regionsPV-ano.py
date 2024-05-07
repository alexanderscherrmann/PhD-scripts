import pandas as pd
import os
import numpy as np
from netCDF4 import Dataset as ds

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')  
df = df.iloc[np.where(df['reg'].values=='MED')[0]]
IDs = df['ID'].values
months = df['months'].values

seasons = ['DJF','SON']
mo = [np.array([12,1,2]),np.array([9,10,11])]
#pre = [250,350,400,450]
pre = [250]
for q,sea in enumerate(seasons):
  for pr in pre:
    av = ds(dp + 'PV%dhPa'%pr + sea + '.nc',mode='r')
    av = av.variables['PV.MEAN'][0,0]

    for ID in os.listdir(ps):
        if ID[-1]=='t' or ID[-1]=='c' or ID[-1]=='g':
            continue

        loc = np.where(IDs==int(ID))[0][0]
        MO = months[loc]
        if not np.any(mo[q]==MO):
            continue

        for f in os.listdir(ps + '%06d/%d/'%(int(ID),pr)):
            if f=='ole' or f[-1]=='c' or f[-1]=='g' or f[-1]=='t' or f[0]=='M':
                continue

            if f[0]=='D':
                d = ds(ps + '%06d/%d/'%(int(ID),pr) + f,'a')
                if np.any(d.variables['PV'][0,0]<-100):
                        d.variables['PV'][0,0]-=av
                d.close()




