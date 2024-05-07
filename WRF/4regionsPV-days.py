import pandas as pd
import os
import pickle
import numpy as np
import sys
sys.path.append('/home/ascherrmann/scripts/')
import helper

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/4regionsPV/'
dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
seasons = ['DJF','SON']

df = pd.read_csv('/atmosdyn2/ascherrmann/011-all-ERA5/data/pandas-basic-data-all-deep-over-sea-12h.csv')
df = df.iloc[np.where(df['reg'].values=='MED')[0]]

dID = df['ID'].values
htminSLP = df['htSLPmin'].values
mdates = df['dates'].values

f = open(ps + '100-region-season.txt','rb')
dd = pickle.load(f)
f.close()


for lev in [450]: #[250,350,400,450]:
    for sea in seasons:
        for k in dd[sea].keys():
            for ID in dd[sea][k]:
                loc = np.where(dID==ID)[0][0]

                if not os.path.isdir(ps + '%06d'%ID):
                    os.mkdir(ps + '%06d'%ID)
                if not os.path.isdir(ps + '%06d'%ID + '/%d'%lev):
                    os.mkdir(ps + '%06d'%ID + '/%d'%lev)

                startdate = helper.change_date_by_hours(mdates[loc],-1 * htminSLP[loc] + 6 - 120)
                
                for k in range(65):
                    d = helper.change_date_by_hours(startdate,k*3)
                    if os.path.isfile(ps + '%06d/%d/D%s'%(ID,lev,d)):
                        continue
                    os.chdir(ps + '%06d/%d/'%(ID,lev))
                    os.system("clim-e5 %s PV@%dhPa.dump test.nc"%(d,lev))


