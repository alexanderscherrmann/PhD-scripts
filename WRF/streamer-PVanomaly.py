import pandas as pd
import os
import numpy as np
from netCDF4 import Dataset as ds
ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/PV300hPa/'
dp = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
we = 'sevendaypriormature'
wi = 'intense-cyclones.csv'
seasons = ['DJF','SON']
pre = [450]#[250,350,400,450]#,300]

for sea in seasons:
  for pr in pre:
    av = ds(ps + 'PV%dhPa'%pr + sea + '.nc',mode='r')
    av = av.variables['PV.MEAN'][0,0]
    sel = pd.read_csv(dp + sea + '-' + wi)
    for ID in sel['ID'].values:
        for f in os.listdir(ps + '%06d/%d/'%(ID,pr)):
            if f=='ole' or f[-1]=='c' or f[-1]=='g' or f[-1]=='t':
                continue
            d = ds(ps + '%06d/%d/'%(ID,pr) + f,'a')
            #try:
            if np.any(d.variables['PV'][0,0]<-100):
                    d.variables['PV'][0,0]-=av
            #except:
            #    d.close()
            #    os.system('rm %s%06d/%d/%s'%(ps,ID,pr,f))
            #    os.system("clim-e5 %s PV@%dhPa.dump stest.nc"%(f[1:],pr))
            #    os.system("mv %s %s"%(f,"%s%06d/%d/%s"%(ps,ID,pr,f)))
            #    d = ds(ps + '%06d/%d/'%(ID,pr) + f,'a')
            #    d.variables['PV'][0,0]-=av

            d.close()


            



