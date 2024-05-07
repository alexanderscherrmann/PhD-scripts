import numpy as np
import pandas as pd
from netCDF4 import Dataset as ds
from wrf import interplevel as intp
import os

ps = '/atmosdyn2/ascherrmann/013-WRF-sim/data/'
sel = pd.read_csv(ps + 'DJF-intense-cyclones.csv')

era5 = '/atmosdyn2/era5/cdf/'


LON = np.linspace(-180,180,721)
LAT = np.linspace(-90,90,361)
minlon = -10
minlat = 30
maxlat = 55
maxlon = 50
lons = np.where((LON>=minlon) & (LON<=maxlon))[0]
lats = np.where((LAT<=maxlat) & (LAT>=minlat))[0]

lo0,lo1,la0,la1 = lons[0],lons[-1]+1,lats[0],lats[-1]+1
lonshort = np.linspace(-10,50,121)
latshort = np.linspace(30,55,51)

for ID,d in zip(sel['ID'].values,sel['dates'].values):
    y = d[:4]
    m = d[4:6]
    ep = era5 +  y + '/' + m + '/'
    S = ds(ep + 'S' + d,mode='r')
    PV = S.variables['PV'][0,:,la0:la1,lo0:lo1]
    hyam=S.variables['hyam']  # 137 levels  #für G-file ohne levels bis
    hybm=S.variables['hybm']
    ak=hyam[hyam.shape[0]-98:] # only 98 levs are used:
    bk=hybm[hybm.shape[0]-98:]

    PS = S.variables['PS'][0,la0:la1,lo0:lo1]
    ps3d=np.tile(PS[:,:],(len(ak),1,1))
    Pr=(ak/100.+bk*ps3d.T).T
    pv300 = intp(PV,Pr,300,meta=False)
    loni = np.where(pv300>=4)[1]
    lati = np.where(pv300>=4)[0]

    save = np.zeros((len(loni),4))
    save[:,3] = 300
    save[:,1] = lonshort[loni]
    save[:,2] = latshort[lati]

    if not os.path.isdir(ps + 'PV-med-traj/'):
        os.mkdir(ps + 'PV-med-traj/')

    np.savetxt(ps + 'PV-med-traj/trajectories-' + d + '-ID-%06d.txt'%ID,save,delimiter=' ',fmt='%f',newline='\n')

